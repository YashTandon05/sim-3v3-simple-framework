"""Player: the control handle for a single robot — users can edit this class directly.

Platform primitives (set_velocity / kick / release_kick / request_mode /
get_up) are delegated to the injected ``_backend`` (framework layer); state
properties (pose / mode / is_fallen / penalty) are read from ``self.context``
or the backend.

Movement behaviors (walk_to / face_to / ensure_ready) are also Player
methods: they're fundamentally "verbs issued to this player," and any
cross-frame state needed for future hysteresis/avoidance can hang directly
off ``self``. Pure coordinate math (dist / angle_to / goal coordinates etc.)
belongs in utils/geom instead.

Instances live for the whole match; ``self.context`` is overwritten by the
framework every frame. To add your own custom moves, add them directly here.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .framework.types import Context, Penalty, Pose2D

from .param import *
from .utils.geom import (
    angle_to,
    clamp,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
)
from .utils.obstacles import collect_obstacles
from .utils.path_planner import plan_global_path
from .utils.tactics import (
    adaptive_outlet_ahead,
    attacking_outlet_spot,
    best_pass_target,
    best_shot,
    carry_direction,
    escape_direction,
    nearest_opponent_dist,
    opponents_overcommitted,
    quick_pass_target,
)
from .utils.worldmodel import goal_line_crossing

if TYPE_CHECKING:
    from .framework.config import SoccerConfig
    from .utils.worldmodel import BallEstimate


__all__ = ["Player"]


_log = logging.getLogger(__name__)

def _heading_clearance(
    px: float, py: float, heading: float, obstacles: list,
) -> float:
    """Used for local obstacle avoidance. Along the lookahead ray from (px,py)
    toward ``heading``, the clearance (dist - radius) to the nearest obstacle.

    Only considers obstacles **ahead** (projection t>0); obstacles behind/to
    the side don't block this direction and are skipped — otherwise being
    close to any obstacle (even directly behind) would drag down the
    clearance for every direction and falsely signal "fully blocked."
    Returns inf when there's no obstacle; larger = more open, negative means
    a collision.
    """
    ux, uy = math.cos(heading), math.sin(heading)
    min_clear = math.inf
    for obs in obstacles:
        t = (obs.x - px) * ux + (obs.y - py) * uy
        if t <= 0.0:
            continue                      # Obstacle is behind/to the side, doesn't block this direction
        if t > PLAN_LOOKAHEAD:
            t = PLAN_LOOKAHEAD
        nx, ny = px + ux * t, py + uy * t
        clear = math.hypot(obs.x - nx, obs.y - ny) - obs.radius
        if clear < min_clear:
            min_clear = clear
    return min_clear


def _behind_ball(
    ball_x: float, ball_y: float, aim: tuple[float, float], offset: float,
) -> tuple[float, float]:
    """Used for computing a positioning target. The point "behind" the ball on
    the ball->``aim`` line: retreat ``offset`` meters from the ball, away from aim.

    Used as the chase target while pursuing the ball: the ball ends up
    between the robot and ``aim``, so on arrival the robot is naturally
    lined up to shoot toward aim. When aim coincides with the ball
    (degenerate case), direction is undefined, so just return the ball's
    position.
    """
    dx, dy = aim[0] - ball_x, aim[1] - ball_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (ball_x, ball_y)
    ux, uy = dx / d, dy / d              # Unit vector from ball toward aim
    return (ball_x - ux * offset, ball_y - uy * offset)   # Retreat offset, away from aim


class Player:
    """Handle for a single player. Add new technical moves here.
    """

    def __init__(
        self,
        player_id: int,
        config: "SoccerConfig",
        _backend: object | None,
    ) -> None:
        self.id: int = player_id
        self.config: "SoccerConfig" = config
        self._backend = _backend           # Wrapper around the robot control interface, usually no need to change
        self.context: Context | None = None

        # Current high-level action name (for visualization/debugging only).
        # Written every frame by the strategy dispatch layer; actions with
        # sub-states (like guard) refine it internally. The visualization
        # pass reads this to label markers, see main.py.
        self.action: str = "init"

        # SDK-cached fields, updated automatically by the framework in the background
        self._mode: str | None = None
        self._fall_down_state: str | None = None

        # Avoidance detour-side memory (cross-frame; None = not currently detouring)
        self._avoid_side: float | None = None

        # Kick hysteresis state (cross-frame)
        self._kicking: bool = False

        # block/guard cross-frame hysteresis state
        self._block_pressing: bool = False
        self._guard_threatened: bool = False
        self._keeper_claiming: bool = False   # keeper sweep/claim hysteresis
        # This frame's kick decision (shoot/pass/clear/carry) or None while
        # chasing. Read by the dispatch to coordinate receiver / rebound crash.
        self._kick_intent: str | None = None
        # Goal-line ball-push hysteresis state
        self._goal_line_push: bool = False
        # Cross-frame state for support temporarily switching to attack when stuck
        self._support_last_pos: tuple[float, float] | None = None
        self._support_stationary_since: float | None = None
        self._support_last_update_at: float | None = None

    # ------------------------------------------------------------------
    # State readers
    # ------------------------------------------------------------------

    @property
    def is_kicking(self) -> bool:
        """Whether currently in the kicking state."""
        return self._kicking

    @property
    def pose(self) -> Pose2D | None:
        ctx = self.context
        if ctx is None:
            return None
        robot = ctx.teammates.get(self.id)
        return None if robot is None else robot.pose

    @property
    def mode(self) -> str | None:
        if self._backend is not None:
            return self._backend.mode
        return self._mode

    @property
    def is_fallen(self) -> bool:
        return self.fall_down_state not in (None, "normal")

    @property
    def fall_down_state(self) -> str | None:
        if self._backend is not None:
            return getattr(self._backend, "fall_down_state", None)
        return self._fall_down_state

    @property
    def penalty(self) -> Penalty:
        ctx = self.context
        if ctx is None or ctx.game is None:
            return Penalty.NONE
        state = ctx.game.get_player_state(self.config.team_id, self.id)
        return Penalty.NONE if state is None else state.penalty

    @property
    def is_penalized(self) -> bool:
        return self.penalty != Penalty.NONE

    # ------------------------------------------------------------------
    # Chassis control
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if self._backend is None:
            _log.debug(
                "player %d set_velocity vx=%.3f vy=%.3f vyaw=%.3f (no backend)",
                self.id, vx, vy, vyaw,
            )
            return
        self._backend.set_velocity(vx, vy, vyaw)

    def stop(self) -> None:
        self.release_kick()
        self.set_velocity(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Kicking
    # ------------------------------------------------------------------

    def kick(
        self,
        kick_direction: float | None = None,
        power: float = KICK_POWER_DEFAULT,
    ) -> None:
        pose = self.pose
        if pose is None:
            _log.warning("player %d kick skipped: pose unknown", self.id)
            return

        ball = self.context.ball if self.context is not None else None
        if ball is None:
            _log.warning("player %d kick skipped: ball unknown", self.id)
            return

        if kick_direction is None:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                _log.warning("player %d kick skipped: kick plan unavailable", self.id)
                return
            kick_direction, power = kick_plan

        if self._backend is None:
            _log.debug(
                "player %d kick ball=(%.3f, %.3f) dir=%.3f (no backend)",
                self.id, ball.x, ball.y, kick_direction,
            )
            return
        # Field coordinates -> body coordinates (using current pose)
        dx = ball.x - pose.x
        dy = ball.y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        ball_x_body = dx * cos_t + dy * sin_t
        ball_y_body = -dx * sin_t + dy * cos_t
        kick_direction = normalize_angle(kick_direction)
        direction_body = normalize_angle(kick_direction - pose.theta)
        power_clamped = max(KICK_POWER_MIN, min(KICK_POWER_MAX, power))
        self._kicking = True
        self._backend.kick(direction_body, power_clamped, ball_x_body, ball_y_body)

    def plan_kick(self) -> tuple[float, float] | None:
        """Compute kick direction and power.

        Kicks from the current ball position toward the center of the
        opponent's goal, power 2.0. Returns ``(kick_direction, kick_power)``;
        returns None when the ball or context is unavailable.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None

        kick_target = opponent_goal(ctx)
        kick_direction = angle_to(ball.x, ball.y, *kick_target)
        kick_target = self._goal_target_for_direction(kick_direction)
        kick_power = (
            KICK_POWER_BACKFIELD if self._in_backfield()
            else KICK_POWER_DEFAULT
        )

        self._draw_kick_target(kick_target)
        return kick_direction, kick_power

    def _goal_target_for_direction(
        self, kick_direction: float,
    ) -> tuple[float, float]:
        """Project the shot direction onto the opponent's goal line, for
        visualizing the kick target."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return (0.0, 0.0)

        dx = math.cos(kick_direction)
        if dx <= 1e-6:
            return opponent_goal(ctx)
        goal_x = ctx.field.length / 2.0
        t = max(0.0, (goal_x - ball.x) / dx)
        return (goal_x, ball.y + math.sin(kick_direction) * t)

    def _in_backfield(self) -> bool:
        """Kick harder by default when the ball is in our own half."""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False
        # own_penalty_edge_x = -ctx.field.length / 2.0 + ctx.field.penalty_area_length
        # return ball.x < own_penalty_edge_x
        return ball.x < 0

    def _draw_kick_target(self, target: tuple[float, float]) -> None:
        """Mark the kick target chosen by plan_kick with an X."""
        from .framework import debugdraw

        x, y = target
        s = KICK_TARGET_MARK_SIZE_M
        debugdraw.line(
            [(x - s, y - s), (x + s, y + s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )
        debugdraw.line(
            [(x - s, y + s), (x + s, y - s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )

    def release_kick(self) -> None:
        self._kicking = False  # Clear the kick hysteresis flag (cancels the cube display)
        if self._backend is None:
            _log.debug("player %d release_kick (no backend)", self.id)
            return
        self._backend.release_kick()

    def kick_can_score(self, kick_direction: float) -> bool:
        """Determine whether kicking from the current ball position along
        ``kick_direction`` would score on a straight trajectory.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False

        goal_x = ctx.field.length / 2.0
        dx = math.cos(kick_direction)
        dy = math.sin(kick_direction)
        if dx <= 1e-6:
            return False

        half_goal = ctx.field.goal_width / 2.0
        if half_goal <= 0.0:
            return False
        if ball.x >= goal_x:
            y_at_goal = ball.y
        else:
            t = (goal_x - ball.x) / dx
            if t < 0.0:
                return False
            y_at_goal = ball.y + dy * t
        return -half_goal <= y_at_goal <= half_goal

    # ------------------------------------------------------------------
    # Slow operations (sync interface invoked asynchronously, so we don't block)
    # ------------------------------------------------------------------

    def request_mode(self, mode: str) -> None:
        if self._backend is None:
            _log.debug("player %d request_mode -> %s (no backend)", self.id, mode)
            return
        self._backend.request_mode(mode)

    def get_up(self) -> None:
        if self._backend is None:
            _log.debug("player %d get_up (no backend)", self.id)
            return
        self._backend.get_up()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def ensure_ready(self) -> bool:
        """Self-recovery: fallen -> get up, not in walk mode -> switch modes
        (both async, produce no movement).

        Returns whether an action can be executed this frame (True = ready).
        """
        if self.is_fallen:
            self.get_up()
            return False
        if self.mode != "walk":
            self.request_mode("walk")
            return False
        return True

    def face_to(self, target_theta: float) -> None:
        """Turn in place to face the target heading."""
        if self.pose is None:
            self.stop()
            return
        err = normalize_angle(target_theta - self.pose.theta)
        self.set_velocity(0.0, 0.0, self._angular(err))

    def walk_to(
        self,
        target: tuple[float, float],
        *,
        face: float | None = None,
        avoid_ball: bool = False,
        avoid_robots: bool = False,
        arrive_dist: float = ARRIVE_DIST,
    ) -> bool:
        """Walk toward the target point. Returns whether it has arrived.

        Obstacle avoidance (local planner, simplified VFH): when
        ``avoid_ball`` / ``avoid_robots`` are enabled, the ball / robots /
        goal structures are collected into circular obstacles, and a sweep of
        candidate directions around the robot picks the one that's "most
        toward the target + clear of obstacles within ``PLAN_LOOKAHEAD``
        ahead." This handles multiple obstacles, concave structures, and
        symmetric conflicts more robustly than single-obstacle detours or
        potential-field repulsion.

        Two walking modes: omnidirectional at close range, turn-then-walk at
        long range. ``face`` specifies the heading to hold on arrival/at
        close range.
        """
        self.release_kick() # Movement overrides kicking, so cancel kick state first
        pose = self.pose
        if pose is None:
            self.stop()
            return False

        tx, ty = target
        dx = tx - pose.x
        dy = ty - pose.y
        distance = math.hypot(dx, dy)

        if distance < arrive_dist:
            # Arrived: turn to face the target heading if requested
            if face is not None:
                err = normalize_angle(face - pose.theta)
                if abs(err) > 0.1:
                    self.set_velocity(0.0, 0.0, self._angular(err))
                else:
                    self.stop()
            else:
                self.stop()
            return True

        # Planning: defaults to global A*, falls back to the old local planner
        # when no path is found.
        goal_dir = math.atan2(dy, dx)
        planned_path: list[tuple[float, float]] | None = None
        waypoint: tuple[float, float] | None = None
        if (avoid_ball or avoid_robots) and self.context is not None:
            obstacles = collect_obstacles(
                self.context, self.id,
                ball=avoid_ball, robots=avoid_robots,
                goals=(avoid_ball or avoid_robots),
            )
            if USE_GLOBAL_PATH_PLANNER:
                planned_path = plan_global_path(
                    self.context,
                    (pose.x, pose.y),
                    (tx, ty),
                    obstacles,
                )
                if planned_path is not None:
                    waypoint = self._path_waypoint(pose, planned_path)
                    heading = angle_to(pose.x, pose.y, waypoint[0], waypoint[1])
                else:
                    heading = self._plan_heading(pose, goal_dir, obstacles)
            else:
                heading = self._plan_heading(pose, goal_dir, obstacles)
        else:
            heading = goal_dir

        # Visualization: target point (green), line to target (gray), planned
        # heading (yellow arrow), forward probe ray (cyan, length=lookahead;
        # the range the planner "looks" ahead, no path/waypoint concept here)
        from .framework import debugdraw
        debugdraw.point(tx, ty, rgb=(0.0, 1.0, 0.0), scale=0.15, ns="target")
        debugdraw.line([(pose.x, pose.y), (tx, ty)], rgb=(0.4, 0.4, 0.4), ns="to_target")
        if planned_path is not None and len(planned_path) >= 2:
            debugdraw.line(planned_path, rgb=(0.2, 0.8, 1.0), ns="global_path")
        if waypoint is not None:
            debugdraw.point(
                waypoint[0], waypoint[1],
                rgb=(0.2, 0.8, 1.0), scale=0.12, ns="global_waypoint",
            )
        debugdraw.arrow(
            pose.x, pose.y,
            pose.x + math.cos(heading) * 0.6, pose.y + math.sin(heading) * 0.6,
            rgb=(1.0, 1.0, 0.0), ns="heading",
        )
        debugdraw.line(
            [(pose.x, pose.y),
             (pose.x + math.cos(heading) * PLAN_LOOKAHEAD,
              pose.y + math.sin(heading) * PLAN_LOOKAHEAD)],
            rgb=(0.0, 0.8, 0.8), ns="lookahead",
        )

        if distance <= OMNI_DIST:
            # Close range: omnidirectional walking, translate along heading
            # while turning toward face
            wdx, wdy = math.cos(heading) * distance, math.sin(heading) * distance
            cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
            vx = LINEAR_GAIN * (wdx * cos_t + wdy * sin_t)
            vy = LINEAR_GAIN * (-wdx * sin_t + wdy * cos_t)
            speed = math.hypot(vx, vy)
            if speed > MAX_LINEAR:
                vx *= MAX_LINEAR / speed
                vy *= MAX_LINEAR / speed
            vyaw = (
                self._angular(normalize_angle(face - pose.theta))
                if face is not None else 0.0
            )
            self.set_velocity(vx, vy, vyaw)
        else:
            # Long range: turn-walk-turn, facing heading
            angle_err = normalize_angle(heading - pose.theta)
            if abs(angle_err) > TURN_THRESHOLD:
                self.set_velocity(0.0, 0.0, self._angular(angle_err))
            else:
                vx = clamp(
                    LINEAR_GAIN * distance * math.cos(angle_err), 0.0, MAX_LINEAR,
                )
                self.set_velocity(vx, 0.0, self._angular(angle_err))
        return False

    def _path_waypoint(
        self, pose: Pose2D, path: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """Pick a short lookahead waypoint from a planned global path."""
        if not path:
            return (pose.x, pose.y)
        prev = (pose.x, pose.y)
        points = path[1:] if len(path) > 1 else path
        for point in points:
            seg_len = dist(prev[0], prev[1], point[0], point[1])
            if seg_len >= GLOBAL_PATH_LOOKAHEAD_M:
                ratio = GLOBAL_PATH_LOOKAHEAD_M / max(seg_len, 1e-6)
                return (
                    prev[0] + (point[0] - prev[0]) * ratio,
                    prev[1] + (point[1] - prev[1]) * ratio,
                )
            prev = point
        return path[-1]

    def _plan_heading(
        self, pose: Pose2D, goal_dir: float, obstacles: list,
    ) -> float:
        """Sweep candidate directions, pick the one most toward the target
        that's clear enough ahead; if none is clear enough, return the most
        open one.

        Candidates are tried in order of increasing deviation ``|offset|``
        from the target direction; which side is tried first is decided by
        player_id parity, to break the symmetry when two players avoid on
        the same side.
        """
        sign_first = 1.0 if self.id % 2 == 0 else -1.0
        best_h = goal_dir
        best_clear = -math.inf

        offsets = [0.0]
        k = 1
        while k * PLAN_STEP <= PLAN_MAX_OFFSET + 1e-9:
            offsets.append(sign_first * k * PLAN_STEP)
            offsets.append(-sign_first * k * PLAN_STEP)
            k += 1

        for off in offsets:
            h = goal_dir + off
            clear = _heading_clearance(pose.x, pose.y, h, obstacles)
            if clear >= PLAN_CLEARANCE:
                return h
            if clear > best_clear:
                best_clear, best_h = clear, h
        return best_h

    # ------------------------------------------------------------------
    # High-level actions (strategy in main.py calls these directly)
    # ------------------------------------------------------------------


    def block_path_projection(
        self, opponent_id: int,
    ) -> tuple[float, float, float, float] | None:
        """The foot of the perpendicular from self onto the opponent->ball
        segment: returns (x, y, perpendicular distance, segment parameter t)."""
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        opponent = ctx.opponents.get(opponent_id) if ctx is not None else None
        if ctx is None or pose is None or ball is None or opponent is None:
            return None
        if opponent.pose is None:
            return None

        ax, ay = opponent.pose.x, opponent.pose.y
        bx, by = ball.x, ball.y
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 < 1e-6:
            return None

        raw_t = ((pose.x - ax) * vx + (pose.y - ay) * vy) / length2
        t = clamp(raw_t, 0.0, 1.0)
        tx = ax + vx * t
        ty = ay + vy * t
        return tx, ty, dist(pose.x, pose.y, tx, ty), raw_t

    def attack(
        self,
        kick_target: tuple[float, float] | None = None,
        passing: bool = True,
        clear_only: bool = False,
    ) -> None:
        """Chase the ball; when in range, decide shoot / pass / carry (QW4/QW5).

        Kick hysteresis: enters kicking state when within ENTER of the ball;
        once in, only exits when farther than EXIT (EXIT > ENTER), to prevent
        boundary jitter. No obstacle avoidance — go straight for the ball. The
        chase target is set **behind the ball** (on the ball->goal line,
        retreating ``CHASE_BEHIND_M`` from the goal), so arrival lines up the
        shot direction.

        ``passing=False`` disables the pass option; ``clear_only=True`` skips
        straight to a hard clearance (used by the keeper's sweep — just boot it
        upfield, never dribble or pass out of our own box).
        """
        ball = self.context.ball if self.context is not None else None
        if ball is None or self.pose is None:
            self.stop()
            return

        self._kick_intent = None                     # reset; _decide_kick sets it when kicking
        d = dist(self.pose.x, self.pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)
        if self._kicking:
            plan = self._decide_kick(passing, clear_only)
            if plan is None:
                self.stop()
                return
            direction, power, aim = plan
            self._draw_kick_target(aim)
            self.kick(direction, power)
        else:
            self.release_kick()
            chase_aim = kick_target if kick_target is not None else opponent_goal(self.context)
            self.walk_to(_behind_ball(ball.x, ball.y, chase_aim, CHASE_BEHIND_M))

    def _decide_kick(
        self, passing: bool, clear_only: bool = False,
    ) -> tuple[float, float, tuple[float, float]] | None:
        """Possession-first shoot / pass / carry / clear decision at the ball.

        Priority (QW4 balanced + QW5 possession):
        1. **Shoot** — within ``SHOT_RANGE_M`` with a clear lane to an open
           corner (away from their keeper) -> full power.
        2. **Forward pass** (if ``passing``) — teammate clearly closer to goal
           with a clear lane -> distance-calibrated power.
        3. **Clear** — ONLY when under pressure AND in our own danger zone
           (near our goal) -> hard boot upfield (safety first).
        4. **Retention pass** (if ``passing``) — under pressure elsewhere, pass
           to any open teammate to keep the ball.
        5. **Carry** — otherwise dribble gently toward goal (retain + advance).

        ``clear_only=True`` short-circuits to a hard clearance. Returns
        ``(direction, power, aim_point)`` or None if the ball is unavailable.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None
        bx, by = ball.x, ball.y
        goal = opponent_goal(ctx)

        if clear_only:
            self._kick_intent = "clear"
            direction = angle_to(bx, by, *goal)
            return (direction, KICK_POWER_CLEAR, self._goal_target_for_direction(direction))

        # 1. Shoot — the priority. Any clear angle to goal within range.
        shot_dir = best_shot(ctx, bx, by, SHOT_RANGE_M, SHOT_LANE_RADIUS_M)
        if shot_dir is not None:
            self._kick_intent = "shoot"
            return (shot_dir, KICK_POWER_SHOT, self._goal_target_for_direction(shot_dir))

        # 2. Forward pass to a clearly-better outlet.
        if passing:
            target = best_pass_target(
                ctx, bx, by, self.id, KEEPER_PLAYER_ID,
                PASS_ADVANCE_MARGIN_M, PASS_LANE_RADIUS_M,
            )
            if target is not None:
                self._kick_intent = "pass"
                return self._pass_plan(bx, by, target)

        # Pressure assessment. Reorienting to kick is slow, so under pressure we
        # RELEASE fast (near-heading pass, or a quick kick into space) rather
        # than dribble — a slow turn lets the defender rob us.
        opp_d = nearest_opponent_dist(ctx, bx, by)
        under_pressure = opp_d is not None and opp_d < PRESSURE_DIST_M
        goal_dir = angle_to(bx, by, *goal)
        heading = self.pose.theta

        if under_pressure:
            # 3a. Deep + pressured -> clear to safety.
            if dist(bx, by, *own_goal(ctx)) < DANGER_RADIUS_M:
                self._kick_intent = "clear"
                return (goal_dir, KICK_POWER_CLEAR, self._goal_target_for_direction(goal_dir))
            # 3b. Quick pass to a near-heading open teammate (minimal turn).
            if passing:
                qt = quick_pass_target(
                    ctx, bx, by, self.id, KEEPER_PLAYER_ID, heading,
                    PASS_LANE_RADIUS_M, math.radians(QUICK_PASS_MAX_TURN_DEG),
                )
                if qt is not None:
                    self._kick_intent = "pass"
                    return self._pass_plan(bx, by, qt)
            # 3c. Quick escape into open space (least reorientation).
            esc = escape_direction(
                ctx, bx, by, heading, goal_dir,
                ESCAPE_AVOID_RADIUS_M, ESCAPE_LOOK_M,
                DIR_SCAN_STEP_DEG, DIR_SCAN_MAX_DEG,
            )
            if esc is not None:
                self._kick_intent = "escape"
                return (esc, KICK_POWER_ESCAPE, self._goal_target_for_direction(esc))

        # 4. Carry: dribble toward goal, steering AROUND any opponent in the near
        # path (don't walk the ball into a defender).
        cd = carry_direction(
            ctx, bx, by, goal_dir,
            CARRY_AVOID_RADIUS_M, CARRY_LOOK_M,
            DIR_SCAN_STEP_DEG, DIR_SCAN_MAX_DEG,
        )
        self._kick_intent = "carry"
        return (cd, KICK_POWER_CARRY, self._goal_target_for_direction(cd))

    def _pass_plan(
        self, bx: float, by: float, target: tuple[float, float],
    ) -> tuple[float, float, tuple[float, float]]:
        """Direction + distance-calibrated power to pass from the ball to ``target``."""
        direction = angle_to(bx, by, target[0], target[1])
        d = dist(bx, by, target[0], target[1])
        power = clamp(
            PASS_POWER_MIN + PASS_POWER_PER_M * d, PASS_POWER_MIN, PASS_POWER_MAX,
        )
        return (direction, power, target)

    def deliver(self, target: tuple[float, float], power: float) -> None:
        """Set-play delivery: approach behind the ball lined up toward ``target``,
        then kick straight at it with fixed ``power`` — a *designed* ball
        (kickoff pass / corner cross / goal kick / free-kick delivery), skipping
        the open-play shoot/pass/carry decision in :meth:`_decide_kick`.

        Same chase + kick-hysteresis as :meth:`attack` (enter kicking within
        ENTER of the ball, exit past EXIT), and no ball avoidance so we can
        reach it. Sets ``_kick_intent='deliver'`` while striking so the dispatch
        can arm the receiver.
        """
        ball = self.context.ball if self.context is not None else None
        if ball is None or self.pose is None:
            self.stop()
            return
        direction = angle_to(ball.x, ball.y, target[0], target[1])
        d = dist(self.pose.x, self.pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)
        if self._kicking:
            self._kick_intent = "deliver"
            self._draw_kick_target(target)
            self.kick(direction, power)
        else:
            self._kick_intent = None
            self.release_kick()
            self.walk_to(_behind_ball(ball.x, ball.y, target, CHASE_BEHIND_M), avoid_ball=False)

    def guard(
        self,
        ball_est: "BallEstimate | None" = None,
        allow_claim: bool = True,
    ) -> None:
        """Goalkeeper: save > claim > arc positioning, chosen each frame.

        1. **save** — a fast shot is heading in: drop to the shallow line and
           slide to the predicted crossing to block it.
        2. **claim** — a loose ball is close, in our half, and we're the closest
           robot to it: come out and clear it. Disabled with ``allow_claim=False``
           during opponent set plays (touching the ball early is a severe foul).
        3. **arc** — default positioning on the [ball -> own-goal] line with
           angle-closing step-out.

        Repositioning faces the travel direction for a large move (fast forward
        walk) and squares up to the ball once settled — see
        :meth:`_keeper_move_to`. Never avoids the ball (that would mean dodging
        incoming shots).
        """
        ctx = self.context
        if ctx is None or self.pose is None:
            self.action = "keeper:stop"
            self.stop()
            return

        gx, gy = own_goal(ctx)                       # Own goal center (-half_l, 0)
        half_goal = ctx.field.goal_width / 2.0
        ball = ctx.ball

        # 1. Save (highest priority): fast shot heading in -> slide to block.
        save_x = gx + KEEPER_SAVE_LINE_M
        crossing = (
            goal_line_crossing(ball_est, save_x) if ball_est is not None else None
        )
        if (
            self._update_keeper_threat(crossing, ball_est, half_goal)
            and crossing is not None
        ):
            self._keeper_claiming = False
            y_cross, _t = crossing
            limit = half_goal + 0.1
            tx, ty = save_x, clamp(y_cross, -limit, limit)
            self.action = "keeper:save"
            self._draw_keeper(ctx, tx, ty, crossing, save=True)
            self._keeper_move_to((tx, ty), ball, avoid_robots=False)
            return

        # 2. Claim a loose ball we can reach first (skipped during opp restarts).
        if allow_claim and ball is not None and self._update_keeper_claim(ctx, ball):
            self.action = "keeper:claim"
            self.attack(passing=False, clear_only=True)  # chase + boot upfield, never dribble/pass
            return
        self._keeper_claiming = False

        # 3. Arc positioning: [ball -> goal center] line with angle-closing step-out.
        bx, by = (ball.x, ball.y) if ball is not None else (0.0, 0.0)
        dx, dy = bx - gx, by - gy
        d = math.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)
        step_out = self._keeper_step_out(d)
        tx = gx + ux * step_out
        ty = clamp(gy + uy * step_out, -(half_goal + KEEPER_LATERAL_MARGIN_M),
                   half_goal + KEEPER_LATERAL_MARGIN_M)
        tx = clamp(tx, gx + 0.1, gx + KEEPER_STEP_OUT_MAX_M)

        self.action = "keeper:arc"
        self._draw_keeper(ctx, tx, ty, crossing, save=False)
        self._keeper_move_to((tx, ty), ball, avoid_robots=True)

    def _keeper_move_to(
        self, target: tuple[float, float], ball, avoid_robots: bool,
    ) -> None:
        """Move to a keeper target, choosing facing for speed. For a large
        reposition, face the travel direction (fast forward walk); once within
        ``KEEPER_SETTLE_DIST_M`` of the spot, square up to the ball (ready to
        react, small strafes only). Never avoids the ball."""
        if self.pose is None:
            self.stop()
            return
        d = dist(self.pose.x, self.pose.y, target[0], target[1])
        if d > KEEPER_SETTLE_DIST_M:
            face = angle_to(self.pose.x, self.pose.y, target[0], target[1])
        elif ball is not None:
            face = angle_to(self.pose.x, self.pose.y, ball.x, ball.y)
        else:
            face = 0.0
        self.walk_to(target, face=face, avoid_ball=False, avoid_robots=avoid_robots)

    def _update_keeper_claim(self, ctx: Context, ball) -> bool:
        """Whether to come out and claim the ball, with hysteresis. Enter when
        the ball is in our half, within ``KEEPER_CLAIM_DIST_M``, and we're the
        closest robot to it (either team) — so we never abandon the net to a
        ball an opponent would reach first. Exit only once the ball is beyond
        ``KEEPER_CLAIM_EXIT_M``."""
        if self.pose is None:
            self._keeper_claiming = False
            return False
        d_self = dist(self.pose.x, self.pose.y, ball.x, ball.y)
        reach = KEEPER_CLAIM_EXIT_M if self._keeper_claiming else KEEPER_CLAIM_DIST_M
        if ball.x >= 0.0 or d_self > reach:
            self._keeper_claiming = False
            return False
        for r in ctx.opponents.values():
            if r.pose is not None and dist(r.pose.x, r.pose.y, ball.x, ball.y) < d_self:
                self._keeper_claiming = False
                return False
        for tid, r in ctx.teammates.items():
            if (
                tid != self.id
                and r.pose is not None
                and dist(r.pose.x, r.pose.y, ball.x, ball.y) < d_self
            ):
                self._keeper_claiming = False
                return False
        self._keeper_claiming = True
        return True

    @staticmethod
    def _keeper_step_out(ball_goal_dist: float) -> float:
        """Step-out distance: larger when the ball is closer to goal (closing
        the angle), smaller when farther (hugging the line). Linear
        interpolation + clamping."""
        near, far = KEEPER_BALL_NEAR_M, KEEPER_BALL_FAR_M
        lo, hi = KEEPER_STEP_OUT_MIN_M, KEEPER_STEP_OUT_MAX_M
        if far <= near:
            return hi
        frac = clamp((ball_goal_dist - near) / (far - near), 0.0, 1.0)
        return hi + (lo - hi) * frac                 # frac 0 (near) -> hi; 1 (far) -> lo

    def _update_keeper_threat(
        self,
        crossing: tuple[float, float] | None,
        ball_est: "BallEstimate | None",
        half_goal: float,
    ) -> bool:
        """Save trigger + hysteresis. Entry is stricter than exit, to avoid
        frame-to-frame jitter near the threshold.

        Entry: ball speed >= ENTER, predicted to arrive within HORIZON, and
        landing point within the goal mouth + MARGIN.
        Exit: ball speed < EXIT, no longer heading toward goal, or landing
        point clearly off target. Writes and returns ``_guard_threatened``.
        """
        if crossing is None or ball_est is None or not ball_est.valid:
            self._guard_threatened = False
            return False

        y_cross, t = crossing
        speed = ball_est.speed
        margin = half_goal + KEEPER_SAVE_MOUTH_MARGIN_M

        if self._guard_threatened:
            self._guard_threatened = (
                speed >= KEEPER_SAVE_EXIT_SPEED
                and t <= KEEPER_SAVE_HORIZON_S * 1.5
                and abs(y_cross) <= margin + 0.3
            )
        else:
            self._guard_threatened = (
                speed >= KEEPER_SAVE_BALL_SPEED
                and t <= KEEPER_SAVE_HORIZON_S
                and abs(y_cross) <= margin
            )
        return self._guard_threatened

    def _draw_keeper(
        self,
        ctx: Context,
        tx: float,
        ty: float,
        crossing: tuple[float, float] | None,
        save: bool,
    ) -> None:
        """Visualization: keeper target point (magenta for save / cyan for
        positioning) + predicted landing X + incoming-ball prediction line."""
        from .framework import debugdraw

        col = (1.0, 0.0, 1.0) if save else (0.0, 0.6, 1.0)
        debugdraw.point(tx, ty, rgb=col, scale=0.22, ns="keeper_target")
        if crossing is not None and ctx.ball is not None:
            gx, _gy = own_goal(ctx)
            cx, cy = gx + KEEPER_SAVE_LINE_M, crossing[0]
            s = 0.18
            debugdraw.line([(cx - s, cy - s), (cx + s, cy + s)],
                           rgb=(1.0, 0.0, 1.0), ns="keeper_cross")
            debugdraw.line([(cx - s, cy + s), (cx + s, cy - s)],
                           rgb=(1.0, 0.0, 1.0), ns="keeper_cross")
            debugdraw.line([(ctx.ball.x, ctx.ball.y), (cx, cy)],
                           rgb=(1.0, 0.4, 0.7), ns="keeper_predict")

    def support(self, dist_m: float | None = None) -> None:
        """Support: stand on the [ball -> own-goal-center] line at ``dist_m``
        from the ball (default ``SUPPORT_DIST_M``), covering defensively.

        The stance point sits on the blocking line between the ball and our own
        goal. A smaller ``dist_m`` tightens it into a close second-line block
        (used when defending in the danger zone).
        """
        ctx = self.context
        if ctx is None:
            self.stop()
            return

        ball = self.context.ball
        gx, gy = own_goal(ctx)
        bx, by = (ball.x, ball.y)
        dx, dy = gx - bx, gy - by
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = -1.0, 0.0
        else:
            ux, uy = dx / d, dy / d
        along = min(SUPPORT_DIST_M if dist_m is None else dist_m, d)
        tx = bx + ux * along
        ty = by + uy * along

        # Don't cross our own end line (x >= goal line + 0.3), clamp laterally within the field
        half_l = ctx.field.length / 2.0
        half_w = ctx.field.width / 2.0 - 0.3
        tx = clamp(tx, -half_l + 0.3, half_l)
        ty = clamp(ty, -half_w, half_w)
        self.move_to_position((tx, ty))

    def support_attack(self) -> None:
        """Advanced attacking support: push up ahead of the ball toward the
        opponent goal and to the open side, as a passing outlet / shooting
        threat. Faces the ball (ready to receive), avoids the ball and robots.

        Used when we're in ATTACK mode; the defensive counterpart is
        :meth:`support` (second-line cover). See the role dispatch in main.py.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None or self.pose is None:
            self.stop()
            return
        opp_ys = [r.pose.y for r in ctx.opponents.values() if r.pose is not None]
        # Adaptive depth: drop deep (short) to offer a safe forward outlet when
        # building up; push high on a fast break when opponents over-commit.
        overcommitted = opponents_overcommitted(
            ctx, OVERCOMMIT_LINE_X, OVERCOMMIT_MIN_COUNT,
        )
        ahead = adaptive_outlet_ahead(
            ctx, ball.x, overcommitted, OUTLET_AHEAD_MIN_M, OUTLET_AHEAD_MAX_M,
        )
        target = attacking_outlet_spot(
            ctx, ball.x, ball.y, opp_ys, ahead, SUPPORT_ATTACK_WIDE_M,
        )
        self.move_to_position(target)

    def crash_net(self) -> None:
        """Crash the box for rebounds: hold a spot just in front of the opponent
        goal, biased to the ball's side, facing the ball to pounce on a loose
        ball after a shot/deflection. Does NOT avoid the ball (we want to be
        near it); once a rebound comes loose, role reassignment makes whoever's
        closest the attacker.
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or self.pose is None:
            self.stop()
            return
        ox, _oy = opponent_goal(ctx)
        half_goal = ctx.field.goal_width / 2.0
        by = ball.y if ball is not None else 0.0
        tx = ox - REBOUND_DEPTH_M
        ty = clamp(by * 0.5, -(half_goal - 0.2), half_goal - 0.2)
        face = (
            angle_to(self.pose.x, self.pose.y, ball.x, ball.y)
            if ball is not None else 0.0
        )
        self.walk_to((tx, ty), face=face, avoid_ball=False, avoid_robots=True)

    def take_kickoff(self, kick_target: tuple[float, float] | None = None) -> None:
        """Our kickoff/restart: if not yet in position, circle around behind
        the ball and wait (avoiding but not touching it); once in position, approach and kick."""
        ball = self.context.ball if self.context is not None else None
        if ball is None or self.pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(self.context)
        kick_dir = angle_to(ball.x, ball.y, *kick_target)
        cos_k, sin_k = math.cos(kick_dir), math.sin(kick_dir)
        rel_x, rel_y = self.pose.x - ball.x, self.pose.y - ball.y
        behind = rel_x * cos_k + rel_y * sin_k          # <0 means behind the ball (our side)
        lateral = abs(-rel_x * sin_k + rel_y * cos_k)
        if behind > KICKOFF_FRONT_MARGIN or lateral > KICKOFF_LATERAL_TOL:
            stage = (
                ball.x - cos_k * KICKOFF_STAGE_M,
                ball.y - sin_k * KICKOFF_STAGE_M,
            )
            self.release_kick()
            self.walk_to(stage, face=kick_dir, avoid_ball=True)
        else:
            self.attack(kick_target)

    def move_to_position(self, target: tuple[float, float] | None) -> None:
        """Walk to a stance point (support/defense/avoidance), facing the
        ball, avoiding both the ball and robots."""
        if target is None:
            self.stop()
            return
        face = None
        ball = self.context.ball if self.context is not None else None
        if ball is not None and self.pose is not None:
            face = angle_to(self.pose.x, self.pose.y, ball.x, ball.y)
        self.release_kick()
        self.walk_to(target, face=face, avoid_ball=True, avoid_robots=True)

    @staticmethod
    def _angular(err: float) -> float:
        return clamp(ANGULAR_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR)
