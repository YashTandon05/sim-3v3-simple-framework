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
from .utils.tactics import attacking_outlet_spot
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

    def attack(self, kick_target: tuple[float, float] | None = None) -> None:
        """Chase the ball and shoot. Kicks toward ``kick_target`` (defaults to
        the opponent's goal center).

        Kick hysteresis: enters kicking state when within ENTER of the ball;
        once in, only exits when farther than EXIT (EXIT > ENTER), to prevent
        boundary jitter. No obstacle avoidance — normal play should go
        straight for the ball. The chase target is set **behind the ball**
        (on the ball->goal line, retreating ``CHASE_BEHIND_M`` away from the
        goal), so arrival naturally lines up the shot direction.
        """
        ball = self.context.ball if self.context is not None else None
        if ball is None or self.pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(self.context)

        d = dist(self.pose.x, self.pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)
        if self._kicking:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                self.stop()
                return
            kick_direction, kick_power = kick_plan
            self.kick(kick_direction, kick_power)
        else:
            self.release_kick()
            self.walk_to(
                _behind_ball(ball.x, ball.y, kick_target, CHASE_BEHIND_M)
            )

    def guard(self, ball_est: "BallEstimate | None" = None) -> None:
        """Goalkeeper: arc positioning + angle-closing step-out + velocity-based saves.

        Two modes (with hysteresis, to avoid frame-to-frame jitter):
        - **arc (default)**: stands on the [ball -> own-goal-center] line, at
          a step-out distance from goal center. The closer/more central the
          ball, the farther the step-out (closing the angle); the farther the
          ball, the closer it hugs the goal line. Lateral y is clamped near
          the goal mouth.
        - **save**: when the ball is moving fast toward goal and the
          predicted landing point is near the goal mouth, drops back to a
          shallow stance line and slides laterally to block the predicted
          landing point. Requires ``ball_est`` (velocity estimate); falls
          back to arc mode without one.

        Always faces the ball (``GUARD_FACE_BALL``) for fast reaction; never
        avoids the ball (avoiding it would mean dodging incoming shots).
        """
        ctx = self.context
        if ctx is None or self.pose is None:
            self.action = "keeper:stop"
            self.stop()
            return

        gx, gy = own_goal(ctx)                       # Own goal center (-half_l, 0)
        half_goal = ctx.field.goal_width / 2.0
        ball = ctx.ball

        # Facing: toward the ball (ball not visible -> toward the opponent's goal direction, default 0).
        face = 0.0
        if GUARD_FACE_BALL and ball is not None:
            face = angle_to(self.pose.x, self.pose.y, ball.x, ball.y)

        # Save trigger (with hysteresis, writes self._guard_threatened).
        save_x = gx + KEEPER_SAVE_LINE_M
        crossing = (
            goal_line_crossing(ball_est, save_x) if ball_est is not None else None
        )
        save_active = self._update_keeper_threat(crossing, ball_est, half_goal)

        if save_active and crossing is not None:
            # Save: drop back to the shallow line, slide laterally to the
            # predicted landing point (clamped inside the goal mouth), skip
            # avoidance for speed.
            y_cross, _t = crossing
            limit = half_goal + 0.1
            tx, ty = save_x, clamp(y_cross, -limit, limit)
            self.action = "keeper:save"
            self._draw_keeper(ctx, tx, ty, crossing, save=True)
            self.walk_to((tx, ty), face=face, avoid_ball=False, avoid_robots=False)
            return

        # arc: stand on the [ball -> goal center] line, stepping out per step_out to close the angle.
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
        self.walk_to((tx, ty), face=face, avoid_ball=False, avoid_robots=True)

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

    def support(self) -> None:
        """Support: stand on the [ball -> own-goal-center] line at
        ``SUPPORT_DIST_M`` from the ball, covering defensively.

        The stance point sits on the blocking line between the ball and our own goal.
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
        along = min(SUPPORT_DIST_M, d)
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
        target = attacking_outlet_spot(
            ctx, ball.x, ball.y, opp_ys,
            SUPPORT_ATTACK_AHEAD_M, SUPPORT_ATTACK_WIDE_M,
        )
        self.move_to_position(target)

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
