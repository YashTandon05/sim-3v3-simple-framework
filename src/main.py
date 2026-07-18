"""SoccerSim strategy entry point — all match strategy logic lives here; change the playbook by editing this file.

Structure (shallow to deep):
- main.py (this file): match strategy. play() dispatches to _act_* based on the
  Phase state machine; each _act_* picks the attacker (closest to ball) and
  calls player actions directly.
- player.py: the Player control handle + high-level actions (attack /
  take_kickoff / move_to_position / walk_to); to add new tricks/technical
  moves, edit this directly.
- utils/: movement/geometry/avoidance tools (opponent_goal / dist / angle_to ...).
- framework/: platform pipeline, not meant to be edited by users.

To change the playbook, mainly edit this file: the Phase state machine, each
_act_* behavior, and positioning formulas.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from .framework.agent import SoccerAgentMixin
from .framework.types import KICKING_TEAM_NONE, Context, GameState, SetPlay
from .param import *
from .player import Player
from .utils import (
    clamp,
    defensive_screen_spot,
    dist,
    opponent_goal,
    own_goal,
    own_goal_area_center,
)
from .utils.tactics import (
    POSSESSION_OURS,
    POSSESSION_THEIRS,
    attacking_corner_target,
    best_pass_target,
    build_out_spot,
    marking_assignment,
    open_side,
    pass_lane_clear,
    read_possession,
    set_play_defense_assignment,
)
from .utils.worldmodel import BallTracker


_log = logging.getLogger(__name__)


# ======================================================================
# Phase state machine — match phase classification
# ======================================================================


class Phase(Enum):
    """Match phase. Top-level state machine determining whether we're currently
    in normal play / kickoff / set play / ready / stopped."""
    NORMAL = "normal"              # PLAYING, normal contest
    OUR_KICKOFF = "our_kickoff"    # Our kickoff (early SET+PLAYING, take_kickoff)
    OPP_KICKOFF = "opp_kickoff"    # Opponent's kickoff (give way)
    OUR_SET_PLAY = "our_set_play"  # Our set play (free kick / corner / goal kick)
    OPP_SET_PLAY = "opp_set_play"  # Opponent's set play (give way)
    READY = "ready"                # READY positioning
    STOPPED = "stopped"            # SET (non-kickoff restart) / INITIAL / FINISHED / stopped


def get_phase(context: Context) -> Phase:
    """Determine the current match phase from the game-controller state."""
    g = context.game
    if g is None:
        return Phase.STOPPED

    state = g.state

    # READY: walk to ready position
    if state == GameState.READY:
        return Phase.READY

    # PLAYING: normal contest, or a kickoff/set play in progress
    if state == GameState.PLAYING and not g.stopped:
        # Set play: set_play != NONE, kicking_team indicates which side
        if g.set_play != SetPlay.NONE and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_SET_PLAY
            else:
                return Phase.OPP_SET_PLAY

        # Kickoff: secondary_time > 0 (countdown window), kicking_team indicates which side
        if g.secondary_time > 0 and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_KICKOFF
            else:
                return Phase.OPP_KICKOFF

        # Normal contest
        return Phase.NORMAL

    # SET / INITIAL / FINISHED / stopped: hold position
    return Phase.STOPPED

def get_set_play_type(context: Context) -> SetPlay:
    """The currently active set-play type; returns ``SetPlay.NONE`` when there's
    no set play (or no game-controller data).

    Reads the game controller's ``set_play`` field directly, without
    distinguishing which side is taking it — that's determined by
    :func:`get_phase` (OUR_SET_PLAY / OPP_SET_PLAY). This function only answers
    "what type of set play is it."

    7 possible return values (see framework.types.SetPlay):
    - ``NONE``: no set play (normal play / kickoff etc.)
    - ``DIRECT_FREE_KICK``: direct free kick (can score directly)
    - ``INDIRECT_FREE_KICK``: indirect free kick (must touch another player before scoring)
    - ``PENALTY_KICK``: penalty kick
    - ``THROW_IN``: throw-in (kicked in)
    - ``GOAL_KICK``: goal kick
    - ``CORNER_KICK``: corner kick
    """
    g = context.game
    if g is None:
        return SetPlay.NONE
    return g.set_play


# ======================================================================
# Agent entry point
# ======================================================================


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent."""

    player_class = Player

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # Previous frame's phase, used to detect phase transitions (edges)
        store.cur_phase = None
        store.kickoff_taker = None    # Locked-in kickoff taker player id (reselected each time we enter a kickoff)
        store.normal_attacker = None
        store.ball_tracker = BallTracker()  # Cross-frame ball velocity estimator (world-model foundation)
        store.ball_est = None               # This frame's ball position+velocity estimate (updated every play() call)
        store.defend_mode = False           # Team mode: True = press+cover (opp has ball in our half), False = attack+outlet
        store.pass_active_until = 0.0        # Time (context.now) until which the receiver commits to a pass
        store.pass_from_id = None            # Player id that played the active pass (so it doesn't chase its own pass)

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase

        # World model: update ball velocity estimate every frame (used by
        # goalkeeper saves / future interception logic).
        store.ball_est = store.ball_tracker.update(context.ball)

        # Draw visualization (every frame)
        _analyze_and_draw(context, players, store)

        # Draw the current phase as a label outside the field.
        from .framework import debugdraw

        # Ball velocity vector (orange arrow): visually verify estimated
        # direction/magnitude.
        est = store.ball_est
        if est is not None and est.moving:
            debugdraw.arrow(
                est.x, est.y, est.x + est.vx * 0.5, est.y + est.vy * 0.5,
                rgb=(1.0, 0.5, 0.0), ns="ball_vel",
            )
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )

        # Self-recovery + filter down to players that can act this frame.
        # ensure_ready: fallen -> get up / switch to walk mode (async, doesn't
        # produce movement); also done for penalized players so they can jump
        # right back in once the penalty clears. Penalized or not-yet-ready
        # players don't take part in dispatch (or role assignment), so we
        # never pick an immobile player as attacker and leave nobody attacking
        # this frame.
        active: list[Player] = []
        for p in players:
            ready = p.ensure_ready()
            if p.is_penalized:
                p.action = "penalized"     # Penalized: can get up / switch modes, but cannot move
                p.stop()
            elif not ready:
                p.action = "fallen" if p.is_fallen else "switching_mode"
            elif p.pose is None:
                p.action = "no_pose"       # Own position unknown: doesn't take part in dispatch (downstream treats pose as known)
                p.stop()
            else:
                active.append(p)

        # Dispatch the whole team once per phase (team-wide computations like
        # role assignment are only done once inside each _act_*).
        if phase == Phase.NORMAL:
            _act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            _clear_normal_sticky(store)
            _act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            _clear_normal_sticky(store)
            _act_opp_kickoff(context, active, store)
        elif phase == Phase.OUR_SET_PLAY:
            _clear_normal_sticky(store)
            _act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            _clear_normal_sticky(store)
            _act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            _clear_normal_sticky(store)
            _act_ready(context, active)
        elif phase == Phase.STOPPED:
            _clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        # Draw teammate visualization uniformly at the end: covers all
        # players (including penalized/not-ready/STOPPED), fixing the issue
        # where the red ball marker/label disappears in states like SET.
        for p in players:
            _draw_teammate_marker(p)


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Distance from a player to the ball's current position."""
    ball = context.ball
    return (
        dist(p.pose.x, p.pose.y, ball.x, ball.y) + _fallen_time_cost(p)
        if ball is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """Select the player with the smallest distance to the ball.

    ``players`` is non-empty, ready, and has a known pose. Shared by normal
    play and kickoffs.
    """
    ranked = [(p, _player_dist_to_ball(context, p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if (
        preferred is not None
        and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M
    ):
        return preferred[0]
    return best


def _assign_keeper(
    context: Context, active: list[Player],
) -> tuple[Player | None, list[Player]]:
    """Split the actionable players into (keeper, field_players).

    The keeper is the fixed ``KEEPER_PLAYER_ID`` whenever that robot is
    available this frame; if it's penalized / fallen / pose-unknown (not in
    ``active``), the closest active player to our goal is promoted so the net
    is never abandoned. With only one player available, nobody is pinned in
    goal — the lone survivor plays as a field player.

    ``field_players`` is ``active`` minus the keeper. Used by every phase so
    the same robot stays in goal across normal play, kickoffs, and set plays.
    """
    if len(active) <= 1:
        return None, list(active)
    keeper = next((p for p in active if p.id == KEEPER_PLAYER_ID), None)
    if keeper is None:
        gx, gy = own_goal(context)
        keeper = min(active, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
    field = [p for p in active if p is not keeper]
    return keeper, field


def _update_team_mode(context: Context, store) -> None:
    """Decide DEFEND vs ATTACK mode from possession + ball zone, with hysteresis.

    DEFEND (store.defend_mode = True) only when the opponent has the ball in our
    half; ATTACK when we have it or the ball is in their half. A "contested"
    read in our half holds the previous mode, so the second field player doesn't
    flip between a deep cover spot and an advanced outlet frame to frame.
    """
    ball = context.ball
    if ball is None:
        return  # keep previous mode
    if ball.x >= 0.0:
        store.defend_mode = False                 # ball in their half -> attack
        return
    poss = read_possession(context)
    if poss == POSSESSION_THEIRS:
        store.defend_mode = True                  # their ball, our half -> defend
    elif poss == POSSESSION_OURS:
        store.defend_mode = False                 # our ball, our half -> build up
    # else contested in our half -> keep previous store.defend_mode


def _ball_near_own_goal(context: Context) -> bool:
    """True when the ball is inside our danger zone (close to our own goal)."""
    ball = context.ball
    if ball is None:
        return False
    gx, gy = own_goal(context)
    return dist(ball.x, ball.y, gx, gy) < DANGER_RADIUS_M


def _ball_in_shot_range(context: Context) -> bool:
    """True when the ball is close enough to the opponent goal that a shot is on
    (so the off-ball player should crash for a rebound)."""
    ball = context.ball
    if ball is None:
        return False
    ox, oy = opponent_goal(context)
    return dist(ball.x, ball.y, ox, oy) < SHOT_RANGE_M


def _ball_collected(context: Context, field: list[Player], exclude_id) -> bool:
    """True when a field player other than ``exclude_id`` is on the ball (a pass
    has been received / the ball is under our control)."""
    ball = context.ball
    if ball is None:
        return False
    for p in field:
        if p.id == exclude_id or p.pose is None:
            continue
        if dist(p.pose.x, p.pose.y, ball.x, ball.y) < BALL_COLLECT_DIST_M:
            return True
    return False


def _act_normal(context: Context, players: list[Player], store) -> None:
    """NORMAL: fixed keeper guards; the two field players take possession-aware
    dynamic roles.

    The nearer field player always goes for the ball (press when defending,
    attack when we have it). The other's role depends on team mode:
    - DEFEND (opp has ball in our half): second-line defensive cover (support()).
    - ATTACK (we have it, or ball in their half): advanced outlet (support_attack()).

    ``players`` is this frame's set of actionable players (ready, pose known).
    """
    keeper, field = _assign_keeper(context, players)
    if keeper is not None:
        keeper.guard(store.ball_est)

    if not field:
        store.normal_attacker = None
        return

    _update_team_mode(context, store)
    danger = _ball_near_own_goal(context)

    # Debug: show team mode + possession + danger near the bottom touchline.
    from .framework import debugdraw
    debugdraw.text(
        0.0, -context.field.width / 2.0 - 0.2,
        f"mode={'DEFEND' if store.defend_mode else 'ATTACK'} "
        f"poss={read_possession(context)}{' DANGER' if danger else ''}",
        rgb=(0.6, 1.0, 0.6), ns="team_mode",
    )

    primary = _select_closest_attacker(
        context, field, getattr(store, "normal_attacker", None),
    )
    store.normal_attacker = primary.id
    primary.action = "press" if store.defend_mode else "attack"
    primary.attack()

    # Coordinate off the primary's kick decision this frame:
    # - a pass opens a short window where the OTHER field player collects it.
    if getattr(primary, "_kick_intent", None) == "pass":
        store.pass_active_until = context.now + PASS_RECEIVE_WINDOW_S
        store.pass_from_id = primary.id
    pass_active = context.now < getattr(store, "pass_active_until", 0.0)
    if pass_active and _ball_collected(context, field, getattr(store, "pass_from_id", None)):
        store.pass_active_until = 0.0
        pass_active = False
    shot_on = _ball_in_shot_range(context)

    # The other field player(s), in ATTACK mode:
    # - receive: go collect an in-flight pass (not the passer itself);
    # - crash: a shot is on -> crash the box for the rebound/deflection;
    # - outlet: otherwise hold an advanced passing/shooting option.
    # In DEFEND mode: second-line cover, tightened to a goal-side block in the
    # danger zone so both robots defend.
    for p in field:
        if p is primary:
            continue
        if store.defend_mode:
            if danger:
                p.action = "defend_deep"
                p.support(SUPPORT_DEEP_DIST_M)
            else:
                # Man-mark the 2nd-most-dangerous opponent if it has advanced
                # into our third; otherwise hold zonal second-line cover.
                mark = (
                    marking_assignment(context, context.ball.x, context.ball.y, MARK_DIST_M)
                    if context.ball is not None else None
                )
                if mark is not None:
                    p.action = "mark"
                    p.move_to_position(mark)
                else:
                    p.action = "cover"
                    p.support()
        elif pass_active and p.id != getattr(store, "pass_from_id", None):
            p.action = "receive"
            p.attack()
        elif shot_on:
            p.action = "crash"
            p.crash_net()
        else:
            p.action = "outlet"
            p.support_attack()


def _act_our_kickoff(context: Context, players: list[Player], store) -> None:
    """OUR_KICKOFF: keep possession with a designed diagonal pass.

    Instead of booting the ball straight into the opponent half (where their
    keeper collects it), the taker plays a controlled pass into space on the
    supporter's (open) wing, and the supporter runs onto it once play opens.
    The fixed keeper guards. The pass is armed via ``store.pass_active_*`` so the
    receiver commits to collecting it as play transitions to NORMAL.
    """
    keeper, field = _assign_keeper(context, players)
    if keeper is not None:
        keeper.guard(store.ball_est)

    if not field:
        return

    field_ids = {p.id for p in field}
    if store.prev_phase != Phase.OUR_KICKOFF or store.kickoff_taker not in field_ids:
        # Just entered the kickoff phase, reselect the kicker
        store.kickoff_taker = _select_closest_attacker(context, field).id

    taker = next((p for p in field if p.id == store.kickoff_taker), None)
    if taker is None:
        return
    supporter = next((p for p in field if p is not taker), None)

    # Diagonal pass into space on the supporter's side (its y sign = the wing it
    # is waiting on), forward into the opponent half so the restart is legal.
    side = 1.0
    if supporter is not None and supporter.pose is not None:
        side = 1.0 if supporter.pose.y >= 0.0 else -1.0
    pass_target = (KICKOFF_PASS_AHEAD_M, side * KICKOFF_PASS_WIDE_M)

    taker.action = "kickoff:pass"
    taker.deliver(pass_target, KICKOFF_PASS_POWER)
    if taker.is_kicking and supporter is not None:
        store.pass_active_until = context.now + PASS_RECEIVE_WINDOW_S
        store.pass_from_id = taker.id

    # Supporter stays legal (own half, outside the circle) on its wing, ready to
    # sprint onto the pass the instant play opens up.
    if supporter is not None:
        supporter.action = "kickoff:outlet"
        supporter.move_to_position((-0.4, side * KICKOFF_PASS_WIDE_M))


def _act_opp_kickoff(context: Context, players: list[Player], store) -> None:
    """Opponent's kickoff: fixed keeper guards; field players hold fixed
    positions in our half, outside the center circle (staying clear of the ball)."""
    keeper, field = _assign_keeper(context, players)
    if keeper is not None:
        keeper.guard(store.ball_est, allow_claim=False)

    r = context.field.circle_radius
    slots = [(-r - 0.5, 0.0), (-r - 2.0, 0.5)]
    for p, target in zip(field, slots):
        p.action = "opp_kickoff:ready"
        p.walk_to(target, avoid_ball=True, avoid_robots=True)


def _arm_pass(context: Context, store, passer_id: int) -> None:
    """Open the receiver window for a set-play pass just played by ``passer_id``
    (the other field player collects it as play transitions to NORMAL)."""
    store.pass_active_until = context.now + PASS_RECEIVE_WINDOW_S
    store.pass_from_id = passer_id


def _setplay_pass_power(bx: float, by: float, target: tuple[float, float]) -> float:
    """Distance-calibrated, deliberately soft power for a set-play pass."""
    d = dist(bx, by, target[0], target[1])
    return clamp(PASS_POWER_MIN + PASS_POWER_PER_M * d, PASS_POWER_MIN, PASS_POWER_MAX)


def _act_our_set_play(context: Context, players: list[Player], store) -> None:
    """OUR_SET_PLAY: a dedicated designed restart per set-play type.

    The fixed keeper always guards. Among the field players, the one closest to
    the ball is the taker; the other supports. Dispatch:
    - PENALTY_KICK: taker shoots a corner away from their keeper; others hold back.
    - CORNER_KICK: taker crosses to the near-post area, supporter crashes it.
    - GOAL_KICK: play out wide to the supporter, else clear up the open wing.
    - DIRECT_FREE_KICK: shoot if there's an angle, else pass/carry (like open play).
    - INDIRECT_FREE_KICK / THROW_IN (no direct goal): force a pass to a teammate.
    """
    keeper, field = _assign_keeper(context, players)
    if keeper is not None:
        keeper.guard(store.ball_est)

    if not field:
        store.normal_attacker = None
        return

    ball = context.ball
    if ball is None:
        for p in field:
            p.action = "our_setplay:hold"
            p.stop()
        return

    set_play = get_set_play_type(context)
    taker = _select_closest_attacker(context, field, getattr(store, "normal_attacker", None))
    store.normal_attacker = taker.id
    supporter = next((p for p in field if p is not taker), None)

    if set_play == SetPlay.PENALTY_KICK:
        _our_penalty(context, taker, supporter)
    elif set_play == SetPlay.CORNER_KICK:
        _our_corner(context, taker, supporter, store)
    elif set_play == SetPlay.GOAL_KICK:
        _our_goal_kick(context, taker, supporter, store)
    elif set_play == SetPlay.DIRECT_FREE_KICK:
        _our_direct_free_kick(context, taker, supporter, store)
    else:
        # INDIRECT_FREE_KICK, THROW_IN: cannot score directly -> must pass first.
        _our_indirect(context, taker, supporter, store)


def _our_penalty(context: Context, taker: Player, supporter: Player | None) -> None:
    """Our penalty: the taker shoots at the open corner (``attack`` aims away
    from their keeper); the supporter waits behind the ball, out of the area."""
    taker.action = "penalty:shoot"
    taker.attack()
    if supporter is not None:
        supporter.action = "penalty:wait"
        supporter.move_to_position((-0.5, 1.5))


def _our_corner(context: Context, taker: Player, supporter: Player | None, store) -> None:
    """Our corner: deliver a cross to the near-post danger area; the supporter
    crashes the box to attack it (and armed as the pass receiver)."""
    target = attacking_corner_target(context, context.ball.y)
    taker.action = "corner:cross"
    taker.deliver(target, CORNER_DELIVERY_POWER)
    if taker.is_kicking:
        _arm_pass(context, store, taker.id)
    if supporter is not None:
        supporter.action = "corner:crash"
        supporter.crash_net()


def _our_goal_kick(context: Context, taker: Player, supporter: Player | None, store) -> None:
    """Our goal kick: play out to a wide outlet if the lane is clear, else clear
    up the open wing (never square across our own goal)."""
    ball = context.ball
    spot = build_out_spot(context)
    if supporter is not None:
        supporter.action = "goalkick:outlet"
        supporter.move_to_position(spot)

    if (
        supporter is not None
        and pass_lane_clear(context, ball.x, ball.y, spot[0], spot[1], PASS_LANE_RADIUS_M)
    ):
        taker.action = "goalkick:pass"
        taker.deliver(spot, _setplay_pass_power(ball.x, ball.y, spot))
        if taker.is_kicking:
            _arm_pass(context, store, taker.id)
    else:
        side = open_side(context)
        clear_target = (0.0, side * (context.field.width / 2.0 - 1.0))
        taker.action = "goalkick:clear"
        taker.deliver(clear_target, KICK_POWER_CLEAR)


def _our_direct_free_kick(
    context: Context, taker: Player, supporter: Player | None, store,
) -> None:
    """Our direct free kick: shoot if there's an angle (a direct FK can score),
    else pass/carry as in open play. Supporter crashes if a shot is on, else
    holds an advanced outlet."""
    taker.action = "freekick:direct"
    taker.attack()
    if getattr(taker, "_kick_intent", None) == "pass":
        _arm_pass(context, store, taker.id)
    if supporter is not None:
        if _ball_in_shot_range(context):
            supporter.action = "freekick:crash"
            supporter.crash_net()
        else:
            supporter.action = "freekick:outlet"
            supporter.support_attack()


def _our_indirect(
    context: Context, taker: Player, supporter: Player | None, store,
) -> None:
    """Our indirect free kick / throw-in: no direct goal, so force a pass to an
    open teammate (who can then shoot in open play). If no clear pass exists,
    nudge the ball into space up-field so a teammate can take the second touch —
    never a wasted direct shot."""
    ball = context.ball
    if supporter is not None:
        supporter.action = "setplay:outlet"
        supporter.support_attack()

    target = best_pass_target(
        context, ball.x, ball.y, taker.id, KEEPER_PLAYER_ID,
        0.0, PASS_LANE_RADIUS_M,
    )
    if target is None and supporter is not None and supporter.pose is not None:
        cand = (supporter.pose.x, supporter.pose.y)
        if pass_lane_clear(context, ball.x, ball.y, cand[0], cand[1], PASS_LANE_RADIUS_M):
            target = cand

    if target is not None:
        taker.action = "setplay:pass"
        taker.deliver(target, _setplay_pass_power(ball.x, ball.y, target))
    else:
        # No clear pass: a short controlled ball toward goal so a teammate can
        # take the required second touch (keeps it legal and in play).
        goal = opponent_goal(context)
        gd = math.atan2(goal[1] - ball.y, goal[0] - ball.x)
        target = (ball.x + math.cos(gd) * 2.0, ball.y + math.sin(gd) * 2.0)
        taker.action = "setplay:nudge"
        taker.deliver(target, KICK_POWER_CARRY)
    if taker.is_kicking:
        _arm_pass(context, store, taker.id)


def _act_opp_set_play(context: Context, players: list[Player], store) -> None:
    """Opponent's set play — defend legally (never breach the keep-clear
    distance to the ball, or it's a 30s send-off + retake).

    The fixed keeper guards (no claiming — an early touch is a severe foul).
    Dispatch by where the restart is:
    - PENALTY_KICK: only the keeper defends; field players wait behind the mark.
    - Ball in OUR half (corner / deep free kick / deep throw): man-mark opponents
      past halfway goal-side, spare defender guards the goal mouth for the cross.
    - Ball in THEIR half (goal kick / high free kick / throw): compact mid-block,
      screening the lane to our goal at a legal distance.
    """
    keeper, field = _assign_keeper(context, players)
    if keeper is not None:
        keeper.guard(store.ball_est, allow_claim=False)

    ball = context.ball
    if ball is None:
        # Ball position unknown: hold still rather than risk drifting too close.
        for p in field:
            p.action = "opp_setplay:hold"
            p.stop()
        return

    set_play = get_set_play_type(context)

    if set_play == SetPlay.PENALTY_KICK:
        # Only the keeper defends a penalty; field players wait behind the mark,
        # well clear of the ball and outside the area, ready to counter a save.
        slots = [(-0.5, -1.5), (-0.5, 1.5)]
        for p, target in zip(field, slots):
            p.action = "opp_penalty:wait"
            p.walk_to(target, face=0.0, avoid_ball=True, avoid_robots=True)
        return

    if ball.x < 0.0:
        # Corner / deep free kick / deep throw: man-mark + guard the box.
        defenders = [(p.id, p.pose.x, p.pose.y) for p in field]
        assignment = set_play_defense_assignment(
            context, defenders, MARK_DIST_M, (ball.x, ball.y), SET_PLAY_KEEP_CLEAR_M,
        )
        for p, (target, label) in zip(field, assignment):
            p.action = label
            p.move_to_position(target)
        return

    # Ball in their half: compact mid-block screening the lane to our goal.
    for i, p in enumerate(field):
        p.action = "opp_setplay:screen"
        target = defensive_screen_spot(
            context, ball.x, ball.y, i, len(field),
            clear=SET_PLAY_KEEP_CLEAR_M, spread=SET_PLAY_DEFENDER_SPREAD_M,
        )
        p.walk_to(target, avoid_ball=True, avoid_robots=True)


def _act_ready(context: Context, players: list[Player]) -> None:
    """READY: the fixed keeper takes its goal, the field players take kickoff
    positions.

    Role-aware so the keeper starts in goal instead of having to sprint back
    when play resumes (the old version assigned spots by list order, which put
    a field player in goal and the keeper upfield). Field slots depend on
    whether it's our kickoff.
    """
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    field_dims = context.field
    keeper, field = _assign_keeper(context, players)

    if keeper is not None:
        keeper.action = "ready:keeper"
        keeper.walk_to(
            own_goal_area_center(context),
            face=0.0, avoid_ball=True, avoid_robots=True,
        )

    if our_kickoff:
        # One field player at the center spot to take the kick, one wide.
        slots = [
            (-field_dims.circle_radius, 0.0),
            (-0.5, field_dims.circle_radius + 2.0),
        ]
    else:
        # Both field players drop into our half, ahead of the keeper.
        slots = [
            (-field_dims.circle_radius - 0.5, 0.0),
            (-field_dims.length / 2.0 + field_dims.penalty_area_length, 0.0),
        ]
    for p, target in zip(field, slots):
        p.action = "ready"
        p.walk_to(target, face=0.0, avoid_ball=True, avoid_robots=True)



# ======================================================================
# On-field visualization — show ball position + player-to-ball distances,
# drawn to the ROS visualizer
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """Visualize a teammate: red. Cube while kicking, sphere otherwise.

    Called uniformly for every player every frame (unaffected by
    phase/penalty/readiness). Two-line label:
    - Top: player id + current high-level action (``p.action``), with
      ``[KICK]`` appended while kicking.
    - Shape (cube vs. sphere) also distinguishes whether it's in kick state.
    """
    from .framework import debugdraw

    if p.pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(p.pose.x, p.pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(p.pose.x, p.pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(p.pose.x, p.pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """Every frame: compute player-to-ball distances and draw visualization.

    No longer depends on the analysis module; distance is now based on the
    ball's current position.
    """
    from .framework import debugdraw

    ball = context.ball

    # Ball not visible: no visualization
    if ball is None:
        return

    # 1. Draw the ball's current position (green dot)
    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    # 2. Player-to-ball distances: ours (red labels) + opponents' (blue labels)
    for p in players:
        if p.pose is None:
            continue
        d = dist(p.pose.x, p.pose.y, ball.x, ball.y)
        debugdraw.text(
            p.pose.x + 0.3, p.pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        if r.pose is None:
            continue
        d = dist(r.pose.x, r.pose.y, ball.x, ball.y)
        debugdraw.text(
            r.pose.x + 0.3, r.pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )
