"""Team-tactics helpers: possession estimate + attacking-support geometry.

Pure functions, no platform dependency (only framework.types + param + geom),
so they're independently unit-testable. Used by the role dispatch in main.py to
decide the two field players' roles, and by Player.support_attack.

Coordinate system matches Context: +x toward the opponent's goal, -x toward our
own goal, field center at (0,0).
"""

from __future__ import annotations

import math

from ..framework.types import Context
from ..param import POSSESSION_MARGIN_M
from .geom import clamp, opponent_goal


__all__ = [
    "POSSESSION_OURS",
    "POSSESSION_THEIRS",
    "POSSESSION_CONTESTED",
    "read_possession",
    "attacking_outlet_spot",
    "opponent_keeper_pos",
    "best_shot",
    "best_pass_target",
    "nearest_opponent_dist",
    "safe_pass_target",
]


POSSESSION_OURS = "ours"
POSSESSION_THEIRS = "theirs"
POSSESSION_CONTESTED = "contested"


def _nearest_dist(robots, bx: float, by: float) -> float | None:
    """Smallest distance from any pose-known robot in `robots` to (bx, by)."""
    best: float | None = None
    for r in robots:
        if r.pose is None:
            continue
        d = math.hypot(r.pose.x - bx, r.pose.y - by)
        if best is None or d < best:
            best = d
    return best


def read_possession(context: Context, margin: float = POSSESSION_MARGIN_M) -> str:
    """Estimate ball possession by comparing the nearest player of each team.

    Returns ``POSSESSION_OURS`` / ``POSSESSION_THEIRS`` / ``POSSESSION_CONTESTED``.
    A ``margin`` band around equality yields "contested", which the caller uses
    to hold the current team mode (avoids flapping near the boundary). With no
    ball or no visible robots on a side, it degrades gracefully (no opponents
    seen -> treat as ours; no ball -> contested).
    """
    ball = context.ball
    if ball is None:
        return POSSESSION_CONTESTED
    ours = _nearest_dist(context.teammates.values(), ball.x, ball.y)
    theirs = _nearest_dist(context.opponents.values(), ball.x, ball.y)
    if ours is None and theirs is None:
        return POSSESSION_CONTESTED
    if theirs is None:
        return POSSESSION_OURS
    if ours is None:
        return POSSESSION_THEIRS
    if ours + margin < theirs:
        return POSSESSION_OURS
    if theirs + margin < ours:
        return POSSESSION_THEIRS
    return POSSESSION_CONTESTED


def attacking_outlet_spot(
    context: Context,
    ball_x: float,
    ball_y: float,
    opp_ys: list[float],
    ahead: float,
    wide: float,
) -> tuple[float, float]:
    """Advanced support position: ahead of the ball toward the opponent goal,
    offset to the more open lateral side, clamped inside the field.

    "Ahead" is along the ball->opponent-goal line so the outlet is a forward
    passing option and a shooting threat. The lateral side is chosen away from
    where the opponents are massed (``opp_ys`` = opponents' y positions); ties
    break to the side opposite the outlet's own y, to spread the attack.
    """
    ox, oy = opponent_goal(context)
    dx, dy = ox - ball_x, oy - ball_y
    d = math.hypot(dx, dy)
    ux, uy = (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)
    ax, ay = ball_x + ux * ahead, ball_y + uy * ahead

    up = sum(1 for y in opp_ys if y > 0.0)
    down = sum(1 for y in opp_ys if y < 0.0)
    if up > down:
        side = -1.0
    elif down > up:
        side = 1.0
    else:
        side = 1.0 if ay <= 0.0 else -1.0

    px, py = -uy, ux                               # perpendicular to ball->goal line
    tx, ty = ax + px * side * wide, ay + py * side * wide

    half_l = context.field.length / 2.0
    half_w = context.field.width / 2.0
    return (clamp(tx, -half_l + 0.3, half_l - 0.3), clamp(ty, -half_w + 0.3, half_w - 0.3))


# ----------------------------------------------------------------------
# Shot / pass selection (QW4)
# ----------------------------------------------------------------------


def opponent_keeper_pos(context: Context) -> tuple[float, float] | None:
    """Guess the opponent's keeper as their robot nearest their own goal (max x)."""
    best: tuple[float, float] | None = None
    best_x = -math.inf
    for r in context.opponents.values():
        if r.pose is not None and r.pose.x > best_x:
            best_x, best = r.pose.x, (r.pose.x, r.pose.y)
    return best


def _shot_on_target(
    ball_x: float, ball_y: float, direction: float, goal_x: float, half_goal: float,
) -> bool:
    """Would a straight kick from the ball along `direction` cross into the goal
    mouth at x = goal_x?"""
    dx, dy = math.cos(direction), math.sin(direction)
    if dx <= 1e-6:
        return False
    if ball_x >= goal_x:
        y = ball_y
    else:
        t = (goal_x - ball_x) / dx
        if t < 0.0:
            return False
        y = ball_y + dy * t
    return -half_goal <= y <= half_goal


def _segment_clear(
    fx: float, fy: float, tx: float, ty: float,
    blockers: list[tuple[float, float]], radius: float,
) -> bool:
    """True if no blocker lies within `radius` of the open segment (fx,fy)->(tx,ty).

    Blockers behind the start or at/beyond the target don't count (the target is
    the receiver/goal, not an obstacle). Used for shot and pass lane checks.
    """
    sdx, sdy = tx - fx, ty - fy
    seg = math.hypot(sdx, sdy)
    if seg < 1e-6:
        return True
    ux, uy = sdx / seg, sdy / seg
    for bx, by in blockers:
        rx, ry = bx - fx, by - fy
        along = rx * ux + ry * uy
        if along <= 0.2 or along >= seg - 0.1:
            continue
        lateral = abs(-rx * uy + ry * ux)
        if lateral < radius:
            return False
    return True


def best_shot(
    context: Context,
    ball_x: float,
    ball_y: float,
    shot_range_m: float,
    lane_radius: float,
) -> float | None:
    """Pick a shooting direction (aim at the open corner away from their keeper)
    if a scoring shot is available and the lane is clear; else None.

    Only considers shots within ``shot_range_m`` of the opponent goal. Tries the
    corner farthest from their keeper first, then the other corner, then dead
    center; returns the first whose straight trajectory scores and whose lane is
    clear of opponents.
    """
    goal_x = context.field.length / 2.0
    half_goal = context.field.goal_width / 2.0
    if math.hypot(goal_x - ball_x, 0.0 - ball_y) > shot_range_m:
        return None

    inset = 0.3
    corners = [
        (goal_x, half_goal - inset),
        (goal_x, -(half_goal - inset)),
        (goal_x, 0.0),
    ]
    keeper = opponent_keeper_pos(context)
    if keeper is not None:
        corners.sort(key=lambda c: -abs(c[1] - keeper[1]))  # farthest from keeper first

    opp_positions = [
        (r.pose.x, r.pose.y)
        for r in context.opponents.values()
        if r.pose is not None
    ]
    for ax, ay in corners:
        direction = math.atan2(ay - ball_y, ax - ball_x)
        if not _shot_on_target(ball_x, ball_y, direction, goal_x, half_goal):
            continue
        if not _segment_clear(ball_x, ball_y, ax, ay, opp_positions, lane_radius):
            continue
        return direction
    return None


def best_pass_target(
    context: Context,
    ball_x: float,
    ball_y: float,
    self_id: int,
    keeper_id: int,
    advance_margin: float,
    lane_radius: float,
) -> tuple[float, float] | None:
    """The best teammate to pass to: clearly closer to the opponent goal (by at
    least ``advance_margin``) with a clear passing lane. Excludes self and our
    own keeper. Returns the most-advanced such teammate, or None.
    """
    goal_x = context.field.length / 2.0
    my_gd = math.hypot(goal_x - ball_x, 0.0 - ball_y)
    opp_positions = [
        (r.pose.x, r.pose.y)
        for r in context.opponents.values()
        if r.pose is not None
    ]
    best: tuple[float, float] | None = None
    best_gain = advance_margin
    for tid, r in context.teammates.items():
        if tid == self_id or tid == keeper_id or r.pose is None:
            continue
        cand_gd = math.hypot(goal_x - r.pose.x, 0.0 - r.pose.y)
        gain = my_gd - cand_gd
        if gain < best_gain:
            continue
        if not _segment_clear(ball_x, ball_y, r.pose.x, r.pose.y, opp_positions, lane_radius):
            continue
        best_gain, best = gain, (r.pose.x, r.pose.y)
    return best


def nearest_opponent_dist(context: Context, x: float, y: float) -> float | None:
    """Distance from the nearest pose-known opponent to (x, y); None if none seen."""
    return _nearest_dist(context.opponents.values(), x, y)


def safe_pass_target(
    context: Context,
    ball_x: float,
    ball_y: float,
    self_id: int,
    keeper_id: int,
    lane_radius: float,
) -> tuple[float, float] | None:
    """A retention outlet: the open teammate (clear lane) with the most space
    around them, regardless of forward progress. Used under pressure to keep the
    ball rather than boot it away. Excludes self and our keeper.
    """
    opp_positions = [
        (r.pose.x, r.pose.y)
        for r in context.opponents.values()
        if r.pose is not None
    ]
    best: tuple[float, float] | None = None
    best_space = -math.inf
    for tid, r in context.teammates.items():
        if tid == self_id or tid == keeper_id or r.pose is None:
            continue
        cand = (r.pose.x, r.pose.y)
        if not _segment_clear(ball_x, ball_y, cand[0], cand[1], opp_positions, lane_radius):
            continue
        space = (
            min(math.hypot(cand[0] - ox, cand[1] - oy) for ox, oy in opp_positions)
            if opp_positions else math.inf
        )
        if space > best_space:
            best_space, best = space, cand
    return best
