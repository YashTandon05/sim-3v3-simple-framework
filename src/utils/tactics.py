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
