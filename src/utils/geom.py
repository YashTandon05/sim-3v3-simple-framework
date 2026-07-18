"""Geometry helpers — pure functions, stateless, take Context / coordinates.

【utils】Sample tools shipped with the framework; users can read, modify, or
fork them, and can add their own utils in this directory (e.g. "is the ball
out of bounds"). They live here because these calculations don't change with
playbook changes, and are reused by nav / strategy.

Coordinate system: team's field-relative view, +x toward the opponent's
goal, -x toward our own goal, field center at (0,0).
"""

from __future__ import annotations

import math

from ..framework.types import Context
from ..param import GOAL_TARGET_DEPTH_M


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def deg2rad(deg: float) -> float:
    """Degrees to radians."""
    return math.radians(deg)


def rad2deg(rad: float) -> float:
    """Radians to degrees."""
    return math.degrees(rad)


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def angle_to(fx: float, fy: float, tx: float, ty: float) -> float:
    """Field angle pointing from (fx,fy) toward (tx,ty)."""
    return math.atan2(ty - fy, tx - fx)


def opponent_goal(ctx: Context) -> tuple[float, float]:
    """Center of the opponent's goal (attacking target); offset slightly
    behind the goal line to avoid aiming backward when the ball is right on
    the line."""
    return (ctx.field.length / 2.0 + GOAL_TARGET_DEPTH_M, 0.0)


def own_goal(ctx: Context) -> tuple[float, float]:
    """Center of our own goal (defensive anchor point)."""
    return (-ctx.field.length / 2.0, 0.0)


def own_goal_area_center(ctx: Context) -> tuple[float, float]:
    """Center of our own goal area (small box) — the default stance point
    when the goalkeeper faces no threat.

    The goal area extends from our goal line along +x by ``goal_area_length``;
    its center is half that depth inside the goal line.
    """
    return (-ctx.field.length / 2.0 + ctx.field.goal_area_length / 2.0, 0.0)


def clamp_inside_field(
    ctx: Context, x: float, y: float, margin: float = 2.0,
) -> tuple[float, float]:
    """Clamp (x,y) inside the field rectangle (leaving a margin)."""
    half_l = ctx.field.length / 2.0 - margin
    half_w = ctx.field.width / 2.0 - margin
    return (clamp(x, -half_l, half_l), clamp(y, -half_w, half_w))


def defensive_screen_spot(
    ctx: Context,
    ball_x: float,
    ball_y: float,
    index: int = 0,
    count: int = 1,
    clear: float = 2.0,
    spread: float = 0.8,
) -> tuple[float, float]:
    """A legal defensive spot for defending an opponent's set play.

    Returns a point on the [ball -> own-goal-center] line, at least ``clear``
    meters from the ball (so we don't breach the required set-play distance
    and get sent off), spread laterally by ``index``/``count`` so multiple
    defenders don't stack, and clamped inside the field. Screens the lane to
    our goal while staying legal. The lateral spread is perpendicular to the
    ball->goal line, which only *increases* distance from the ball, so the
    clearance floor is preserved (except where field-boundary clamping pulls a
    spot back in — near a corner the effective clearance can shrink slightly).
    """
    gx, gy = own_goal(ctx)
    dx, dy = gx - ball_x, gy - ball_y
    d = math.hypot(dx, dy)
    ux, uy = (dx / d, dy / d) if d > 1e-6 else (-1.0, 0.0)
    tx = ball_x + ux * clear
    ty = ball_y + uy * clear
    if count > 1:
        px, py = -uy, ux                       # perpendicular to the ball->goal line
        offset = (index - (count - 1) / 2.0) * spread
        tx += px * offset
        ty += py * offset
    half_l = ctx.field.length / 2.0
    half_w = ctx.field.width / 2.0 - 0.3
    return (clamp(tx, -half_l + 0.3, half_l - 0.3), clamp(ty, -half_w, half_w))
