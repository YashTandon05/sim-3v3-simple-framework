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
from ..param import (
    BOX_GUARD_DEPTH_M,
    BOX_GUARD_SPREAD_M,
    CORNER_DELIVERY_DEPTH_M,
    CORNER_DELIVERY_WIDE_M,
    POSSESSION_MARGIN_M,
    SHOT_AIM_EDGE_M,
)
from .geom import clamp, normalize_angle, opponent_goal, push_clear_of_ball


__all__ = [
    "POSSESSION_OURS",
    "POSSESSION_THEIRS",
    "POSSESSION_CONTESTED",
    "read_possession",
    "attacking_outlet_spot",
    "opponent_keeper_pos",
    "best_shot",
    "forced_shot_direction",
    "best_pass_target",
    "nearest_opponent_dist",
    "safe_pass_target",
    "quick_pass_target",
    "escape_direction",
    "carry_direction",
    "clearance_direction",
    "marking_assignment",
    "opponents_overcommitted",
    "adaptive_outlet_ahead",
    "open_side",
    "pass_lane_clear",
    "box_guard_spots",
    "set_play_defense_assignment",
    "attacking_corner_target",
    "build_out_spot",
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
    self_id: int | None = None,
) -> float | None:
    """Pick a shooting direction if a scoring shot is available and the lane is
    clear; else None.

    Only considers shots within ``shot_range_m`` of the opponent goal. Aim
    points are CENTRAL (span +/- ``SHOT_AIM_EDGE_M`` from goal center — kick
    noise makes corner aims miss wide; on target beats top bins), tried
    farthest-from-their-keeper first. The lane must be clear of opponents AND
    of our own teammates (never smash the ball into our own robot — pass
    ``self_id`` to exclude the shooter itself).
    """
    goal_x = context.field.length / 2.0
    half_goal = context.field.goal_width / 2.0
    if math.hypot(goal_x - ball_x, 0.0 - ball_y) > shot_range_m:
        return None

    edge = min(SHOT_AIM_EDGE_M, half_goal - 0.3)
    corners = [
        (goal_x, edge),
        (goal_x, -edge),
        (goal_x, edge / 2.0),
        (goal_x, -edge / 2.0),
        (goal_x, 0.0),
    ]
    keeper = opponent_keeper_pos(context)
    if keeper is not None:
        corners.sort(key=lambda c: -abs(c[1] - keeper[1]))  # farthest from keeper first

    blockers = _opp_positions(context)
    blockers += [
        (r.pose.x, r.pose.y)
        for tid, r in context.teammates.items()
        if tid != self_id and r.pose is not None
    ]
    for ax, ay in corners:
        direction = math.atan2(ay - ball_y, ax - ball_x)
        if not _shot_on_target(ball_x, ball_y, direction, goal_x, half_goal):
            continue
        if not _segment_clear(ball_x, ball_y, ax, ay, blockers, lane_radius):
            continue
        return direction
    return None


def forced_shot_direction(
    context: Context, ball_x: float, ball_y: float,
) -> float:
    """Shot direction with NO lane requirement: aim at the goal-mouth corner
    farthest from their keeper and just hit it.

    Used close to their goal when no clean lane exists — a blocked shot still
    produces deflections and rebounds in the box (where our crasher waits),
    which beats turning away from goal. Volume of shots > purity of shots.
    Aims centrally (``SHOT_AIM_EDGE_M``) so the strike stays on target."""
    goal_x = context.field.length / 2.0
    edge = min(SHOT_AIM_EDGE_M, context.field.goal_width / 2.0 - 0.3)
    corners = [(goal_x, edge), (goal_x, -edge)]
    keeper = opponent_keeper_pos(context)
    if keeper is not None:
        corners.sort(key=lambda c: -abs(c[1] - keeper[1]))
    ax, ay = corners[0]
    return math.atan2(ay - ball_y, ax - ball_x)


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


# ----------------------------------------------------------------------
# Pressure escape / marking / adaptive outlet (QW7)
# ----------------------------------------------------------------------


def _opp_positions(context: Context) -> list[tuple[float, float]]:
    return [
        (r.pose.x, r.pose.y)
        for r in context.opponents.values()
        if r.pose is not None
    ]


def quick_pass_target(
    context: Context,
    ball_x: float,
    ball_y: float,
    self_id: int,
    keeper_id: int,
    heading: float,
    lane_radius: float,
    max_turn: float,
) -> tuple[float, float] | None:
    """Fastest-release pass: the open teammate whose pass direction is closest
    to the carrier's current ``heading`` (least reorientation), within
    ``max_turn`` radians, and not backward toward our own goal. Excludes self /
    keeper. None if the only options need a big (slow) turn or are blocked.
    """
    opp = _opp_positions(context)
    best: tuple[float, float] | None = None
    best_dev = math.inf
    for tid, r in context.teammates.items():
        if tid == self_id or tid == keeper_id or r.pose is None:
            continue
        cand = (r.pose.x, r.pose.y)
        if cand[0] < ball_x - 0.5:                      # notably behind -> toward our goal, skip
            continue
        if not _segment_clear(ball_x, ball_y, cand[0], cand[1], opp, lane_radius):
            continue
        pd = math.atan2(cand[1] - ball_y, cand[0] - ball_x)
        dev = abs(normalize_angle(pd - heading))
        if dev > max_turn:
            continue
        if dev < best_dev:
            best_dev, best = dev, cand
    return best


def escape_direction(
    context: Context,
    ball_x: float,
    ball_y: float,
    heading: float,
    goal_dir: float,
    avoid_radius: float,
    look: float,
    scan_step_deg: float,
    scan_max_deg: float,
) -> float | None:
    """A quick-release kick direction under pressure: the direction closest to
    the carrier's ``heading`` (least turn) whose short lookahead is clear of
    opponents, restricted to the forward hemisphere (within 90 deg of
    ``goal_dir`` — never escape toward our own goal). None if nothing is open.
    """
    opp = _opp_positions(context)
    best: float | None = None
    best_dev = math.inf
    n = int(scan_max_deg / scan_step_deg)
    for i in range(-n, n + 1):
        d = heading + math.radians(i * scan_step_deg)
        if abs(normalize_angle(d - goal_dir)) > math.pi / 2.0:
            continue                                    # not forward -> skip
        px, py = ball_x + math.cos(d) * look, ball_y + math.sin(d) * look
        if not _segment_clear(ball_x, ball_y, px, py, opp, avoid_radius):
            continue
        dev = abs(normalize_angle(d - heading))
        if dev < best_dev:
            best_dev, best = dev, d
    return best


def carry_direction(
    context: Context,
    ball_x: float,
    ball_y: float,
    goal_dir: float,
    avoid_radius: float,
    look: float,
    scan_step_deg: float,
    scan_max_deg: float,
) -> float:
    """Dribble direction: toward ``goal_dir``, but if an opponent is in the near
    path, deflect to the nearest clear side. Returns the clear direction closest
    to ``goal_dir`` (falls back to ``goal_dir`` if nothing is clearer).
    """
    opp = _opp_positions(context)
    if not opp:
        return goal_dir
    n = int(scan_max_deg / scan_step_deg)
    order = [0]
    for k in range(1, n + 1):
        order.extend((k, -k))
    for k in order:
        d = goal_dir + math.radians(k * scan_step_deg)
        px, py = ball_x + math.cos(d) * look, ball_y + math.sin(d) * look
        if _segment_clear(ball_x, ball_y, px, py, opp, avoid_radius):
            return d
    return goal_dir


def _ray_clearance(
    bx: float, by: float, direction: float,
    opp: list[tuple[float, float]], look: float,
) -> float:
    """Smallest perpendicular distance to any opponent lying ahead
    (``0 < t <= look``) along the ray from (bx,by) in ``direction``; ``inf`` if
    the corridor is empty (the more open, the larger)."""
    ux, uy = math.cos(direction), math.sin(direction)
    best = math.inf
    for ox, oy in opp:
        rx, ry = ox - bx, oy - by
        t = rx * ux + ry * uy
        if t <= 0.0 or t > look:
            continue
        lateral = abs(-rx * uy + ry * ux)
        if lateral < best:
            best = lateral
    return best


def clearance_direction(
    context: Context,
    ball_x: float,
    ball_y: float,
    prefer_dir: float,
    scan_max_deg: float,
    scan_step_deg: float,
    look: float,
) -> float:
    """Best direction to hoof a clearance under pressure in our own half.

    Scans a wide forward arc around ``prefer_dir`` (toward the opponent goal)
    and returns the lane with the most room from the nearest opponent,
    tie-broken toward ``prefer_dir``. The ball goes forward when that's open, or
    sideways toward a touchline when the middle is congested — never backward
    toward our own goal (the arc is limited to +/- ``scan_max_deg`` <= 90)."""
    opp = _opp_positions(context)
    if not opp:
        return prefer_dir
    # Any lane with >= this much room is "clear enough" — treat them as equally
    # safe so we don't needlessly deflect away from a defender who's only near
    # the wide edge of the forward lane; the tie-break then keeps us forward.
    good = 1.0
    best_dir = prefer_dir
    best_score = -math.inf
    n = int(scan_max_deg / scan_step_deg)
    for i in range(-n, n + 1):
        d = prefer_dir + math.radians(i * scan_step_deg)
        clr = min(_ray_clearance(ball_x, ball_y, d, opp, look), good)
        score = clr - 0.001 * abs(i)         # tie-break toward forward (prefer_dir)
        if score > best_score:
            best_score, best_dir = score, d
    return best_dir


def marking_assignment(
    context: Context, ball_x: float, ball_y: float, mark_dist: float,
) -> tuple[float, float] | None:
    """Man-marking spot for the second field defender: goal-side of the most
    goal-threatening opponent that is NOT the ball-carrier AND is inside our own
    defensive third (else they aren't a real threat -> None, use zonal cover).
    """
    opps = _opp_positions(context)
    if not opps:
        return None
    third_x = -context.field.length / 2.0 + context.field.length / 3.0
    gx, gy = -context.field.length / 2.0, 0.0
    carrier = min(opps, key=lambda o: math.hypot(o[0] - ball_x, o[1] - ball_y))
    threats = [o for o in opps if o is not carrier and o[0] < third_x]
    if not threats:
        return None
    threat = min(threats, key=lambda o: math.hypot(o[0] - gx, o[1] - gy))
    dx, dy = gx - threat[0], gy - threat[1]
    d = math.hypot(dx, dy)
    ux, uy = (dx / d, dy / d) if d > 1e-6 else (-1.0, 0.0)
    return (threat[0] + ux * mark_dist, threat[1] + uy * mark_dist)


def opponents_overcommitted(
    context: Context, line_x: float, min_count: int,
) -> bool:
    """True when at least ``min_count`` opponents have pushed onto our side of
    ``line_x`` (over-committed) — the cue to fast-break into the space behind them."""
    count = sum(
        1 for r in context.opponents.values()
        if r.pose is not None and r.pose.x < line_x
    )
    return count >= min_count


def adaptive_outlet_ahead(
    context: Context,
    ball_x: float,
    overcommitted: bool,
    ahead_min: float,
    ahead_max: float,
) -> float:
    """How far ahead of the ball the attacking outlet should sit. Drops deep
    (short) when the ball is in our half (retain / offer a safe forward outlet),
    pushes high (long) as we advance; snaps to max on a fast break."""
    if overcommitted:
        return ahead_max
    half_l = context.field.length / 2.0
    frac = clamp((ball_x + half_l) / (2.0 * half_l), 0.0, 1.0)  # 0 deep, 1 advanced
    return ahead_min + (ahead_max - ahead_min) * frac


# ----------------------------------------------------------------------
# Set pieces — designed restarts (kickoff / our + opponent set plays)
# ----------------------------------------------------------------------


def open_side(context: Context) -> float:
    """The lateral side with fewer opponents (the open side to attack into):
    ``+1`` (positive y) or ``-1`` (negative y). Ties resolve to ``+1``."""
    up = sum(
        1 for r in context.opponents.values()
        if r.pose is not None and r.pose.y > 0.0
    )
    down = sum(
        1 for r in context.opponents.values()
        if r.pose is not None and r.pose.y < 0.0
    )
    if up > down:
        return -1.0
    return 1.0


def pass_lane_clear(
    context: Context, fx: float, fy: float, tx: float, ty: float, radius: float,
) -> bool:
    """True if no opponent is within ``radius`` of the segment (fx,fy)->(tx,ty).
    A public wrapper over the shared lane check, for set-play delivery decisions."""
    return _segment_clear(fx, fy, tx, ty, _opp_positions(context), radius)


def _goal_side_spot(
    gx: float, gy: float, ox: float, oy: float, mark_dist: float,
) -> tuple[float, float]:
    """A point ``mark_dist`` from opponent (ox,oy) toward our goal (gx,gy) —
    i.e. goal-side of them, denying the pass/shot lane."""
    dx, dy = gx - ox, gy - oy
    d = math.hypot(dx, dy)
    ux, uy = (dx / d, dy / d) if d > 1e-6 else (-1.0, 0.0)
    return (ox + ux * mark_dist, oy + uy * mark_dist)


def box_guard_spots(
    context: Context, count: int,
    depth: float = BOX_GUARD_DEPTH_M, spread: float = BOX_GUARD_SPREAD_M,
) -> list[tuple[float, float]]:
    """``count`` positions in front of our goal to defend a cross, spread evenly
    across the goal mouth (a single defender sits central)."""
    gx = -context.field.length / 2.0
    x = gx + depth
    if count <= 0:
        return []
    if count == 1:
        return [(x, 0.0)]
    return [(x, -spread + 2.0 * spread * (i / (count - 1))) for i in range(count)]


def set_play_defense_assignment(
    context: Context,
    defenders: list[tuple[int, float, float]],
    mark_dist: float,
    ball_xy: tuple[float, float],
    keep_clear: float,
) -> list[tuple[tuple[float, float], str]]:
    """Assign our field defenders for an opponent set play in our half (corner /
    deep free kick / deep throw).

    Man-mark opponents that have advanced past halfway (goal-side, most
    dangerous first) — excluding the taker on the ball — and send any spare
    defender to guard the goal mouth for the cross. Every returned target is at
    least ``keep_clear`` from the ball (legal, no set-piece send-off).

    ``defenders`` is ``[(id, x, y), ...]``; returns a list aligned to it of
    ``(target, label)``, where label is ``"mark:setplay"`` or ``"box"``.
    """
    gx, gy = -context.field.length / 2.0, 0.0
    bx, by = ball_xy
    opps = _opp_positions(context)
    n = len(defenders)
    if n == 0:
        return []

    # Mark opponents past halfway, excluding the taker (nearest opp to the ball).
    marks: list[tuple[tuple[float, float], str]] = []
    if opps:
        taker = min(opps, key=lambda o: math.hypot(o[0] - bx, o[1] - by))
        cand = [o for o in opps if o is not taker and o[0] < 0.0]
        cand.sort(key=lambda o: math.hypot(o[0] - gx, o[1] - gy))  # nearest our goal first
        marks = [
            (_goal_side_spot(gx, gy, o[0], o[1], mark_dist), "mark:setplay")
            for o in cand
        ]
    marks = marks[:n]

    n_box = n - len(marks)
    box = [(s, "box") for s in box_guard_spots(context, n_box)]
    tasks = [
        (push_clear_of_ball(t, bx, by, keep_clear), lbl)
        for (t, lbl) in (marks + box)
    ][:n]

    # Greedy nearest assignment: highest-priority task (mark nearest threat)
    # first claims its closest free defender, minimizing travel / crossing.
    result: list[tuple[tuple[float, float], str] | None] = [None] * n
    used: set[int] = set()
    for target, lbl in tasks:
        best_i, best_d = None, math.inf
        for i, (_did, dx, dy) in enumerate(defenders):
            if i in used:
                continue
            dd = math.hypot(dx - target[0], dy - target[1])
            if dd < best_d:
                best_d, best_i = dd, i
        if best_i is not None:
            used.add(best_i)
            result[best_i] = (target, lbl)

    fallback = push_clear_of_ball((gx + BOX_GUARD_DEPTH_M, 0.0), bx, by, keep_clear)
    return [r if r is not None else (fallback, "box") for r in result]


def attacking_corner_target(
    context: Context, ball_y: float,
) -> tuple[float, float]:
    """Where to deliver our corner: a point just in front of the opponent goal
    on the near-post (corner) side, where the crashing teammate attacks it."""
    gx = context.field.length / 2.0
    half_goal = context.field.goal_width / 2.0
    side = 1.0 if ball_y >= 0.0 else -1.0
    ty = side * min(CORNER_DELIVERY_WIDE_M, half_goal - 0.2)
    return (gx - CORNER_DELIVERY_DEPTH_M, ty)


def build_out_spot(context: Context) -> tuple[float, float]:
    """A wide outlet in our half for playing out from a goal kick: up the open
    wing, around the edge of our defensive third."""
    half_l = context.field.length / 2.0
    x = -half_l + context.field.length / 3.0        # our defensive-third boundary
    side = open_side(context)
    y = side * (context.field.width / 2.0 - 1.5)
    return (x, y)
