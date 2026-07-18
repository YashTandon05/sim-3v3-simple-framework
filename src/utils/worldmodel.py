"""World-model foundation: temporal quantities the raw Context doesn't carry.

The platform ``Context`` / ``BallState`` are position-only and rebuilt fresh
every frame, so anything derived from *motion* has to be estimated here. This
module adds the first such piece: ball **velocity**, obtained by differencing
ball position across frames. From velocity we derive speed / heading, a linear
future-position prediction, and the ball's crossing point on a vertical plane
(e.g. the goal line) — the inputs a reactive goalkeeper or interceptor needs.

Cross-frame state lives in :class:`BallTracker` (held on ``store``); the pure
geometry helpers take an immutable :class:`BallEstimate` snapshot so they stay
side-effect free and unit-testable.

Coordinate system matches Context: +x toward the opponent's goal, -x toward
our own goal, field center at (0,0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..framework.types import BallState
from ..param import (
    BALL_VEL_RESET_DT_S,
    BALL_VEL_SMOOTH_ALPHA,
    BALL_VEL_STATIONARY_SPEED,
)


__all__ = [
    "BallEstimate",
    "BallTracker",
    "goal_line_crossing",
    "predict_position",
]


@dataclass(frozen=True)
class BallEstimate:
    """Immutable per-frame ball snapshot enriched with velocity.

    ``valid`` is True only when a trustworthy velocity estimate exists (ball
    seen this frame and last, with a small enough time gap). Consumers must fall
    back to position-only behavior when ``valid`` is False.
    """

    x: float
    y: float
    vx: float
    vy: float
    valid: bool

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def heading(self) -> float:
        """Direction of travel (field angle); meaningless when not moving."""
        return math.atan2(self.vy, self.vx)

    @property
    def moving(self) -> bool:
        """True when the velocity estimate is trustworthy AND above noise floor."""
        return self.valid and self.speed >= BALL_VEL_STATIONARY_SPEED


class BallTracker:
    """Cross-frame ball velocity estimator (finite difference + EMA smoothing).

    Call :meth:`update` once per tick, early in ``play()``. Velocity is measured
    against the ball's own ``last_seen_at`` timestamp rather than the control
    tick, so it reflects the real observation interval (the ball feed may update
    slower than the 30 Hz loop; on ticks with no new sample we reuse the last
    estimate instead of computing a bogus zero).
    """

    def __init__(self) -> None:
        self._px: float | None = None
        self._py: float | None = None
        self._pt: float | None = None
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._valid: bool = False

    def update(self, ball: BallState | None) -> BallEstimate | None:
        """Fold a new ball observation in; return the enriched estimate.

        Returns None when the ball is unseen/stale (framework already filtered
        it to None), which also clears history so a later reappearance doesn't
        produce a huge spurious velocity from an old sample.
        """
        if ball is None:
            self._px = self._py = self._pt = None
            self._vx = self._vy = 0.0
            self._valid = False
            return None

        t = ball.last_seen_at

        # First observation after a (re)start: seed history, no velocity yet.
        if self._pt is None:
            self._px, self._py, self._pt = ball.x, ball.y, t
            self._valid = False
            return BallEstimate(ball.x, ball.y, 0.0, 0.0, valid=False)

        dt = t - self._pt

        # No new sample this tick: reuse the last velocity estimate as-is.
        if dt <= 0.0:
            return BallEstimate(ball.x, ball.y, self._vx, self._vy, valid=self._valid)

        # Gap too large (ball was lost a while): distrust the difference, reseed.
        if dt > BALL_VEL_RESET_DT_S:
            self._px, self._py, self._pt = ball.x, ball.y, t
            self._vx = self._vy = 0.0
            self._valid = False
            return BallEstimate(ball.x, ball.y, 0.0, 0.0, valid=False)

        raw_vx = (ball.x - self._px) / dt
        raw_vy = (ball.y - self._py) / dt
        a = BALL_VEL_SMOOTH_ALPHA
        self._vx = a * raw_vx + (1.0 - a) * self._vx
        self._vy = a * raw_vy + (1.0 - a) * self._vy
        self._px, self._py, self._pt = ball.x, ball.y, t
        self._valid = True
        return BallEstimate(ball.x, ball.y, self._vx, self._vy, valid=True)


def predict_position(est: BallEstimate, t: float) -> tuple[float, float]:
    """Linearly extrapolate ball position ``t`` seconds ahead.

    No rolling-friction model — over the short horizons a keeper reacts on
    (< ~2 s), constant velocity is close enough and errs toward the keeper
    committing early, which is the safe direction.
    """
    return (est.x + est.vx * t, est.y + est.vy * t)


def goal_line_crossing(
    est: BallEstimate, line_x: float,
) -> tuple[float, float] | None:
    """Where/when the ball crosses the vertical plane ``x = line_x``.

    Returns ``(y_cross, t_cross)`` with ``t_cross >= 0``, or None when the ball
    isn't moving toward that plane (stationary in x, or heading away). The
    keeper uses this to find a shot's arrival ``y`` on its save line.
    """
    if not est.valid:
        return None
    if abs(est.vx) < 1e-6:
        return None
    t = (line_x - est.x) / est.vx
    if t < 0.0:
        return None  # ball moving away from the plane
    return (est.y + est.vy * t, t)
