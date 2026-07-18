"""Framework runtime: 30Hz main loop, Context construction, Player instance management.

Context data comes from the injected ContextSource (Phase 2's ROS data
source); when not injected (dev machine / unit tests), an empty Context is
built every frame. Freshness filtering (stale -> None) is handled uniformly
at this layer, see docs/new_design.md section 9.3.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

from .config import SoccerConfig
from .types import (
    ADULT_FIELD_DIMENSIONS,
    BallState,
    Context,
    GameControlState,
    RobotState,
    WorldSnapshot,
)

if TYPE_CHECKING:
    from ..player import Player
    from .agent import SoccerAgentMixin


__all__ = ["ContextSource", "SoccerRuntime"]


_log = logging.getLogger(__name__)


class ContextSource(Protocol):
    """Data source protocol: runtime pulls a raw snapshot from it every frame; itself does not depend on any concrete ROS implementation."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_snapshot(self) -> WorldSnapshot: ...


class SoccerRuntime:
    """30Hz control loop + Player lifecycle management.

    Invisible to the user: users only touch SoccerAgent / Player / Context
    / play(), and don't need to know this class exists.

    When ``context_source`` is None (dev machine / unit tests), an empty
    Context is built every frame; when a ROS data source is provided,
    Context is built from real data with freshness filtering applied.
    """

    def __init__(
        self,
        agent: "SoccerAgentMixin",
        context_source: "ContextSource | None" = None,
    ) -> None:
        self._agent = agent
        self._config: SoccerConfig = agent.config
        self._source = context_source
        self._store = SimpleNamespace()
        self._players: list[Player] = [
            agent.player_class(player_id=pid, config=self._config, _backend=None)
            for pid in self._config.player_ids
        ]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._init_store_called = False
        self._last_now: float | None = None
        self._tick_id = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            _log.info("runtime already running, ignore start")
            return
        if self._source is not None:
            self._source.start()
        if not self._init_store_called:
            self._agent.init_store(self._store)
            self._init_store_called = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="soccer_runtime", daemon=True,
        )
        self._thread.start()
        _log.info(
            "SoccerRuntime started: team_id=%d control_hz=%.1f players=%d source=%s",
            self._config.team_id, self._config.control_hz, len(self._players),
            type(self._source).__name__ if self._source else "None",
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._source is not None:
            self._source.stop()
        self._close_backends()
        _log.info("SoccerRuntime stopped")

    def _close_backends(self) -> None:
        """Close the SDK backend for every player."""
        for player in self._players:
            if player._backend is not None:
                try:
                    player._backend.close()
                except Exception as exc:
                    _log.warning(
                        "player %d backend close failed: %s", player.id, exc,
                    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self._config.control_hz)
        while not self._stop.is_set():
            started_at = time.monotonic()
            try:
                self._tick(started_at)
            except Exception as exc:
                _log.exception("control loop tick failed: %s", exc)
                # On exception, stop every player (call directly, not via the play path)
                for p in self._players:
                    try:
                        p.stop()
                    except Exception:
                        pass

            elapsed = time.monotonic() - started_at
            self._stop.wait(max(0.0, period - elapsed))

    def _tick(self, now: float) -> None:
        self._tick_id += 1
        dt = 0.0 if self._last_now is None else (now - self._last_now)
        self._last_now = now

        ctx = self._build_context(now, dt)
        for p in self._players:
            p.context = ctx

        # Debug visualization: open a frame, draw the persistent world (ball/teammates/opponents), then let play() append strategy markers
        from . import debugdraw
        debugdraw.begin_frame()
        self._draw_world(ctx)

        # Call the user's play(); the framework does not constrain its behavior
        self._agent.play(ctx, self._players, self._store)

        debugdraw.flush()

        # Log one heartbeat line every ~2s, for visually verifying the main loop + data path
        if self._tick_id % 60 == 0:
            self._log_heartbeat(ctx, dt)

    def _draw_world(self, ctx: Context) -> None:
        """Persistent visualization: field/goals (dark), ball (orange), our team (red + number + facing), opponents (blue + facing)."""
        from . import debugdraw
        import math

        self._draw_field(ctx)

        if ctx.ball is not None:
            debugdraw.point(
                ctx.ball.x, ctx.ball.y, rgb=(1.0, 0.5, 0.0), scale=0.2, ns="ball",
            )
        # Our own team's markers (color/shape depending on kick state, label
        # including chaser) are drawn by the strategy layer's main.py,
        # because kick state lives on player and chaser lives in play().
        # Here we only draw facing + opponents.
        for r in ctx.teammates.values():
            if r.pose is not None:
                self._draw_facing(r.pose)
        for r in ctx.opponents.values():
            if r.pose is not None:
                debugdraw.point(r.pose.x, r.pose.y, rgb=(0.2, 0.4, 1.0),
                                scale=0.3, ns="opponent")
                self._draw_facing(r.pose)

    def _draw_facing(self, pose) -> None:
        """Robot facing: short white arrow (0.4m), ns=facing. Distinguished from the yellow velocity heading."""
        from . import debugdraw
        import math

        debugdraw.arrow(
            pose.x, pose.y,
            pose.x + math.cos(pose.theta) * 0.4,
            pose.y + math.sin(pose.theta) * 0.4,
            rgb=(1.0, 1.0, 1.0), ns="facing",
        )

    def _draw_field(self, ctx: Context) -> None:
        """Static field: outer boundary, midline, center circle, goal frames on both sides. Dark gray."""
        from . import debugdraw
        import math

        f = ctx.field
        hl, hw = f.length / 2.0, f.width / 2.0
        gray = (0.5, 0.5, 0.5)
        # Outer boundary
        debugdraw.line(
            [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)],
            rgb=gray, ns="field_bounds",
        )
        # Midline
        debugdraw.line([(0.0, -hw), (0.0, hw)], rgb=gray, ns="field_midline")
        # Center circle (polygon approximation)
        r = f.circle_radius
        circle = [
            (r * math.cos(a), r * math.sin(a))
            for a in [i * math.pi / 12 for i in range(25)]
        ]
        debugdraw.line(circle, rgb=gray, ns="field_circle")
        # Goal frames on both sides (half goal width x depth 0.6)
        gw = f.goal_width / 2.0
        depth = 0.6
        for sx in (-1.0, 1.0):
            fx = sx * hl
            bx = sx * (hl + depth)
            debugdraw.line(
                [(fx, -gw), (bx, -gw), (bx, gw), (fx, gw)],
                rgb=gray, ns="field_goal",
            )

    def _log_heartbeat(self, ctx: Context, dt: float) -> None:
        ball_repr = (
            "None" if ctx.ball is None else f"({ctx.ball.x:.2f},{ctx.ball.y:.2f})"
        )
        seen = sum(1 for r in ctx.teammates.values() if r.pose is not None)
        opp_seen = sum(1 for r in ctx.opponents.values() if r.pose is not None)
        _log.info(
            "tick #%d dt=%.3f game=%s ball=%s teammates_seen=%d/%d opponents_seen=%d/%d",
            self._tick_id, dt,
            "None" if ctx.game is None else ctx.game.state.value,
            ball_repr,
            seen, len(ctx.teammates),
            opp_seen, len(ctx.opponents),
        )

    # ------------------------------------------------------------------
    # Context construction + freshness filtering
    # ------------------------------------------------------------------

    def _build_context(self, now: float, dt: float) -> Context:
        snap = self._source.get_snapshot() if self._source is not None else WorldSnapshot()
        return Context(
            now=now,
            dt=dt,
            team_id=self._config.team_id,
            field=ADULT_FIELD_DIMENSIONS,
            game=self._fresh_game(snap.game, now),
            ball=self._fresh_ball(snap.ball, now),
            teammates={
                pid: self._fresh_robot(r, now) for pid, r in snap.teammates.items()
            },
            opponents={
                pid: self._fresh_robot(r, now) for pid, r in snap.opponents.items()
            },
        )

    def _fresh_game(
        self, game: GameControlState | None, now: float,
    ) -> GameControlState | None:
        if game is None:
            return None
        if now - game.last_seen_at > self._config.game_state_max_age_sec:
            return None
        return game

    def _fresh_ball(self, ball: BallState | None, now: float) -> BallState | None:
        if ball is None:
            return None
        if now - ball.last_seen_at > self._config.ball_max_age_sec:
            return None
        return ball

    def _fresh_robot(self, robot: RobotState, now: float) -> RobotState:
        """If the pose is stale, clear it to None (the robot object itself is kept), see doc section 9.3."""
        if (
            robot.pose is not None
            and now - robot.last_seen_at > self._config.robot_pose_max_age_sec
        ):
            return dataclasses.replace(robot, pose=None)
        return robot
