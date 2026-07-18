"""BoosterRobot SDK wrapper: one backend handle per player.

[Platform layer, Docker-only] Depends on boosteros.robots.booster, only
imported in a runtime environment with the SDK installed. Player calls the
chassis/kick/slow operations via ``_backend``, but Player itself
(player.py) does not import this module -- it is injected by agent at
runtime construction time, keeping player.py platform-agnostic.

SDK method names match the validated calls from the old code.

Slow operations (request_mode / get_up) are second-scale synchronous SDK
calls, executed on one worker thread per backend so the main loop stays
non-blocking. The intent queue has length 1 (overwrite semantics): only
the latest of consecutive requests is kept.

Mode is managed by the user via request_mode (see docs/new_design.md
section 5); set_velocity / kick are only issued when ``_mode == "walk"``,
otherwise skipped -- this avoids flooding logs with 400 responses from the
SDK when not in walk mode.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, cast

from boosteros.robots.booster import BoosterRobot, SoccerKickManager


__all__ = ["RobotBackend"]


_log = logging.getLogger(__name__)

_GET_UP_THROTTLE_SEC = 1.0


class RobotBackend:
    """SDK wrapper + slow-operation worker thread for a single player.

    Lifecycle: created by the mixin and injected into Player; closed
    uniformly (stopping the worker) when runtime stops.
    """

    def __init__(self, player_id: int, robot_name: str) -> None:
        self._player_id = player_id
        self._robot_name = robot_name
        self._robot = BoosterRobot(
            virtual_robot_name=robot_name,
            enable_tf_listener=False,
            timeout=10.0,
        )
        self._kick_manager = SoccerKickManager(self._robot)
        self._mode: str | None = None   # confirmed SDK mode (updated by the worker)
        self._fall_down_state: str | None = None
        self._fall_down_recoverable: bool = False
        self._kicking = False

        # Slow-operation worker: length-1 overwrite intent slot + wake event
        self._pending: tuple[str, object] | None = None
        self._slot_lock = threading.Lock()
        self._wake = threading.Event()
        self._worker_stop = threading.Event()
        self._last_get_up_at = 0.0
        self._actions_probed = False   # one-shot startup log of the SDK's predefined action list
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"backend_worker_{player_id}",
            daemon=True,
        )
        self._worker.start()

        _log.info(
            "RobotBackend created: player_id=%d robot_name=%s",
            player_id, robot_name,
        )

    def close(self) -> None:
        """Release SDK resources: stop the worker, stop kicking, stop the chassis, close the connection."""
        self._worker_stop.set()
        self._wake.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.release_kick()
        try:
            self._robot.set_velocity(vx=0.0, vy=0.0, vyaw=0.0)
        except Exception as exc:
            _log.warning(
                "player %d set_velocity(0,0,0) on close failed: %s",
                self._player_id, exc,
            )
        try:
            close_fn = getattr(self._robot, "_close", None)
            if callable(close_fn):
                cast(Callable[[], None], close_fn)()
        except Exception as exc:
            _log.warning("player %d SDK close failed: %s", self._player_id, exc)

    @property
    def mode(self) -> str | None:
        """Currently confirmed SDK mode; updated once the worker completes a switch."""
        return self._mode

    @property
    def fall_down_state(self) -> str | None:
        """Current SDK fall-down state; None means unknown."""
        return self._fall_down_state

    @property
    def fall_down_recoverable(self) -> bool:
        return self._fall_down_recoverable

    # ------------------------------------------------------------------
    # Chassis control (step 1)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Chassis velocity. Skipped while kicking (kicking has exclusive control of the chassis); skipped outside walk mode (call request_mode first)."""
        if self._kicking:
            return
        if self._mode != "walk":
            _log.debug(
                "player %d set_velocity skipped: mode=%s (call request_mode first)",
                self._player_id, self._mode,
            )
            return
        try:
            self._robot.set_velocity(vx=vx, vy=vy, vyaw=vyaw)
        except Exception as exc:
            _log.warning(
                "player %d set_velocity(%.3f,%.3f,%.3f) failed: %s",
                self._player_id, vx, vy, vyaw, exc,
            )

    # ------------------------------------------------------------------
    # Kicking (step 2) -- parameters are in body frame
    # ------------------------------------------------------------------

    def kick(
        self, direction: float, power: float, ball_x: float, ball_y: float,
    ) -> None:
        """Start or update a kick (body frame). Skipped outside walk mode (call request_mode first)."""
        if self._mode != "walk":
            _log.debug(
                "player %d kick skipped: mode=%s (call request_mode first)",
                self._player_id, self._mode,
            )
            return
        try:
            if not self._kicking:
                self._kick_manager.start()
                self._kicking = True
                _log.info("player %d kick started", self._player_id)
            self._kick_manager.update_command(direction=direction, power=power)
            self._kick_manager.update_ball(x=ball_x, y=ball_y)
        except Exception as exc:
            _log.warning("player %d kick failed: %s", self._player_id, exc)
            self._kicking = False

    def release_kick(self) -> None:
        """End the kick; the chassis resumes accepting set_velocity."""
        if not self._kicking:
            return
        try:
            self._kick_manager.stop()
            _log.info("player %d kick released", self._player_id)
        except Exception as exc:
            _log.warning("player %d kick stop failed: %s", self._player_id, exc)
        finally:
            self._kicking = False

    # ------------------------------------------------------------------
    # Slow operations (step 3) -- non-blocking, executed by the worker thread
    # ------------------------------------------------------------------

    def request_mode(self, mode: str) -> None:
        """Asynchronously request an SDK mode switch. Short-circuits if already in the target mode."""
        if self._mode == mode:
            return
        self._enqueue(("mode", mode))

    def get_up(self) -> None:
        """Asynchronously trigger get-up. Throttled to ~1s; safe to call unconditionally every frame."""
        now = time.monotonic()
        if now - self._last_get_up_at < _GET_UP_THROTTLE_SEC:
            return
        self._last_get_up_at = now
        self._enqueue(("get_up", None))

    def do_action(self, action_id: object) -> None:
        """Asynchronously trigger a predefined SDK motion (e.g. a goalkeeper
        dive). The canned motion owns the body while it runs; afterwards the
        mode is unknown, so ``ensure_ready`` re-requests walk and the robot
        self-recovers. Action ids come from the startup ``list_actions`` log."""
        self._enqueue(("action", action_id))

    def _enqueue(self, intent: tuple[str, object]) -> None:
        with self._slot_lock:
            self._pending = intent   # overwrite semantics: only the latest is kept
        self._wake.set()

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            self._wake.wait(timeout=0.5)
            if self._worker_stop.is_set():
                break
            self._wake.clear()
            # Poll the real SDK mode to keep _mode reflecting reality (rather
            # than an optimistic cache). After a match restart the simulator
            # resets robots out of walk mode; polling lets _mode follow along,
            # so higher-level ensure_ready, seeing p.mode != "walk", will
            # automatically re-call request_mode and self-heal.
            self._poll_mode()
            self._poll_fall_down_state()
            self._probe_actions_once()
            with self._slot_lock:
                intent = self._pending
                self._pending = None
            if intent is None:
                continue
            kind, arg = intent
            if kind == "mode":
                self._exec_set_mode(cast(str, arg))
            elif kind == "get_up":
                self._exec_get_up()
            elif kind == "action":
                self._exec_action(arg)

    def _poll_mode(self) -> None:
        try:
            mode = self._robot.get_mode()
        except Exception as exc:
            _log.debug("player %d get_mode failed: %s", self._player_id, exc)
            return
        if isinstance(mode, str):
            self._mode = mode
            if mode == "walk":
                self._fall_down_state = "normal"
                self._fall_down_recoverable = False

    def _poll_fall_down_state(self) -> None:
        if self._mode == "walk":
            return
        try:
            fall_down_state = self._robot.get_fall_down_state()
        except Exception as exc:
            _log.debug(
                "player %d get_fall_down_state failed: %s", self._player_id, exc,
            )
            return
        state_value = getattr(fall_down_state, "state", None)
        recoverable_value = getattr(fall_down_state, "recoverable", False)
        self._fall_down_state = state_value if isinstance(state_value, str) else None
        self._fall_down_recoverable = (
            recoverable_value if isinstance(recoverable_value, bool) else False
        )

    def _exec_set_mode(self, mode: str) -> None:
        try:
            self._robot.set_gait("soccer")
            self._robot.set_mode(mode)
            self._mode = mode   # optimistic immediate feedback; the next _poll_mode will correct it with the real value
            _log.info("player %d entered %s mode", self._player_id, mode)
        except Exception as exc:
            _log.warning("player %d set_mode(%s) failed: %s", self._player_id, mode, exc)

    def _exec_get_up(self) -> None:
        try:
            self._robot.get_up()
            self._mode = None   # mode is unknown after getting up, needs a fresh request_mode
            self._fall_down_state = None
            self._fall_down_recoverable = False
            _log.info("player %d get_up done", self._player_id)
        except Exception as exc:
            _log.warning("player %d get_up failed: %s", self._player_id, exc)

    def _probe_actions_once(self) -> None:
        """Log the SDK's predefined action list once at startup, so the real
        action ids (e.g. a dive) can be read from the match log and configured
        in param.py (``KEEPER_DIVE_ACTION_LEFT/RIGHT``)."""
        if self._actions_probed:
            return
        self._actions_probed = True
        try:
            list_fn = getattr(self._robot, "list_actions", None)
            if callable(list_fn):
                _log.info(
                    "player %d available SDK actions: %s",
                    self._player_id, list_fn(),
                )
            else:
                _log.info("player %d SDK has no list_actions", self._player_id)
        except Exception as exc:
            _log.info("player %d list_actions failed: %s", self._player_id, exc)

    def _exec_action(self, action_id: object) -> None:
        """Run a predefined SDK motion synchronously on the worker. Kicking is
        released first (it owns the chassis); afterwards mode is unknown, so the
        normal ensure_ready path re-requests walk mode and the robot recovers."""
        try:
            self.release_kick()
            self._robot.do_action(action_id)
            self._mode = None
            _log.info("player %d action %r done", self._player_id, action_id)
        except Exception as exc:
            _log.warning(
                "player %d do_action(%r) failed: %s",
                self._player_id, action_id, exc,
            )
