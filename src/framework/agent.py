"""SoccerAgent framework mixin: the source of framework behavior for the user's entry class.

Platform constraint: Booster's build validation only checks whether the
entry class's *direct base class* is ``booster_agent_framework.AgentBase``;
it does not walk up multiple levels of inheritance. So the framework can't
just provide a ``SoccerAgent(AgentBase)`` for the user to single-inherit
from -- that would make the user class's direct base SoccerAgent, and
validation would fail.

Solution: put the framework behavior in a mixin that does *not* inherit
from AgentBase, so the user's entry class is written as
``class MyAgent(SoccerAgentMixin, AgentBase)``, making AgentBase one of the
direct base classes.

See docs/new_design.md section 8 for the detailed API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from booster_agent_framework import AgentFeatures

from .config import SoccerConfig
from .runtime import SoccerRuntime

if TYPE_CHECKING:
    from types import SimpleNamespace

    from ..player import Player
    from .types import Context


__all__ = ["SoccerAgentMixin"]


_log = logging.getLogger(__name__)


class _PlatformLogHandler(logging.Handler):
    """Forward standard Python logging records to the Booster platform logger (``self.logger``).

    The platform logger is rclcpp-style, offering only
    ``.info/.warn/.error(msg: str)``. Standard logging has no handler by
    default and INFO is dropped, so neither framework nor user code logs
    would be visible; once this bridge is installed,
    ``logging.getLogger(__name__).info(...)`` is correctly routed to the
    console/log file.

    Platform coupling is concentrated here only; runtime / player etc.
    still use platform-agnostic standard logging.
    """

    def __init__(self, platform_logger: object) -> None:
        super().__init__()
        self._platform = platform_logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                self._platform.error(msg)  # type: ignore[attr-defined]
            elif record.levelno >= logging.WARNING:
                warn = getattr(self._platform, "warn", None) or getattr(
                    self._platform, "warning", None
                )
                (warn or self._platform.info)(msg)  # type: ignore[attr-defined]
            else:
                self._platform.info(msg)  # type: ignore[attr-defined]
        except Exception:
            self.handleError(record)


class SoccerAgentMixin:
    """Framework behavior mixin; does not inherit from AgentBase, must be combined with AgentBase.

    Usage::

        from booster_agent_framework import AgentBase
        from .soccer.agent import SoccerAgentMixin

        class MyAgent(SoccerAgentMixin, AgentBase):
            player_class = MyPlayer

            @staticmethod
            def play(context, players, store): ...

            def init_store(self, store): ...

    The MRO is ``[MyAgent, SoccerAgentMixin, AgentBase, object]``, so
    ``super().__init__`` correctly reaches ``AgentBase.__init__``.
    """

    # ------------------------------------------------------------------
    # Slots filled in by the user
    # ------------------------------------------------------------------

    # Slot 1: the Player class. main.py must set ``player_class = Player``
    #         (imported from src.player). The framework does not import the
    #         user's player.py -- dependency injection keeps the dependency
    #         pointing downward.
    player_class: "type[Player]"

    # Slot 2: play -- called every frame (no-op by default)
    @staticmethod
    def play(
        context: "Context",
        players: "list[Player]",
        store: "SimpleNamespace",
    ) -> None:
        """Called at 30Hz. Does nothing by default; subclasses override as needed."""

    # Slot 3: init_store -- called once before the match starts (optional)
    def init_store(self, store: "SimpleNamespace") -> None:
        """No-op by default; subclasses override as needed."""

    # ------------------------------------------------------------------
    # Framework internals -- users normally don't touch these
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Follow the MRO to AgentBase.__init__(AgentFeatures())
        super().__init__(AgentFeatures())  # type: ignore[call-arg]
        self._setup_logging()
        self.config = SoccerConfig.from_env()
        # ROS data source (Docker-only); deferred import to avoid polluting
        # dev machines without rclpy
        from .ros_source import RosContextSource

        source = RosContextSource(self.config)
        self.runtime = SoccerRuntime(self, context_source=source)
        # Create a backend (SDK wrapper) for each player and inject it
        self._create_backends()
        _log.info(
            "SoccerAgent initialized: team_id=%d robots=%s",
            self.config.team_id, list(self.config.robot_names),
        )

    def _create_backends(self) -> None:
        """Create an SDK backend for each player. Docker-only, deferred import."""
        from .robot_backend import RobotBackend

        for player in self.runtime._players:
            robot_name = self.config.robot_names[player.id - 1]
            player._backend = RobotBackend(player.id, robot_name)

    def _setup_logging(self) -> None:
        """Bridge standard logging to the platform logger so framework and user logs are visible.

        ``self.logger`` is provided by AgentBase after ``super().__init__``.
        The bridge is only installed once; on repeated activation, the old
        bridge is removed first to avoid duplicate output.
        """

        platform_logger = getattr(self, "logger", None)
        if platform_logger is None:
            # No platform logger (shouldn't happen in theory): fall back to
            # stderr, since the framework has already redirected the std streams.
            logging.basicConfig(level=logging.INFO)
            return
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for handler in list(root.handlers):
            if isinstance(handler, _PlatformLogHandler):
                root.removeHandler(handler)
        bridge = _PlatformLogHandler(platform_logger)
        bridge.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(bridge)

    def on_agent_activated(self) -> None:
        _log.info("SoccerAgent activated")
        self.runtime.start()

    def on_agent_close(self) -> None:
        _log.info("SoccerAgent closing")
        self.runtime.stop()
