"""Core data types -- the framework's "data contract layer".

Field definitions map directly to docs/new_design.md section 9. This
module has no ROS / boosteros dependencies, so it can be imported,
unit-tested, and reloaded independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


__all__ = [
    "ADULT_FIELD_DIMENSIONS",
    "MAX_NUM_PLAYERS",
    "BallState",
    "CompetitionType",
    "Context",
    "FieldDimensions",
    "GameControlState",
    "GamePhase",
    "GameState",
    "KickingTeam",
    "Penalty",
    "PlayerState",
    "Pose2D",
    "RobotState",
    "SetPlay",
    "TeamState",
    "WorldSnapshot",
]


MAX_NUM_PLAYERS = 20
KICKING_TEAM_NONE = 255


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------


class CompetitionType(str, Enum):
    SMALL = "SMALL"
    MIDDLE = "MIDDLE"
    LARGE = "LARGE"


class GamePhase(str, Enum):
    NORMAL = "NORMAL"
    PENALTY_SHOOT_OUT = "PENALTY_SHOOT_OUT"
    EXTRA_TIME = "EXTRA_TIME"
    TIMEOUT = "TIMEOUT"


class GameState(str, Enum):
    INITIAL = "INITIAL"
    READY = "READY"
    SET = "SET"
    PLAYING = "PLAYING"
    FINISHED = "FINISHED"


class SetPlay(str, Enum):
    NONE = "NONE"
    DIRECT_FREE_KICK = "DIRECT_FREE_KICK"
    INDIRECT_FREE_KICK = "INDIRECT_FREE_KICK"
    PENALTY_KICK = "PENALTY_KICK"
    THROW_IN = "THROW_IN"
    GOAL_KICK = "GOAL_KICK"
    CORNER_KICK = "CORNER_KICK"


class Penalty(str, Enum):
    NONE = "NONE"
    ILLEGAL_POSITIONING = "ILLEGAL_POSITIONING"
    MOTION_IN_SET = "MOTION_IN_SET"
    LOCAL_GAME_STUCK = "LOCAL_GAME_STUCK"
    INCAPABLE_ROBOT = "INCAPABLE_ROBOT"
    PICKED_UP = "PICKED_UP"
    BALL_HOLDING = "BALL_HOLDING"
    LEAVING_THE_FIELD = "LEAVING_THE_FIELD"
    PLAYING_WITH_ARMS_HANDS = "PLAYING_WITH_ARMS_HANDS"
    PUSHING = "PUSHING"
    SENT_OFF = "SENT_OFF"
    SUBSTITUTE = "SUBSTITUTE"


class KickingTeam(int, Enum):
    """Marker for the special value of the game controller's kicking_team field."""

    NONE = KICKING_TEAM_NONE


# ----------------------------------------------------------------------
# Basic data types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass(frozen=True)
class FieldDimensions:
    """Field geometry dimensions, values only, no methods.

    Geometry helpers (opponent_goal etc.) live in stdlib or user code, see
    docs/new_design.md section 9.3.
    """

    length: float
    width: float
    penalty_dist: float
    goal_width: float
    circle_radius: float
    penalty_area_length: float
    penalty_area_width: float
    goal_area_length: float
    goal_area_width: float


ADULT_FIELD_DIMENSIONS = FieldDimensions(
    length=14.0,
    width=9.0,
    penalty_dist=2.1,
    goal_width=2.6,
    circle_radius=1.5,
    penalty_area_length=3.0,
    penalty_area_width=6.0,
    goal_area_length=1.0,
    goal_area_width=4.0,
)


# ----------------------------------------------------------------------
# Observation types (ball / robot / game controller)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BallState:
    x: float = 0.0
    y: float = 0.0
    last_seen_at: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class RobotState:
    player_id: int
    pose: Pose2D | None = None
    last_seen_at: float = 0.0


@dataclass(frozen=True)
class PlayerState:
    penalty: Penalty = Penalty.NONE
    secs_till_unpenalised: int = 0
    warnings: int = 0
    cautions: int = 0


@dataclass(frozen=True)
class TeamState:
    team_number: int = 1
    field_player_colour: int = 0
    goalkeeper_colour: int = 0
    goalkeeper: int = 0
    score: int = 0
    penalty_shot: int = 0
    single_shots: int = 0
    message_budget: int = 0
    players: tuple[PlayerState, ...] = field(
        default_factory=lambda: tuple(PlayerState() for _ in range(MAX_NUM_PLAYERS))
    )


@dataclass(frozen=True)
class GameControlState:
    packet_number: int = 0
    players_per_team: int = 0
    competition_type: CompetitionType = CompetitionType.MIDDLE
    stopped: bool = False
    game_phase: GamePhase = GamePhase.NORMAL
    state: GameState = GameState.INITIAL
    set_play: SetPlay = SetPlay.NONE
    first_half: bool = True
    kicking_team: int = KICKING_TEAM_NONE
    secs_remaining: int = 0
    secondary_time: int = 0
    teams: tuple[TeamState, ...] = field(
        default_factory=lambda: (TeamState(team_number=1), TeamState(team_number=2))
    )
    last_seen_at: float = 0.0

    def get_team_state(self, team_id: int) -> TeamState | None:
        for team in self.teams:
            if team.team_number == team_id:
                return team
        return None

    def get_player_state(self, team_id: int, player_id: int) -> PlayerState | None:
        team = self.get_team_state(team_id)
        if team is None or player_id < 1 or player_id > len(team.players):
            return None
        return team.players[player_id - 1]


# ----------------------------------------------------------------------
# Context -- the first argument of play(context, players, store)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """Read-only snapshot built by the framework every frame, passed as play()'s argument.

    See docs/new_design.md section 9 for detailed field semantics.
    """

    now: float
    dt: float
    team_id: int
    field: FieldDimensions
    game: GameControlState | None = None
    ball: BallState | None = None
    teammates: dict[int, RobotState] = field(default_factory=dict)
    opponents: dict[int, RobotState] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldSnapshot:
    """Framework-internal: the raw snapshot the data source provides every frame (not yet freshness-filtered).

    The data source (e.g. ROS ground truth) is only responsible for
    filling in the latest observations along with ``last_seen_at``;
    freshness filtering (stale -> None) is done uniformly by runtime when
    constructing Context, see docs/new_design.md section 9.3.
    """

    game: GameControlState | None = None
    ball: BallState | None = None
    teammates: dict[int, RobotState] = field(default_factory=dict)
    opponents: dict[int, RobotState] = field(default_factory=dict)
