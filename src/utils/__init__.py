"""Utility layer (utils) — sample tools shipped with the framework + tools users add themselves.

Pure functions, stateless, no platform dependency (only depends on the
framework.types data contract), independently reusable. Users can edit these
directly, or add their own util modules (e.g. out-of-bounds detection, pass
scoring, etc.).

- geom: geometry helpers (opponent_goal / dist / angle_to / clamp / clamp_inside_field)
- obstacles: obstacle avoidance (Obstacle / collect_obstacles / detour)

Note: movement/liveness (walk_to / face_to / ensure_ready) are "verbs issued
to a player" and need cross-frame state, so they live as Player methods in
src/player.py, not in utils.
"""

from .geom import (
    angle_to,
    clamp,
    clamp_inside_field,
    defensive_screen_spot,
    deg2rad,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
    own_goal_area_center,
    push_clear_of_ball,
    rad2deg,
)
from .obstacles import Obstacle, collect_obstacles, detour

__all__ = [
    "Obstacle",
    "angle_to",
    "clamp",
    "clamp_inside_field",
    "collect_obstacles",
    "defensive_screen_spot",
    "deg2rad",
    "detour",
    "dist",
    "normalize_angle",
    "opponent_goal",
    "own_goal",
    "own_goal_area_center",
    "push_clear_of_ball",
    "rad2deg",
]
