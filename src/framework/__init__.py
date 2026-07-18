"""Framework layer -- platform plumbing that users almost never modify.

Contains: data contracts (types), configuration (config), the runtime main
loop (runtime), the SDK wrapper (robot_backend), the ROS data source
(ros_source), game controller decoding (game_codec), and the Agent entry
mixin (agent).

Import directly from submodules as needed (e.g.
``from .framework.types import Context``); ``__init__`` does not do eager
imports, to avoid import failures on dev machines without the SDK, since
agent/robot_backend/ros_source depend on the SDK.
"""
