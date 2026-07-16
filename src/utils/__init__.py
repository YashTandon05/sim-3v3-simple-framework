"""工具层(utils)—— 框架预置的样例工具 + 用户自己加的工具。

纯函数、无状态、无平台依赖,可独立复用。
用户可直接改这里,或新增自己的 util 模块(如出界判定、传球评分等)。

- geom:几何计算工具(dist / angle_to / clamp / normalize_angle)
- obstacles:避障(Obstacle / collect_obstacles / detour)

注:走位/活性(walk_to / face_to / ensure_ready)是"对 player 下命令的动词"、
且需要跨帧状态,已作为 Player 方法放在 src/player.py,不在 utils。
"""

from .geom import angle_to, clamp, dist, normalize_angle
from .obstacles import Obstacle, collect_obstacles, detour

__all__ = [
    "Obstacle",
    "angle_to",
    "clamp",
    "collect_obstacles",
    "detour",
    "dist",
    "normalize_angle",
]
