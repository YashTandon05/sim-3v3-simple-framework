"""几何计算工具 —— 纯函数,无状态,只依赖基础数值类型。

提供距离、角度、归一化等基础几何计算,不依赖 Context 或任何框架类型。
"""

from __future__ import annotations

import math


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def angle_to(fx: float, fy: float, tx: float, ty: float) -> float:
    return math.atan2(ty - fy, tx - fx)
