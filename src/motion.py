"""底层执行层 —— MotionMixin 提供机器人低层次执行能力。

Player 通过继承 MotionMixin 获得以下能力:
- 基础控制: set_velocity(速度指令)、stop(停止)、request_mode(模式切换)、get_up(起身)、ensure_ready(就绪检查)
- 行走: walk_to(支持避障)、face_to(原地转向)
- 踢球: kick(执行踢球)、plan_kick(规划踢球)、release_kick(释放踢球)、kick_can_score(进球判断)
- 辅助: block_path_projection(拦截点计算)

设计原则:
- 不涉及策略逻辑,只负责"怎么执行"
- 通过 self.context / self.pose / self.id / self.set_velocity 访问 Player 状态
- 异常时调用 self.stop() 兜底
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .framework.types import Action, Pose2D
from .framework import debugdraw

from .param import *
from .utils.geom import angle_to, clamp, dist, normalize_angle
from .utils.obstacles import collect_obstacles
from .utils.path_planner import plan_global_path

if TYPE_CHECKING:
    from .framework.types import Context

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# 全局工具函数
# ------------------------------------------------------------------------

def _heading_clearance(
    px: float, py: float, heading: float, obstacles: list,
) -> float:
    """计算沿某方向的最小障碍距离。"""
    ux, uy = math.cos(heading), math.sin(heading)
    min_clear = math.inf
    for obs in obstacles:
        t = (obs.x - px) * ux + (obs.y - py) * uy
        if t <= 0.0:
            continue
        if t > PLAN_LOOKAHEAD:
            t = PLAN_LOOKAHEAD
        nx, ny = px + ux * t, py + uy * t
        clear = math.hypot(obs.x - nx, obs.y - ny) - obs.radius
        if clear < min_clear:
            min_clear = clear
    return min_clear


def _behind_ball(
    ball_x: float, ball_y: float, aim: tuple[float, float], offset: float,
) -> tuple[float, float]:
    """计算球后方 offset 距离处的点(沿球到目标的方向)。"""
    dx, dy = aim[0] - ball_x, aim[1] - ball_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (ball_x, ball_y)
    ux, uy = dx / d, dy / d
    return (ball_x - ux * offset, ball_y - uy * offset)


# ------------------------------------------------------------------------
# MotionMixin 类
# ------------------------------------------------------------------------

class MotionMixin:
    """机器人低层次执行功能集，Player 通过继承获得。"""

    _backend: object | None = None
    _kicking: bool = False

    @property
    def is_kicking(self) -> bool:
        return self._kicking

    # ------------------------------------------------------------------------
    # 基础控制
    # ------------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """发送速度指令到机器人。

        参数:
            vx: 前进速度 (m/s)
            vy: 侧向速度 (m/s)
            vyaw: 转向角速度 (rad/s)
        """
        if self._backend is None:
            _log.debug(
                "player %d set_velocity vx=%.3f vy=%.3f vyaw=%.3f (no backend)",
                getattr(self, "id", "?"), vx, vy, vyaw,
            )
            return
        self._backend.set_velocity(vx, vy, vyaw)

    def stop(self, action = Action.STOPPED) -> None:
        """停止所有动作。"""
        self.action = action
        self.release_kick()
        self.set_velocity(0.0, 0.0, 0.0)

    def request_mode(self, mode: str) -> None:
        """请求切换机器人模式。

        参数:
            mode: 目标模式名称
        """
        if self._backend is None:
            _log.debug("player %d request_mode -> %s (no backend)", getattr(self, "id", "?"), mode)
            return
        self._backend.request_mode(mode)

    def get_up(self) -> None:
        """执行起身动作。"""
        if self._backend is None:
            _log.debug("player %d get_up (no backend)", getattr(self, "id", "?"))
            return
        self._backend.get_up()

    def ensure_ready(self) -> bool:
        """确保球员处于可行动状态。

        检查是否摔倒或模式不对，必要时起身或切换模式。

        返回:
            bool: 已就绪返回 True；正在起身或切换模式返回 False
        """
        if self.is_fallen:
            self.get_up()
            return False
        if self.mode != "walk":
            self.request_mode("walk")
            return False
        return True

    # ------------------------------------------------------------------------
    # 行走控制
    # ------------------------------------------------------------------------

    def face_to(self, target_theta: float, action: Action = Action.TURN) -> None:
        """原地转向至指定角度。

        参数:
            target_theta: 目标朝向角度(弧度)
            action: 设置球员 action 状态, None 表示不修改
        """
        self.action = action
        try:
            pose = self.pose
            if pose is None:
                self.stop()
                return
            err = normalize_angle(target_theta - pose.theta)
            self.set_velocity(0.0, 0.0, self._angular(err))
        except Exception:
            self.stop()

    def walk_to(
        self,
        target: tuple[float, float],
        *,
        face: float | None = None,
        avoid_ball: bool = False,
        avoid_robots: bool = False,
        arrive_dist: float = ARRIVE_DIST,
        action: Action | None = Action.WALK,
    ) -> bool:
        """行走至目标点。

        支持全局路径规划(A*)和局部避障(VFH)，近距离使用全向控制。

        参数:
            target: 目标坐标 (x, y)
            face: 到达后转向的目标角度(弧度), None 表示不转向
            avoid_ball: 是否避开球
            avoid_robots: 是否避开机器人
            arrive_dist: 到达距离阈值
            action: 设置球员 action 状态, None 表示不修改

        返回:
            bool: 已到达返回 True；正在移动返回 False
        """
        self.action = action
        try:
            self.release_kick()
            pose = self.pose
            if pose is None:
                self.stop()
                return False

            tx, ty = target
            dx = tx - pose.x
            dy = ty - pose.y
            distance = math.hypot(dx, dy)

            # 到达判断
            if distance < arrive_dist:
                if face is not None:
                    err = normalize_angle(face - pose.theta)
                    if abs(err) > 0.1:
                        self.set_velocity(0.0, 0.0, self._angular(err))
                        return True
                self.stop()
                return True

            # 方向规划
            goal_dir = math.atan2(dy, dx)
            planned_path: list[tuple[float, float]] | None = None
            waypoint: tuple[float, float] | None = None
            ctx: Context | None = getattr(self, "context", None)
            if (avoid_ball or avoid_robots) and ctx is not None:
                obstacles = collect_obstacles(
                    ctx, self.id,
                    ball=avoid_ball, robots=avoid_robots,
                    goals=(avoid_ball or avoid_robots),
                )
                if USE_GLOBAL_PATH_PLANNER:
                    planned_path = plan_global_path(
                        ctx, (pose.x, pose.y), (tx, ty), obstacles,
                    )
                    if planned_path is not None:
                        waypoint = self._path_waypoint(pose, planned_path)
                        heading = angle_to(pose.x, pose.y, waypoint[0], waypoint[1])
                    else:
                        heading = self._plan_heading(pose, goal_dir, obstacles)
                else:
                    heading = self._plan_heading(pose, goal_dir, obstacles)
            else:
                heading = goal_dir

            self._debug_draw_walk_info(pose, tx, ty, heading, planned_path, waypoint)

            # 速度计算
            if distance <= OMNI_DIST:
                # 近距离全向控制
                wdx, wdy = math.cos(heading) * distance, math.sin(heading) * distance
                cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
                vx = LINEAR_GAIN * (wdx * cos_t + wdy * sin_t)
                vy = LINEAR_GAIN * (-wdx * sin_t + wdy * cos_t)
                speed = math.hypot(vx, vy)
                if speed > MAX_LINEAR:
                    vx *= MAX_LINEAR / speed
                    vy *= MAX_LINEAR / speed
                vyaw = (
                    self._angular(normalize_angle(face - pose.theta))
                    if face is not None else 0.0
                )
                self.set_velocity(vx, vy, vyaw)
            else:
                # 远距离先转向再前进
                angle_err = normalize_angle(heading - pose.theta)
                if abs(angle_err) > TURN_THRESHOLD:
                    self.set_velocity(0.0, 0.0, self._angular(angle_err))
                else:
                    vx = clamp(
                        LINEAR_GAIN * distance * math.cos(angle_err), 0.0, MAX_LINEAR,
                    )
                    self.set_velocity(vx, 0.0, self._angular(angle_err))
            return False
        except Exception:
            self.stop()
            return False

    # ------------------------------------------------------------------------
    # 踢球控制
    # ------------------------------------------------------------------------

    def kick(
        self,
        kick_direction: float | None = None,
        power: float = KICK_POWER_DEFAULT,
        action: Action | None = None,
    ) -> bool:
        """执行踢球。

        支持自动规划踢球方向，只需传入 kick_direction=None。

        参数:
            kick_direction: 踢球方向(弧度),None 表示自动规划
            power: 踢球力度
            action: 设置球员 action 状态, None 表示不修改

        返回:
            bool: 成功执行踢球返回 True,否则返回 False
        """
        if action is not None:
            self.action = action
        try:
            pose = self.pose
            if pose is None:
                _log.warning("player %d kick skipped: pose unknown", self.id)
                return False

            ctx = getattr(self, "context", None)
            ball = ctx.ball if ctx is not None else None
            if ball is None:
                _log.warning("player %d kick skipped: ball unknown", self.id)
                return False

            # 自动规划踢球方向
            if kick_direction is None:
                kick_plan = self.plan_kick()
                if kick_plan is None:
                    _log.warning("player %d kick skipped: kick plan unavailable", self.id)
                    return False
                kick_direction, power = kick_plan

            if self._backend is None:
                _log.debug(
                    "player %d kick ball=(%.3f, %.3f) dir=%.3f (no backend)",
                    self.id, ball.x, ball.y, kick_direction,
                )
                return True

            # 转换到机器人坐标系
            dx = ball.x - pose.x
            dy = ball.y - pose.y
            cos_t = math.cos(pose.theta)
            sin_t = math.sin(pose.theta)
            ball_x_body = dx * cos_t + dy * sin_t
            ball_y_body = -dx * sin_t + dy * cos_t
            kick_direction = normalize_angle(kick_direction)
            direction_body = normalize_angle(kick_direction - pose.theta)
            power_clamped = max(KICK_POWER_MIN, min(KICK_POWER_MAX, power))
            self._kicking = True
            self._backend.kick(direction_body, power_clamped, ball_x_body, ball_y_body)
            return True
        except Exception:
            self.stop()
            return False

    def plan_kick(self) -> tuple[float, float] | None:
        """规划踢球方向和力度。

        方向指向对方球门，力度根据球是否在后场调整。

        返回:
            tuple[float, float]: (踢球方向弧度, 踢球力度), 失败返回 None
        """
        ctx = getattr(self, "context", None)
        ball = ctx.ball if ctx else None
        if not (ctx and ball):
            return None

        kick_target = ctx.field.opponent_goal
        kick_direction = angle_to(ball.x, ball.y, *kick_target)
        kick_target = self._goal_target_for_direction(ctx, kick_direction)
        kick_power = (
            KICK_POWER_BACKFIELD if self._in_backfield(ctx)
            else KICK_POWER_DEFAULT
        )

        self._draw_kick_target(kick_target)
        return kick_direction, kick_power

    def release_kick(self) -> None:
        """释放踢球状态。"""
        self._kicking = False
        if self._backend is None:
            _log.debug("player %d release_kick (no backend)", self.id)
            return
        self._backend.release_kick()

    def kick_can_score(self, kick_direction: float) -> bool:
        """判断当前踢球方向能否进球。"""
        ctx = getattr(self, "context", None)
        ball = ctx.ball if ctx else None
        if not (ctx and ball):
            return False

        goal_x = ctx.field.half_length
        dx = math.cos(kick_direction)
        dy = math.sin(kick_direction)
        if dx <= 1e-6:
            return False

        half_goal = ctx.field.goal_width / 2.0
        if half_goal <= 0.0:
            return False
        if ball.x >= goal_x:
            y_at_goal = ball.y
        else:
            t = (goal_x - ball.x) / dx
            if t < 0.0:
                return False
            y_at_goal = ball.y + dy * t
        return -half_goal <= y_at_goal <= half_goal

    # ------------------------------------------------------------------------
    # 踢球辅助方法(私有)
    # ------------------------------------------------------------------------

    def _goal_target_for_direction(
        self, ctx: Context, kick_direction: float,
    ) -> tuple[float, float]:
        """计算沿踢球方向到达球门线的交点。"""
        ball = ctx.ball
        if ball is None:
            return (0.0, 0.0)

        dx = math.cos(kick_direction)
        if dx <= 1e-6:
            return ctx.field.opponent_goal
        goal_x = ctx.field.half_length
        t = max(0.0, (goal_x - ball.x) / dx)
        return (goal_x, ball.y + math.sin(kick_direction) * t)

    def _in_backfield(self, ctx: Context) -> bool:
        """判断球是否在后场(己方半场)。"""
        ball = ctx.ball
        return ball is not None and ball.x < 0

    def _draw_kick_target(self, target: tuple[float, float]) -> None:
        """绘制踢球目标标记。"""
        x, y = target
        s = KICK_TARGET_MARK_SIZE_M
        debugdraw.line(
            [(x - s, y - s), (x + s, y + s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )
        debugdraw.line(
            [(x - s, y + s), (x + s, y - s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )

    # ------------------------------------------------------------------------
    # 辅助功能
    # ------------------------------------------------------------------------

    def block_path_projection(
        self, opponent_id: int,
    ) -> tuple[float, float, float, float] | None:
        """计算对手到球路径上的拦截点。

        返回:
            tuple: (拦截点x, 拦截点y, 距离, 投影参数t), 失败返回 None
        """
        ctx = getattr(self, "context", None)
        pose = self.pose
        ball = ctx.ball if ctx else None
        opponent = ctx.opponents.get(opponent_id) if ctx else None
        if not (ctx and pose and ball and opponent and opponent.pose):
            return None

        ax, ay = opponent.pose.x, opponent.pose.y
        bx, by = ball.x, ball.y
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 < 1e-6:
            return None

        raw_t = ((pose.x - ax) * vx + (pose.y - ay) * vy) / length2
        t = clamp(raw_t, 0.0, 1.0)
        tx = ax + vx * t
        ty = ay + vy * t
        return tx, ty, dist(pose.x, pose.y, tx, ty), raw_t

    # ------------------------------------------------------------------------
    # 内部实现(私有方法)
    # ------------------------------------------------------------------------

    def _path_waypoint(
        self, pose: Pose2D, path: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """从规划路径中提取前方 LOOKAHEAD 距离处的路径点。"""
        if not path:
            return (pose.x, pose.y)
        prev = (pose.x, pose.y)
        points = path[1:] if len(path) > 1 else path
        for point in points:
            seg_len = dist(prev[0], prev[1], point[0], point[1])
            if seg_len >= GLOBAL_PATH_LOOKAHEAD_M:
                ratio = GLOBAL_PATH_LOOKAHEAD_M / max(seg_len, 1e-6)
                return (
                    prev[0] + (point[0] - prev[0]) * ratio,
                    prev[1] + (point[1] - prev[1]) * ratio,
                )
            prev = point
        return path[-1]

    def _debug_draw_walk_info(
        self,
        pose: Pose2D,
        tx: float,
        ty: float,
        heading: float,
        planned_path: list[tuple[float, float]] | None,
        waypoint: tuple[float, float] | None,
    ) -> None:
        """绘制行走调试信息(目标点、路径、方向等)。"""
        debugdraw.point(tx, ty, rgb=(0.0, 1.0, 0.0), scale=0.15, ns="target")
        debugdraw.line([(pose.x, pose.y), (tx, ty)], rgb=(0.4, 0.4, 0.4), ns="to_target")
        if planned_path is not None and len(planned_path) >= 2:
            debugdraw.line(planned_path, rgb=(0.2, 0.8, 1.0), ns="global_path")
        if waypoint is not None:
            debugdraw.point(
                waypoint[0], waypoint[1],
                rgb=(0.2, 0.8, 1.0), scale=0.12, ns="global_waypoint",
            )
        debugdraw.arrow(
            pose.x, pose.y,
            pose.x + math.cos(heading) * 0.6, pose.y + math.sin(heading) * 0.6,
            rgb=(1.0, 1.0, 0.0), ns="heading",
        )
        debugdraw.line(
            [(pose.x, pose.y),
             (pose.x + math.cos(heading) * PLAN_LOOKAHEAD,
              pose.y + math.sin(heading) * PLAN_LOOKAHEAD)],
            rgb=(0.0, 0.8, 0.8), ns="lookahead",
        )

    def _plan_heading(
        self, pose: Pose2D, goal_dir: float, obstacles: list,
    ) -> float:
        """局部避障方向规划(VFH)。

        在目标方向附近扫描候选方向，选择最开阔且距离目标最近的方向。
        """
        sign_first = 1.0 if self.id % 2 == 0 else -1.0
        best_h = goal_dir
        best_clear = -math.inf

        offsets = [0.0]
        k = 1
        while k * PLAN_STEP <= PLAN_MAX_OFFSET + 1e-9:
            offsets.append(sign_first * k * PLAN_STEP)
            offsets.append(-sign_first * k * PLAN_STEP)
            k += 1

        for off in offsets:
            h = goal_dir + off
            clear = _heading_clearance(pose.x, pose.y, h, obstacles)
            if clear >= PLAN_CLEARANCE:
                return h
            if clear > best_clear:
                best_clear, best_h = clear, h
        return best_h

    @staticmethod
    def _angular(err: float) -> float:
        """角度误差转角速度(带限幅)。"""
        return clamp(ANGULAR_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR)