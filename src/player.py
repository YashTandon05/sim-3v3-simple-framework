"""动作层 —— Player 是单个机器人的控制 handle。

三层架构中的中间层:
- 上层 main.py(策略层):决定"做什么"
- 本层(动作层):定义"怎么做"(attack / guard / support / take_kickoff)
- 下层 motion.py(执行层):负责"怎么执行"(walk_to / kick / release_kick)

Player 通过继承 MotionMixin 获得行走、踢球等低层次执行能力。
高层动作(attack / guard / support / take_kickoff)在此定义。

实例跨整场比赛存活;``self.context`` 每帧被框架覆写。用户想加自己的动作直接在这里加。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .framework.types import Action, Context, Penalty, Phase, Pose2D
from .framework import debugdraw

from .param import *
from .motion import MotionMixin, _behind_ball
from .utils.geom import angle_to, clamp, dist

if TYPE_CHECKING:
    from .framework.config import SoccerConfig


__all__ = ["Player"]


_log = logging.getLogger(__name__)


class Player(MotionMixin):
    """单个球员的 handle。高层动作定义在这里。"""

    def __init__(
        self,
        player_id: int,
        config: "SoccerConfig",
        _backend: object | None,
    ) -> None:
        self._backend = _backend
        self.id: int = player_id
        self.config: "SoccerConfig" = config
        self.context: Context | None = None

        self.action: Action = Action.INIT

        self._mode: str | None = None
        self._fall_down_state: str | None = None

        self._avoid_side: float | None = None

        self._block_pressing: bool = False
        self._guard_threatened: bool = False
        self._goal_line_push: bool = False
        self._support_last_pos: tuple[float, float] | None = None
        self._support_stationary_since: float | None = None
        self._support_last_update_at: float | None = None

    @property
    def pose(self) -> Pose2D | None:
        ctx = self.context
        return getattr(ctx.teammates.get(self.id), "pose", None) if ctx else None

    @property
    def mode(self) -> str | None:
        return getattr(self._backend, "mode", None) if self._backend else self._mode

    @property
    def is_fallen(self) -> bool:
        return self.fall_down_state not in (None, "normal")

    @property
    def fall_down_state(self) -> str | None:
        return getattr(self._backend, "fall_down_state", None) if self._backend else self._fall_down_state

    @property
    def penalty(self) -> Penalty:
        ctx = self.context
        state = ctx.game.get_player_state(self.config.team_id, self.id) if (ctx and ctx.game) else None
        return state.penalty if state else Penalty.NONE

    @property
    def is_penalized(self) -> bool:
        return self.penalty != Penalty.NONE

    def attack(self, kick_target: tuple[float, float] | None = None, action = Action.ATTACK) -> None:
        """进攻动作:接近球并尝试射门。

        距球足够近时踢球,否则走到球后方准备射门位置。

        参数:
            kick_target: 踢球目标点,None 表示射向对方球门(默认行为)
        """
        self.action = action
        kick_target = kick_target if kick_target is not None else self.context.field.opponent_goal

        ball = self.context.ball
        power = KICK_POWER_BACKFIELD if ball.x < 1 else KICK_POWER_DEFAULT
        power = KICK_POWER_OUR_KICKOFF if self.context.game.phase == Phase.OUR_KICKOFF else power

        d = dist(self.pose.x, self.pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)

        if self._kicking:
            kick_direction = angle_to(ball.x, ball.y, *kick_target)
            if not self.kick(kick_direction, power):
                self.stop()
        else:
            self.release_kick()
            self.walk_to(
                _behind_ball(ball.x, ball.y, kick_target, CHASE_BEHIND_M)
            )

    def guard(self) -> None:
        """防守动作:回到己方球门区域中心待命。

        站位在己方球门区域中心,面向球(可选)。
        """
        home = self.context.field.own_goal_area_center
        face = 0.0
        if GUARD_FACE_BALL and self.context.ball is not None:
            face = angle_to(
                self.pose.x, self.pose.y, self.context.ball.x, self.context.ball.y,
            )

        self.action = Action.GUARD
        debugdraw.point(
            home[0], home[1], rgb=(0.0, 0.6, 1.0), scale=0.2, ns="guard_home",
        )
        self.walk_to(home, face=face, avoid_ball=True, avoid_robots=True)

    def support(self, action = Action.SUPPORT) -> None:
        """支援动作:站在球与己方球门之间的支援位置。

        保持在球的后方一定距离处,随时准备接应或防守。
        """
        self.action = action
        ctx = self.context
        ball = ctx.ball
        gx, gy = ctx.field.own_goal
        bx, by = (ball.x, ball.y)
        dx, dy = gx - bx, gy - by
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = -1.0, 0.0
        else:
            ux, uy = dx / d, dy / d
        along = min(SUPPORT_DIST_M, d)
        tx = bx + ux * along
        ty = by + uy * along

        half_l = ctx.field.half_length
        half_w = ctx.field.half_width - 0.3
        tx = clamp(tx, -half_l + 0.3, half_l)
        ty = clamp(ty, -half_w, half_w)
        face = angle_to(self.pose.x, self.pose.y, ball.x, ball.y)
        self.release_kick()
        self.walk_to((tx, ty), face=face, avoid_ball=True, avoid_robots=True)

    def take_kickoff(self, kick_target: tuple[float, float] | None = None, action = Action.KICKOFF) -> None:
        """开球动作:走到开球位置并执行踢球。

        参数:
            kick_target: 踢球目标点,None 表示射向对方球门(默认行为)
        """
        self.action = action
        ball = self.context.ball
        kick_target = kick_target if kick_target is not None else self.context.field.opponent_goal
        
        # 计算踢球方向(从球指向目标)
        kick_dir = angle_to(ball.x, ball.y, *kick_target)
        cos_k, sin_k = math.cos(kick_dir), math.sin(kick_dir)
        
        # 计算球员相对于球的位置在踢球方向上的投影
        # behind: 沿踢球方向的距离(正=在球前方,负=在球后方)
        # lateral: 垂直踢球方向的偏差
        rel_x, rel_y = self.pose.x - ball.x, self.pose.y - ball.y
        behind = rel_x * cos_k + rel_y * sin_k
        lateral = abs(-rel_x * sin_k + rel_y * cos_k)
        
        # 如果位置不在球后方合适区域,先走到球后方 KICKOFF_STAGE_M 处
        # 否则直接踢球
        if behind > KICKOFF_FRONT_MARGIN or lateral > KICKOFF_LATERAL_TOL:
            stage = (
                ball.x - cos_k * KICKOFF_STAGE_M,
                ball.y - sin_k * KICKOFF_STAGE_M,
            )
            self.release_kick()
            self.walk_to(stage, face=kick_dir, avoid_ball=True)
        else:
            self.kick(kick_dir, KICK_POWER_OUR_KICKOFF)

    @staticmethod
    def walk_to_slots(
        players: list["Player"],
        positions: list[tuple[float, float]],
        action: Action = Action.STAY,
        face: float | None = None,
    ) -> None:
        """将多个球员分配到指定站位。

        使用最优匹配算法(穷举排列)使总移动距离最短。

        参数:
            players: 球员列表
            positions: 站位坐标列表(按重要性从高到低排列),可传入临时列表
            action: 动作状态(默认STAY)
            face: 到达后转向的目标角度(弧度), None 表示不转向
        """
        from itertools import permutations

        n = min(len(players), len(positions))
        if n == 0:
            return

        dist_matrix = [
            [dist(p.pose.x, p.pose.y, pos[0], pos[1]) for pos in positions[:n]]
            for p in players[:n]
        ]

        best_perm = None
        best_total = float("inf")
        for perm in permutations(range(n)):
            total = sum(dist_matrix[i][perm[i]] for i in range(n))
            if total < best_total:
                best_total = total
                best_perm = perm

        if best_perm is not None:
            for i, pos_idx in enumerate(best_perm):
                players[i].walk_to(
                    positions[pos_idx],
                    avoid_ball=True,
                    avoid_robots=True,
                    action=action,
                    face = face,
                )