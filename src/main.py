"""SoccerSim 策略入口 —— 比赛策略主逻辑都在这里,改打法就改这个文件。

三层架构:
1. main.py(策略层):比赛策略。play() 按 Phase 状态机分派到 _act_*;各 _act_* 选出
   attacker(离球最近)并直接调 player 动作。
2. player.py(动作层):Player 控制 handle + 高层动作(attack / guard / support);
   想加拐棍/技术动作直接改它。
3. motion.py(走位层):走位/避障工具(walk_to / face_to / ensure_ready)。

utils/:几何工具(dist / angle_to / clamp ...)。
framework/:平台管线,用户不改。

改打法主要改本文件:Phase 状态机、各 _act_* 行为、站位公式。

数据访问路径:
- context.game.phase: 当前比赛阶段(NORMAL/OFF_KICK等)
- context.game.strategy_state: 策略跨帧状态容器
- context.pre_context: 上一帧上下文
"""

from __future__ import annotations

import logging

from booster_agent_framework import AgentBase

from .framework.agent import SoccerAgentMixin
from .framework.types import Action, Phase, SetType, Context
from .param import *
from .player import Player
from .utils import dist


_log = logging.getLogger(__name__)


def get_set_play_type(context: Context) -> SetType:
    g = context.game
    if g is None:
        return SetType.NONE
    return g.set_play


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent。"""

    player_class = Player

    @staticmethod
    def play(context: Context, players: list[Player]) -> None:
        """每帧策略入口。根据比赛阶段分派球员动作。"""
        phase = context.game.phase if context.game else Phase.STOPPED

        _analyze_and_draw(context, players)

        from .framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.half_width + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )

        active: list[Player] = []
        for p in players:
            ready = p.ensure_ready()
            if p.is_penalized:
                p.action = Action.PENALIZED
                p.stop()
            elif not ready:
                p.action = Action.FALLEN if p.is_fallen else Action.SWITCHING_MODE
            elif p.pose is None:
                p.action = Action.NO_POSE
                p.stop()
            else:
                active.append(p)

        if phase == Phase.NORMAL:
            _act_normal(context, active)
        elif phase == Phase.OUR_KICKOFF:
            _act_our_kickoff(context, active)
        elif phase == Phase.OPP_KICKOFF:
            _act_opp_kickoff(context, active)
        elif phase == Phase.OUR_SET_PLAY:
            _act_our_set_play(context, active)
        elif phase == Phase.OPP_SET_PLAY:
            _act_opp_set_play(context, active)
        elif phase == Phase.READY:
            _act_ready(context, active)
        elif phase == Phase.STOPPED:
            for p in active:
                p.stop()

        for p in players:
            _draw_teammate_marker(p)


def _select_closest_attacker(context: Context, players: list[Player]) -> Player:
    """选择到球距离最近的球员作为进攻者(含粘性选择和摔倒惩罚)。

    从 context.game.strategy_state.normal_attacker 读取偏好攻击者ID,实现粘性选择。
    """
    ball = context.ball
    if ball is None:
        return players[0]

    def dist_to_ball(p: Player) -> float:
        return dist(p.pose.x, p.pose.y, ball.x, ball.y) + (FALLEN_COST if p.is_fallen else 0.0)

    preferred_id = context.game.strategy_state.normal_attacker
    ranked = [(p, dist_to_ball(p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if preferred is not None and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M:
        return preferred[0]
    return best


def _act_normal(context: Context, players: list[Player]) -> None:
    """NORMAL阶段:距球最近者attack,剩下人里离己方门最近者guard,其余support。"""
    if not players:
        return

    state = context.game.strategy_state
    attacker = _select_closest_attacker(context, players)
    state.normal_attacker = attacker.id
    attacker.attack()

    rest = [p for p in players if p is not attacker]
    if rest:
        gx, gy = context.field.own_goal
        guard = min(rest, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
        guard.guard()
        rest = [p for p in rest if p is not guard]

    for p in rest:
        p.support()


def _act_our_kickoff(context: Context, players: list[Player]) -> None:
    """OUR_KICKOFF阶段:锁定距离最小者开球,其余人站到开球站位。"""
    if not players:
        return

    state = context.game.strategy_state
    state.normal_attacker = None
    
    prev_phase = (
        context.pre_context.game.phase
        if context.pre_context and context.pre_context.game
        else None
    )
    
    if prev_phase != Phase.OUR_KICKOFF or state.kickoff_taker is None:
        state.kickoff_taker = _select_closest_attacker(context, players).id

    attacker = next((p for p in players if p.id == state.kickoff_taker), None)
    if attacker is None:
        attacker = _select_closest_attacker(context, players)
        state.kickoff_taker = attacker.id

    attacker.take_kickoff()

    rest = [p for p in players if p is not attacker]
    Player.walk_to_slots(rest, OUR_KICKOFF_SLOTS, Action.KICKOFF)


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    """OPP_KICKOFF阶段:一人守门,其余人站到开球站位等待。"""
    if not players:
        return
    context.game.strategy_state.normal_attacker = None
    gx, gy = context.field.own_goal
    guard = min(players, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
    guard.guard()

    rest = [p for p in players if p is not guard]
    Player.walk_to_slots(rest, OPP_KICKOFF_SLOTS, Action.OPP_KICKOFF_READY, face=0.0)


def _act_our_set_play(context: Context, players: list[Player]) -> None:
    """OUR_SET_PLAY阶段:按定位球类型分派动作。
    可以让发球队员使用take_kickoff方法，其他队员使用walk_to_slots方法。
    """
    context.game.strategy_state.normal_attacker = None
    set_play = get_set_play_type(context)
    if set_play == SetType.THROW_IN:
        _act_normal(context, players)
        return
    if set_play == SetType.CORNER_KICK:
        _act_normal(context, players)
        return
    if set_play == SetType.GOAL_KICK:
        _act_normal(context, players)
        return
    _act_normal(context, players)


def _act_opp_set_play(context: Context, players: list[Player]) -> None:
    """OPP_SET_PLAY阶段:对方定位球,全部静止等待。"""
    for p in players:
        p.stop()


def _act_ready(context: Context, players: list[Player]) -> None:
    """READY阶段:各自走到准备位置。站位坐标由 param.py 配置。"""
    context.game.strategy_state.normal_attacker = None
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    positions = READY_OUR_POSITIONS if our_kickoff else READY_OPP_POSITIONS
    Player.walk_to_slots(players, positions, Action.READY, face=0.0)


def _draw_teammate_marker(p: Player) -> None:
    from .framework import debugdraw

    if p.pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(p.pose.x, p.pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(p.pose.x, p.pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(p.pose.x, p.pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player]) -> None:
    from .framework import debugdraw

    ball = context.ball

    if ball is None:
        return

    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    for p in players:
        if p.pose is None:
            continue
        d = dist(p.pose.x, p.pose.y, ball.x, ball.y)
        debugdraw.text(
            p.pose.x + 0.3, p.pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        if r.pose is None:
            continue
        d = dist(r.pose.x, r.pose.y, ball.x, ball.y)
        debugdraw.text(
            r.pose.x + 0.3, r.pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )