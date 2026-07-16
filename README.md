# Sim-3v3-Simple-Framework

> 3v3 足球仿真机器人策略框架 | 基于 Booster Agent Framework | Python

---

## 项目说明

本仓库为**参赛队员内部共享代码**，用于 3v3 足球仿真机器人比赛的策略开发与交流。框架提供了一套简洁的策略模板，降低参赛队员的上手门槛，使其能够专注于策略本身的设计与优化。

**核心理念：你写策略，框架帮你跑比赛。**

- **低上手门槛**：策略逻辑集中在 `main.py`，参数集中在 `param.py`
- **职责清晰**：策略层 / 动作层 / 执行层 三层分离
- **灵活扩展**：可轻松添加新动作、修改站位、调整策略

---

## 三层架构

```
┌─────────────────────────────────────────────────────────┐
│  main.py      策略层 — 决定"做什么"                        │
│  Phase 状态机 → 选择进攻者 → 分配角色                       │
│  NORMAL / KICKOFF / READY / SET_PLAY ...                 │
├─────────────────────────────────────────────────────────┤
│  player.py    动作层 — 定义"怎么做"                        │
│  attack() / guard() / support() / take_kickoff()         │
│  walk_to_slots() 批量站位（最优匹配）                       │
├─────────────────────────────────────────────────────────┤
│  motion.py    执行层 — 负责"怎么执行"                      │
│  walk_to() / face_to() / kick() / set_velocity()         │
│  支持 A* 全局规划 + VFH 局部避障                             │
└─────────────────────────────────────────────────────────┘
```

**数据流向**：每帧（30Hz）框架构造 `Context` 快照（球/机器人/比赛状态），传入 `play()` → 策略层决策 → 逐层调用 → SDK 发送指令。

---

## 目录结构

```
sim-3v3-simple-framework/
├── agent.toml              # Agent 元信息（ID、版本、入口）
├── build.toml              # 构建配置（依赖、支持平台）
├── docs/                   # 文档
│   ├── 新手入门指南.md
│   ├── Booster Agent Framework Python API.md
│   └── BoosterOS 开发者接口文档 - V1.0.md
├── res/
│   └── logo.png
└── src/
    ├── main.py             # ★ 策略入口（改打法改这里）
    ├── player.py           # ★ Player 类，高层动作定义
    ├── motion.py           # ★ MotionMixin，行走/踢球执行
    ├── param.py            # ★ 所有可调参数（调参改这里）
    ├── utils/              # 工具函数
    │   ├── geom.py         # 几何工具（距离、角度、归一化）
    │   ├── obstacles.py    # 障碍物收集与绕行
    │   └── path_planner.py # A* 全局路径规划器
    └── framework/          # 平台层（一般不改）
        ├── types.py       # 核心数据类型（Context, Phase, Action）
        ├── agent.py       # 框架入口
        ├── runtime.py     # 30Hz 控制主循环
        └── ...
```

**新手只需关注 3 个文件**：`main.py`、`player.py`、`param.py`

---

## 快速开始

### 1. 阅读文档

- [新手入门指南](docs/新手入门指南.md) — 从零开始了解框架
- [Booster Agent Framework Python API](docs/Booster Agent Framework Python API.md) — 平台 API 参考

### 2. 环境要求

- Python 3.13+
- `py_trees==2.4.0`
- `booster_agent_framework`（Booster 平台）
- 支持平台：`sim_x86_64` / `sim_aarch64` / `real_jetson`

### 3. 构建与运行

```bash
# 构建 Agent 包
booster-agent build

# 在 Booster Studio 中加载运行
```

### 4. 修改策略

1. **调参数**：修改 `src/param.py` 中的常量（踢球力度、站位坐标等）
2. **改策略**：修改 `src/main.py` 中的 `_act_*` 函数
3. **加动作**：在 `src/player.py` 中添加新方法

---

## 核心概念

| 概念 | 说明 |
|------|------|
| `Context` | 每帧由框架构造的只读快照，包含球位、机器人位、比赛状态 |
| `Phase` | 比赛阶段（NORMAL / KICKOFF / SET_PLAY / READY / STOPPED） |
| `Action` | 球员动作意图（ATTACK / GUARD / SUPPORT / KICKOFF ...） |
| `Player` | 单个机器人的控制句柄，跨整场比赛存活 |
| `strategy_state` | Context 中的跨帧状态容器，用于记忆进攻者 ID 等 |

---

## 常用修改场景

| 需求 | 修改位置 |
|------|----------|
| 调整踢球力度 | `param.py` → `KICK_POWER_*` |
| 修改站位坐标 | `param.py` → `*_POSITIONS` / `*_SLOTS` |
| 更换进攻/防守分配逻辑 | `main.py` → `_act_normal()` |
| 添加新的球员动作 | `player.py` → 新增方法 |
| 实现角球/任意球战术 | `main.py` → `_act_our_set_play()` |
| 调整避障参数 | `param.py` → 路径规划相关常量 |

---

## 相关资源

- [Booster 官方文档](docs/Booster Agent Framework Python API.md)
- [SDK 接口参考](docs/BoosterOS 开发者接口文档 - V1.0.md)

---

## 交流规范

- **问题反馈**：请附上相关截图、日志和复现步骤
- **策略分享**：建议保留原作者注释，方便后续维护
- **代码风格**：遵循现有代码风格（简洁优先、适当注释）
