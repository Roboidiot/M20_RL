# M20_rl — IsaacLab 2.3 轮足式四足机器人强化学习训练项目

基于 [IsaacLab](https://isaac-sim.github.io/IsaacLab/) **2.3.2** 与 [RSL-RL](https://github.com/leggedrobotics/rsl_rl)（`isaaclab_rl`）训练 **M20** 轮足式四足机器人的速度追踪（Locomotion）策略。

- 任务：**平坦地形速度追踪**（`Isaac-Velocity-Flat-M20-v0`，主训练目标），并附**崎岖地形**变体（`Isaac-Velocity-Rough-M20-v0`）。
- 机器人资产：USD / URDF / MJCF 三种格式均位于 `source/M20_rl/assets/M20/`。

---

## 1. M20 机器人

M20 是轮足式四足机器人，4 条腿 × 4 关节 = **16 个执行关节**：

| 腿 | 关节 | 类型 | 控制方式 |
|----|------|------|----------|
| `fl/fr/hl/hr` | `hipx` | 横滚（revolute，绕 `-x`） | 位置 |
| `fl/fr/hl/hr` | `hipy` | 髋俯仰（revolute，绕 `-y`） | 位置 |
| `fl/fr/hl/hr` | `knee` | 膝俯仰（revolute，绕 `-y`） | 位置 |
| `fl/fr/hl/hr` | `wheel` | 轮（continuous，绕 `-y`） | **速度** |

- 基座质量 15.88 kg，轮半径 0.09 m。
- 轮子不可转向，因此机器人是**非完整约束**：只能前进/后退 + 原地转向，不能横移（侧向速度指令设为 0）。
- 关节名遵循 URDF 命名：`fl_hipx_joint`、`fl_hipy_joint`、`fl_knee_joint`、`fl_wheel_joint` … `hr_wheel_joint`。

> **执行器设计**：腿关节用隐式 PD（`ImplicitActuatorCfg`，非零 `stiffness`/`damping`）做位置控制；轮关节用 `stiffness=0` + 非零 `damping` 做**速度控制**——阻尼项将指令的关节速度转换为力矩（`torque = damping * (v_cmd - v)`）。

---

## 2. 目录结构

```
my_training/
├── source/
│   └── M20_rl/
│       ├── assets/
│       │   ├── m20.py                 # M20 的 ArticulationCfg（机器人配置）
│       │   └── M20/                   # 模型文件（USD/URDF/MJCF + STL 网格）
│       └── tasks/
│           └── locomotion/velocity/
│               └── config/m20/
│                   ├── __init__.py            # gym 环境注册
│                   ├── rough_env_cfg.py       # 崎岖地形基础配置
│                   ├── flat_env_cfg.py        # 平坦地形配置（主）
│                   └── agents/
│                       └── rsl_rl_ppo_cfg.py  # RSL-RL PPO 超参
├── scripts/
│   ├── inspect_robot.py                # 打印关节/刚体名，验证资产
│   └── rsl_rl/
│       ├── train.py                    # 训练脚本
│       └── play.py                     # 推理脚本
├── setup.py
└── README.md
```

---

## 3. 环境要求

- **Isaac Sim 4.5 / IsaacLab 2.3.2**（或兼容的 2.3.x），Python 3.10。
- `isaaclab`、`isaaclab_tasks`、`isaaclab_rl`（RSL-RL ≥ 3.0）、`isaaclab_assets`、`rsl-rl-lib`、`gymnasium`。

## 4. 安装

在 IsaacLab 的 Python 环境里，以**可编辑模式**安装本项目：

```bash
# 假设使用 IsaacLab 官方安装脚本（isaaclab.sh）
./isaaclab.sh -p -m pip install -e .

# 或独立 conda 环境
pip install -e .
```

安装后，`import M20_rl.tasks` 会向 gymnasium 注册以下环境：

| 环境 ID | 说明 |
|---------|------|
| `Isaac-Velocity-Flat-M20-v0` | 平坦地形训练 |
| `Isaac-Velocity-Flat-M20-Play-v0` | 平坦地形推理 |
| `Isaac-Velocity-Rough-M20-v0` | 崎岖地形训练 |
| `Isaac-Velocity-Rough-M20-Play-v0` | 崎岖地形推理 |

---

## 5. 验证机器人资产

先运行检查脚本，确认 USD 能被 IsaacLab 正确解析，并打印出实际关节名：

```bash
./isaaclab.sh -p scripts/inspect_robot.py
```

预期输出 16 个关节（`fl_hipx_joint` … `hr_wheel_joint`）和对应刚体名。若关节名与预期不同（例如导入器改成了 `fl_hipx` 而非 `fl_hipx_joint`），请据此修改 `source/M20_rl/assets/m20.py` 与
`source/M20_rl/tasks/locomotion/velocity/config/m20/rough_env_cfg.py` 中的 `joint_names_expr` 正则。

## 6. 训练

```bash
# 平坦地形（推荐先跑通）
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-M20-v0 --num_envs 4096

# 崎岖地形
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-M20-v0 --num_envs 4096

# 从 checkpoint 续训
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-M20-v0 --resume
```

- 日志与 checkpoint 默认写入 `logs/rsl_rl/<experiment_name>/<时间戳>/`。
- 可用 `--max_iterations`、`--seed` 覆盖超参；`--headless` 关闭渲染窗口。

## 7. 推理

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-M20-Play-v0 --num_envs 16 --real-time
```

推理脚本默认加载 `logs/rsl_rl/<experiment_name>/` 下最新的 checkpoint。

---

## 8. 关键可调参数

| 位置 | 参数 | 说明 |
|------|------|------|
| `assets/m20.py` | `legs.stiffness/damping` | 腿关节 PD 增益（默认 40 / 1.0） |
| `assets/m20.py` | `wheels.damping` | 轮子速度控制阻尼（默认 1.5，越大响应越快越费力） |
| `rough_env_cfg.py` | `ActionsCfg.legs.scale` | 腿位置动作缩放（默认 0.5，即 ±0.5 rad） |
| `rough_env_cfg.py` | `ActionsCfg.wheels.scale` | 轮速度动作缩放（默认 10 rad/s） |
| `rough_env_cfg.py` | `base_height_l2.target_height` | 目标基座高度（默认 0.58 m） |
| `rough_env_cfg.py` | `CommandsCfg` 速度范围 | 前进/转向指令范围 |
| `agents/rsl_rl_ppo_cfg.py` | `max_iterations`、网络宽度 | PPO 训练规模 |

## 9. 常见问题

- **机器人没出现在场景里**：检查 `inspect_robot.py` 输出，确认 `prim_path`（`{ENV_REGEX_NS}/Robot`）与 USD 的 `defaultPrim`（`M20`）是否一致。
- **动作维度对不上（不是 16）**：说明关节名正则没匹配全，用 `inspect_robot.py` 核对后修正 `joint_names_expr`。
- **训练不收敛 / 机器人倒下**：优先调 `flat_orientation_l2`、`base_height_l2`、`joint_deviation_l1` 权重，以及腿关节 `stiffness/damping`。

## 10. 许可证

代码采用 [BSD-3-Clause](LICENSE)。M20 机器人模型文件版权归其原作者所有。
