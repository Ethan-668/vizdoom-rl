# ViZDoom 强化学习项目工作总结

本文档整理当前项目已经完成的主要工作、关键实验结果、DeadlyCorridor 复杂环境分析，以及最终可展示材料。

## 1. 项目环境

- 系统：Windows 11 + WSL2 Ubuntu 22.04
- 项目路径：`~/projects/vizdoom-rl`
- Conda 环境：`vizdoom-rl`
- Python：3.10.20
- GPU：RTX 4060 Laptop GPU
- 主要库：ViZDoom、Gymnasium、Stable-Baselines3、OpenCV、TensorBoard、Sample Factory

训练和评估命令均在 WSL 中执行：

```bash
cd ~/projects/vizdoom-rl
conda activate vizdoom-rl
```

## 2. 主实验：DefendCenter 多方法对比

项目首先在 `VizdoomDefendCenter-v1` 上完成了基础强化学习算法对比。该环境任务相对清晰：智能体站在中心位置，主要学习转向和射击。

### 2.1 对比方法

- Random
- A2C
- DQN
- PPO

### 2.2 关键结果

| 方法 | 训练步数 | 平均奖励 | 标准差 | 最低分 | 最高分 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Random | 0 | 0.20 | 0.98 | -1 | 3 | 基本不会玩 |
| A2C | 500k | 0.90 | 0.70 | 0 | 3 | 略优于随机，但未学会稳定策略 |
| DQN | 500k | 11.35 | 4.56 | 5 | 19 | 能学到有效射击行为 |
| PPO | 500k | 21.20 | 2.82 | 12 | 25 | 表现最好，出现满分局 |
| PPO | 1M | 21.15 | 6.13 | 5 | 25 | 平均值未提升，波动更大 |

### 2.3 结论

PPO 在 DefendCenter 中表现最好。500k 步 PPO 已能稳定转向和射击，并在视频中出现满分局。继续训练到 1M 步后平均奖励没有明显提升，且方差增大，说明训练步数并非越多越好，需要结合奖励、方差、最低分和视频行为综合判断。

相关文件：

- 结果表：`results/final_eval_results.md`
- 模型：`models/ppo_defendcenter_500k.zip`、`models/best_ppo_defendcenter.zip`
- 视频：`videos/hd_ppo_500k_general/`、`videos/hd_500k_v2/`

## 3. DeadlyCorridor 初始实验与问题

`VizdoomDeadlyCorridor-v1` 是更复杂的走廊战斗环境。智能体不仅需要前进，还要转向、躲避、射击和处理远处敌人。该环境暴露出单纯依赖 raw reward 的问题。

### 3.1 默认 Gym 环境 PPO

默认环境使用 `Discrete(8)` 动作空间。PPO-500k 在评估中得到较高 raw reward：

| 指标 | 数值 |
|---|---:|
| mean reward | 231.26 |
| std reward | 216.06 |
| min reward | -93.11 |
| max reward | 815.61 |
| mean steps | 63.80 |

但是诊断发现策略几乎只重复单一动作。重复执行每个动作的测试显示，`action=4` 本身就能获得较高奖励：

| 动作 | mean reward |
|---|---:|
| action 4 | 327.66 |
| 其他动作 | 约 -116 |

因此该结果不是“真正会玩”，而是利用了环境 reward shortcut。

### 3.2 NoBackward / NoShortcut 尝试

后续尝试移除 shortcut 动作：

- 移除 action 4 后，策略塌缩到 action 7，mean reward 约 -115.97。
- 移除 action 4 和 7 后，PPO-500k 平均奖励约 26.17，视频中出现转向和攻击，但无法形成稳定推进策略。

结论：仅靠删动作无法稳定解决 DeadlyCorridor，需要更合适的动作空间、记忆结构和训练规模。

## 4. 自定义 Reward Shaping 与 Curriculum 尝试

项目随后参考 RL-Doom / DeadlyCorridor 常见做法，创建了自定义环境和课程学习版本：

- 7 个基础动作：`MOVE_LEFT`、`MOVE_RIGHT`、`ATTACK`、`MOVE_FORWARD`、`MOVE_BACKWARD`、`TURN_LEFT`、`TURN_RIGHT`
- 84x84 灰度输入
- 4 帧堆叠
- PPO
- reward shaping
- curriculum learning：skill 1 -> skill 2 -> skill 3 -> skill 4 -> skill 5

相关文件包括：

- `scripts/deadlycorridor_rldoom_env_v2.py`
- `scripts/train_ppo_deadlycorridor_rldoom_v2_curriculum.py`
- `scripts/eval_ppo_deadlycorridor_rldoom_v2.py`

### 4.1 v1 问题

短版 curriculum 训练后，模型仍然塌缩为一直 `MOVE_FORWARD`：

| skill | mean shaped reward | mean raw reward | mean health loss | mean damage | mean hits | 主要动作 |
|---|---:|---:|---:|---:|---:|---|
| 5 | 32.98 | 232.98 | 75.00 | 0.00 | 0.00 | MOVE_FORWARD |
| 1 | 1890.50 | 2033.30 | 66.40 | 0.00 | 0.00 | MOVE_FORWARD |

原因是 raw reward 权重过大，推进奖励压倒了战斗行为。

### 4.2 后续 shaping 尝试

后续又尝试了多版 reward shaping 和课程策略，包括：

- `renotte_scaled`
- `progress_from_renotte`
- `survival_progress`
- `gatepush`
- `battle_gate`
- `killaware`

这些版本中，一些模型已经能杀死出生点附近敌人和第一个路口部分敌人，但仍容易出现以下问题：

- 在第一个路口前后移动拖时间
- 偏向硬冲而不是找角度击杀
- 能攻击但推进不稳定
- 视频行为比默认 PPO 更合理，但还达不到强智能体水平

结论：在 DeadlyCorridor 中，简单 PPO + 手写 reward shaping 很难稳定得到真正强策略。

## 5. 前人强基线复现：Sample Factory

经过调研，表现最强且有公开权重的路线是 Sample Factory 的 ViZDoom DeadlyCorridor 模型。这类模型不是普通 PPO，而是：

- APPO（Asynchronous PPO）
- CNN 视觉编码
- RNN 记忆网络（GRU 或 LSTM）
- 组合动作空间
- 大规模并行采样
- 大训练规模

组合动作空间是关键差异。它允许智能体在同一时刻完成多个控制，例如：

- 转向
- 前进/后退
- 左右平移
- 攻击

这比 SB3 中单步只能选一个离散动作更接近真实 FPS 控制方式，因此能表现出更自然的走位和战斗。

### 5.1 已下载并复现的公开模型

公开模型下载到：

`models/sample_factory_deadlycorridor_external/`

规范化后的对比目录：

`models/sample_factory_deadlycorridor_compare_train_dir/`

对比视频目录：

`videos/deadlycorridor/sample_factory_compare/`

对比结果：

`results/sample_factory_deadlycorridor_compare/summary.md`

### 5.2 Sample Factory 模型对比

同一环境 `doom_deadly_corridor`，同一录像脚本，native 640x480，10 episodes。

| 模型来源 | 本地名 | 算法 | RNN | 并行规模 | checkpoint 约对应帧数 | 10局平均奖励 |
|---|---|---|---|---:|---:|---:|
| edbeeching/doom_deadly_corridor_1111 | `edbeeching_1111` | APPO | LSTM-512 | 20 x 12 | 4.75B | 22.828 |
| edbeeching/doom_deadly_corridor_2222 | `edbeeching_2222` | APPO | LSTM-512 | 20 x 12 | 3.97B | 19.275 |
| MattStammers/vizdoom_deadly_corridor | `mattstammers` | APPO | GRU-512 | 8 x 4 | 11.8M | 17.587 |
| edbeeching/doom_deadly_corridor_3333 | `edbeeching_3333` | APPO | LSTM-512 | 20 x 12 | 3.97B | 12.884 |
| Apocalypse-19/doom_deadly_corridor | `default_experiment` | APPO | GRU-512 | 16 x 8 | 50.0M | 11.575 |
| RamonAnkersmit/rl_course_doom_deadly_corridor | `ramon` | APPO | GRU-512 | 8 x 4 | 10.0M | 9.175 |
| execbat/rl_course_vizdoom_doom_deadly_corridor | `execbat` | APPO | GRU-512 | 32 x 4 | 10.0M | 8.070 |

### 5.3 主要发现

1. 强模型不是靠重复单一动作获得奖励，而是能转向、移动、攻击和推进。
2. `edbeeching_1111` 当前表现最好，10 局平均奖励达到 22.828。
3. 训练规模影响明显。数十亿帧级别的 LSTM 模型明显优于 10M 到 50M 帧级别模型。
4. RNN 对 DeadlyCorridor 很重要，因为环境存在部分可观测性，智能体需要记住敌人位置、路口结构和近期动作。
5. 组合动作空间比单一离散动作更适合 FPS 行为。

## 6. 最终展示材料

### 6.1 推荐主展示视频

优先展示：

`videos/deadlycorridor/sample_factory_compare/edbeeching_1111_native640_eval.mp4`

备选视频：

- `videos/deadlycorridor/sample_factory_compare/edbeeching_2222_native640_eval.mp4`
- `videos/deadlycorridor/sample_factory_compare/mattstammers_native640_eval.mp4`
- `videos/deadlycorridor/sample_factory_pretrained/native640_candidates/sf_native640_candidate_31.mp4`
- `videos/deadlycorridor/sample_factory_pretrained/native640_candidates/sf_native640_candidate_30.mp4`
- `videos/deadlycorridor/sample_factory_pretrained/native640_candidates/sf_native640_candidate_23.mp4`

### 6.2 报告中建议写法

可以将项目分为两个层次：

1. **基础算法对比**：在 DefendCenter 上比较 Random、A2C、DQN、PPO，证明基础强化学习流程完整，PPO 表现最佳。
2. **复杂环境扩展**：在 DeadlyCorridor 中分析 raw reward shortcut，说明高 reward 不等于真正会玩。
3. **前人强方法复现**：引入 Sample Factory APPO + RNN + 组合动作空间，复现多个公开强模型，并通过视频验证其行为质量。

推荐表述：

> 在 DeadlyCorridor 中，默认 reward 容易诱导智能体学习 MOVE_FORWARD shortcut，使得 raw reward 较高但行为并不合理。本文先后尝试了动作空间约束、reward shaping 与 curriculum learning，发现简单 PPO 难以稳定得到真实玩家式策略。进一步参考公开强基线后，本文复现了 Sample Factory 的 APPO + RNN + 组合动作空间模型。该方法能同时完成转向、移动、平移和攻击，视频中表现出更接近真实玩家的走廊推进和战斗行为。

## 7. 最终结论

本项目完成了从简单 ViZDoom 环境到复杂 ViZDoom 环境的强化学习实验链路：

- 在 DefendCenter 中，PPO 能稳定学会转向射击，并优于 Random、A2C 和 DQN。
- 在 DeadlyCorridor 中，单纯看 raw reward 会误判策略能力，默认 PPO 容易学到 shortcut。
- 手写 reward shaping 和 curriculum learning 可以改善行为，但仍难以稳定解决复杂走廊战斗。
- 前人强基线表明，DeadlyCorridor 更需要 APPO、RNN、组合动作空间和大规模并行训练。
- 最终复现的 Sample Factory 模型能展示真正更像玩家的智能体行为，可作为报告中的复杂环境扩展成果。
