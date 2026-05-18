# Sample Factory DeadlyCorridor 外部模型对比

同一环境 `doom_deadly_corridor`，同一录像脚本，native 640x480，10 episodes。

| 模型来源 | 本地实验名 | 平均奖励 | 备注 |
|---|---|---:|---|
| edbeeching/doom_deadly_corridor_1111 | `edbeeching_1111` | 22.828 | APPO + RNN + combo action |
| edbeeching/doom_deadly_corridor_2222 | `edbeeching_2222` | 19.275 | APPO + RNN + combo action |
| MattStammers/vizdoom_deadly_corridor | `mattstammers` | 17.587 | APPO + RNN + combo action |
| edbeeching/doom_deadly_corridor_3333 | `edbeeching_3333` | 12.884 | APPO + RNN + combo action |
| Apocalypse-19/doom_deadly_corridor | `default_experiment` | 11.575 | APPO + RNN + combo action |
| RamonAnkersmit/rl_course_doom_deadly_corridor | `ramon` | 9.175 | APPO + RNN + combo action |
| execbat/rl_course_vizdoom_doom_deadly_corridor | `execbat` | 8.070 | APPO + RNN + combo action |

视频目录：`videos/deadlycorridor/sample_factory_compare/`。

说明：Sample Factory enjoy 命令在保存视频后会返回非零码，但日志中有 `(0, reward)` 和 `Replay video saved`，视频已成功生成。

## 训练方法与配置对比

这些外部模型都不是普通 SB3 PPO，而是 Sample Factory 的 ViZDoom 方案。共同点是：

- 算法：APPO（Asynchronous PPO）
- 输入：ViZDoom 图像，经 CNN 编码
- 记忆：RNN，处理走廊中部分可观测、需要记住敌人和路线的问题
- 动作：Sample Factory 的组合动作空间，可同时转向、前后移动、左右平移、攻击
- 环境：`doom_deadly_corridor`

| 模型来源 | 本地名 | 算法 | RNN | 并行规模 | 配置训练步数 | checkpoint 约对应帧数 | 学习率 | 10局平均奖励 |
|---|---|---|---|---:|---:|---:|---:|---:|
| edbeeching/doom_deadly_corridor_1111 | `edbeeching_1111` | APPO | LSTM-512 | 20 workers x 12 envs | 10B | 4.75B | 1e-4 | 22.828 |
| edbeeching/doom_deadly_corridor_2222 | `edbeeching_2222` | APPO | LSTM-512 | 20 workers x 12 envs | 10B | 3.97B | 1e-4 | 19.275 |
| MattStammers/vizdoom_deadly_corridor | `mattstammers` | APPO | GRU-512 | 8 workers x 4 envs | 10M | 11.8M | 1e-4 | 17.587 |
| edbeeching/doom_deadly_corridor_3333 | `edbeeching_3333` | APPO | LSTM-512 | 20 workers x 12 envs | 10B | 3.97B | 1e-4 | 12.884 |
| Apocalypse-19/doom_deadly_corridor | `default_experiment` | APPO | GRU-512 | 16 workers x 8 envs | 50M | 50.0M | 1e-4 | 11.575 |
| RamonAnkersmit/rl_course_doom_deadly_corridor | `ramon` | APPO | GRU-512 | 8 workers x 4 envs | 10M | 10.0M | 1e-4 | 9.175 |
| execbat/rl_course_vizdoom_doom_deadly_corridor | `execbat` | APPO | GRU-512 | 32 workers x 4 envs | 10M | 10.0M | 3e-3 | 8.070 |

## 可写入报告的结论

DeadlyCorridor 中表现最好的外部模型并不是简单 PPO，而是 Sample Factory 的 APPO + RNN 架构。该方法使用组合动作空间，使智能体可以在同一时刻完成转向、移动、平移和攻击，因此行为更接近真实玩家。对比结果显示，训练规模和 RNN 类型对表现影响明显：edbeeching 的 LSTM 模型在数十亿帧级别训练后，平均奖励和视频行为都显著优于 10M 到 50M 帧级别的 GRU 模型。
