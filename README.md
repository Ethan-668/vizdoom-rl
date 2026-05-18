# ViZDoom Reinforcement Learning

这是一个基于 ViZDoom 的强化学习实验项目，主要包含 DefendCenter 和 DeadlyCorridor 两类环境中的训练、评估、录像和结果分析脚本。

## 项目内容

- `scripts/`: 训练、评估、录像、诊断和绘图脚本
- `configs/`: DeadlyCorridor 自定义场景配置
- `results/`: 实验结果、图表和项目总结

大体积训练产物没有放进 Git 仓库，包括 `models/`、`videos/`、`logs/` 和 `final_artifacts/`。这些文件适合后续用 GitHub Release、网盘或 Git LFS 单独管理。

## 环境

项目在 WSL2 Ubuntu 22.04、Python 3.10、RTX 4060 Laptop GPU 上运行过。建议使用 Conda：

```bash
conda create -n vizdoom-rl python=3.10
conda activate vizdoom-rl
pip install -r requirements.txt
```

## 常用命令

训练 DefendCenter PPO：

```bash
python scripts/train_ppo_defendcenter.py --timesteps 500000
```

评估 DefendCenter PPO：

```bash
python scripts/eval_ppo_defendcenter.py --model models/ppo_defendcenter_500k.zip
```

训练 DeadlyCorridor curriculum：

```bash
python scripts/train_ppo_deadlycorridor_rldoom_v2_curriculum.py
```

## 结果摘要

详见：

- `results/project_summary.md`
- `results/final_eval_results.md`
- `results/sample_factory_deadlycorridor_compare/summary.md`

项目结论包括：

- PPO 在 DefendCenter 中表现稳定，500k 步已经能学到较好的转向和射击策略。
- DeadlyCorridor 中默认 reward 容易出现 shortcut，高分不一定代表真正学会通关行为。
- Sample Factory 的 APPO + RNN + 组合动作空间是 DeadlyCorridor 更强的公开基线方案。
