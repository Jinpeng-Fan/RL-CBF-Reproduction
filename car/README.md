# 车辆跟随算例

论文图 5 的车辆跟随任务：五辆车沿直线行驶，只控制第 4 辆车的加速度，CBF 要求它与前后车的间距不小于 2 m。结论、图表与数据的对应关系和复现命令见仓库根目录的 `README.md`，方法与结果见 `report/main.pdf` 第 1–3 节与附录 B–D。

## 代码

四种算法实际都运行 `TRPO-CBF/` 中的共享实现：

| 入口 | 说明 |
|---|---|
| `TRPO-CBF/sim.py` | TRPO-CBF；全部超参数为其 argparse 默认值，物理常数在 `TRPO-CBF/config.py` |
| `TRPO/sim.py` | 调用 `TRPO-CBF/sim.py` 并加 `--no_cbf` |
| `DDPG-CBF/ddpg.py` | DDPG 主体（Actor/Critic、回放池、OU 噪声），其余模块从 `TRPO-CBF/` 导入 |
| `DDPG/ddpg.py` | 调用 `DDPG-CBF/ddpg.py` 并加 `--no_cbf` |

这些文件与 `results/car_final/*/seed*/run_config.json` 中记录的 SHA256 一致。

## 与作者源码的主要差别

- 模型：驾驶员制动改为 `-kb*max(0, 6-gap)`（论文式 (27) 字面为 `-kb*gap`）；CBF 用含相对速度的两行屏障 `h = Δv + 2(Δp - 2)`，闭式求解并返回松弛量。
- 代码修正：位置积分改用旧速度（作者为 `p + v dt + 1.5 a dt²`）；DDPG 回放池中的状态被同一数组覆盖、BN 在单样本下使 Actor 与状态无关；补偿网络按 Algorithm 1 每次迭代训练并使用；TRPO-CBF 先采样后过滤；GAE、KL 检查、TRPO 观测错位等。
- 环境调整（改变了任务）：五车噪声 σ = 1，每个 episode 200 步，随机初态，相撞即终止并罚 `500 + 100 × 剩余步数`，燃油项用实际加速度，观测为 10 维缩放特征；CBF 版本另用 400 步标定 GP。

完整对照见报告附录表 B.2，完整参数见附录表 D.2。

## 结果

- `results/car_final/<ALG>/seed<k>/`：修改后代码，Python 3.7.9 + TensorFlow 1.15（CPU），每个运行 3×10⁵ 步。`checkpoint/` 为该运行最后保存的模型（TRPO 基线为第 250–275 次迭代，DDPG 种子 2 为第 325 次，其余为第 300 次）。
- `results/car_author_default/<ALG>/seed<k>/`：作者源码在其默认设置下重跑（TRPO 1.8×10⁶ 步，TRPO-CBF 1.2×10⁶ 步，DDPG 两种各 1000 个 episode），`episodes.csv` 由 `report/figures/extract_car_original.py` 从作者输出的 `.mat` 中提取（`.mat` 未放入仓库）。
- `results/car_eval/`：上述最终策略在真实动力学上的 10 s 闭环仿真（`scripts/simulate_final.py`，同一初态与噪声序列），逐步数据 `sim_seed<k>.npz`、汇总 `sim_summary.json`，以及四种算法并排的动画 `animation_seed<k>.gif`（`scripts/animate_final.py`）。
