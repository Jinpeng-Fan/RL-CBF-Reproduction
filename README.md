# RL-CBF 复现：基于控制障碍函数的端到端安全强化学习

本仓库复现 Cheng, Orosz, Murray, Burdick, *End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks*, AAAI 2019（作者代码：<https://github.com/rcheng805/RL-CBF>）。

包含倒立摆（`pendulum/`）与车辆跟随（`car/`）两个算例。复现报告见 `report/main.pdf`（正文）和 `report/supplement.pdf`（参考文献与附录）。

## 快速开始：双击启动（Windows）

| 双击文件 | 执行内容 |
|---|---|
| **`start_training.bat`** | 从头训练车辆 **TRPO-CBF 与 TRPO**，种子 **1**，每个运行 **20,000 步**，并自动打开实时 TensorBoard 面板。两个进程间隔约 20 秒启动。 |
| **`view_tensorboard.bat`** | 一次打开倒立摆、车辆两个独立的历史面板，各 **4 个算法 × 3 个种子 = 12 组**；不启动训练。 |
| **`view_pendulum.bat`** | 只打开倒立摆历史面板。 |
| **`view_car.bat`** | 只打开车辆历史面板。 |

操作步骤：双击对应文件 → 等待初始化 → 浏览器自动打开 **Scalars** 页面。在左侧筛选 `seed1`、`seed2`、`seed3` 或 `TRPO-CBF` 等算法名称。历史面板首次加载需要等待，曲线齐全后即可演示。

- **首次准备**：需要可从 PATH 找到的 Python 3.9+（本机已验证 Python 3.13）。启动器会在 `E:/RL-CBF-Data/viewer/` 建立独立看板环境并安装 TensorBoard 2.21.0，首次安装需要联网；安装完成后可以离线使用。首次使用前请分别双击一次，完成环境准备。
- **训练环境**：仍使用 Python 3.7.9 / TensorFlow 1.15。启动器自动查找仓库内 `.venv-train/Scripts/python.exe`，以及本机仓库旁的 `../.runtime/python37_cpu/python.exe`，并检查依赖。其他电脑需先准备 `requirements.txt` 所述训练环境，将解释器路径填入 `demo_config.json` 的 `training.python`，也可通过 `RLCBF_TRAIN_PY` 环境变量指定。训练环境不包含在仓库中。
- **输出与端口**：每次训练自动创建新目录，位于 `E:/RL-CBF-Data/<仓库标识>/runs/<时间戳>/`，窗口会显示具体路径。面板从原始 `episodes.csv` 生成精简事件文件，放在 `sessions/<本次启动>/curves/`，兼容中文仓库路径。倒立摆历史面板默认端口 6007，车辆历史面板 6009，车辆实时面板 6008；端口被占用时自动选择空闲端口，以窗口显示的地址为准。
- **E 盘存储约定**：`demo_config.json` 的 `storage_root` 默认是 `E:/RL-CBF-Data`，可改为 E 盘上的其他英文绝对路径，也可用 `RLCBF_DATA_ROOT` 环境变量覆盖。启动器拒绝 C 盘输出目录；看板环境、安装临时文件、pip 缓存、TensorBoard 服务信息与 Matplotlib 缓存也写在此 E 盘目录中，不修改系统全局 TEMP 设置。C 盘旧版入口产生的本次训练记录已迁到 E 盘相同仓库标识目录，记录内的旧命令路径作为历史信息保留。
- **停止**：保持启动窗口打开。训练结束后，面板仍可查看；按 Enter 关闭服务。训练过程中可按 Ctrl+C 或关闭启动窗口，结束本次启动的训练与看板进程，已写入的数据保留。浏览器关闭不会停止训练。
- **日志**：启动与错误日志放在 `E:/RL-CBF-Data/<仓库标识>/sessions/`；各算法日志在对应新训练目录的 `logs/` 中。
- **修改演示参数**：编辑根目录 `demo_config.json`。`training.algorithms` 支持 `TRPO-CBF`、`TRPO`、`DDPG-CBF`、`DDPG`；`seeds` 是种子列表；`steps` 是每个运行的环境步数预算，实际可能在最后一批结束时略超预算；`workers` 控制并行数。20,000 步用于现场演示训练过程，报告主实验使用 300,000 步。短训练不保证产生模型快照，完整模型演示使用已保存的 checkpoint。
- **作者源码对照曲线**：默认历史面板仅显示主结果。需要同时查看全部 48 组实验时，在 `history_groups` 列表中追加 `car_author_default` 和 `author_code_default`，然后重新双击历史面板入口；两个面板仍按算例分开；作者源码实验使用 `pendulum_author/seed1/TRPO-CBF`、`car_author/seed1/TRPO-CBF` 等名称，避免与主实验混淆。

**精简面板**：运行名称统一为 `pendulum/seed1/TRPO-CBF`、`car/seed1/TRPO-CBF` 等，不显示 `final`、`final_600ep`。两个算例使用独立服务和页面；车辆实时训练采用相同命名和指标。

| 面板 | 保留指标（横轴均为 episode） |
|---|---|
| 倒立摆：4 项 | `reward/episode_return` 回报；`safety/max_abs_angle_rad` 最大绝对摆角；`safety/unsafe_episodes_cumulative` 累计越界回合数；`cbf/correction_rms` CBF 修正量 RMS |
| 车辆：5 项 | `reward/episode_return` 回报；`safety/min_headway_m` 最小车距；`safety/collision_episodes_cumulative` 累计碰撞回合数；`safety/unsafe_episodes_cumulative` 累计越界回合数；`cbf/correction_rms` CBF 修正量 RMS |

倒立摆越界阈值为最大绝对摆角 > 1 rad（沿用历史导出的 1e-6 数值容差）；车辆车距 < 2 m 为越界、≤ 0 为碰撞。累计指标统计**回合数**，不是单回合内发生次数。CBF 修正量仅对 CBF 算法显示。保留单一原始回报曲线，可用 TensorBoard 的 Smoothing 控件平滑；不再另列移动平均、单回合越界标记、训练损失、松弛量等诊断图。原始 CSV、完整事件日志和模型保留。当前复现 CSV 没有统一记录耗时与逐步违规比例，因此不添加这些曲线。

**建议演示顺序**：先打开历史曲线说明完整结果，再启动短训练展示实时更新，最后打开 `car/results/car_eval/animation_seed1.gif` 说明已训练策略的闭环行为。动画是已生成的仿真回放。更改算法和种子后，已有实验数据仍可用于对照。

## 主要结论（倒立摆，3 个种子 × 600 个 episode）

| 算法 | 后 100 个 episode 回报 | 越界 episode 数（种子 1/2/3） |
|---|---|---|
| TRPO（修改后，统一环境） | −3.33 ± 0.07 | 174 / 155 / 143 |
| DDPG（修改后，统一环境） | −115.8 ± 54.5 | 217 / 209 / 295 |
| **TRPO-CBF（修改后）** | **−3.52 ± 0.65** | **0 / 0 / 0** |
| **DDPG-CBF（修改后）** | **−4.16 ± 1.25** | **0 / 0 / 1** |
| TRPO-CBF（作者源码，默认设置） | −69.7 ± 107.6 | 1 / 9 / 332 |
| DDPG-CBF（作者源码，默认设置） | −82.3 ± 17.2 | 127 / 67 / 37 |

修改后代码的 TRPO-CBF 在三个种子下均无越界，DDPG-CBF 在一个种子下有一次越界；CBF 版本改善了学习早期的安全表现，修正量随学习消退（从约 3.5 N·m 降到 0.02 N·m）。最终性能与学习速度的比较依算法而异，详见报告。作者源码在默认设置下只部分复现。修改内容与原因见报告第 2 节，代码对照见附录 B。

## 主要结论（车辆跟随，3 个种子）

碰撞 = 第 4 辆车与前车或后车间距 ≤ 0；越界 = 间距 < 2 m（含碰撞）。两组的回合长度、噪声、奖励与终止规则不同，回报只能组内比较。

| 算法 | 末 100 个 episode 回报 | 碰撞 episode 数（种子 1/2/3） | 越界 episode 数（种子 1/2/3） |
|---|---|---|---|
| TRPO（修改后，3×10⁵ 步） | −7837 ± 10457 | 2564 / 1908 / 1432 | 2620 / 2099 / 1471 |
| DDPG（修改后） | −2586 ± 208 | 69 / 151 / 35 | 100 / 177 / 69 |
| **TRPO-CBF（修改后）** | **−740 ± 122** | **0 / 0 / 0** | **0 / 0 / 0** |
| **DDPG-CBF（修改后）** | **−5856 ± 3745** | **0 / 0 / 0** | **0 / 0 / 0** |
| TRPO（作者源码，默认 22515 个 episode） | −2156 ± 443 | 1247 / 825 / 7093 | 3634 / 3575 / 9824 |
| DDPG（作者源码，1000 个 episode） | (−3.2 ± 4.9)×10⁵ | 462 / 688 / 878 | 623 / 804 / 927 |
| TRPO-CBF（作者源码，15015 个 episode） | −2305 ± 1067 | 0 / 0 / 0 | 1 / 0 / 1 |
| DDPG-CBF（作者源码，1000 个 episode） | −28937 ± 2359 | 0 / 0 / 0 | 0 / 10 / 0 |

两组的 CBF 版本都没有碰撞。修改后的 TRPO-CBF 在 1.1–1.4 万步内收敛，CBF 修正量 RMS 从 1.38 降到 0.01 m/s²；修改后的 DDPG-CBF 补偿网络输出趋于 ±100 饱和，回报不如 DDPG 基线（尚未解决，见报告 3.2 节）。修改后的环境（噪声、回合长度、初态、奖励与终止、驾驶员制动规则）与论文和作者源码不同，见报告表 2。

**训练所得策略的闭环仿真**：用 `car/results/car_final/<算法>/seed<k>/checkpoint/` 中各运行最后保存的模型，在真实动力学上从同一初态、同一噪声序列仿真 10 s（确定性动作）。动画 `car/results/car_eval/animation_seed<k>.gif`：上方四条车道并排显示四种算法，下方实时绘制第 4 辆车的最小车距、车速与输入。种子 1 中 TRPO 在 7.0 s 被追尾，其余三种算法最小车距 3.1–3.3 m；DDPG-CBF 种子 2、3 的 CBF 在 55–62% 的步数上介入（报告图 4、附录表 C.4、图 C.7）。

## 目录结构

```
RL-CBF-Reproduction/
├── README.md
├── requirements.txt              训练环境（Python 3.7 / TensorFlow 1.15）
├── start_training.bat           一键启动车辆训练与实时 TensorBoard（Windows）
├── view_tensorboard.bat          一键打开两个独立历史面板（Windows）
├── view_pendulum.bat / view_car.bat  分别打开单个历史面板
├── demo_config.json             演示算法、种子、步数、历史实验组与端口
├── scripts/                     一键启动器与独立看板环境管理
├── tensorboard/                  主结果与作者源码重跑的 TensorBoard 日志（两个算例共 48 个运行）
├── report/                       复现报告 PDF、LaTeX 源码、图表与绘图脚本
│   └── figures/                  make_pendulum_figs.py, extract_original.py, make_car_figs.py,
│                                 extract_car_original.py, *.pdf, pendulum_metrics.json, car_metrics.json
├── pendulum/
│   ├── DDPG/ DDPG-CBF/ TRPO/ TRPO-CBF/    修改后的算法代码（与报告所用运行的源码快照逐字节一致）
│   ├── scripts/                  启动、评估、导出与 TensorBoard 转换脚本
│   └── results/
│       ├── final_600ep/          报告主结果：修改后代码，统一环境，种子 1–3，前 600 个 episode
│       ├── author_code_default/  作者源码在其默认设置下的重跑（逐 episode 记录与日志）
│       └── checkpoint_rerun_seed1/  种子 1 带参数快照的重跑，以及首末策略部署评估（报告图 2）
└── car/
    ├── DDPG/ DDPG-CBF/ TRPO/ TRPO-CBF/    修改后的算法代码（与报告所用运行记录的 SHA256 一致）
    ├── scripts/                  run_final.py, launch_orig.py, run_orig.py, csv_to_tensorboard.py
    └── results/
        ├── car_final/            报告主结果：修改后代码，种子 1–3，每个 3×10^5 步
        ├── car_author_default/   作者源码在其默认设置下的重跑（逐 episode 记录与日志）
        └── car_eval/             最终策略的闭环仿真：sim_seed<k>.npz、sim_summary.json、animation_seed<k>.gif（及末帧 png）
```

倒立摆的每个 `results/<组>/<算法>/seed<k>/` 目录包含：

- `episodes.csv`：逐 episode 的 `reward`、`max_angle`、`violation`，CBF 版本另有 `correction_rms`（以及修改后版本的 `slack`）。
- `run_config.json`：完整命令行参数和各源码文件的 SHA256。
- `source/`：运行时的源码快照（修改后代码）。
- `train_log.mat`：仅 TRPO 类，作者代码格式的逐迭代日志，含轨迹。
- `ckpt/first`、`ckpt/last`、`eval_first_last.npz`：仅 `checkpoint_rerun_seed1`，分别为训练前后的参数和首末策略评估结果。

作者源码重跑输出的 `.mat` 文件共约 515 MB，未放入仓库。仓库只保留由它们提取的 `episodes.csv`，可按下文命令重新生成。

车辆跟随的每个 `car/results/<组>/<算法>/seed<k>/` 目录包含 `episodes.csv`（逐 episode 的 `return`、`min_headway`、`collision`、`length`、`collisions_cum`、`exit_2m`，CBF 版本另有 `cbf_rms`、`comp_rms`，修改后版本另有 `slack_max`、`infeasible_steps`、`cbf_cond_viol`；作者代码不记录松弛量，这几列为空）、`run_config.json`（命令与源码 SHA256），以及控制台日志（修改后为 `train.log`，作者源码为 `stdout.log`）。作者源码的 `.mat` 未放入仓库；`car_author_default/qp_errors.json` 记录各运行打印 “Error in QP” 的次数（均为 0）。

## 报告图表与数据的对应

| 报告内容 | 数据 | 生成脚本 |
|---|---|---|
| 图 1、图 C.1、图 C.2、表 3 下半部分、表 C.1 | `pendulum/results/final_600ep/` | `report/figures/make_pendulum_figs.py` |
| 图 2（首末策略） | `pendulum/results/checkpoint_rerun_seed1/*/seed1/eval_first_last.npz` | `scripts/run_ckpt.py` → `scripts/eval_ckpt.py` |
| 图 C.3、表 3 上半部分、表 C.2 | `pendulum/results/author_code_default/` | `scripts/launch_orig.py` → `report/figures/extract_original.py` |
| 倒立摆全部数值 | `report/figures/pendulum_metrics.json` | `make_pendulum_figs.py` |
| 图 3、图 C.5、表 4 下半部分 | `car/results/car_final/` | `report/figures/make_car_figs.py` |
| 图 C.4、表 4 上半部分 | `car/results/car_author_default/` | `car/scripts/launch_orig.py` → `report/figures/extract_car_original.py` → `make_car_figs.py` |
| 图 C.6、表 C.3 | 以上两组 | `make_car_figs.py` |
| 图 4、图 C.7、表 C.4 | `car/results/car_eval/` | `car/scripts/simulate_final.py` → `report/figures/make_car_sim_figs.py` |
| 动画 GIF | `car/results/car_eval/` | `car/scripts/animate_final.py` |
| 车辆跟随全部数值 | `report/figures/car_metrics.json` | `make_car_figs.py` |

## 查看训练曲线（TensorBoard）

在 Windows 上查看训练曲线，可直接双击 `view_tensorboard.bat`，使用步骤见开头。下面是手动查看方式，应在独立 Python 3.9+ 看板环境中执行（无需 TensorFlow）：

```bash
python -m pip install tensorboard==2.21.0
python -m tensorboard.main --logdir tensorboard --host 127.0.0.1 --port 6007 --load_fast false
```

浏览器打开 <http://127.0.0.1:6007>。该手动命令用于检查原始完整日志，会混合全部 48 个运行及所有指标。查看主实验结果时，建议使用上方一键入口：两个独立精简面板各 12 组主结果。以下标签说明仅针对原始完整日志。

- 包含 `final_600ep`（修改后代码）和 `author_code_default`（作者源码）两组；`checkpoint_rerun_seed1` 只用于报告图 2，不放入 TensorBoard。运行名称格式为 `<结果组>/seed<k>/<算法>`，可在左侧按正则筛选，例如 `final_600ep.*CBF` 只看修改后的两种 CBF 版本。
- 标量包括：`return/episode_return`、`return/episode_return_ma10`、`safety/max_abs_angle_rad`、`safety/unsafe_episode`、`safety/unsafe_episodes_cum`、`cbf/correction_rms`、`cbf/max_slack`。横轴为训练 episode。
- 日志由 `pendulum/scripts/csv_to_tensorboard.py` 从 `episodes.csv` 生成，可随时重新生成。
- 车辆跟随为 `car_final`（修改后代码）与 `car_author_default`（作者源码）两组，运行名同样为 `<结果组>/seed<k>/<算法>`，例如 `car_final/seed1/TRPO-CBF`。标量包括 `return/episode_return(_ma10)`、`safety/min_headway_m`、`safety/collision(s_cum)`、`safety/below_2m(_cum)`、`progress/env_steps`、`cbf/correction_rms`、`cbf/compensator_rms`、`cbf/max_slack`（仅修改后版本），横轴为训练 episode（作者源码每个 80 步，修改后至多 200 步）。日志由 `car/scripts/csv_to_tensorboard.py` 生成。
- 已知限制：在 Windows 上，如果仓库路径含中文等非 ASCII 字符，TensorBoard 可能无法读写日志目录。一键入口会自动使用英文缓存路径；手动操作时，可把仓库放在英文路径下，或用 `--out` / `--logdir` 指向英文路径。

## 复现运行

环境：Python 3.7.9，依赖见 `requirements.txt`，只用 CPU。以下命令都在 `pendulum/` 下运行。

```bash
# 修改后代码：单个算法、单个种子（超参数已在脚本中显式给出）
python scripts/run_one.py --alg TRPO-CBF --seed 1 --episodes 600 --run-dir runs/TRPO-CBF/seed1

# 修改后代码：4 个算法 × 3 个种子并行
python scripts/launch_all.py --root runs_all --episodes 600 --seeds 1 2 3

# 首末策略评估（报告图 2）
python scripts/run_ckpt.py  --alg DDPG-CBF --seed 1 --episodes 600 --run-dir runs_ckpt/DDPG-CBF/seed1
python scripts/eval_ckpt.py --run-dir runs_ckpt/DDPG-CBF/seed1

# 作者源码重跑（作者文件原样运行，只在外部补设随机种子并关闭渲染）
git clone https://github.com/rcheng805/RL-CBF ../../RL-CBF-master
python scripts/launch_orig.py --root runs_orig --orig-root ../../RL-CBF-master/pendulum
python ../report/figures/extract_original.py runs_orig

# 重画报告图表与汇总指标
cd ../report && python figures/make_pendulum_figs.py
```

车辆跟随的命令在 `car/` 下运行：

```bash
# 修改后代码：4 个算法 × 3 个种子（只传入种子与输出目录，其余为 TRPO-CBF/sim.py 与 config.py 的默认值）
python scripts/run_final.py --root runs/final --seeds 1 2 3 --workers 5

# 作者源码重跑（作者文件原样运行，只在外部补设种子、设置 MPLBACKEND=Agg）
python scripts/launch_orig.py --root runs_orig --orig-root ../../RL-CBF-master/car
python ../report/figures/extract_car_original.py --runs runs_orig   # 写入 results/car_author_default/

# 重画报告图表与汇总指标；重新生成 TensorBoard 日志
cd ../report && python figures/make_car_figs.py
cd ../car && python scripts/csv_to_tensorboard.py

# 最终策略的闭环仿真（模型在 results/car_final/*/seed*/checkpoint/）、动画与报告图
python scripts/simulate_final.py          # 写 results/car_eval/sim_seed<k>.npz、sim_summary.json
python scripts/animate_final.py          # 写 results/car_eval/animation_seed<k>.gif
cd ../report && python figures/make_car_sim_figs.py
```

说明：
- 车辆跟随报告中的作者源码重跑有 4 个进程（TRPO 种子 1、DDPG 种子 1、DDPG-CBF 种子 2 和 3）在启动约 5 s 时同时出现 Windows 原生崩溃，已用 `scripts/run_orig.py` 逐个原样重跑；`run_final.py` 因此错开了前几个进程的启动时间。
- TensorFlow 多线程计算不完全确定。重跑时 TRPO 能逐 episode 复现，DDPG 和 TRPO-CBF 会有细微差别（见报告 3.1 节）。
- 报告中的主结果运行启动时设为 2000 个 episode，在第 600 个之后停止，`results/final_600ep/` 由 `scripts/export_first_n.py` 截取前 600 个 episode 得到。直接用 `--episodes 600` 运行，得到的是同样的前 600 个 episode（不确定性差异除外）。

## 引用

```
@inproceedings{cheng2019end,
  title     = {End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks},
  author    = {Cheng, Richard and Orosz, G{\'a}bor and Murray, Richard M. and Burdick, Joel W.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {33}, pages = {3387--3395}, year = {2019}
}
```

作者源码的版权归原作者所有。本仓库只包含修改后的代码，不包含作者原始文件的副本；运行作者源码请从其仓库获取。
