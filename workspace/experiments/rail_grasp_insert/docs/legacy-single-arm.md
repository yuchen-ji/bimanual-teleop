# 旧单手任务：实现与历史验收

本页仅记录 2026-10-21 旧场景、旧单手任务。下列命令使用 config.yaml，不适用于当前双手接管任务。当前入口和验证范围见 ../README.md。

## 1. 录制、无窗口运行和连续验收

以下每组命令均可独立复制。带窗口命令在 Remmina 桌面终端运行；无窗口命令可在 SSH 终端运行。一次只运行一个 Isaac 实例。

### 录制三路视频和原始状态

当前服务器完整回合实测约 30 分钟。任务自动执行；窗口保留时录制仍可继续。查看完成后关闭窗口，或点击 **Save and stop recording** 结束录制并写完文件。

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/config.yaml \
  --mode auto \
  --until insert \
  --record \
  --snapshots
```

视频在本次输出目录的 `recordings/` 内。`--record` 开启三路 RGB 和原始状态录制，`--snapshots` 保存阶段 PNG。B 键只记录无名称分段点，任务名称和分组在离线阶段指定。

### 无窗口执行一次完整任务

约 10 分钟，任务结束后自动退出。终端输出本次目录，检查其中的 `result.json`。

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/config.yaml \
  --headless \
  --mode auto \
  --until insert \
  --snapshots
```

### 无窗口连续验收 5 次

约 51 分钟。在同一个 Isaac 进程内重置场景后逐次执行；任一失败即停止。`summary.json` 应显示 `requested_trials`、`completed_trials`、`passed_trials` 均为 5，且每次状态为 `success`。

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/config.yaml \
  --headless \
  --mode auto \
  --until insert \
  --repeat 5
```

## 2. 参数与局部诊断

| 参数 | 用途或限制 |
|---|---|
| `--config` | 实验 YAML；回放时使用原运行保存的 `input.yaml` |
| `--output` | 必须是本项目 `workspace/outputs/` 下尚不存在的目录；常规操作省略，由入口自动创建唯一目录 |
| `--headless` | 不显示窗口，必须配合 `--mode auto` |
| `--repeat` | 连续次数；大于 1 必须为自动模式 |
| `--steps` | 物理步数上限；提前达到上限不代表任务成功 |
| `--replay` | 自动模式下单次回放；不与中间终点、重复运行或局部初始化组合 |
| `--initial-state` / `--start-phase` | 成对使用；局部初始化必须是无接触的张手状态；阶段可选 `slot`、`approach`、`wrap`、`close` |

局部初始化只设置回合起点，不在执行中改变物体位姿。局部运行只报告 `diagnostic_passed`，不计入完整任务成功。回放也只报告 `diagnostic_passed / trajectory_replayed_only`，播放完成本身不认证物理任务成功；使用原回合的配置，示例见运行首页。

实验子进程将 OpenBLAS 设为单线程，参数写入 `metadata.json`。此设置降低多线程开销和 CPU 占用，不改变物理步长，也不应据此预期完整回合显著加速。频率和动作时长按仿真时间计算。

## 3. 代码文件

| 文件 | 用途 |
|---|---|
| run.py | 独立入口；合并实验覆盖，保存配置与版本，启动容器，支持自动/分步、重复和回放 |
| config.yaml | 窄边初态、双臂待机、握持模板、手指姿态、路径、速度与验收阈值 |
| controller.py | 阶段控制、末端路径与逆运动学、实际位姿/接触反馈、张手退出、日志和轨迹 |
| evaluate.py | CAD 孔销配准、窄边摆放、对向接触与最终就位检查 |
| tests/test_rail_grasp_insert.py | 坐标、限位/限速、阶段与故障、重置、张手、回放和默认配置隔离检查 |
| README.md | 启动、查看、输出字段和验收证据 |

通用模块增加可选动作源接入、实验输出目录绑定，并修复任务监视器的 JSON 标量类型；默认配置、模型资源和相机标定不变。

## 4. 动作生成和配置

`config.yaml` 的 simulation 覆盖环境字段，task 保存抓握、路径、速度、容差和超时。手指角度按模型的 20 关节顺序，单位 rad。

窄边摆放使用重新计算的握持方向和食指/中指姿态；分段接近路径使用落稳导轨坐标系，每个中间点确认到位后推进。抬出时向桌面中心移动并升高，避免只沿原位置上升造成末端不可达。这些动作通过真实接触物理运行验收。

实验中的右臂以抬高姿态固定待机，为导轨转平留出空间；左臂也使用实验初始待机姿态。两者均由实验 YAML 覆盖，默认配置不改。

抬出后先移动到 `transfer_center_m` 指定的空域，再一边转平一边移到定位销上方高位，随后垂直对准。该路径避免以倾斜姿态一直移到工作区近端，并为转平后的腕部姿态留出空间。逆运动学保留 5° 关节限位余量；较大的关节跟踪误差持续超过 0.25 秒即停止。

提起稳定检查对速度和支撑力使用 100 ms 指数平滑，再要求持续 1 秒；同时保留瞬时速度、支撑力上限和对向接触检查。该窗口用于处理物理步接触微振动，原始数值与平滑数值均保存，不将持续支撑或明显摆动计为抓稳。

模型名 `l_wrist` 的网格包含手掌基座，可在对向手指仍有效夹持时辅助接触。搬运期间持续检查拇指与食指/中指对向接触、物体相对手的滑移；支架和机械臂接触导轨仍会停止任务。

搬运、转平、对准和插入直接反馈导轨实际位姿，补偿夹持接触的弹性变化。握持完整性检查限制平移滑动 8 mm、相对转动约 14°，并持续要求对向手指接触；最终孔销对准另按 CAD 间隙和导轨实际姿态验收。

转平阶段允许保持对向接触的受控滚动，按导轨实际路径误差检查，超过 10 mm 或约 11.5° 持续 0.25 秒即停止。到达目标姿态后，物体和相对握持关系稳定至少 1 秒，才更新握持方向基准并进入对准。其余搬运阶段仍执行相对转动限制。

转平与对准的稳定窗口要求：物体位移不超过 1 mm、姿态变化不超过约 0.29°，相对握持位移不超过 1 mm、转动不超过约 0.57°；转平保持 1 秒、对准保持 0.2 秒，窗口内持续满足目标误差和接触条件。瞬时及平滑速度仍记录，用于检查接触微振动。

松手后沿导轨横向向外、略向上撤离，使张开的手指先退出底板下方。`retreat_direction_in_rail` 为该方向，`retreat_m` 为撤离距离。

松手姿态根据实际握持关系生成：拇指尖退出槽口上方，食指和中指尖退出底板下方；使用模型指尖运动学求解关节角，再按关节速度限制张开。这样可适应转平时改变的握持方向。

张开后检查实际导轨坐标系中的指尖位置：拇指 Z ≥ 24 mm，食指/中指 Z ≤ −24 mm；指尖和其他部件的导轨接触力不超过 0.05 N。食指/中指近节可保留不超过 5 N 的边缘接触，随后沿横向撤离来消除；最终稳定验收仍要求全部机械手和机器人接触力不超过 0.05 N。

## 5. 输出和判定口径

终端显示本次目录，位于 `workspace/outputs/rail_grasp_insert/<run_id>/`：

- input.yaml、effective.yaml、metadata.json：输入、完整配置、版本和默认配置哈希。
- result.json：最新结果。milestone_passed 只代表所选阶段通过；完整任务要求 success。
- trial_000/trace.jsonl：实际状态、目标、接触和误差；每次 Reset 新建 trial。
- trial_000/events.jsonl：阶段与失败事件。
- trial_000/trajectory.json：兼容原 --trajectory 的 54 维关节目标，保留 test_data=true。
- trial_000/snapshots/<phase>/camera_0.png 等：头部、左腕、右腕图像和对应 state.json。
- recordings/：可选原始录制，保留来源 UUID。

录制元数据的 action_source 包含实验回合、任务参数、驱动覆盖和代码哈希，便于独立追溯；Zarr/视频格式与无名称 B 分段格式保持原样。

步数上限或窗口关闭不等于任务成功。检查 result.json 和实际物理状态；异常碰撞、滑移、卡住或超时均停止推进并记录原因。

原始录制的 complete 表示文件采集完整；任务是否完成仍以实验 result.json 的 success 为准。

原始 `task_metrics.task_completed` 保留默认环境的完整任务定义（还包含成品箱阶段）。本实验终点为定位夹具，因此该字段可以为 false；本实验使用独立的就位、释放和稳定检查判定 success。

## 6. 已完成的验收（2026-10-21）

以下验收目录均位于 `/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/`。

使用仿真真值、预设握持模板、逆运动学及真实接触。右臂待机；不连接实机驱动。默认配置和相机标定只读，实验覆盖在内存中合并。

初始摆放：导轨斜靠左侧料箱**靠近机器人的窄边**（箱体 X 负方向、沿 Y 的箱沿），低端在箱内，高端伸向机器人。正常初态完整任务已连续 5 次通过，完整三路录制、实际回放与默认场景回归均已通过。总报告为 `workspace/outputs/rail_grasp_insert/validation-2026-10-21.json`；此前宽边摆放的调试不计入验收。

窄边场景已于 2026-10-21 通过一次自然落稳验证：`workspace/outputs/rail_grasp_insert/narrow-stage1-scene/result.json` 为 milestone_passed；落稳中心约 `[0.6631, 0.5611, 0.8315] m`，倾角约 24.6°，箱体支撑约 7.71 N，无机械手接触。`scene_geometry.png` 是根据实际状态绘制的俯视与侧视图；三相机快照位于该目录的 `trial_000/snapshots/settle/`。

窄边抓取于 `narrow-full-01` 的 `trial_000/events.jsonl` 中通过 lift：导轨中心抬升约 19 mm，对向接触并稳定至少 1 秒。快照为 `trial_000/snapshots/lift/`；完整成功与连续验收记录见下文。

`narrow-full-11` 通过搬运、转平与对准。`trial_000/snapshots/align/state.json` 的实际孔销横向误差约 0.023 mm，轴线倾角约 0.006°，导轨位于支撑面上方 25 mm。该指标仅证明插入前的对准。

`narrow-full-12/result.json` 报告正常初态全程 success：松手撤离后稳定 1 秒，最终孔销误差约 0.176 mm、支撑间隙 0.573 mm、插入深度 12.79 mm。未使用物体吸附、固定约束或运行中的物体位姿重设。

同目录的 `actual_rail_path.png` 展示实际导轨中心轨迹与初末姿态；`quality.json` 记录对最后连续 121 个状态（1 秒）的独立复核、物理时钟和原生轨迹解析检查。

`narrow-acceptance-5/summary.json` 已报告正常初态连续 5/5 次 success；每次在同一进程内重置，实际驱动属性一致。每个 trial 下的 `quality.json` 独立复核最后 1 秒全部就位条件、物理时钟和轨迹解析。五次最终孔销误差为 0.176–0.432 mm，支撑间隙为 0.573–0.579 mm，插入深度为 12.79–12.85 mm。此结果仅覆盖当前固定场景。

`narrow-recorded-success` 又完成一次正常初态全程 success。`recording-quality.json` 核对三路视频各 3988 帧（640×480、30 Hz）、31904 条状态（240 Hz）、所有视频帧解码、状态/图像对应关系和来源 UUID；`trial_000/quality.json` 通过最后 1 秒独立就位复核。阶段快照位于 `trial_000/snapshots/`，完整视频位于该回合 `recordings/` 下的 episode 目录。

`narrow-replay-validation/replay-quality.json` 比较实际物理回放的 15952 个状态：关节目标最大差异约 2.84×10⁻¹¹ rad，实际导轨位置和姿态与录制逐样本一致，来源 UUID 和最后 1 秒就位检查均通过。回放报告 diagnostic_passed，独立于上述 5 次闭环成功。

`default-regression/quality.json` 记录原有 `python bimanual_teleop/sim/run.py --headless --steps 600` 入口的回归结果：无实验动作源，正常保持默认姿态并结束，54 个关节与导轨状态有效。默认 YAML、相机标定和 scene.py 与 HEAD 字节一致；默认 YAML 哈希与本任务开始时一致。

## 7. 软件检查

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python scripts/run_tests.py discover -s /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/tests -v
python scripts/run_tests.py discover -s /home/jiyuchen/project/WYS/bimanual-teleop/bimanual_teleop/sim/tests -q
```

物理验收仅覆盖当前固定场景；质量和摩擦沿用已有仿真假设。

相关软件检查：实验 21 项通过；仿真模块 54 项中 53 项通过、1 项因主机缺少 Torch 跳过。此前全项目 816 项测试中的 3 项契约检查失败，在未修改 HEAD 的归档中也可复现（扫描现有仿真测试时禁止导入 cv2 等外部依赖）；日志为 `bimanual_teleop/sim/.runtime/rail-project-regression.log` 和 `rail-baseline-contracts.log`。

## 8. 宽边历史调试记录

`stage1-scene/` 和 `stage2-grasp-*` 使用了宽边摆放，仅保留为调试记录，不作为当前窄边任务验收证据。此前仿真回归运行 53 项，其中 52 项通过、1 项因缺少 Torch 跳过。
