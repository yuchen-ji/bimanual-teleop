# 导轨任务：左手抓端部取出，右手抓中部插入

2026-10-23 验证版任务顺序：**左手包握导轨一端 → 从料箱取出 → 右手握住中部并稳定 → 左手松开撤离 → 右手对准并插入导轨台 → 右手松开撤离**。

导轨大部分放在左侧料箱内，斜靠靠近机器人的窄边，仅端头略过箱沿。左手从箱口伸入，在距端头约 7.3 cm 处包握。实验读取当前桌子、料箱和导轨台的布局，任务配置为 `handover.yaml`。导轨重量使用用户实测的 764.2 g，写入本任务覆盖配置。默认仿真配置和相机标定不由本任务改写。

采用纯物理验收：通过手指及掌面接触抓取，保持原关节扭矩上限，不建立物体固定约束。当前手部静、动摩擦系数均为未实测的实验值 `1.2`，导轨摩擦系数为 `0.5`。本任务逐个校验实际材质绑定，结果写入 `hand_material_binding.json`。当前抓握组合已通过左手取出，不能据此把改善单独归因于摩擦系数。

[查看实际绑定后的 0.8 / 1.2 对照结果](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-demo-analysis/friction-comparison.md)：两次均在合手阶段因参与夹持的手指数量不足而停止，未进入承重阶段，不能据此比较提起后的滑移。

## 先查看参考示范

- [三路视频关键帧与两臂关节角对照](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-demo-analysis/README.md)
- [完整的两臂、两手实测关节与末端位姿](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-demo-analysis/reference.json)
- [新旧场景配置变化](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-demo-analysis/scene_changes.json)

示范取自 `recordings/rail_place3/episode_000010`，并对照查看了 episode_000011。两臂各 7 关节，两手各 20 关节；按三路视频的时间对齐读取实测数据。新手型采用示范中的五指包握和张手，臂部示范作为逆运动学选解参考。右手中部接管是本次任务要求，位置会重新计算。

## 操作前

在 **Remmina 远程桌面里的终端**执行带窗口命令。每个代码块均可整段复制；入口自动启动容器。一次只运行一个 Isaac 实例，先关闭上一次窗口。

**完整物理验证已通过。** `handover-physical-24/trial_000` 从正常初态完成全部 20 个阶段，最终 `status: success`、`phase: verify`。右手撤离后，导轨在无机器人接触的状态下稳定至少 1 秒。当前配置已完成一次全程验证，63 项软件检查通过。

`handover.yaml` 已采用两手的实测指节弯曲和小幅根部屈伸预紧。左手食指最后合拢，提起时保持腕姿竖直上提 28 cm。右手在导轨上方 8 cm 先完成预握，再接近中部；双手稳定后，左手张开，沿导轨法向离开并沿轴回收。安装座与手指只在本任务内使用凸分解碰撞体。

右手插入后四指张开、拇指向外退让，保持该手型沿导轨法向抬离 10 cm，再恢复普通张手。承重依靠手指及掌面的真实接触，不使用吸附、固定约束或运动学搬运；导轨质量 764.2 g 和手部原扭矩上限保持不变，通用默认配置未被本任务覆盖写入。

[查看成功结果](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-physical-24/result.json)，[查看独立核验报告](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-physical-24/physical_validation.json)，[查看最终右手已离物的腕部截图](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-physical-24/trial_000/snapshots/verify/camera_2.png)。最后 1 秒孔销误差不超过 0.276 mm、轴线倾角不超过 0.133°、支撑面间隙不超过 0.530 mm、插入深度至少 12.877 mm，两手接触力为 0。

[查看实际承重与箱沿净空曲线](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-grasp-batch-15/trial_000/extraction_validation.png)，[查看各阶段截图与状态](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-grasp-batch-15/trial_000/snapshots)。固定相机在左手高位时可能拍不全，窗口全景更适合查看整个动作。

[查看箱内摆放截图](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/handover-inside-scene-01/trial_000/snapshots/settle/camera_0.png)；对应回合为 `handover-inside-scene-01`，`result.json` 的状态为 `milestone_passed`、阶段为 `settle`。

## 1. 查看已通过的左手取出

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/handover.yaml \
  --mode auto \
  --until extract \
  --snapshots
```

打开窗口后，在 **Simulation** 面板点击 **Frame entire scene** 查看全景。左手自动取出导轨，状态应为 `milestone_passed`，阶段为 `left_clear`。动作在此停止，窗口保留供查看，最后手动关闭。无需遥操作或手动控制手指。

## 2. 查看完整双手任务

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/handover.yaml \
  --mode auto \
  --until insert \
  --snapshots
```

此命令从正常初态自动生成并执行完整动作；无需遥操作。窗口打开后点击 **Frame entire scene** 查看全景，动作完成后窗口保留。只有插入、松手、撤离和稳定检查全部通过才报告 `success`；遇到失败会停止并保存原因。右手握持未稳定时程序禁止左手松开。若只查看右手接管，把 `--until insert` 改为 `--until handover`。其他可选终点如下：

| 取值 | 终点 |
|---|---|
| `scene` | 导轨自然落稳，双手准备完成 |
| `grasp` | 左手抓住端部，按所选配置完成提起，完全脱离料箱支撑并稳定至少 1 秒 |
| `extract` | 左手把整根导轨抬出箱沿 |
| `handover` | 右手中部接管已确认，左手松开并撤离 |
| `transport` | 右手完成搬运，在定位销上方对准 |
| `insert` | 右手插入、松手、撤离，最终稳定检查通过 |

所选终点达到后，本次动作结束；它不是从该阶段开始运行的选项。只有 `insert` 完整通过时报告 `success`。

## 3. 逐段查看手臂与手指动作

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/handover.yaml \
  --mode step \
  --until insert \
  --snapshots
```

场景落稳后等待操作。出现 `Next stage` 提示时，点击同名按钮执行下一段。

| 按钮 | 作用 |
|---|---|
| Next stage | 当前段完成后执行下一段 |
| Run automatically | 从当前位置改为自动执行到所选终点 |
| Pause task / Resume | 暂停或恢复动作推进；物理仿真仍运行 |
| Capture snapshot | 保存当前三路相机图像和状态 |
| Frame entire scene | 恢复全景视角 |
| Reset scene | 恢复本版初态，并新建下一回合 |

暂停整个物理环境使用 Isaac 时间轴的 **Pause**。

## 4. 快速预检：不启动物理仿真

此命令可在 SSH 终端执行，检查左右抓握、左手提起终点和取出终点的名义逆运动学，以及左手预抓取的 Lula 路径和碰撞采样。它不证明已抓稳、完整搬运路径无碰撞或已完成换手。

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/handover.yaml \
  --mode auto \
  --preflight
```

结果查看本次目录的 `preflight.json` 和 `planning/`。预检通过报告 `preflight_passed`。

## 5. 结果目录与完成标志

终端第一行 **Experiment output:** 显示本次输出目录的绝对路径；每次自动创建新目录，位置统一在：

```text
/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_grasp_insert/
```

| 文件 | 用途 |
|---|---|
| `result.json` | 最新状态、失败原因和阶段 |
| `trial_000/events.jsonl` | 抓取通过、接管确认、阶段完成和失败事件 |
| `trial_000/trace.jsonl` | 两臂、两手实际关节与目标、导轨位姿、接触和误差 |
| `trial_000/snapshots/` | 阶段截图和对应 `state.json` |
| `trial_000/trajectory.json` | 新生成的 54 维关节目标轨迹及来源 |
| `planning/` | Lula 模型、路径、耗时和碰撞预检记录 |
| `grasp_templates.json` | 两手的示范适配手型与抓握变换 |
| `hand_material_binding.json` | 本回合手部摩擦系数实际绑定的校验结果 |
| `collision_approximations.json` | Isaac 实际碰撞形状、材质和摩擦系数 |
| `collision_shape_overrides.json` | 仅本任务生效的安装座、手指凸分解覆盖；不改默认场景 |
| `input.yaml`、`effective.yaml`、`metadata.json` | 本次配置、完整场景、代码版本和哈希 |

截图中的 `camera_0`、`camera_1`、`camera_2` 分别为头部、左腕、右腕。重点查看 `left_close`、`left_clear`、`right_close`、`left_retreat`、`right_align` 和 `verify` 目录。Reset 后使用 `trial_001` 等新回合目录。

| 状态 | 含义 |
|---|---|
| `running` | 正在运行，分步模式也可能在等下一步 |
| `milestone_passed` | 指定中间终点通过 |
| `success` | 完整物理任务通过就位、释放和稳定检查 |
| `preflight_passed` | 仅运动学与初始路径预检通过 |
| `diagnostic_passed` | 轨迹回放或局部诊断完成 |
| `failed` / `interrupted` | 失败或提前停止，查看 `reason` |

连续视频需要额外使用 `--record`；`--snapshots` 只保存截图。无窗口运行需同时使用 `--headless --mode auto`。具体命令、实现与验收记录见[技术说明](/home/jiyuchen/project/WYS/bimanual-teleop/workspace/experiments/rail_grasp_insert/docs/implementation.md)。

## 旧单手轨迹

`config.yaml` 和 `narrow-*` 输出属于旧版左手中部抓握任务。旧轨迹依赖旧桌子、料箱和导轨台位置；仅提供旧 `input.yaml` 不会恢复完整旧场景，入口会拒绝这种不匹配的旧回放。旧视频和结果保留为历史参考。
