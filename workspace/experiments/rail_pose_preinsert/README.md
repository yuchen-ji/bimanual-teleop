# 导轨预插入仿真实验

模拟递料、左手抓握、位姿估计和搬运，在目标位姿沿孔轴外移 20 mm 处保持；不执行插入或真机控制。当前抓稳和 FoundationPose 方向稳定性尚未通过验收。

## 启动

当前默认执行保持导轨朝向的五指抓握（`task.until: grasp`、`orientation_locked_grasp: true`）。握紧前以模拟递料支撑保持初始朝向，规划器允许导轨平移，并分别拟合四指末端指节和拇指接触；现有碰撞体和碰撞过滤保持不变。通过实际接触与完整网格检查后，才启用法兰与导轨的固定关节。

`grasp_contact_surface: bottom` 指定四指末端内侧贴合导轨底面：最近点必须落在导轨的下表面中央区域、位于末端指腹内侧，且内侧朝向底面（夹角不超过约 20°）。侧面间隙再小也不满足连接条件。先调整拇指避让，再平移导轨，最后弯曲四指；每段在执行前检查完整手部几何，并检查不同手指的中节、末端指节和指腹之间的间距；四指张开角度参与求解，避免互相挤压造成目标偏移。`bottom_grasp_seed.json` 是底面包握的求解初值，与历史两侧夹持初值分开保存。

```bash
python workspace/experiments/rail_pose_preinsert/run.py --mode auto --until grasp --grasp contact_fixed
```

当前 `grasp_attachment_criterion: distal_clearance` 要求五指末端与导轨的间隙均不超过 0.2 mm，实际 PhysX 指节位姿下完整网格不相交，且至少两次检查间隔 0.5 秒；不使用接触力阈值强行压入现有碰撞体。连接后做小幅机械臂往返运动，检查导轨相对法兰的位姿误差。`constrained_grasp_complete` 表示几何贴合已验证且仿真固定约束已建立，**不是纯摩擦无约束抓稳验证**；`physical_grasp_validated` 保持 false。诊断数据保存在 `trials/trial_*/locked_grasp/`，完成图像在 `captures/trial_*/locked_grasp/`。默认只检查抓握，不执行搬运到预插入位姿。

首次合拢在约 1 mm 的规划间隙处停止，再按实际指节表面距离逐指微调，给重力和关节跟踪偏差留余量。`hand_profiles/locked_grasp_seed.json` 只用于求解初值，每次都会重新检查场景和路径。底面约束抓握姿态及法兰—导轨相对变换保存在 `hand_profiles/left_bottom_grasp.json`，历史侧面握姿保存在 `hand_profiles/left_locked_grasp.json`，其中保留纯摩擦抓稳尚未验证的标记。

早期四指尝试的 `left_four_fingers_supported.json` 已标记为用户拒绝，仅保留诊断记录。下面的四指预闭合是保留的独立诊断模式，其接触统计覆盖整根手指，不能代替上述末端指节接触判据。

**四指预闭合**可显式选择（`--until four_fingers`）。保留模拟递料支撑和现有碰撞体；只驱动四指屈伸，拇指、侧向张合、手臂目标及导轨位姿不变。每轮最多弯曲 2.5°，平滑运动后等待 0.6 秒，再以实际关节与法兰位姿检查完整实体网格。距离导轨不足 2 mm 或已触碰时减为 0.5°；接触力达到 0.3 N 即停止该指当轮运动，稳定接触超过 0.2 N 时保持。

单指超过 3 N、手部位移超过 1 mm 或检查发现实体网格相交，会退回已检查的角度并减小步幅；最细 0.1° 仍不能安全继续时，该指停止尝试。各轮角度、接触力、几何检查和退回原因均保留。`four_finger_flexion_limit_rad` 只定义此阶段三个屈伸关节的行程上限，不覆盖完整抓握参考。此阶段不执行目标示教、FoundationPose、重新递料、搬运或撤去支撑。

启动后点击 **Start / Next stage**；完成后保存三路图像、关节角和各指接触力，并暂停物理：

```bash
python workspace/experiments/rail_pose_preinsert/run.py --mode step --until four_fingers
```

四指接触持续满足条件时记录 `supported_four_finger_contact`；缺少接触时记录 `four_finger_preview_complete`，并列出缺少接触的手指。两者都不代表无支撑抓稳。结果在 `captures/trial_*/four_fingers/` 和 `trials/trial_*/four_fingers_pose.json`。需要恢复完整流程时显式指定 `--until preinsert`。

在 Remmina 远程桌面的终端运行：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python workspace/experiments/rail_pose_preinsert/run.py --mode step --perception foundationpose
```

启动器会打开 Isaac Sim 5.1。共享 overlay 同时只能运行一个实例。只检查抓握、暂不调用 FoundationPose 时：

```bash
python workspace/experiments/rail_pose_preinsert/run.py --mode step --until grasp
```

启动后自动恢复四视图：左侧全景，右上头部相机，右下左右腕部相机；四个画面均为 **4:3**。Camera Inspector 在左侧，Stage/Simulation 在右侧，Content/Console 在底部。布局与全景视角保存在 `task.desktop_layout`；`enabled: false` 可关闭启动时恢复。图像保持比例，面板较窄时显示黑边。

这里的 Camera Inspector 是任务提供的只读位姿查看器，显示所选相机的世界/局部位姿，不改写相机标定或物理对象。

## 执行与失败重试

操作集中在 **Simulation** 面板。

- **Start / Next stage**：开始当前阶段；任务已经结束时，重置并开始新一轮。
- **Run automatically**：自动推进。
- **Restart task**：复位并从头重新执行，无需关闭 Isaac；分步模式仍会在后续阶段等待 Next。
- **Apply config / Restart task**：重新读取 YAML；普通参数只复位任务，场景结构或资源改变时自动重建场景。
- **Apply config / Rebuild scene**：强制重新读取配置和资源，在同一个 Isaac 应用内重建场景。
- **Pause / Resume physics**：暂停或继续尚未结束的任务。任务结束后保持暂停；原生工具栏 Play 也不会继续执行失去检测的物理场景。
- **View head camera / Frame entire scene**：查看头部相机或全景。
- **Restore default layout**：恢复任务配置中的四视图布局、相机绑定和 4:3 比例。重试任务会保留当前窗口布局。

抓握、感知或碰撞检查失败后会暂停物理并保留软件，可直接点击 **Restart task** 重试。场景加载后的 Python 运行异常也会保留窗口，并在重试时重新绑定物理控制句柄。构建场景失败时，在错误窗口点 **Apply config / Rebuild scene** 可修正配置后重试。

参数通过 `config.yaml` 或 `--config <配置路径>` 设置。加 `--record` 录制三路视频。无窗口运行使用 `--headless --mode auto`，失败后仍返回非零退出码。

## 修改参数再试

保持 Isaac 打开，在编辑器中修改并保存启动时的配置文件，然后点 **Apply config / Restart task**。读取期间暂停任务；无效配置会显示拒绝原因，并保留原设置。**Restart task** 只重试已应用的设置，不读取文件。命令行指定的 `--until` 始终优先。

默认 `config.yaml` 已保存用户调整后的七个初始关节角、预抓握手势、闭合目标和手轨相对位姿。直接点 **Apply config / Restart task** 即可；这是任务复位，不是从当前姿态执行旋转轨迹。

| 修改内容 | YAML 字段 | 处理方式 |
| --- | --- | --- |
| 默认初始机械臂姿态 | `simulation.robot.ready_deg.left` | 7 个关节角，单位度；应用并重置，无需重新启动软件 |
| 导轨初始生成位姿 | `task.initial_rail_pose_world: {xyz_m: [...], rpy_deg: [...]}` | 直接在指定世界位姿生成导轨；不改变导轨台上的目标位姿 |
| 自动模板的递料初始位置 | `task.initial_rail_offset_world_m` | 仅自动模板模式：重算初始 IK 并重置 |
| 自动模板的法兰 Z 轴朝向 | `task.initial_flange_z_axis_world: [1, 0, 0]` | 需要已加载轴对齐代码的启动器：保持法兰初始位置，最小旋转对齐并生成关节角 |
| 手内导轨位置、方向 | `task.rail_in_flange_override: {xyz_m: [...], rpy_deg: [...]}` | 应用并重置 |
| 开手、闭合握姿 | `task.hand_open_override_rad`、`task.hand_grasp_override_rad` | 各 20 个弧度值，按手部关节顺序 |
| 时长、规划速度、检查阈值 | `task` 中对应原有字段 | 同步规划器与仿真，再重置 |
| 质量、摩擦、碰撞偏移 | `simulation.rail` 中 `mass_kg`、`friction`、`finger_friction`、`contact_offset_m`、`rest_offset_m` | 更新物理属性；质量变化按比例更新惯量 |
| 机械臂驱动 | `simulation.robot.arm_stiffness`、`arm_damping` | 更新实际关节驱动 |
| 左手驱动 | `task.left_hand_stiffness`、`left_hand_damping` | 各 20 个正值 |
| 模型、碰撞几何、相机或桌面场景 | 对应仿真配置或资源文件 | 校验后自动重建场景，保留 Isaac 应用 |

重建会停止当前任务和录制，释放旧相机输出、控制器和物理对象，再加载场景并恢复四视图。分步模式重建后等待 **Start / Next stage**。旧轮次和配置历史保留；配置校验不通过时不拆除旧场景。场景构建和碰撞计算仍需要时间，但不再重复启动整个 Isaac。

机器人资源仍需满足本实验的双臂 54 关节、官方运动学和资源清单一致性；重建不负责将任意新机器人自动适配到规划器。修改应用启动环境、GPU 或已加载的普通 Python 代码不属于此配置重建入口。

默认配置使用已有的显式位姿模式（内部名 `legacy_geometry`），直接读取关节角，不再用模板覆盖它们。闭合参考来自人类示范，来源保存在 `task.grasp_reference`；本轮四指调整已撤销，历史实际保持角度见每轮的 `four_fingers_pose.json`。`initial_flange_z_axis_world: null` 表示不额外求解，保留手动调整后的手臂姿态。不要修改输出目录里的 `effective.yaml`；它用于记录实际生效参数。

左手“预抓握手势”保存于 `hand_profiles/left_pregrasp.json`，标签为 `left_pregrasp`：拇指根部第二关节后转约 50.918°，其余 19 个关节目标为零。当前 `hand_open_override_rad` 已使用此手势；以后复用时，将该文件中 `candidates[0].command_position_rad` 填入此字段并应用。文件同时保留实测关节角和仿真来源，复用目标值使用 `command_position_rad`。

自动模板配置保留在 `config.demonstrated.yaml`，下次启动时可用 `--config workspace/experiments/rail_pose_preinsert/config.demonstrated.yaml` 选择。该模式下可修改初始偏移和目标轴，点击应用时自动求解；正常参数迭代同样不需要重启 Isaac。

FoundationPose 在首次测量时加载一次，后续阶段、重试和场景重建复用同一个模型进程；感知模型或启动配置改变时才重新创建感知进程。每次测量独立初始化位姿。进程异常或超时后，下次测量重新启动；退出实验时释放进程。

## 结果

启动时打印输出目录，默认位于：

```text
/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_pose_preinsert/<本次运行>/
```

`result.json` 查看结果，`failure_state.json` 查看失败瞬间状态；`runtime.log`、`foundationpose.log` 查看日志；`captures/` 查看图像。每轮参数和结果保存在 `trials/trial_*/`，配置版本保存在 `configurations/`，实际物理参数见 `applied_physics.json` 和 `actual_drives.json`。界面打开或重试成功不代表抓握与搬运验收通过。

### 后台服务与窗口重连

后台求解服务随 Isaac 窗口持续运行，不因某一轮成功、失败而退出。测试结束后点击 **Start / Next stage** 或 **Restart task** 开始新一轮；原生工具栏播放不会重启已结束的任务。长时间求解期间独立线程继续更新 `rpc/worker.json`，后台断开超过 5 秒时，等待规划的任务会暂停并显示 `Planner offline`，在物理已经暂停时也能检测。

若桌面窗口仍在、原主机启动器已退出，可以在项目 Conda 环境中重新连接当前抓握场景（输出目录取当前控制器目录，PID 取主机可见的 Isaac Python 进程）：

```bash
python workspace/experiments/rail_pose_preinsert/bottom_grasp_worker.py <当前输出目录> --scene-pid <Isaac进程PID>
```

重连服务支持本实验的 oracle 抓握、重启与配置重载，按保存的轮次设置恢复配置版本；窗口退出后服务自动结束，同一目录不允许两个后台同时处理请求。FoundationPose 场景应重新启动完整的 `run.py` 主机流程。断线后先连接后台，再点击任务重启。

### 人类示范抓握适配

`extract_recorded_grasp.py` 从“左手握紧导轨”录制中提取实测手指关节与同步机械臂姿态；`adapt_recorded_grasp.py` 在导轨方向固定、碰撞设置保持原样的条件下拟合左手与末端。参考样本、SDK/仿真基座变换和关节顺序保存在 `hand_profiles/recorded_left_tight_grasp.json`。

2026-10-25 的已验证姿态保存在 `hand_profiles/left_recorded_adapted_grasp.json`，包含实测角、带接触预紧量的目标角、世界系末端位姿、手轨相对位姿和验证结果。无固定连接、无外部支撑保持 5.1 秒通过；末端上移 5 毫米并旋转 1°、返回并稳定保持也通过。末端方向相对示范约差 5.75°；手指角度均方根差约 18.39°，个别末节最大约差 42.08°。拇指目前主要由中段接触。这是当前仿真几何下的适配结果，不是逐关节复现，也不是实机验证。

`adapt_contact_sim.py --output <试验输出目录>` 使用该目录的 `effective.yaml`、`runtime.json` 启动独立接触试验；`adapt_contact_control.py` 执行 `segment.json` 中的短轨迹，限制右臂/右手目标、锁定闭合阶段导轨方向，并在超力、环境碰撞或无支撑滑移时暂停。`squeeze_recorded_grasp.py` 负责微量闭合，`balance_recorded_grasp.py --operation preload|balance` 生成预紧或指根卸载/拇指夹持步骤。每一步都应读取上一段的实测状态；仅触碰或闭合到目标角不等于抓握验收通过。

本次记录位于 `workspace/outputs/rail_pose_preinsert/adapt_recorded_20261025/`，`validation.json` 为汇总，`native/*_state.json`、`*_trace.json`、`*_result.json` 保留检查点与测试过程。配置中的目标角含接触预紧，不能直接当作无接触初始姿态；复用应经过带接触限值的闭合轨迹。原始可视网格与 PhysX 简化碰撞面有细小差异，检查记录同时保留两者结果，未修改碰撞体来消除差异。

当前接触试验的初始状态另存为 `hand_profiles/left_grasp_initial_state.json`。`adapt_contact_sim.py` 启动及 **Restart task** 都会读取它：物理关节恢复保存的实测角，驱动恢复保存的夹持目标，导轨恢复保存的位姿；支撑和固定连接均关闭。初始化完成后暂停，点击 **Play** 或 **Pause / Resume physics** 可持续观察抓握；不会自动重新执行预抓握与递料。恢复后已实际播放约 5 秒验证通过。该入口的持续保持保留滑移和接触超限暂停检查。
