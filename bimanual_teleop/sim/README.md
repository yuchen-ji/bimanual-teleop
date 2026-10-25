# 双臂仿真：在 Remmina 中直接查看 Isaac

默认使用一个 Isaac 进程，不启动网络桥接或网页查看器。通用仿真环境保留在本目录，具体任务实验放在项目根目录的 `workspace/experiments/`，真机遥操与采集入口保持独立。

需要运行导轨任务时，查看[导轨双手接力任务：运行与查看](../../workspace/experiments/rail_grasp_insert/README.md)，当前流程为左手抓端部取出、右手抓中部接管并插入；其中提供完整运行、分步查看和快速预检命令。默认启动不启用实验动作源，默认配置不改写。

当前默认场景使用 `assets/source/rail_task/` 中复制的导轨任务 CAD：480 mm 长导轨、长导轨台、完整桌子、两侧各五层转运箱，以及两个腕部的力传感器、相机与支架。机器人面向桌子窄边，导轨默认在导轨台上，Reset 恢复同一位置。[模型与装配说明](docs/task-models.md)

2026-10-22 实测布局：五层箱沿离地 83 cm，箱口净长／净宽／内深为 44.5／31.8／16.5 cm；上层箱口靠机器人侧外沿比桌子近边向 +X 偏移 15.3 cm；桌子近边距腰／腿部立柱前表面 29.5 cm。桌面仍为 73 cm，导轨台与默认导轨随桌子同步移动。

头部相机已按真实导轨视频末帧设置位置、朝向、内参和镜头畸变；配置位于 `configs/default.yaml` 的 `cameras.head`。[相机对齐说明](../../camera_calibration/docs/head-alignment.md)

“View head camera”默认使用 1280×960 原生预览，三路采集仍为 640×480。`cameras.preview_scale` 控制预览倍数；实时光照采样与抗锯齿设置集中在 `rendering.py`。腕部相机实体采用当前 CAD 装配位置；光学位姿已应用 2026-10-20 的实测手眼标定，内参和畸变使用对应 D436 参数，默认加载 `configs/wrist_camera_calibrated.yaml`。[腕部标定结果](../../camera_calibration/docs/wrist-results.md) · [标定采样与操作步骤](../../camera_calibration/README.md)

桌面采用暖棕木纹贴图，转运箱采用黑色塑料细纹贴图，桌架采用深色，桌上增加仅用于显示的黑色垫板，地面采用暖灰色。颜色和室内光照在 `configs/default.yaml` 的 `scene.appearance` 中调整，颜色值为线性 RGB；`textures: false` 可切回纯色，`table_mat: false` 可隐藏垫板。贴图保存在 `assets/textures/` 并由 Git 跟踪。外观按真实视频目视近似，不代表实测反射率或办公室背景重建。修改后重新启动 Isaac 生效。[腕部相机数据与光心修正说明](../../camera_calibration/docs/wrist-results.md)

机器人使用实拍的主要配色：双臂与上部躯干为白色喷涂外壳，立柱、底座和连接部位为深灰，手掌与手指采用深色哑光近似实拍中的手套，力传感器为灰色金属。`robot.appearance` 控制配色，覆盖 CAD 原有的红绿蓝调试色；该覆盖只作用于显示材质。手套的厚度、布料形状和机器人贴纸尚未建模。

## 场景物理参数与预设

2026-10-23 起，导轨默认质量采用用户实测 **764.2 g**，`mass_is_assumption: false`。左右手安装座默认采用保留凹槽的凸分解碰撞体；手部物理材质和接触偏移会写入实际碰撞形状，包括 URDF 导入的实例内部形状，并在启动时核验。手部接触偏移为 0.2 mm、静止偏移为 0。默认手部摩擦系数仍为 0.8；双手任务的实验值 1.2 继续由任务覆盖。

| 启动方式 | 导轨初态 | 手指碰撞 |
|---|---|---|
| 普通启动 | 导轨台上，保持原初态 | 凸包，计算量较低 |
| `configs/hand_contact.yaml` | 导轨台上，保持原初态 | 精细凸分解 |
| `configs/rail_bin_pick.yaml` | 大部分在左侧料箱内，斜靠机器人侧窄边 | 精细凸分解 |

在 Remmina 终端执行以下命令查看箱内初态：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python /home/jiyuchen/project/WYS/bimanual-teleop/bimanual_teleop/sim/run.py \
  --config /home/jiyuchen/project/WYS/bimanual-teleop/bimanual_teleop/sim/configs/rail_bin_pick.yaml
```

窗口选择 **Frame entire scene** 查看布局。此预设只加载场景，双臂保持待机；自动取出、换手和插入使用[双手任务入口](../../workspace/experiments/rail_grasp_insert/README.md)。若只想查看导轨台上的精细手部接触场景，将配置文件名换为 `hand_contact.yaml`。

手指精度由 `robot.collision.finger_approximation` 选择 `convexHull` 或 `convexDecomposition`。安装座凹槽默认采用凸分解；底座、传感器及相机支架保留原成功回合实际使用的凸包表示。旧场景曾声明这些硬件采用凸分解，但导入节点类型使声明没有实际生效；本次不将其变成额外的几何变化。刚度、阻尼、抓点、手型、抓放顺序及速度仍属于任务控制配置。

普通启动的实际物理参数检查结果写入 `bimanual_teleop/sim/data/local/scene_physics.json`；任务启动则写入该次输出目录的 `scene_physics.json`。其中包括实际导轨质量、每个碰撞体的表示、接触偏移和材质，以及绑定核验结果。场景配置变化后旧轨迹会进行场景签名检查，已有成功记录保留其原始完整配置和代码哈希。

## 日常使用

在 **Remmina 远程桌面内的终端**运行：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python bimanual_teleop/sim/run.py
```

直接打开已装配完整场景的 Isaac 原生窗口。默认只查看、计算物理和保持机器人待机，不连接外部设备，不录制三相机。首次加载需等待应用和模型准备完成；窗口出现后短暂黑屏不代表失败。

录制时加一个参数：

```bash
python bimanual_teleop/sim/run.py --record
```

启用三路 640×480 RGB，直接在同一 Isaac 进程中写 MP4、Zarr 和事件。原生窗口右侧的 **Simulation** 面板提供录制暂停 / 恢复、B 分段、保存、重置与恢复观察视角。按 **B** 记录无名称边界；“Save and stop recording”完成当前示范并停止录制，之后再点“Record / Pause recording”新建下一条。关闭窗口会保存收尾。

**默认动作源是保持待机，录制只用于检查图像和数据链路，标记为测试数据，正常学习导出会拒绝。** 不接外部设备也能用审核过的关节轨迹驱动：

```bash
python bimanual_teleop/sim/run.py --record \
  --trajectory bimanual_teleop/sim/data/my_trajectory.json
```

轨迹用仿真时间、54 个绝对关节目标和明确的关节名称；位置 / 速度限制及起点待机姿态均检查。不是自动生成抓取策略，任务完成仍需审核。

## 常用参数

| 参数                       | 用途                                                        |
| -------------------------- | ----------------------------------------------------------- |
| `--config PATH`          | 当前机器人场景的 YAML 覆盖配置                              |
| `--record`               | 启用三相机与记录器                                          |
| `--trajectory PATH`      | JSON 关节轨迹，单位 rad，时间 s                             |
| `--headless --steps 480` | 无窗口运行固定物理步数；适合自动检查                        |
| `--codec libx264`        | NVENC 不可用时使用 CPU 编码                                 |
| `--display :10`          | 从 SSH / 无 DISPLAY 的终端指定自己的 X11 桌面；通常无需设置 |

配置与轨迹文件放在项目目录内，启动器自动映射为容器路径。共享的可写 overlay 一次只运行一个 Isaac 实例；先关闭旧窗口再启动下一次。

例如 `--config bimanual_teleop/sim/configs/wide_workspace.yaml` 增加两侧箱堆间距，保留真实 CAD 桌子尺寸。它演示同一任务的布局变化；新物体与新任务条件的支持范围见架构说明。

## 首次安装与资料

[从零搭建手册](docs/zero-to-simulation.md) · [简化架构与数据主干](docs/architecture.md) · [验证记录](docs/validation.md)

主机使用 `bimanual-teleop` Conda；Isaac 使用官方 5.1.0 容器。模型已经随 Git 跟踪，克隆后可直接使用；首次运行 `docker/bootstrap-apptainer.sh` 准备 Isaac 运行时。本机已完成。已删除重复的 Three.js 网页全景实现、浏览器资源安装入口及 8081 预览转发；日常查看不再依赖它们。

模型在 `assets` 中，原始 CAD、最小 URDF / STL 源模型和直接运行网格均进入 Git；详见 [资源清单](assets/README.md)。`data/recordings`、`data/datasets` 与 `.runtime` 中的数据和运行缓存不进入 Git。原始示范和离线方案保留源 UUID；B 点仍然不暂停、不自动命名任务。

## 可选：以后接入外部设备

`client.py` / `server.py` / `protocol.py` 保留供 Quest、Wuji glove 与脚踏板遥操使用，默认入口不会启动它们。配置 LAN token 后，服务端运行 `python bimanual_teleop/sim/run.py server --config ...`，引擎用 `docker/run-apptainer.sh --config /workspace/...`，本地运行 `python bimanual_teleop/sim/run.py client --host ... --config ...`。这一模式仍在服务端记录，控制输入来自外部设备；不能与本地独立模式共用同一个运行中的 overlay。

## 检查

```bash
python scripts/run_tests.py discover -s bimanual_teleop/sim/tests -v
python scripts/run_tests.py discover -s tests -q
```

停止当前 Isaac 后，才可运行 `docker/exec-apptainer.sh /isaac-sim/python.sh /workspace/bimanual-teleop/bimanual_teleop/sim/run.py validate_isaac` 或 `python bimanual_teleop/sim/run.py smoke`。前者检查真实物理，后者检查保留的远程桥接链路。配置的 240 / 120 / 30 Hz 是仿真时间频率，实际墙钟性能以目标设备实测为准。

检查模型完整性与 Git 跟踪：`python bimanual_teleop/sim/run.py resources --check-git`。重新生成模型使用仓库内的源文件：`python bimanual_teleop/sim/run.py assets`，无需原始聊天附件。
