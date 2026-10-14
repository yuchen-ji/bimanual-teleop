# 双臂仿真：在 Remmina 中直接查看 Isaac

默认使用一个 Isaac 进程，不启动网络桥接或网页查看器。所有仿真文件保留在本目录，真机遥操与采集入口保持独立。

当前默认场景使用 `assets/source/rail_task/` 中复制的导轨任务 CAD：480 mm 长导轨、长导轨台、完整桌子、两侧各五层转运箱，以及两个腕部的力传感器、相机与支架。机器人面向桌子窄边，导轨默认在导轨台上，Reset 恢复同一位置。[模型与装配说明](docs/task-models.md)

## 日常使用

在 **Remmina 远程桌面内的终端**运行：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python -m bimanual_teleop.sim
```

直接打开已装配完整场景的 Isaac 原生窗口。默认只查看、计算物理和保持机器人待机，不连接外部设备，不录制三相机。首次加载需等待应用和模型准备完成；窗口出现后短暂黑屏不代表失败。

录制时加一个参数：

```bash
python -m bimanual_teleop.sim --record
```

启用三路 640×480 RGB，直接在同一 Isaac 进程中写 MP4、Zarr 和事件。原生窗口右侧的 **Simulation** 面板提供录制暂停 / 恢复、B 分段、保存、重置与恢复观察视角。按 **B** 记录无名称边界；“Save and stop recording”完成当前示范并停止录制，之后再点“Record / Pause recording”新建下一条。关闭窗口会保存收尾。

**默认动作源是保持待机，录制只用于检查图像和数据链路，标记为测试数据，正常学习导出会拒绝。** 不接外部设备也能用审核过的关节轨迹驱动：

```bash
python -m bimanual_teleop.sim --record \
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

`client.py` / `server.py` / `protocol.py` 保留供 Quest、Wuji glove 与脚踏板遥操使用，默认入口不会启动它们。配置 LAN token 后，服务端运行 `python -m bimanual_teleop.sim.server --config ...`，引擎用 `docker/run-apptainer.sh --config /workspace/...`，本地运行 `client --host ... --config ...`。这一模式仍在服务端记录，控制输入来自外部设备；不能与本地独立模式共用同一个运行中的 overlay。

## 检查

```bash
python -m unittest discover -s bimanual_teleop/sim/tests -v
python -m unittest discover -s tests -q
```

停止当前 Isaac 后，才可运行 `docker/exec-apptainer.sh /isaac-sim/python.sh -m bimanual_teleop.sim.validate_isaac` 或 `python -m bimanual_teleop.sim.smoke`。前者检查真实物理，后者检查保留的远程桥接链路。配置的 240 / 120 / 30 Hz 是仿真时间频率，实际墙钟性能以目标设备实测为准。

检查模型完整性与 Git 跟踪：`python -m bimanual_teleop.sim.resources --check-git`。重新生成模型使用仓库内的源文件：`python -m bimanual_teleop.sim.assets`，无需原始聊天附件。
