# 从零搭建与日常操作：原生 Isaac 简化版

更新：2026-10-14。采用当前已验证的 Linux x86_64、官方 Isaac Sim 5.1.0 容器、Apptainer 1.5.4。使用完整项目代码复现环境，不需要从头编写仿真器。

## 1. 第一次在新设备安装

服务器需要可用的 NVIDIA RTX 驱动和图形桌面；`nvidia-smi` 应正确显示 GPU。Ubuntu 22.04 / 24.04、32 GB 以上内存、16 GB 以上显存、至少 50 GB 空闲空间作为 Isaac 5.1 官方基线；数据另留空间。[官方要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

安装 Conda，例如按 [Miniforge 官方说明](https://github.com/conda-forge/miniforge) 安装到用户目录。系统需要 curl、bsdtar、squashfs 与 FUSE 工具；在有管理权限的新 Ubuntu 上：

```bash
sudo apt update
sudo apt install -y curl ca-certificates libarchive-tools squashfs-tools e2fsprogs util-linux fuse3
```

把**完整项目**复制到新设备，例如 `$HOME/projects/bimanual-teleop`。截至本次整理，sim 目录仍未提交，不能假定克隆旧仓库已有本次代码；独立复制 sim 也缺少它复用的项目模块。镜像 / 数据被忽略 Git，迁移时另行保留。

在新项目根目录执行：

```bash
cd "$HOME/projects/bimanual-teleop"
PIP_USER=false conda env create -f environment.yml
conda activate bimanual-teleop
PIP_USER=false python -m pip install -r bimanual_teleop/sim/requirements-assets.txt
python -m bimanual_teleop.sim.doctor
```

主机 Python 始终使用 `bimanual-teleop` Conda；Isaac 使用容器自带 Python 3.11。不要使用系统 Python、pip --user 或 sudo pip 安装项目依赖。

将四个原始附件放在 `bimanual_teleop/sim/assets/incoming/`，明确指定路径生成模型：

```bash
python -m bimanual_teleop.sim.assets \
  --robot bimanual_teleop/sim/assets/incoming/Pro-m6-696-夹爪.rar \
  --hands bimanual_teleop/sim/assets/incoming/hand2_beta2.zip \
  --fixture bimanual_teleop/sim/assets/incoming/导轨台2.step \
  --camera bimanual_teleop/sim/assets/incoming/D435i_Solid.SLDPRT
```

成功时生成 `assets/generated/robot.urdf`、`fixture.obj`、`rail/rail.obj` 与 `manifest.json`，报告 70 个保持通孔空心的凸碰撞体。原附件不会覆盖；默认附件路径只适用于旧服务器，换设备用以上明确路径。

准备运行时：

```bash
bimanual_teleop/sim/docker/bootstrap-apptainer.sh
```

它安装用户目录 Apptainer，下载固定 5.1.0 镜像和容器依赖，创建 4 GiB overlay。下载约 7 GB，临时文件还需额外空间。脚本接受 NVIDIA 镜像许可。新系统需允许用户命名空间 / FUSE。[容器说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)

如果已有旧镜像而仅更新本次代码，先关闭 Isaac，再运行一次 `bimanual_teleop/sim/docker/install-runtime-deps.sh`，补齐容器本地记录依赖。本机已完成。容器安装临时文件写到 `.runtime/tmp`，不使用私有的 64 MiB /tmp。

## 2. 每次只需一个启动命令

连接 Remmina，**在远程桌面里打开终端**，执行：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python -m bimanual_teleop.sim
```

换设备时改项目位置。程序直接打开完整 Isaac 原生场景，不需要先开 server，不需要浏览器或设备客户端。等待启动和模型准备；可见窗口比场景就绪更早，短暂黑屏不要立刻当作失败。

程序持续运行时终端不返回提示符。退出用正常关闭 Isaac 窗口，或回到启动终端 Ctrl+C。重新启动前等待旧容器退出；同一 overlay 不同时运行多个实例。

从 SSH / 编程工具启动而没有 DISPLAY 时，可用自己的桌面编号：

```bash
python -m bimanual_teleop.sim --display :10
```

`:10` 是当前用户本机 XRDP 会话，换电脑以 Remmina 终端 `echo "$DISPLAY"` 为准。启动器传递当前用户的 X11 授权，不修改 xhost 权限。

## 3. 观察场景与基础操作

Isaac 左侧为视口、右侧为 Stage / Property / Simulation。可查看机器人、桌面、料箱和导轨台的层级与属性。鼠标在视口中操作：Alt+左键拖动环绕，中键拖动平移，滚轮缩放；选中物体后 F 聚焦。[官方快捷键](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/gui/reference_keyboard_shortcuts.html)

Simulation 面板中的“Frame entire scene”恢复全景。左侧时间线按钮暂停 / 继续物理；它与“Pause recording”不同，后者只暂停记录。重置使用 Simulation 的“Reset scene”；若物理已暂停，先恢复 Play。

默认机器人保持待机，导轨受重力落稳，没有自动执行完整取放任务。桌面视口是观察画面，不是保存的学习相机。当前场景由代码创建，界面临时编辑不会自动改写源码配置。

检查文件 `sim/data/local/status.json`：state=running、gui=true、54 个 q 值、tick 持续增加，说明独立物理在运行。窗口不出图时同时检查此文件和启动终端，区分未加载完成、程序异常与显示问题。

## 4. 不接设备，怎样动作与录制

查看无需任何参数。录制加 `--record`：

```bash
python -m bimanual_teleop.sim --record
```

三路 RGB 和原始状态直接在同一个 Isaac 进程里写盘。Simulation 面板：

| 操作 | 行为 |
| --- | --- |
| Record / Pause recording | 暂停 / 续录；保存后再点它，新建下一条 |
| B 键或 B 按钮 | 在下一张成功写入的图像处记无名称边界，不暂停；750 ms 内重复抑制 |
| Save and stop recording | 结束并保存当前条目，停止记录，物理继续 |
| Reset scene | 保存当前条目，停止记录，重置机器人与导轨 |
| 关闭窗口 | 保存并退出 |

只有 `--record` 才启用三路图像渲染产品；查看模式中的录制按钮会提示重启时添加参数。当前保持待机的示范标记为 synthetic，正常学习导出拒绝。

有自己的关节轨迹时：

```bash
python -m bimanual_teleop.sim --record \
  --trajectory bimanual_teleop/sim/data/my_trajectory.json
```

JSON 结构为 `joint_names`（与 RobotModel.names 或已录制 episode.json 的 metadata.joint_names 一致）和 `waypoints` 列表；每个点有 `time_s`、`positions_rad`（54 个数）。至少两点，时间从 0 严格递增，起点等于待机姿态；线性插值，并检查位置 / 速度限制。动作按仿真时间执行。调试轨迹增加 `test_data: true`，其记录默认禁止正常学习导出。轨迹文件只提供关节目标，不会自动解决碰撞、双手抓持、孔位对齐和任务成功。

命令支持 `--headless --steps 480` 做定长自动检查；`--codec libx264` 作为 NVENC 不可用时的 CPU 编码选项。正常退出后检查 status.json 的 last_completed 或原 episode.json 为 complete；失败条目不当作学习示范。

## 5. 原始数据与离线子任务

```text
sim/data/recordings/日期_local_标识/episode_000000/
  camera_0.mp4        头部
  camera_1.mp4        左腕
  camera_2.mp4        右腕
  raw.zarr/          states 和与图像逐帧对应的 frames
  events.jsonl       B 点及暂停 / 恢复
  episode.json       状态、源 UUID、配置、模型哈希
```

用真实 SESSION 替换以下占位符。先生成方案，人工填写所选范围的 task；不需要子任务时可直接 `--mode full`：

```bash
python -m bimanual_teleop.sim.segments \
  --input bimanual_teleop/sim/data/recordings/SESSION \
  --output bimanual_teleop/sim/data/schemes/scheme_01.yaml
python -m bimanual_teleop.sim.export \
  --input bimanual_teleop/sim/data/recordings/SESSION \
  --output bimanual_teleop/sim/data/datasets/both_01 \
  --mode both --scheme bimanual_teleop/sim/data/schemes/scheme_01.yaml
```

方案范围为从 0 开始的图像帧索引 `[start,end)`，不是 tick / 秒；保留源 UUID。可合并、移动或删除范围；选中的 task 不得为 null。导出不覆盖已有目录，改方案后用新输出名。整任务窗口能跨 B，子任务按所选范围切开，真实暂停 / 缺帧都切开。训练 / 验证统一按原始示范 UUID 划分。

三相机与状态按物理 tick 同步。默认物理 240 Hz、控制 120 Hz、图像 30 Hz 均为仿真时间频率；视频播放长度不等于墙钟录制时长。六维数据为固定手安装关节反力，含重力 / 惯性；深度、实际关节扭矩、手指触觉尚未实现。

## 6. 改环境与首次验收

复制 `configs/default.yaml` 或写一份仅覆盖参数的 `configs/my_scene.yaml`，使用 `--config` 启动。可以调整当前场景的桌面、箱子、导轨台布局；导轨几何、手安装角改变后，用同一配置重新运行 assets。所有参数位置在 [架构说明](architecture.md)。

当前 engine / task 仍含导轨与 54 关节假设；增加新物体、改成新任务时需同步改场景、状态读取、动作与成功条件。仅改 YAML 名称不是新任务实现。任意 USD 任务加载、策略闭环和多环境训练没有现成入口。

首次换设备先做软件检查，再顺序做物理与采集链路检查；以下容器验证不能与日常 Isaac 同时运行：

```bash
python -m unittest discover -s bimanual_teleop/sim/tests -v
bimanual_teleop/sim/docker/exec-apptainer.sh \
  /isaac-sim/python.sh -m bimanual_teleop.sim.validate_isaac
```

以 `data/isaac-validation/report.json` 的 passed=true 和日志共同判断，不能仅凭退出码。外部设备桥接另有 `python -m bimanual_teleop.sim.smoke` 检查。保留新设备报告，旧报告不替代新设备验收。

## 7. 常见问题与迁移清单

| 问题 | 检查与处理 |
| --- | --- |
| DISPLAY missing / X11 授权失败 | 在 Remmina 桌面终端运行；确认自己的 DISPLAY 和 XAUTHORITY |
| overlay 占用 | 关闭本人旧 Isaac / 验证进程，等退出；不删除占用中的 overlay |
| 窗口先黑屏 | 等应用和模型准备，查看终端与 status.json；不能只凭 NGX 报错判断原因 |
| 模型缺失 | 四个附件路径是否正确，assets 是否成功；generated 忽略 Git |
| NVENC 不可用 | 用 `--codec libx264`，检查 episode 保存结果 |
| B 无效 | 加了 --record、正在录制、终端 / 视口焦点正确；等待下一张成功图像 |
| 导出拒绝 | complete 状态、真实动作来源、方案 task / UUID / frame_count 与输出新目录 |

迁移保留完整源码、四个原始附件、自用配置、原始 recordings、分段方案和源 UUID 划分。SIF 可复制减少下载，overlay 建议在新设备重建；复制它之前必须关闭容器。`.runtime` 与 `data` 不在 Git。原有 Quest / 手套 / 脚踏板分支保留，准备接设备时再按 `client/server` 的参数与 SDK 文档联调。
