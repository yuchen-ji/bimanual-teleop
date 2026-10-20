# 腕部相机实测标定与仿真视角对齐

本次使用 Ubuntu 22.04 本地电脑连接两台腕部相机，用户通过示教器或拖动模式调整机械臂。本地只读取相机和实测关节，服务器离线解算并更新仿真。2026-10-20 已收到并应用实测样本，数值结果见 [腕部标定结果](wrist-cameras.md)。本文保留重新采样和复现步骤。

## 标定板与准备

照片中的标定板为 20 点非对称圆点阵。OpenCV 排列为 4 列 × 5 行；用户实测同一直线相邻圆心距离为 100 mm，交错行步长为 50 mm。程序已用用户照片检查完整检测和旋转后的编号一致性。照片可用于确认板型，手眼标定需要腕部相机本身拍摄的图像。

1. 将板固定在两只手腕能够看到的位置，整次采样中保持板和机器人基座不动。板的世界坐标无需事先测量。板若移动，重新采一组；左右腕可以分别采样，但每组内必须固定。
2. 保持相机、支架和手的安装不变。擦净圆点区域，避开反光；画面完整显示 20 个圆点并留边。
3. 用示教器或拖动模式调整姿态。每只腕部相机采约 20 个不同姿态，至少 15 个有效姿态；改变观察位置，也改变俯仰、偏航和滚转，避免只平移或只绕一个轴旋转。若操作空间允许，在多个方向改变约 20–40°，并让标定板覆盖画面中心与四周。
4. 每个姿态停稳 1–2 秒，手离开画面，保持圆点完整可见。左右臂可以轮流操作，已经采满的一侧会停止增加样本。不要用移动标定板来替代手腕运动。

## 独立采样包：本地没有项目时

独立包包含当前采样程序、已有天机 SDK 适配层、校验清单及官方 SDK 的必要文件与许可证，无需克隆或安装整个项目。生成包的代码在 `wrist_calibration/bundle.py`，输出到被 Git 忽略的 `data/wrist-handeye/`。包只提供本地 collect/detect；solve/apply 在服务器完整项目内运行。

在**接 USB 相机的本地电脑终端**执行以下命令。`8.135.8.154:6022` 是服务器地址，SSH 登录到服务器后不能访问本地 USB。

```bash
mkdir -p ~/wrist-handeye
cd ~/wrist-handeye
scp -P 6022 jiyuchen@8.135.8.154:/home/jiyuchen/project/WYS/bimanual-teleop/bimanual_teleop/sim/data/wrist-handeye/wrist-calibration-local.zip .
unzip wrist-calibration-local.zip
cd wrist-calibration-local
```

本地需有 Conda。尚无 `bimanual-teleop` 环境时创建最小环境：

```bash
conda env create -f environment.yml
conda activate bimanual-teleop
```

已有同名环境时使用 `conda env update -n bimanual-teleop -f environment.yml`，然后激活。依赖只安装到这个 Conda 环境。采样包不需要 Isaac、Wuji、Quest 或机器人运动入口。

关闭占用腕部相机的 RealSense Viewer，以及使用天机 SDK 的其他本地遥操程序，再运行：

```bash
python -m bimanual_teleop.sim.wrist_calibration collect \
  --same-row-mm 100 --samples 20 --output capture_real
```

默认机器人 IP 为 `192.168.1.190`；实际地址不同时加 `--robot-ip 地址`。这需要本地电脑能访问机器人局域网，与连接服务器的公网 SSH 地址无关。

默认左相机序列号 `408322072104`，右相机 `408322070938`。如果更换了相机，使用 `--left-serial` / `--right-serial` 指定；程序会列出未匹配的实际设备，不会任意选一台代替。只采一侧可加 `--side left` 或 `--side right`。每次使用一个新的输出目录，避免混入上一组移动过标定板的样本。

窗口显示左右腕图像、圆点检测和有效样本计数。左侧预览按原采集约定旋转 180°，保存的 PNG 保留传感器原图；右侧不旋转。停稳、完整检测、重投影合格、关节姿态与已存姿态有差异时自动保存。按 Q、关闭窗口或 Ctrl+C 结束并保留已采数据。没有本地桌面时可加 `--no-preview`，终端会显示计数及未通过原因。

本地采样完成后，在同一个本地目录执行：

```bash
scp -P 6022 -r capture_real \
  jiyuchen@8.135.8.154:/home/jiyuchen/project/WYS/bimanual-teleop/bimanual_teleop/sim/data/wrist-handeye/
```

目录包含 `session.json`（板型、相机内参/序列号、完成状态）、`samples.jsonl`（图像与实测关节对应关系）以及 `left/`、`right/` 下的原始 PNG。也可把整个目录压缩后交付，不要只传预览截图。

## 手工离线拍摄时需要交付什么

也可以用 RealSense Viewer 等工具保存腕部彩色原图，并在每个停稳姿态记录示教器的**实际关节反馈 J1–J7**。一个图像对应一行记录，左右臂分别约 20 组：

```text
capture_manual/
  left/0001.png ...
  right/0001.png ...
  joints.csv
  camera_info/          # 内参、分辨率、序列号；也可保存设备标定 JSON / bag
```

```csv
side,image,j1,j2,j3,j4,j5,j6,j7,unit
left,left/0001.png,实际J1,实际J2,实际J3,实际J4,实际J5,实际J6,实际J7,deg
right,right/0001.png,实际J1,实际J2,实际J3,实际J4,实际J5,实际J6,实际J7,deg
```

每张原图与对应关节读数都必须在同一停稳姿态下取得；用相同编号保存示教器照片也可以，但需要清楚显示七个带正负号的实测关节角及单位。记录图像是否旋转或镜像。保留 640×480 原始 RGB、避免裁剪、缩放、手机转拍和聊天压缩。若使用其他分辨率，必须提供对应内参；现有仿真按 640×480 运行。

若仍是原来的两台 D436、原分辨率且没有更改镜头标定，可结合已有 `rail_task_episode` 内参核对，但需要确认序列号和图像方向。只有普通视频、没有对应关节角时，无法确定相机相对于法兰的安装变换。手工样本需另外整理核对时间对应关系，不能直接冒充自动采样程序的同步记录。

## 本地只读范围与同步检查

采样工具复用 `ControlSDK`，机器人侧只调用 `open`、`poll_feedback`、`close`；不调用回位、伺服使能、状态切换、拖动模式、阻抗或运动目标接口。示教器／拖动模式由用户操作。SDK 使用 UDP 4730，工具检测端口占用，避免与另一遥操进程争用。SDK 建连仍需要机器人网络可达；服务器不打开真机驱动。

相机使用 RealSense GLOBAL_TIME，映射到本地单调时钟，与实测反馈匹配。每帧要求前后约 200 ms 的反馈窗口、足够的独立反馈包、关节变化不超过 0.002 rad、反馈间隙不超过 40 ms、最近图像／关节观测时间差不超过 25 ms。采样采用停稳姿态，减少传输延迟影响；观测时间不是控制器硬件触发同步时间。检测到本地时钟跳变则终止本组，保留原因。

## 服务器解算与应用

下列命令在服务器完整项目根目录、`bimanual-teleop` Conda 环境执行。安装额外离线依赖只影响仿真标定工具：

```bash
PIP_USER=false python -m pip install -r bimanual_teleop/sim/requirements-calibration.txt
python -m bimanual_teleop.sim.wrist_calibration solve \
  bimanual_teleop/sim/data/wrist-handeye/capture_real \
  --output bimanual_teleop/sim/data/wrist-handeye/calibrated_real
```

以实测关节进行当前仿真 URDF 正运动学，得到法兰位姿，估计板到相机的 PnP 位姿，再拟合“法兰 → 彩色相机”。固定板的世界坐标同时估计，无需将真实桌子或板的位置硬套到仿真。使用源图像、SDK 模型一致的镜头反投影，并检查 RealSense 与 Isaac/OpenCV 畸变映射误差小于 0.1 像素。

20% 的样本留作独立验证，不参与拟合或异常点剔除。工程验收门槛包括：单帧 PnP RMS ≤1 像素、充分的非平行旋转、验证集最大固定板偏差 <10 mm / 2°、验证集平移中位数 <5 mm、验证集重投影最大 RMS <4 像素且中位数 <2 像素。训练异常点过多、验证失败或样本不足会拒绝应用。这些是本任务检查阈值，实际标定精度仍需结合未参与采样的实拍与仿真画面对比。

查看 `calibrated_real/result.json` 以及 `left_heldout_*.png` / `right_heldout_*.png`，绿色圆为检测圆心，紫色叉为预测投影。通过后应用：

```bash
python -m bimanual_teleop.sim.wrist_calibration apply \
  bimanual_teleop/sim/data/wrist-handeye/calibrated_real/result.json
python -m bimanual_teleop.sim
```

应用只写仿真目录：生成 `configs/wrist_camera_calibrated.yaml` 和来源／质量记录，并将 `configs/default.yaml` 的 `cameras.wrist_calibration_file` 指向结果。会检查 URDF 哈希，防止把旧机器人模型的标定直接套到新模型。相机实体 CAD 继续保留当前装配位置，渲染光学位姿由实测结果覆盖临时值。左图额外处理 180° 旋转对应的光轴滚转、主点反射和切向畸变符号，再转换 OpenCV → USD 坐标。

重启 Isaac 后切换 `sim_camera_1` / `sim_camera_2` 查看。后续还需将机械臂与手指置于对照姿态，核对手掌、手指及标定板在画面中的位置。若实测光心与简化 CAD 相机外壳相交，应修正模型的几何表达，不应偏移实测光心来迎合 CAD。

回退到临时 CAD 光学设置：将 `cameras.wrist_calibration_file` 改回 `null`，重启仿真。真机遥操入口、配置及录制格式均不改动。

## 依据与软件检查

- [OpenCV 非对称圆点阵与标定板](https://docs.opencv.org/doc/doxygen/html/da/d0d/tutorial_camera_calibration_pattern.html)
- [OpenCV PnP 和手眼标定接口](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Isaac Sim 5.1 OpenCV 镜头模型](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html#opencv-pinhole)

`tests/test_wrist_handeye.py` 检查板型／尺度与旋转编号、PnP、已知手眼变换恢复、噪声和训练异常点、独立验证失败、退化运动、图像旋转／USD 坐标转换、只读 SDK 调用及运动／同步拒绝。另用合成圆点图像、URDF 关节、离线完整命令链检查解算。合成数据只验证软件，不能作为实测标定应用。
