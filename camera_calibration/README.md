# 主视角相机标定

腕部相机请使用 [键盘位置控制、C 手动采样](docs/wrist.md)：腕部相机随末端运动，标定板固定在环境中；不自动判断停稳，预览实时显示圆点及顺序编号。

主视角（第三视角）相机固定不动，4×5 非对称圆点板刚性固定在所选机械臂末端。使用键盘移动机器人，停稳后按 `C` 记录一组原始图像和实际关节反馈。这是眼在手外标定；同时求解相机相对所选臂基座的外参、标定板相对法兰的安装位姿，无需事先测量板的安装位置。

已移除依赖腕部相机标定结果、两台相机同时观察标定板的旧主视角流程。旧 `external_camera_capture_v1` 数据不接受新求解，旧参数 `--wrist-result` 不再支持。

在项目根目录运行：

```bash
conda activate bimanual-teleop
PIP_USER=false pip install -e '.[calibration,realsense]'
python camera_calibration/run.py external collect --side left --output camera_calibration/data/main_left
```

默认使用 [configs/recording.yaml](../configs/recording.yaml) 中第一台相机（`camera_0`，当前为 D436 `405622076394`），只打开这一台相机。右臂安装板时改用 `--side right`。板的同一行相邻圆心距离默认 **100 mm**，实物尺寸不同时必须用 `--same-row-mm` 指定；不是相邻错位行之间的距离。

标定中的机器人移动使用控制器的位置控制模式（`position`）；键盘仍指定末端位姿，经受限运动规划转换为关节位置目标。其他遥操作和独立点动入口保持原来的笛卡尔阻抗控制模式。

启动前按终端提示确认，再按一次 `Enter` 接合机器人；出现“已接合所选机械臂”后，移动键才生效。暂停后再按 `Enter` 仍以位置模式恢复。

不会自动回零或移动到预备姿态；沿用 [天机配置](../configs/tianji_teleop.yaml) 的所选臂负载、速度和加速度参数。安装板和支架后，应使用包含它们的实际工具负载参数。请关闭占用机器人 SDK/UDP 4730 的厂家上位机和其他控制进程。

**所有按键在运行命令的终端输入。** 字母不区分大小写。预览实时显示圆点位置、顺序连线和 `0–19` 编号：0 号红色，其余绿色；检测失败时显示 `0/20`，不残留上一帧的标注。保存的 PNG 始终是未画标注的原始图像。

| 按键 | 动作 |
| --- | --- |
| `Enter` | 接合或从暂停恢复，基于当前实际位姿重新开始 |
| `W / S` | 原生基座 X 正 / 负向平移 |
| `A / D` | 原生基座 Y 正 / 负向平移 |
| `R / F` | 原生基座 Z 正 / 负向平移 |
| `I / K` | 绕原生基座 X 正 / 负向旋转 |
| `J / L` | 绕原生基座 Y 正 / 负向旋转 |
| `U / O` | 绕原生基座 Z 正 / 负向旋转 |
| `C` | 立即记录当前原始图像和最新实际关节反馈，由操作者确认已停稳 |
| `Space` | 暂停机器人 |
| `Q` / `Ctrl+C` / 关闭预览 | 退出并关闭控制与相机，保留已记录的数据 |

默认每次平移 5 mm、旋转 2°，平滑过渡时间 0.25 秒，可用 `--step-mm`、`--step-deg`、`--transition-s` 调整。左臂基座 +Y 向下，右臂基座 +Y 向上；X/Y/Z 使用各臂自身基座，不能当作统一世界坐标。移动复用现有笛卡尔点动、逆运动学、关节/耦合限位及看门狗。

每到一个新位姿，由操作者确认机器人停稳、整块板可见，再按 `C`。程序不做自动停稳判断、前后反馈时间窗、图像稳定性或重复位姿拦截；即使圆点未检测完整也会保存，离线求解时再检查图像。一次按键对应一组，不按键就不记录。

按键时即取出图像与实际关节值，检测和写盘不会改成后来才读到的位姿。图像或有效机器人反馈尚未到达时会明确提示。机器人的位置控制、限位、暂停和运动看门狗继续使用现有实现。

默认目标 25 组，可用 `--samples` 设置任意正整数；离线求解至少需要 15 组。至少绕两个不平行轴改变朝向，总相对转角达到 20°，建议覆盖 20–40°；同时变化位置与距离，让板覆盖图像不同区域。只做平移、只绕一个轴旋转、重复同一位姿都不能完成有效标定。板相对末端和主视角相机相对机器人基座全程不能松动。

`Q` 可提前结束并保留所有已记录数据，达到目标数量会结束。数量、朝向覆盖度和独立验证由离线求解判定。输出目录必须不存在，以防覆盖既有数据。没有桌面时使用 `--no-preview`，仍需交互终端。

```bash
python camera_calibration/run.py external solve camera_calibration/data/main_left --output camera_calibration/data/main_left_solve
python camera_calibration/run.py external export camera_calibration/data/main_left_solve/result.json --output camera_calibration/data/main_left_extrinsics.json
```

求解重读原始图像、校验图像哈希，并通过实际关节和采样时同一份天机模型重新计算法兰位姿。使用自定义 `--model` 时，采样和求解必须使用相同文件；两阶段均支持 `--sdk-root`。使用新输出目录可以在保留原始数据的同时重新求解。

每第五组（组号 0、5、10……）固定为独立验证组，不参与初始化、优化和训练异常剔除；最多剔除 20% 的训练组。报告包含验证重投影误差、位姿一致性、局部条件不确定度及验证叠加图。验证组损坏或超限会使标定失败；只有通过验证的实测结果可导出。默认阈值为验证像素 RMS 中位数 ≤1.5 px、最大值 ≤3 px，平移一致性中位数 ≤5 mm、最大值 ≤10 mm，旋转一致性最大值 ≤1°。这些数值不等于机器人绝对定位精度。

结果中的 `T_base_camera_opencv` 将相机坐标变换到所选臂原生基座，`T_flange_board` 将板坐标变换到法兰；平移单位为米。相机坐标为 OpenCV 光学坐标：X 向右、Y 向下、Z 向前。两者满足：

```text
T_base_camera_opencv @ T_camera_board = T_base_flange @ T_flange_board
```

基座坐标不能直接作为仿真世界坐标；需要世界坐标外参时，导出时使用 `--world-from-base 文件.json`，文件内容为明确测定的 `T_world_base` 4×4 矩阵。导出不会改写机器人配置或仿真配置。

算法坐标约定参考 [OpenCV 手眼标定文档](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html)。检查圆点编号可使用 `python camera_calibration/run.py detect 图像.png --output 检测图.png`。

无硬件测试：

```bash
python -m unittest discover -s camera_calibration/tests -v
```
