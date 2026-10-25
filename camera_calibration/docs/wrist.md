# 腕部相机：键盘位置控制、按键采样

腕部相机保持固定安装在机器人末端，标定板固定在环境中。使用与主相机标定相同的位置模式键盘控制所选机械臂，观察图像、人工确认停稳后按 `C` 保存。程序不自动判断是否停稳，也不拦截重复位姿或未完整检测到板的图像。

程序通过机器人 SDK 建立控制连接并读取实际关节。运行前断开 MarvinPlatform 上位机与机器人的连接，并退出其他机器人控制程序，避免本机 UDP 4730 端口或控制连接冲突。相机通过 USB 接入运行程序的 Linux 电脑。直接使用 Conda 环境，无需 `sudo` 或原始套接字权限。

左腕采样：

```bash
conda activate bimanual-teleop
python /home/yuchen/project/bimanual-teleop/camera_calibration/run.py wrist collect --side left
```

右腕将 `--side left` 改为 `--side right`。每次只控制并采集所选侧，左右臂分别运行，不支持 `--side both`。默认 25 组，可用 `--samples` 修改，输出目录自动按时间命名并在结束时打印。默认序列号取自 [相机配置](../../configs/recording.yaml)：左腕 `408322072104`，右腕 `408322070938`，也可用 `--left-serial` / `--right-serial` 覆盖。

- 启动后先按回车确认，程序不会自动回位；再按 `Enter` 接合所选机械臂。
- `Enter`：接合/恢复；`Space`：暂停；`Q` / `Ctrl+C` / 关闭窗口：停止键盘控制并退出，保留已采集数据。
- `W/S`、`A/D`、`R/F`：沿所选臂原生基座 X/Y/Z 正负方向平移。
- `I/K`、`J/L`、`U/O`：绕所选臂原生基座 X/Y/Z 正负方向旋转。
- `C`：立即保存当前原图和最新实际关节反馈。

终端和预览窗口均支持以上按键。默认每键平移 5 mm、旋转 2°、平滑时间 0.25 s，可用 `--step-mm`、`--step-deg`、`--transition-s` 修改。

预览实时画出检测到的 20 个圆点、顺序连线和 `0–19` 编号，0 号红色，其余绿色。左腕预览按现有安装方向旋转 180°，文字在旋转后绘制，便于直接阅读。保存的 PNG 始终是传感器原图，不包含标注或旋转；元数据记录预览方向。

一次 `C` 对应一组；不按键就不保存。程序只要求当前图像和实际反馈可用，不等待前后时间窗、不检查速度或低速标志、不检查位姿变化量，也不自动补拍。图像质量和姿态覆盖度由离线求解检查。同一行相邻圆心距离默认 100 mm，实物不同时用 `--same-row-mm` 指定。

每侧至少采集 15 组，建议 25 组，并变化位置、距离及两个以上轴的朝向；整块标定板和机器人基座在整个采样过程中保持固定。

采样文件包括 `session.json`、`samples.jsonl` 和所选侧 `left/` 或 `right/` 下的原始 PNG。关节值来自 SDK 实际反馈 `m_FB_Joint_Pos`，保存为弧度，不读取运动目标作为实际位姿。控制与键盘读取独立于图像检测和文件保存，元数据记录 `robot_control: keyboard`、`robot_control_mode: position`。

在 `bimanual-teleop` 环境离线求解：

```bash
python /home/yuchen/project/bimanual-teleop/camera_calibration/run.py wrist solve \
  "采样结束时打印的完整目录" --output "新的求解输出目录"
```

结果为 `result.json`、`report.md` 和独立验证叠加图。`T_flange_sensor_opencv` 将原始相机光学坐标变换到所选臂法兰；`T_flange_output_opencv` 对应预览/录制图像方向。平移单位为米，满足：

```text
T_base_flange @ T_flange_sensor_opencv @ T_sensor_board = T_base_board
```

求解使用与采样相同的天机运动学模型和实际关节，固定每第五组为独立验证，不需要仿真模块。`passed: true` 表示通过验证；通过的实测结果可用 `wrist export 结果.json --output 新外参.json` 导出。采样端没有自动停稳或采样筛选。

单张圆点检测成功不代表图像与关节位姿整体一致。联合拟合超限时，报告会保留候选外参、每组的单张 PnP 误差、联合重投影误差和位姿差，并列明超限组号；结果仍标为 `passed: false`，不能导出。无需先重采即可查看 `report.md` 和验证叠加图，区分检测失败、重复姿态、训练一致性不足和独立验证超限。失败输出目录保留，重新求解时指定新的 `--output`。
