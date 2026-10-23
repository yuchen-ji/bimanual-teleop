# 腕部相机手眼标定

## 准备

使用已有 4×5 非对称圆点板，同一直线相邻圆心距离为 100 mm，交错步长为 50 mm。板与机器人基座在整次采样中固定，相机安装保持不变。两只手腕可以分别采样。

先按[总览](../README.md)安装环境。关闭占用相机或机器人 UDP 4730 的程序。

## 采样：本地 USB 电脑

```bash
python camera_calibration/run.py wrist collect \
  --same-row-mm 100 --samples 20 \
  --output camera_calibration/data/wrist_capture
```

默认连接机器人 `192.168.1.190`，左右相机序列号分别为 `408322072104`、`408322070938`。可用 `--robot-ip`、`--left-serial`、`--right-serial` 覆盖；单侧用 `--side left` 或 `right`。无桌面时使用 `--no-preview`。

每侧至少 15 组，建议 20～30 组。改变位置、距离和俯仰／偏航／滚转，避免只平移或只绕一个轴转动；每次停稳 1～2 秒。完整显示 20 个圆点并留边，避免反光与极端斜视。

程序自动检查静止、反馈对应和重复姿态。按 Q、关闭窗口或 Ctrl+C 结束。上传整个输出目录，包含 `session.json`、`samples.jsonl` 和原始 PNG。原图不旋转；左腕 180° 录制方向在输出适配时转换。

手工拍摄可参考[关节记录模板](wrist-manual-template.csv)，需要整理成同样的采样会话格式后解算。

## 求解与验收：完整项目

```bash
python camera_calibration/run.py wrist solve \
  camera_calibration/data/wrist_capture \
  --output camera_calibration/data/wrist_result
```

查看 `result.json` 和 `*_heldout_*.png`。独立验证样本不参加手眼拟合；失败会以非零状态退出。报告中的 `T_flange_sensor_opencv` 是原始彩色光学坐标到法兰的变换，`T_world_board` 是板到项目世界坐标的变换。

## 显式应用

```bash
python camera_calibration/run.py wrist apply camera_calibration/data/wrist_result/result.json
```

只接受通过验证的实测结果，并检查机器人 URDF 哈希。应用后更新仿真腕部配置及来源记录，重启仿真生效。回退时将仿真配置的 `cameras.wrist_calibration_file` 设为 `null`。
