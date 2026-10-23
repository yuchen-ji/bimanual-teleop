# 主视角（第三视角）相机标定

使用已标定腕部相机作为参考，固定腕部手眼外参和两台相机内参，联合求解第三视角外参与各组板位姿。

这里的第三视角就是现有主视角 `camera_0`，型号为 RealSense D436；命令名称沿用 `external`。

## 准备

- 主视角 D436 相对机器人基座固定，当前配置序列号为 `348122072137`。
- 选一台已标定 D436 腕部相机，保持其安装不变。
- 使用与腕部标定一致的 4×5 非对称圆点板、实际尺寸和编号，两台相机观察同一面。
- 使用 `wrist solve` 的 `result.json`，或现有 `bimanual_teleop/sim/configs/wrist_camera_calibration_reference.json`。独立采样包中对应文件为 `wrist-result.json`。

## 采样：本地 USB 电脑

```bash
python camera_calibration/run.py external collect \
  --wrist-result bimanual_teleop/sim/configs/wrist_camera_calibration_reference.json \
  --side left --external-serial 348122072137 --samples 25 \
  --output camera_calibration/data/external_capture
```

每组停稳 1～2 秒，板和手腕可以在组间移动，第三视角保持固定。也可以先固定机械臂，仅移动板。覆盖工作区的近／远、中心／四周和两个不同轴的约 15～40° 倾角，完整显示圆点。适当靠近可提高圆点像素尺寸；避免模糊、反光和过大的斜视角。

第三视角默认使用 640×480。设备支持时可加 `--external-resolution 1280 720` 提高圆点像素尺寸，程序读取该分辨率对应的内参；腕部维持原标定分辨率。高分辨率结果可通用导出，仿真覆盖导出要求与当前仿真分辨率一致。

默认保存至少 25 组不同观测，只有角度覆盖合格才自动结束。按 Q 或 Ctrl+C 可提前结束；至少 15 组且覆盖合格才标记采样完成。状态提示会说明停稳、同步、可见性或重复姿态未通过的原因。

`--robot-ip` 可修改控制器地址，`--no-preview` 用于无桌面采样。采样只读取反馈。上传整个采样目录及本次使用的腕部标定文件；求解时检查它们的哈希一致。

时间对应使用相机 GLOBAL_TIME 和主机反馈时间，允许的双相机／关节对应误差均不超过 25 ms；它不是硬件触发同步，停稳检查用于减小延迟影响。镜头模型在采样开始前进行原生 SDK 与 OpenCV 的映射检查，超过 0.1 px 时拒绝继续。

## 求解与验收：完整项目

```bash
python camera_calibration/run.py external solve \
  camera_calibration/data/external_capture \
  --wrist-result bimanual_teleop/sim/configs/wrist_camera_calibration_reference.json \
  --output camera_calibration/data/external_result
```

初值采用 `T_world_C = T_world_F × T_F_W × T_W_M × inverse(T_C_M)`，其中 W 为腕部相机、C 为第三视角、M 为板。求解使用双相机像素残差和稳健损失；训练观测剔除超过 20% 时失败。

每五组预留一组验证，编号在检测前确定。验证时仅用腕部图像恢复板位姿，再预测第三视角图像；第三视角验证图像不参与优化或板位姿拟合。

| 独立验证指标 | 默认门槛 |
| --- | --- |
| 重投影 RMS，中位数／最大值 | 1.5 / 3 px |
| 外参位置一致性，中位数／最大值 | 5 / 10 mm |
| 外参角度一致性，最大值 | 1° |

查看 `report.md`、`result.json` 和 `validation_*.png`：绿色圆是检测位置，紫色叉是腕部预测位置。JSON 包含优化前后逐组误差、拒绝原因和局部条件不确定度。后者固定了腕部外参、正运动学和内参，不包含它们的系统误差，也不代表绝对定位精度。

门槛集中定义在 `external.Limits`。可用 `--limits limits.json` 覆盖，例如 `{"validation_max_px": 2.0}`；报告会记录最终门槛。应根据任务要求收紧，避免通过放宽门槛掩盖问题。

失败先检查板尺寸、图像方向、设备序列号和支架，再增加有差异的姿态。持续偏差可能来自腕部手眼外参、机器人模型或内参，需要独立测量定位原因。

## 导出

```bash
python camera_calibration/run.py external export \
  camera_calibration/data/external_result/result.json \
  --output camera_calibration/data/external_camera.yaml
```

只导出通过独立验证的实测结果。文件保存相机内参、原始 OpenCV 光学坐标下的 `T_world_camera_opencv`、`T_base_camera_opencv`、基座到世界的变换和来源。基座使用所选侧的 `RobotModel.base_pose(side)` 约定；左右臂各自基座和项目世界坐标不可混用。输出文件必须尚不存在。

可选地导出仿真覆盖文件并显式加载：

```bash
python camera_calibration/run.py external export \
  camera_calibration/data/external_result/result.json \
  --simulation --output camera_calibration/data/head_camera.yaml
python bimanual_teleop/sim/run.py --config camera_calibration/data/head_camera.yaml
```

导出不改默认配置，也不移动相机 CAD。新第三视角的真实精度仍需用未参与采样的实体目标验收。
