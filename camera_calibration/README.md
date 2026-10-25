# 相机标定

独立于遥操作和仿真运行的标定工具。支持腕部手眼标定，以及通过已标定腕部相机求固定第三视角相机外参。

第三视角就是现有的主视角相机 `camera_0`，型号为 RealSense D436，配置序列号为 `348122072137`。命令中的 `external` 指这台主视角相机。

## 安装与入口

在 `bimanual-teleop` 项目根目录执行：

```bash
conda activate bimanual-teleop
PIP_USER=false pip install -e '.[calibration,realsense]'
python camera_calibration/run.py --help
```

| 操作 | 说明 |
| --- | --- |
| [腕部标定](docs/wrist.md) | 固定板、移动手腕，求相机到法兰的外参 |
| [主视角（第三视角）标定](docs/external.md) | 腕部相机与主视角 D436 观察同一块板，求主视角外参 |
| `python camera_calibration/run.py detect 图像 --output 检测图.png` | 检查板检测及圆点编号 |
| `python camera_calibration/bundle.py --output camera_calibration/data/local.zip` | 生成独立本地采样包 |

采样仅在 USB 连接相机的 Linux 电脑进行，读取原图与实际关节反馈；机器人姿态由操作者通过独立示教器调整，停稳后自动保存，不用按拍照键。程序不会使能机械臂或开启拖动模式，不能与本项目 Quest／点动程序并行使用。启动后的逐步操作见[腕部操作说明](docs/wrist.md#启动后怎样操作机械臂)。服务器执行离线求解与导出。

本地没有完整项目时，解压独立采样包，在包根目录创建／更新其中的 `environment.yml` 环境，再执行相同采样命令。包内 `wrist-result.json` 是打包时的腕部标定快照；上传样本时一并保留。离线求解在完整项目中执行。

## 结构

- `board.py`、`geometry.py`：圆点检测、镜头模型、PnP 和坐标变换。
- `wrist.py`、`external.py`：数值求解与独立验证。
- `collect.py`、`external_collect.py`、`streams.py`：本地只读采样。
- `workflow.py`、`external_workflow.py`：文件输入、报告与导出。
- `robot.py`、`simulation.py`：正运动学及显式仿真配置适配。
- `tests/`：数值、失败检测和设备模拟测试。

`T_A_B` 将 B 坐标转换到 A 坐标，平移单位为米。相机使用 OpenCV 光学坐标：X 右、Y 下、Z 前。项目世界坐标与单臂基座坐标分别保存。

新数据放在被 Git 忽略的 `camera_calibration/data/`。旧腕部数据继续保留在原目录。标定统一通过 `python camera_calibration/run.py ...` 运行，打包通过 `python camera_calibration/bundle.py ...` 运行。

## 参考与测试

[已有腕部实测结果](docs/wrist-results.md) · [历史头部视觉对齐](docs/head-alignment.md)

以下测试在完整项目中执行：

```bash
python scripts/run_tests.py discover -s camera_calibration/tests -v
python scripts/run_tests.py bimanual_teleop.sim.tests.test_camera_calibration -v
```

验证像素误差、位姿重复一致性和条件不确定度均不等于机器人绝对定位精度；真实精度需要未参与标定的实拍／实体测量验收。
