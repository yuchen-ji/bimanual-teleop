# Tianji 记录导出为 LeRobot v3.0

`scripts/convert_lerobot.py` 将已整理的 `raw.zarr` 与三路 MP4 转为 LeRobotDataset v3.0。原始目录保持原样，结果发布到输入目录外的新目录。转换成功前只写隐藏临时目录；出现数据或编码错误时清理临时文件，不发布半成品，也不覆盖已有结果。

转换只使用数据文件，无需连接机器人、相机、手套或 Quest。脚本不自动上传 Hugging Face。

## 安装和运行

在项目的 `bimanual-teleop` Conda 环境安装导出依赖：

```bash
PIP_USER=false conda run -n bimanual-teleop python -m pip install -e '.[recording,lerobot-export]'
```

导出依赖是 Datasets、Arrow 和现有录制库；不要求安装 LeRobot 的完整训练、硬件或可视化依赖。输出遵循官方 [v3.0 文件布局](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)，可由 `LeRobotDataset` 读取。

先检查对齐、修复和丢弃情况：

```bash
conda run -n bimanual-teleop python scripts/convert_lerobot.py \
  --input recordings/<session> --action-space joint --repo-id local/tianji \
  --dry-run --report /tmp/lerobot-preview.json
```

正式导出：

```bash
conda run -n bimanual-teleop python scripts/convert_lerobot.py \
  --input recordings/<session> --output datasets/<session>_lerobot \
  --action-space joint --repo-id local/tianji
```

任务句子来自每个 session 的 `dataset.json`；旧数据没有该文件时，通过 `--task "Pick and place the car seat rails"` 指定。显式 `--task` 覆盖所有输入条目的任务。输入可以是一个 session、多个 session 的共同父目录或单个 episode。源 manifest 存在时核对相机序列号与 episode 快照一致。

| 参数 | 行为 |
| --- | --- |
| `--action-space joint` | 54 维已提交关节目标，推荐 Tianji 当前数据使用 |
| `--action-space eef` | 52 维已提交末端目标与手关节目标 |
| `--fps N` | 覆盖转换配置中的整数频率；默认 30 Hz |
| `--conversion-config PATH` | 使用现有转换配置；要求 `mode: repair` |
| `--quality-config PATH` | 沿用现有质量等级阈值 |
| `--no-raw` | 不复制原始 Zarr，此时结果没有原始深度数据 |
| `--max-frames-per-shard N` | 每个数据／视频分片的目标最大帧数，默认 10,000；单条连续 episode 保持完整，可超过该目标 |
| `--encoder-threads N` | 每路 H.264 编码线程数，默认 2；RGB 使用 CRF 18、GOP 2、无 B 帧 |
| `--allow-mixed-metadata` | 明确允许合并标定、模型或控制参数不一致的原始数据 |
| `--dry-run` | 与正式转换使用相同的有效帧规划，不解码视频、不写数据集 |
| `--report PATH`、`--verbose` | 另存报告、打印每路流的详细统计 |

## 状态、动作与相机

| LeRobot 字段 | 维度／形状 | 定义 |
| --- | --- | --- |
| `observation.state` | 54 | 实测左臂 J1..J7、右臂 J1..J7、左手 20 关节、右手 20 关节；单位 rad |
| `action`，joint | 54 | 对应上述顺序的成功提交目标；取采样时刻之前最近的有效命令 |
| `action`，eef | 52 | 左臂 xyz + rotvec、右臂 xyz + rotvec、左手 20 关节、右手 20 关节；位置 m、旋转 rad |
| `observation.eef_pose` | 12 | 实测关节 FK 得到的左右法兰位姿，xyz m + rotvec rad |
| `observation.wrench` | 12 | 左右 Fx,Fy,Fz,Tx,Ty,Tz；力 N、力矩 N·m |
| `observation.images.camera_0/1/2` | H×W×3，视频 | 三路 RGB；相机角色、序列号和实际标定留在 `meta/tianji.json` 的来源快照中 |

手部顺序为 finger1..5，每根 joint1..4；与原始记录一致。字段分量名称记录在 `meta/info.json` 的 `features.*.names`。

## 时间、缺帧与 episode 边界

复用 DP 导出的 `convert_recordings` 和 `plan_episode`，使用配置指定的固定训练网格。状态在真实样本之间插值，位姿旋转采用四元数插值；动作只保持过去成功提交的命令。过期命令、无效状态或无法修复的图像会剔除。默认设置允许少量相机画面复用，其次数和图像年龄写入报告。

每个有效连续片段独立成为一个 LeRobot episode。暂停、无效帧造成的时间缺口保持为边界；只有已有明确复核结论的暂停才按原有规划合并。标准 `timestamp` 从零开始，为 `frame_index / fps`；原始采样时刻保留在 `source.*` 字段中。`strict` 的真实主相机时间不构成固定网格，所以该脚本拒绝 strict 配置。

LeRobot episode 重新从 0 编号。`meta/tianji.json` 保留它与原始 `dataset_id`、`episode_id` 和片段范围的对应关系；原始编号空缺保持在来源信息中。

`source.*` 表列包括来源记录索引、重建的含暂停时间 ns、扣除暂停的录制时间 ns、录制块、选中的 RGB 行号、画面复用标记、相机时间偏差与设备时间 ms、状态插值标记、命令年龄，以及原始深度行号。`state_interpolated_fields` 与 `command_age_fields` 的列顺序由 `meta/tianji.json` 的 `provenance` 描述。主机单调时钟时间不代表日历日期。

## 原始深度与高频数据

默认复制所有实际导出来源的 `raw.zarr`，其压缩文件、dtype、原始时间戳、全部深度帧和高频状态／命令逐字节保留。该附件独立于 30 Hz 训练视图；不会因深度缺帧丢弃有效 RGB 样本。

标准 LeRobot 加载器读取 RGB 与表列。原始深度通过 `meta/tianji.json` 中的 `sources` 定位：

```python
import json
from pathlib import Path
import zarr

root = Path("datasets/<session>_lerobot")
mapping = json.loads((root / "meta/tianji.json").read_text())
# sample 是 LeRobotDataset 返回的一帧。
record = mapping["sources"][int(sample["source.recording_index"].item())]
depth_row = int(sample["source.depth_row"].item())
if depth_row >= 0:
    raw = zarr.open_group(str(root / record["raw_path"]), mode="r")
    depth_uint16 = raw["cameras/camera_0/depth/image"][depth_row]
    # 物理距离按该来源 metadata.cameras.camera_0 中的深度比例换算。
```

深度行号以原始主机时间匹配，使用转换配置的相机匹配容差，默认 20 ms；超出容差为 `-1`。深度数值未量化为视频，也没有替换成彩色预览。所有附件采用相对路径，移动或上传整个结果目录后仍可定位。

## 输出与读取

```text
<session>_lerobot/
  README.md
  data/chunk-000/file-000.parquet
  videos/observation.images.camera_0/chunk-000/file-000.mp4
  videos/observation.images.camera_1/chunk-000/file-000.mp4
  videos/observation.images.camera_2/chunk-000/file-000.mp4
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/tianji.json
  meta/conversion_report.json
  raw/recording-000000/
    source_episode.json
    source_dataset.json         # 源 session 存在 dataset.json 时保存
    raw.zarr/
```

一个分片可包含多个完整 episode，视频通过每条 episode 的时间偏移定位，表通过全局行范围定位。新分片只在 episode 边界创建。数值特征的均值、标准差、最小值和最大值使用全部导出行；分位数按 episode 的样本数加权估计。RGB 统计使用最多 64 个均匀时间样本及空间网格。跨分片只保留统计累加器，不缓存整个数据集的数值列。

在安装了官方 LeRobot 的环境中读取：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/tianji", root="datasets/<session>_lerobot", video_backend="pyav")
sample = dataset[0]
print(sample["observation.state"].shape, sample["action"].shape)
```

也可使用官方 [数据可视化工具](https://huggingface.co/docs/lerobot/using_dataset_tools)。原始附件的快照命名为 `source_episode.json`，避免后续录制发现逻辑将导出附件再次当作待转换 episode。
