# 数据采集操作

本文只写操作步骤；实现、数据格式和转换规则见[数据采集参考](recording_reference.md)。

## 流程总览

所有命令在 `conda activate bimanual-teleop` 后于项目根目录运行：

| 步骤 | 命令 | 必须 | 结果 |
| --- | --- | --- | --- |
| 采集 | `teleop_quest_tianji.py --record --task <task> --viewer` | 是 | `recordings/<task>/episode_*/`，状态 `captured` |
| 整理 | `finalize_recording.py` | 是 | 生成 `raw.zarr`，状态 `complete`；删除作废和失败条目 |
| 试运行 | `convert_recording.py --dry-run` | 否 | 打印每条的等级、丢弃和修复，不写文件 |
| 审片 | `review_recording.py` | 否 | `review.json`：剔除条目、决定暂停接缝是否接上 |
| 导出 | `convert_recording.py --output` | 是 | `datasets/<name>.zarr`，默认修复模式 |
| 训练 | `BimanualImageDataset` | — | 读取导出的数据集 |

最少只需采集、整理、导出三步：

```bash
python scripts/teleop_quest_tianji.py --record --task rail_place --viewer   # B 标记、S 保存、X 作废、Q 保存并退出
python scripts/finalize_recording.py --input recordings/rail_place
python scripts/convert_recording.py --input recordings/rail_place \
  --output datasets/rail_place_eef.zarr --action-space eef
```

不审片时所有条目按保留处理，暂停一律在接缝处切开。

## 1. 开始前检查

- 机械臂、灵巧手和三台 RealSense 周围无人员和障碍物；实体急停已释放且随手可按。
- `nvidia-smi -L` 能列出显卡。采集只用 NVENC 硬件编码，检查不通过时程序会在机械臂运动前退出。
- [采集配置](../configs/recording.yaml)中的相机序列号与实物一致，顺序固定为 `camera_0` 主视角 D436（`405622076394`）、`camera_1` 机器人左臂（`408322072104`）、`camera_2` 机器人右臂（`408322070938`）；`frame_capacity` 已按主机内存设置。
- `recordings/` 所在磁盘空间充足，主视角原始深度约占 1.1 GB/分钟。

2026-10-14 起左右臂相机编号调整为上述顺序；旧配置采集的数据是 `camera_1` 右臂、`camera_2` 左臂，不会自动改名。2026-10-16 起主视角 `camera_0` 为 D436 `405622076394`，此前是 D435 `348122072137`。混用前须按元数据中的序列号统一对应关系，详见[相机配置与编号](recording_reference.md#配置)。

## 2. 启动

```bash
conda activate bimanual-teleop
python scripts/teleop_quest_tianji.py --record --task rail_place --viewer
```

程序依次做 NVENC 检查、相机检查、运动确认和初始回位，终端会打印本次运行日志的路径。

- `--task` 是任务名，必须指定（或写在采集配置的 `task` 中），只能用字母、数字、`_`、`-`。同一任务的条目都放在 `recordings/<task>/` 下，编号跨多次启动连续：Ctrl+C 或退出后重新启动，新条目接着上次的编号，不会新建目录。
- 换任务时换一个任务名。同一任务目录同时只能有一个采集程序，第二个会在启动时报错退出。

## 3. 采集

| 操作 | 效果 |
| --- | --- |
| 接合（Enter；开启手势时也可双手比 V） | 等待 `start_delay_s`（默认 0）后开始新的一条；有暂停中的条目则继续该条 |
| 脱离（Enter、Space、摇滚手势或设备故障） | 暂停当前条；再次接合续录同一条，暂停时间不计入 |
| H | 暂停当前条并回位；再次接合仍续录同一条 |
| B | 录制中记录一个动作阶段边界时间点，保持录制连续；支持大小写 |
| S | 保存当前条，遥操作继续 |
| X | 作废当前条 |
| Q | 保存当前条并退出 |
| Ctrl+C | 退出；当前条记为失败，不能导出 |

- 换任务、回位或场景变化前先按 S，避免前后画面拼进同一条。
- 保存时若仍在接合状态，不会自动开始新条；脱离后再接合。
- 接合后提示"等待新鲜的采集数据"时无需操作，数据就绪后自动开始。
- B 每次记录一个时间点，成功写入后终端显示标记编号和本条录制中的秒数；未开始、暂停、恢复等待、保存中或采集故障时不记录。标记随原始数据和导出数据保留，阶段名称及划分由后续处理决定，不自动切开训练片段。

## 4. 离线整理

退出遥操作后执行：

```bash
python scripts/finalize_recording.py --input recordings/<task>
```

- 作废（X）和采集失败的条目在这一步被**直接删除**，不可恢复。正在采集时中断、状态仍为 `capturing` 的条目会跳过并保留。整理成功的条目才能导出。删除后编号留空，新条目不会复用。
- 整理成功后，条目内的原始缓存 `raw_spool/` 移到 `raw_spools/<task>/<episode>`（相对当前目录），原处留符号链接；`--spool-archive DIR` 可改位置，但不能放在 `--input` 之内。
- 整理某一条出错时继续整理其余条目，最后以非零退出码结束；排除问题后重跑同一命令即可。

## 5. 可选：试运行与审片

想先看导出结果、剔除个别条目或把暂停接缝接上时，在导出前执行；不需要时直接做第 6 步。

```bash
python scripts/convert_recording.py --input recordings/<task> --action-space eef --dry-run
python scripts/review_recording.py --input recordings/<task>
```

- 试运行与正式导出流程相同、打印内容相同（见第 6 步），只是不解码视频、不写任何文件，可省略 `--output`。它会一次列出所有会让正式导出中止的错误，例如元数据不一致、三台相机识别出的暂停不一致，此时退出码为 1；正式导出遇到第一个错误就停止。
- 审片逐条生成三路并排的 `preview.mp4` 并用 ffplay 播放，缺帧处显示为暗红色块。按 y 保留、n 剔除、r 重播、s 跳过、q 退出，结论写入条目旁的 `review.json`；剔除的条目导出时跳过。已有结论的条目默认不再询问，加 `--all` 重审。
- 条目有暂停时，审片会列出每个接缝及其在预览中的时间；保留该条后逐个回答合并（m）或保留边界（k）。只有接缝两侧动作确实连续才选合并；没有回答的接缝导出时切开。

## 6. 导出训练数据

整理后（或试运行、审片后）导出，默认不需要人工：

```bash
python scripts/convert_recording.py \
  --input recordings/<task> \
  --output datasets/<task>_eef.zarr \
  --action-space eef
```

- 默认按[转换配置](../configs/recording_conversion.yaml)的修复模式处理：相机偶尔缺 1～2 帧时沿用前一张图像；缺得更多，或状态、指令缺失的帧剔除；暂停处自动切开。
- 终端逐条打印 `[OK]`、`[WARN]`、`[FAIL]` 或 `[SKIP]` 和写入帧数，下面几行是：
  - “丢弃”：首尾裁剪是条目开头或结尾缺少状态或指令的帧，每条通常 1～2 帧，属正常；中间剔除会把条目切成多个片段。括号内是原因。
  - “修复”：沿用前一张图像的帧数及图像最旧多少毫秒、放宽插值的帧数、因暂停在哪一帧切开。
  - “问题”：超出[质量配置](../configs/recording_quality.yaml)阈值的项，如相机丢帧或断档、跨相机偏差、低维流频率偏低、时长过短。等级只是标注，不影响写入哪些帧。

  最后汇总全部条目的等级、丢弃和修复；`--verbose` 另列每路统计。
- 输出路径必须不存在；关节动作空间改用 `--action-space joint`；需要深度时加 `--include-depth`。
- `--input recordings` 一次导出其下所有任务。合并的条目之间，相机序列号与内参、运动学模型、天机和 Wuji 控制参数必须一致，否则报错并列出不同的字段；确认可以混用时加 `--allow-mixed-metadata`。
- 训练前建议在转换配置中把 `min_segment_frames` 设为不小于训练窗口长度（DP 配置的 `horizon`，默认 16）。

## 7. 训练接入

导出的数据集用 `bimanual_teleop.recording.dataset.BimanualImageDataset` 读取，训练环境需另装官方 `diffusion_policy`（本项目环境不含 PyTorch）。

- [`configs/dp_eef.yaml`](../configs/dp_eef.yaml) 对应 `--action-space eef`（动作 52 维），[`configs/dp_joint.yaml`](../configs/dp_joint.yaml) 对应 `--action-space joint`（动作 54 维）。它们只是数据读取配置片段，合并到训练配置根节点，并把 `dataset_path` 改成实际导出的路径。
- 适配器按片段采样，训练窗口不跨暂停或剔除造成的缺口；采样和训练／验证划分规则见[数据采集参考](recording_reference.md#dp-训练接入)。

## 8. 故障恢复

相机、编码、写盘失败或数据断流时，当前条记为失败，但遥操作不会自动停止：

1. 主动脱离。
2. 根据终端提示和运行日志排除相机、显卡或磁盘问题。
3. 在脱离状态按 C，等待"三路相机采集已就绪"。
4. 重新接合，自动开始新的一条。

若提示采集进程被强制终止，须退出并重启程序。排查时保留运行日志和整个条目目录。
