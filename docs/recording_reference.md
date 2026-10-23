# 数据采集参考

本文说明采集实现、数据格式、时间语义和转换规则；操作步骤和按键见[数据采集操作](recording_guide.md)。

## 流程

```text
teleop_quest_tianji.py --record  ->  recordings/<task>/episode_*/     MP4 + raw_spool   状态 captured
finalize_recording.py            ->  同一目录生成 raw.zarr                             状态 complete
convert_recording.py --dry-run   ->  （可选）逐条等级、丢弃与修复，不写任何文件          OK / WARN / FAIL / SKIP
review_recording.py              ->  （可选）同一目录生成 preview.mp4 和 review.json    keep / reject
convert_recording.py             ->  datasets/<name>.zarr，默认 repair 模式            DP ReplayBuffer
```

整理后可直接导出，默认不需要人工：repair 模式自动复用短缺帧、剔除无法修复的帧，未审片的暂停在接缝处切开，没有 `review.json` 的条目按保留处理。试运行和审片只在想先看结果、剔除条目或合并暂停接缝时使用。质量检查只在导出（含试运行）时进行，采集过程中不做。

采集只支持双臂双手联合遥操作，不能与 `--arms-only` 或单侧 `--side` 同用。

## 配置

[`configs/recording.yaml`](../configs/recording.yaml)，可用 `--recording-config PATH` 替换：

| 字段 | 含义 |
| --- | --- |
| `cameras` | 三台相机序列号，依次为 `camera_0` 主视角、`camera_1` 机器人左臂、`camera_2` 机器人右臂；主视角（第三视角）D436 是时间基准 |
| `main_depth` | 是否采集 `camera_0` 深度，默认 `true` |
| `state_hz` | 实测状态的记录频率上限，默认 200；不影响控制和保护 |
| `output_dir` | 输出根目录，默认 `recordings`，相对启动目录 |
| `task` | 任务名，条目写入 `<output_dir>/<task>/`；默认留空，启动时须用 `--task NAME` 指定，命令行优先。1～64 个字母、数字、`_` 或 `-`，以字母或数字开头 |
| `start_delay_s` | 接合后多久开始或继续录制，0～60 s，默认 0 |
| `frame_capacity` | 每路图像帧池长度，16～2048，默认 256（约 825 MiB、8.5 s）；按主机内存设置 |
| `controls` | 边界标记（`mark_key`，默认 B）、保存（S）、作废（X）、保存并退出（Q）、恢复采集进程（C）的按键；标记键不得与其他采集键或遥操作键冲突 |

相机固定为 640×480、30 Hz：三路 RGB，深度只来自 `camera_0`。

| 图像编号 | 安装位置（按机器人自身左右） | 序列号 | RGB 旋转 |
| --- | --- | --- | --- |
| `camera_0` | 主视角（第三视角）D436 | `348122072137` | 0° |
| `camera_1` | 左臂 D436 | `408322072104` | 180° |
| `camera_2` | 右臂 D436 | `408322070938` | 0° |

编号按 `cameras` 列表顺序与序列号绑定，与 USB 插入顺序无关；录制预览标签、MP4 文件名、原始流和导出数据集图像键使用同一编号。
`camera_1`（左臂）的 RGB 在采集后统一旋转 180°，实时预览、MP4 和后续导出的图像方向一致；`camera_0`、`camera_2` 及主视角深度保持原方向。未录制时的独立预览按采集配置中的序列号识别左臂相机，采用相同旋转。

2026-10-14 调整了左右臂编号：此前默认配置中 `camera_1` 是右臂（`408322070938`），`camera_2` 是左臂（`408322072104`）。已有录制及导出数据不会自动交换两路图像；读取旧数据时以该条 `episode.json` 的 `metadata.cameras` 序列号为准。合并新旧录制会报相机元数据不一致，须先统一图像及对应元数据，不能仅跳过检查后混用于训练。

## 在线采集

### 启动检查

启用 `--record` 后，在机械臂运动前依次完成：

1. 检查 NVIDIA 驱动，并发运行三路 NVENC 编解码测试。失败直接退出，不回退到 CPU 编码。
2. 打开三台相机，每路须连续收到 3 帧带 `GLOBAL_TIME` 时间戳的新帧；10 s 内未就绪则退出。

### 条目生命周期

- **开始**：接合后等待 `start_delay_s` 自动开始。要求 8 路低维流（双臂、双手的实测和指令）都在 100 ms 内有新数据，否则提示等待并自动重试。
- **暂停与继续**：任何脱离（包括 H 回位和设备故障）只暂停本条，期间不写数据；再次接合后继续同一条，所有流的时间戳减去暂停时长，时间轴从暂停处接上。暂停区间本身不单独记录，转换时从相机时间识别（见[暂停识别](#两种模式)）。
- **结束**：保存时录制进程最多再等约 2 s，确认三路 RGB（启用时含深度）已越过结束时刻、低维队列已排空，再核对相机帧的生产数与写盘数、低维队列的发送数与消费数；全部一致才记为 `captured`。
- **边界标记**：正在录制时，每收到一次 B 按键事件记录一个时间点。主进程读取按键时取得单调时钟时间，录制进程扣除累计暂停时长并原子更新 `episode.json`，写入后才确认成功；不改变条目生命周期或设备流计数。未在有效录制中时拒绝标记并提示。

| 状态 | 含义 |
| --- | --- |
| `capturing` | 正在采集；进程被强杀或断电时停留在此状态 |
| `captured` | 原始数据已保存，等待离线整理 |
| `finalizing` | 正在整理 |
| `complete` | 整理完成，可以转换 |
| `failed` | 采集、写盘或收尾出错，或 Ctrl+C 退出时未结束；不能转换 |
| `discarded` | 操作员作废，整理时删除 |

### 进程结构

- 设备适配器只把实测反馈和成功提交的指令复制进共享内存队列（每路 4096 条，约 20 s），不做图像处理、正运动学、压缩或磁盘操作。双臂数据来自主进程，双手数据来自 Wuji 子进程。
- 录制进程持有三台相机，每台一个采集线程，同时消费低维队列并顺序写盘。
- 每条开始时，三路 RGB 各启动一个 NVENC 编码进程，深度启动一个写盘进程；图像经共享帧池（`frame_capacity`）传递。
- 录制相关进程不绑核，只降低调度优先级（录制进程 `nice +5`，由它启动的预览进程再加 10），把 CPU 让给运动控制。
- 预览是独立进程，只读取约 5 Hz 的最新 RGB，卡顿或关闭不影响录制；未开始条目且无预览时不复制像素。
- 预览窗口贴屏幕顶端，三路无边距并排、铺满一块显示器的宽度（高度不超过屏幕的 55%），下方留给终端。

### 故障处理

以下任一情况会把当前条记为 `failed` 并结束录制进程：

- 低维队列、相机队列或帧池已满。程序不覆盖未写盘的数据。
- 录制中任一低维流或相机流超过 0.5 s 无新数据。
- 相机时间戳离开 `GLOBAL_TIME`、时间或帧序号倒退，或主机实时时钟跳变超过 10 ms。
- 编码、写盘或收尾失败。

采集故障不暂停运动；设备故障、安全保护和控制看门狗仍立即暂停。脱离后按 C 会重新做 NVENC 检查并启动新的录制进程；旧进程只能被强制终止时，须重启整个程序。

## 原始数据

### 目录

```text
recordings/<task>/
  .next_episode                 # 下一个条目编号
  .recording.lock               # 采集进程持有的文件锁
  episode_000000/
    episode.json                  # 状态、[start_ns, end_ns)、各流计数、元数据、边界标记
    camera_0.mp4 … camera_2.mp4   # NVENC H.264，每个真实帧编码一次
    raw_spool/
      streams/<流名>.bin           # 流名中的 / 换成 __，如 arms__left.bin；int64 time_ns、int64 sequence、float64 数值，小端
      cameras/camera_<i>_rgb.bin   # 每帧：int64 time_ns、int64 sequence、float64 source_time_ms
      cameras/camera_0_depth.raw   # 启用深度时：逐帧 uint16 480×640
      cameras/camera_0_depth.bin   # 深度帧时间表，格式同 RGB
    raw.zarr/                     # 离线整理生成
    review.json、preview.mp4      # 审片生成，可选
  episode_000001/ …
```

MP4 的播放时间不作同步依据，每帧真实时间在 `camera_<i>_rgb.bin` 中，整理后写入 `raw.zarr`。

同一任务的条目平铺在任务目录下，编号跨多次运行连续递增：启动采集时取 `.next_episode` 与已有条目（含中断留下的 `.episode_*.finalizing-*`）最大编号加 1 中的较大者，每开始一条就写回计数。因此整理删除作废条目、手动删除条目或 C 恢复后，编号都不会被重复使用，作废和失败条目会在编号中留下空缺。启动时对任务目录加排他文件锁，同一任务目录同时只允许一个采集进程；进程退出或崩溃时锁自动释放。Ctrl+C 中断后重新启动，新条目接在同一任务目录下继续编号。

### 时间语义

- 所有 `time_ns` 都是主机单调时钟（int64），暂停时长已扣除。
- 机械臂：主机读到 SDK 新反馈的时刻；手部：SDK 出队时刻；指令：SDK 成功提交的时刻，不代表已执行。
- 相机：`GLOBAL_TIME` 帧时间映射到主机单调时钟，不是曝光中点；`source_time_ms` 保留设备原值，不扣除暂停。
- 三台相机没有硬件同步，软件对齐不能消除曝光差和设备延迟。
- `[start_ns, end_ns)` 是有效窗口；收尾时可能多写少量窗口外的帧，转换时排除。
- B 标记的 `time_ns` 是主进程处理按键事件的时间，使用同一条目的录制时间轴；不代表设备收到指令或图像曝光的时刻。

### 边界标记

`episode.json` 的可选顶层字段 `phase_markers` 为列表，每项是 `{"index": 1, "time_ns": 123456789000}`。`index` 在每条录制内从 1 递增，`time_ns` 是扣除暂停时长的整数纳秒时间戳；减去本条 `start_ns` 并除以 `1e9` 得到相对秒数。标记按时间排序，位于有效窗口内，不存储阶段名称。

离线整理和 `--refinalize` 保留该列表，同时写入 `raw.zarr.attrs['phase_markers']`。标记不参与设备流 `counts` 或暂停接缝的识别。旧记录缺少该字段时按空列表处理，schema 版本保持不变。

### 元数据

`episode.json` 的 `metadata` 保存天机与 Wuji 配置、采集配置（含 `recording.task`）、相机内参（深度另含到 RGB 的外参和 `depth_scale`）、运动学模型摘要和各字段说明。`metadata.session` 是本次运行的标识（`YYYYmmdd_HHMMSS_8位随机`），同一次启动采集的条目相同，可据此找回运行分组；不参与转换时的元数据一致性检查。
每台相机的 `image_rotation_deg` 记录 RGB 图像相对传感器原图的旋转角度，当前配置的 `camera_1`（左臂）为 180，其余为 0；旧录制以其自身元数据为准。`rgb_intrinsics` 保留传感器原始标定；投影到旋转后的左臂图像时，像素坐标按 `u' = width - 1 - u`、`v' = height - 1 - v` 转换。旧数据缺少旋转字段时按 0 处理，不会自动旋转；转换时将图像旋转角度纳入元数据一致性检查。

## 离线整理

```bash
python scripts/finalize_recording.py --input recordings/<task> [--spool-archive DIR] [--refinalize] [--sdk-root PATH]
```

- 递归处理 `--input` 下的全部条目，跳过以 `.` 开头的临时目录。`discarded` 条目被删除，`failed` 和 `capturing` 条目跳过并保留。某条失败时继续处理其余条目，最后以非零退出码结束。
- 对 `captured` 条目核对：低维和相机序号严格递增、MP4 解码帧数与时间表一致、深度文件完整、各流计数与 `episode.json` 一致、运动学模型摘要与采集时相同。
- 由实测关节计算双臂法兰正运动学得到 `eef_pose`，全部数据无损压缩写入 `raw.zarr`。
- 结果先写临时目录再原子替换。失败时条目保持 `captured` 并记录 `finalize_error`，修复后重跑即可。
- 成功后 `raw_spool/` 移到 `<spool-archive>/<task>/<episode>`（默认 `./raw_spools`，不得位于 `--input` 内），原处留符号链接。跨文件系统时先完整复制再删除，中断后重跑可继续。转换只读 `raw.zarr` 和 MP4。
- `--refinalize` 把 `complete` 条目改回 `captured` 并从 `raw_spool` 重建 `raw.zarr`；符号链接失效时到 `--spool-archive` 中查找，仍找不到则保持 `complete` 并报错。

### raw.zarr

每个流都有 `time_ns`、`sequence` 和下列字段：

| 流 | 字段与形状 |
| --- | --- |
| `arms/left`、`arms/right` | `joint_pos (N,7)`、`eef_pose (N,7)`、`wrench (N,6)` |
| `hands/left`、`hands/right` | `joint_pos (N,20)` |
| `arm_commands/left`、`arm_commands/right` | `joint_pos (N,7)`、`eef_pose (N,7)` |
| `hand_commands/left`、`hand_commands/right` | `joint_pos (N,20)` |
| `cameras/camera_<i>/rgb` | `source_time_ms (N,)`；第 k 行对应 MP4 解码的第 k 帧 |
| `cameras/camera_0/depth`（启用时） | `source_time_ms (N,)`、`image (N,480,640) uint16` |

- 位姿为各臂基座下的法兰 `[x,y,z,qx,qy,qz,qw]`，位置单位米；关节单位弧度，顺序为每臂 J1～J7、每只手手指 1～5 各关节 1～4。
- 力为 `[Fx,Fy,Fz,Tx,Ty,Tz]`（N、N·m），沿原生传感器轴，不去零、不滤波、不做重力补偿。
- 深度值乘 `depth_scale` 得米，0 表示无效；未重投影到 RGB。
- 机械臂指令约 200 Hz，手部指令约 120 Hz。`arm_commands.eef_pose` 是约束求解前送入笛卡尔控制器的目标，`joint_pos` 是成功提交的关节目标；二者对应不同动作空间，不能互相替代。

## 审片

```bash
python scripts/review_recording.py --input PATH [--all] [--no-play] [--player CMD] [--scale 0.5] [--refresh-preview]
```

- 对每条 `complete` 条目生成 `preview.mp4`：三路相机并排，每路缩放到 `--scale`（默认 320×240），每个主相机原始帧对应预览中的一帧；其他相机在 20 ms 内没有对应帧时显示暗红色块。已有预览不重复生成，`--refresh-preview` 强制重建。
- 默认用 ffplay 播放，找不到时用 xdg-open；`--player` 指定其他播放命令，`--no-play` 只提问。默认只问还没有 keep/reject 结论的条目，`--all` 全部重审。
- 保留的条目若有暂停，逐个询问接缝是否合并，写入 `review.json` 的 `pause_reviews`；暂停识别阈值默认取 `configs/recording_conversion.yaml` 的 `pause_offset_tolerance_ms`，导出另用配置时用 `--pause-offset-tolerance-ms` 保持一致。

`review.json` 的字段都可省略：

```json
{"decision": "keep", "note": "抓取完整", "reviewed_at": "2026-10-05T14:03:11+08:00",
 "pause_reviews": {"679": {"merge": true, "reason": "审片确认接缝两侧动作连续"}}}
```

`decision` 为 `reject` 的条目在转换时跳过并写入报告。`pause_reviews` 的键是暂停后主相机首帧的原始行号，格式与转换配置中的 `pause_reviews` 相同；同一接缝两处都有结论时以 `review.json` 为准。

## 转换为 DP 数据集

```bash
python scripts/convert_recording.py --input PATH --output datasets/<name>.zarr \
  --action-space {eef,joint} [--include-depth] [--conversion-config PATH] [--quality-config PATH] \
  [--dry-run] [--allow-mixed-metadata] [--verbose] [--report PATH]
```

- 只转换 `complete` 条目，其他状态和 `review.json` 中 `reject` 的条目跳过并写入报告。`--input` 可以是单条、一个任务目录或多个任务的上级目录；以 `.` 开头的临时目录不扫描。
- 输出路径必须不存在。结果先写临时目录，成功后原子发布；任一条目出错则整次失败，不留半成品。
- `--dry-run` 走同一套规划和质量检查，只是不解码视频、不写数据集，可省略 `--output`；打印的数字与正式转换完全一致。正式转换遇到第一个错误即中止，试运行则把每条的错误记入报告后继续，一次列出全部问题，有错误时退出码为 1。`--verbose` 打印每路统计，`--report PATH` 另存完整质量报告 JSON。
- 默认不读深度；加 `--include-depth` 才检查深度匹配，且要求所有条目一致地有或没有深度。
- 合并的条目须在以下元数据上一致：`model_sha256`、位姿与时间语义说明、`recording.state_hz`、天机 `profile` 与 `quest`、Wuji `profile_id`／`parameters`／`control_hz`、每台相机的序列号、型号、分辨率、帧率和 RGB 内参（`--include-depth` 时另加主相机深度内参、外参和 `depth_scale`）。不一致时报错并列出字段；`--allow-mixed-metadata` 允许合并，差异写入报告的 `metadata_differences`。操作员名、网络地址、按键和超时设置不参与比较。这是默认流程中唯一需要人确认后才能继续的情况。

### 两种模式

默认读 [`configs/recording_conversion.yaml`](../configs/recording_conversion.yaml)，其中为 `mode: repair`；改这个文件即改默认值。`--conversion-config PATH` 改用另一份配置，省略的项取下表括号中的默认值（单位 ms，帧率为 Hz，缺帧为帧数）。配置写 `mode: strict` 为 strict 模式：不复用任何图像，以主相机真实帧时刻为时间轴，有暂停的条目不应使用。

| 规则 | repair（默认） | strict |
| --- | --- | --- |
| 时间基准 | 每个录制块内，从首个主相机帧开始的固定网格，项目YAML当前 `target_fps=30`（原始RGB仍为30 Hz） | 主相机真实帧时刻 |
| 相机匹配 | 最近帧，相差 ≤`camera_match_tolerance_ms`（20）；连续不超过 `max_missing_camera_frames`（2）个缺帧时复用前一帧 | `camera_1`、`camera_2` 取最近帧，相差 ≤20 ms，不复用 |
| 状态插值两端间隔 | ≤`max_state_interp_gap_ms`（100） | ≤50 ms |
| 动作命令龄 | ≤`max_command_age_ms`（50） | ≤50 ms |
| 暂停 | 识别后默认在暂停处切开 | 识别后只在报告中记 WARN，不切开 |
| 片段切分 | 无效帧、网格间隔超过 1.5 帧、录制块边界 | 无效帧，或主相机帧间隔 >50 ms |

两种模式由同一份规划代码执行，strict 只是一组固定参数：以主相机真实帧为查询时刻、整条为一个录制块、不复用缺帧、不在暂停处切开。

共同规则：

- 状态：关节、力和手部线性插值，位姿用 SLERP，不在观测范围外外推。
- 动作：取该时刻及之前最近一次成功提交的目标，不插值。
- 任一条件不满足的帧被剔除。
- 短于 `min_segment_frames`（默认 0，不过滤）的片段不写入，计入报告的 `short_segments`。配置文件写 `mode: strict` 时其他阈值仍用 strict 的固定值，只有该项和 `pause_offset_tolerance_ms` 生效。

**暂停识别**：两种模式相同，任一相机的 `source_time_ms` 与 `time_ns` 之差突增超过 `pause_offset_tolerance_ms`（默认 100 ms）即判为一次暂停。repair 要求三路相机的判断一致，否则该条报错；暂停处默认作为片段边界，只有 `pause_policy: checked_compress` 且对该接缝写明 `merge: true` 和理由时才合并，合并后时间戳压缩为连续。结论可写在转换配置的 `pause_reviews` 或条目的 `review.json` 中，同一接缝以 `review.json` 为准。接缝以"暂停后第一帧在主相机原始数据中的行号"标识，即报告中的 `pauses[].after_main_row`。

repair 模式另在 `meta/` 下写入逐样本追溯数组：`source_time_ns`、`recording_time_ns`、`recording_block`、`camera_source_row`、`camera_reused`、`camera_time_offset_ns`、`camera_source_time_ms`、`state_interpolated`、`command_age_ns`。

### 丢弃与修复

转换不伪造控制数据：状态只在两个真实样本之间插值，不外推；动作只取已提交的指令，不插值，也不使用超过命令龄上限的指令。在此前提下，repair 修复以下情况，这些帧照常写入：

| 情况 | 修复方式 | 记录 |
| --- | --- | --- |
| 某路相机（含主相机）在网格时刻没有匹配帧，连续缺失不超过 `max_missing_camera_frames` | 沿用该相机前一张真实图像；图像年龄不超过（缺帧上限 + 1）个帧周期，默认 100 ms | `meta/camera_reused`、`meta/camera_source_row`；报告 `camera_repairs` |
| 状态两侧真实样本相隔 50 ms 到 `max_state_interp_gap_ms` | 照常插值；strict 会剔除这些帧 | 报告 `state_repairs` |
| 暂停 | 未审片的接缝切成两个片段；审片确认合并的接缝接上，时间戳压缩为连续 | 报告 `pauses` |

修复不了的帧不写入，分四类：

| 类别 | 内容 | 报告字段 |
| --- | --- | --- |
| 整条跳过 | 状态不是 `complete`，或审片剔除 | `status`、`reason`、`skipped` |
| 首尾裁剪 | 首个有效帧之前、最后一个有效帧之后的帧，通常是开头还没有状态样本的第 0 帧 | `edge_trimmed_frames`、`edge_trimmed_reasons` |
| 中间剔除 | 相机连续缺帧超过上限、状态缺失或两侧间隔超过上限、指令缺失或过期；会把条目切成多个片段 | `interior_invalid_frames`、`interior_invalid_reasons` |
| 短片段 | 短于 `min_segment_frames` 的片段 | `short_segments` |

终端逐条打印等级、写入帧数、“丢弃”“修复”和“问题”，最后汇总全部条目，试运行与正式转换输出相同。例如：

```text
  [WARN] 20261004_222629_255443a0/episode_000001：参考 1650 帧，写入 1648 帧（99.9%），2 个片段
      丢弃：首尾 1+0 帧（左臂关节缺失或间隔过大 1，…）；中间 1 帧（camera_1 缺帧 1，camera_2 缺帧 1，…）
      修复：主相机第 679 帧前暂停 27.9 s，在此切开
      问题：hands/left 138 Hz
丢弃：整条跳过 4 条（failed 4）；首尾 5 帧，中间 1 帧，短片段 0 段 / 0 帧
修复：缺帧沿用前一张图 camera_1 5 帧，camera_2 5 帧；放宽插值 0 帧；暂停在此切开 1 处，按审片结论接上 0 处
```

### 质量等级

规划每条时顺带统计原始流的健康状况，与 [`configs/recording_quality.yaml`](../configs/recording_quality.yaml) 中的 `{warn, fail}` 阈值比较，得出 OK、WARN 或 FAIL。统计复用规划已读出的数据，与转换模式无关；暂停识别阈值和跨相机匹配容差沿用转换模式的设置。

| 检查 | 内容 |
| --- | --- |
| 时长 | `[start_ns, end_ns)` 窗口时长低于下限 |
| 每台相机 | 帧号跳变累计的丢帧数、相邻帧间隔超过 `gap_periods` 个帧周期的断档次数、最大帧间隔；暂停接缝两侧不计。帧号连续但时间间隔超标也算断档 |
| 暂停 | 三台相机识别出的暂停数不同记 FAIL（repair 直接报错）；strict 下有暂停记 WARN，提示改用默认的 repair |
| 跨相机 | `camera_1`、`camera_2` 对每个主相机真实帧的最近帧时间差：已对应帧的 p50／p99 和全部帧的最大值，按 p99 定级；对不上的帧数另列，它们造成的剔除见剔除原因 |
| 低维流 | 8 路流各自的频率（样本数除以时长）和最大间隔 |
| 元数据 | 加 `--allow-mixed-metadata` 合并的不一致条目记 WARN |

等级只标注报告，不改变写入哪些帧；要排除某条，在审片中剔除。`failed`、`discarded` 和审片剔除的条目记 SKIP，`captured` 等尚待整理的条目记 WARN。试运行中会让正式转换中止的问题记 FAIL，原因写在该条的 `error`。

### 输出 schema v2

`data/` 下的字段：

| 字段 | 形状 / 类型 |
| --- | --- |
| `camera_0`、`camera_1`、`camera_2` | `(T,480,640,3)` uint8 RGB |
| `camera_0_depth`（仅 `--include-depth` 且原始数据有深度） | `(T,480,640)` uint16 |
| `robot_eef_pose` | `(T,12)`：每臂 xyz + 旋转向量 |
| `robot_joint` | `(T,14)` |
| `hand_joint` | `(T,40)` |
| `wrench` | `(T,12)` |
| `action` | eef 模式 `(T,52)`，joint 模式 `(T,54)` |
| `timestamp` | `(T,)` float64，相对原始条目开始的秒数；repair 模式为网格时刻，已合并的暂停被压缩 |

- 低维字段均为 float32，各组先左后右；`action` 依次为左臂、右臂、左手、右手。
- `meta/episode_ends`：每条保留了有效帧的原始演示占一个元素。
- `meta/segment_ends`：每个连续片段占一个元素，训练窗口不跨片段，原始演示边界必是片段边界。例如一条演示保留 100 帧、第 40 帧后有缺口，则 `episode_ends=[100]`、`segment_ends=[40,100]`。
- `meta.attrs['segments']` 与 `segment_ends` 一一对应，记录来源条目和输出范围。
- `meta.attrs['phase_markers']` 保存有导出帧的条目中的全部边界标记，每项为 `{"source_episode": "episode_000000", "index": 1, "recording_time_ns": 123456789000, "output_boundary": 42}`。`recording_time_ns` 保留原始标记时间；`output_boundary` 是从 0 开始的全局切分位置，表示在第 42 帧前切分。映射取本条实际保留帧中录制时间首次不早于标记的位置；早于首帧映射到条目起点，晚于末帧映射到条目结束位置（不含该位置）。重采样、缺帧剔除、短片段过滤或暂停合并后仍按原始录制时间映射，多个标记可对应同一位置。B 标记不会修改 `episode_ends`、`segment_ends` 或训练采样边界。
- `meta.attrs['quality_report']` 是完整报告，`--report PATH` 另存同一份 JSON。每条记录参考帧数、有效帧数、写入帧数、片段数、主相机缺口数和剔除原因：`invalid_reasons` 统计全部剔除帧，首尾和中间分开的字段见[丢弃与修复](#丢弃与修复)；repair 模式另含 `pauses`、`camera_repairs` 和 `state_repairs`。另含 `short_segments`、审片结论 `review`、被跳过时的 `skipped`，以及质量等级 `level`、各项检查 `checks` 和统计 `duration_s`、`cameras`、`cross_camera`、`streams`。顶层记录所用的 `conversion_config` 和 `quality_thresholds`。
- 报告中每条的 `phase_markers` 与导出标记使用相同字段；未导出的条目仍保留标记，其 `output_boundary` 为 `null`。试运行使用同一映射逻辑，报告中的位置与正式转换一致。

只读 `episode_ends` 的通用 DP loader 会忽略片段内缺口，须使用下面的适配器或自行遵守 `segment_ends`。适配器仍兼容把片段结束写在 `episode_ends` 中的旧 schema v1。

## DP 训练接入

适配器为 `bimanual_teleop.recording.dataset.BimanualImageDataset`，需另装官方 `diffusion_policy`（本项目环境不含 PyTorch）。

- 按 `segment_ends` 采样，同一原始演示的所有片段划入同一训练／验证分区；只有一条演示时没有验证集。
- 只有 `shape_meta.obs` 中选择的字段进入模型，深度不会自动作为输入。
- [`configs/dp_eef.yaml`](../configs/dp_eef.yaml)、[`configs/dp_joint.yaml`](../configs/dp_joint.yaml) 是数据读取配置片段，不是完整训练配置；分别以末端位姿或关节为观测，都使用三路 RGB、手部关节和六维力。

## 实机验收

- `--record --viewer` 连续遥操作 15 分钟，无机械臂看门狗超时、Wuji 数据过期或控制暂停。
- 三路 RGB 30 Hz、深度 30 Hz、低维状态 200 Hz 连续录制 30 分钟，采集链路无内部丢弃；设备源数据缺口在转换时被切分。
- 关闭或卡住预览窗口不影响采集。
- 连续至少 100 次开始、暂停续录、保存和作废。
- 编码、深度或低维写入失败时，当前条记为 `failed`、遥操作继续，按 C 后可继续采集。
- 异常退出后，离线整理不发布伪完整数据；整理结果可直接转换。

设计参考：[DP 实机代码](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/real_world/real_env.py)、[DP 论文 §7.1](https://arxiv.org/html/2303.04137v5#S7.SS1)、[UMI 双臂对齐](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/bimanual_umi_env.py)、[ALOHA 录制](https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/record_episodes.py)、[TeleVision 后处理](https://github.com/OpenTeleVision/TeleVision/blob/main/scripts/post_process.py)。
