# 双臂机器人遥操作

Quest 手柄控制天机机械臂，Wuji Glove 控制 Hand2，可同步采集相机与机器人数据用于 Diffusion Policy 训练。

## 快速开始

首次使用按以下步骤进行，各步细节见后文。

### 1. 安装

运行主机为 Linux x86_64，需要 Conda 和 ADB。在项目根目录执行：

```bash
PIP_USER=false conda env create -f environment.yml
conda activate bimanual-teleop
```

Quest 开启开发者模式、连接 USB 并在头显中授权调试后安装客户端：

```bash
adb install -r quest_app/artifacts/quest-capture-debug.apk
```

### 2. 配置

- [天机配置](configs/tianji_teleop.yaml)：控制器地址 `controller_ip`。
- [Wuji 配置](configs/wuji_teleop.yaml)：左右设备地址 `devices`、已标定用户名 `sdk_user_name`；尚未标定先做[手套标定](#手套标定)。
- [采集配置](configs/recording.yaml)：三台相机序列号，只在采集时需要。

### 3. 检查设备

各命令分别运行，确认数据正常后关闭：

```bash
python scripts/view_quest.py                   # 头显与手柄位姿
python scripts/view_wuji_glove.py --side left  # 手套，右手用 --side right
python scripts/home_tianji.py --inspect        # 天机当前位姿，不运动
```

### 4. 遥操作

```bash
python scripts/teleop_quest_tianji.py
```

1. 确认机器人周围无人、实体急停已释放且随手可按，按回车；机械臂先清错并回初始位姿。
2. 拿起手柄、戴上头显，按 **Enter** 接合，双臂和双手开始跟随。**左手柄控制右臂，右手柄控制左臂。**
3. **Space** 暂停，再按 Enter 以当前位姿为基准继续；**H** 停止跟随并回 `ready_pose`；**Q** 退出。

### 5. 采集数据

```bash
python scripts/teleop_quest_tianji.py --record --viewer
```

接合后自动开始录制一条；脱离只暂停，再接合续录同一条。**S** 保存、**X** 作废、**Q** 保存并退出。换任务或回位前先按 S。

### 6. 整理并导出

```bash
python scripts/finalize_recording.py --input recordings/<session>
python scripts/convert_recording.py --input recordings/<session> \
  --output datasets/episodes_eef.zarr --action-space eef
```

整理时会删除作废的条目。导出默认用修复模式，不需要人工：短缺帧沿用前一张图像，无法修复的帧剔除，暂停处自动切开，并逐条打印丢弃和修复了什么。导出前可选 `--dry-run` 试运行（不写文件）和 `review_recording.py` 审片（剔除条目、合并暂停接缝）。全部步骤一览见[数据采集操作的流程总览](docs/recording_guide.md#流程总览)。

本项目使用 `bimanual-teleop` 环境完成采集、整理和转换；导出的zarr交给相邻 [diffusion_policy项目](../diffusion_policy/docs/tianji-rail/README.md)训练和评估，模型侧使用 `tianji-rail-diffusion-policy` 环境，直接读取本项目的数据路径。训练读取接口见[训练接入](docs/recording_guide.md#7-训练接入)。

## 文档

| 文档                                                   | 内容                                                                             |
| ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| [数据采集操作](docs/recording_guide.md)                 | 流程总览、采集前检查、按键、整理、可选的试运行与审片、导出、训练接入、故障恢复   |
| [数据采集参考](docs/recording_reference.md)             | 采集架构、数据格式、时间语义、审片文件、转换模式、丢弃与修复、质量等级、实机验收 |
| [开发参考](docs/development.md)                         | 模块、数据约定、控制时序、设备接口、测试与实机验收                               |
| [Quest 客户端](quest_app/README.md)                     | 连接、追踪、APK 构建                                                             |
| [官方天机 SDK](bimanual_teleop/vendor/tianji/README.md) | 随包 SDK 来源与更新                                                              |
| [历史记录](docs/history/README.md)                      | 故障分析、迁移验证和设计计划，仅供追溯                                           |

## 安装与配置

已有环境用 `conda env update -n bimanual-teleop -f environment.yml` 更新后重新激活。天机官方 SDK 随项目提供，迁移时复制整个项目并创建环境即可，无需下载或编译。

| 文件                                  | 内容                                                                                                                   |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [天机配置](configs/tianji_teleop.yaml) | `controller_ip`、运动参数 `profile`、回位目标 `ready_pose`、参考系 `quest.coordinate_frame`、按键 `controls` |
| [Wuji 配置](configs/wuji_teleop.yaml)  | 左右设备地址`devices`、已标定用户名 `sdk_user_name`（空则用 SDK 默认用户）、Hand2 反馈频率 `feedback_hz`         |
| [采集配置](configs/recording.yaml)     | 相机序列号、帧缓冲`frame_capacity`、开录延迟 `start_delay_s`、录制按键                                             |

单位和数组顺序见 YAML 注释，修改后重启程序。换主机时，采集只需按内存调整 `frame_capacity`（默认 256 帧约 825 MiB）。

常用参数（完整列表见各入口 `--help`）：

| 参数                                                                | 适用入口                         | 作用                                                                                                                              |
| ------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--side left\|right\|both`                                          | 大多数入口                       | 选择机器人侧；点动、手套查看和标定只能选`left` 或 `right`，这三个入口和 Hand2 回零必须指定                                    |
| `--tianji-config`、`--wuji-config`、`--recording-config PATH` | 对应入口                         | 改用其他配置文件                                                                                                                  |
| `--robot-ip`（遥操作）、`--ip`（点动、回位、清错）              | 天机入口                         | 临时覆盖控制器 IP                                                                                                                 |
| `--sdk-root PATH`                                                 | 天机入口                         | 改用外置的同版本官方 SDK                                                                                                          |
| `--user-name NAME`                                                | 手套查看、手部遥操作、联合遥操作 | 临时选用已标定的 Wuji 用户                                                                                                        |
| `--serial SERIAL`                                                 | Quest 查看、联合遥操作           | 多台 ADB 设备时选择头显                                                                                                           |
| `--viewer`                                                        | 两个遥操作入口                   | 另开窗口显示全部 RealSense 彩色画面；与`--record` 同用时只显示三台采集相机（约 5 Hz）。无相机时提示后继续，关闭窗口不影响遥操作 |
| `-v` / `--verbose`                                              | 遥操作、天机回位                 | 显示调试日志、跟随受限提示和完整故障诊断；`NO_COLOR=1` 关闭颜色                                                                 |
| `--log-file PATH`                                                 | 联合遥操作                       | 指定运行日志路径（文件须不存在）                                                                                                  |

## 查看设备

除快速开始中的三个查看命令外，`python scripts/read_tianji_force.py` 读取腕部六维力（默认双臂）。同一设备同时只能被一个入口占用。

- 手套窗口：压力颜色是相对值，不是牛顿；`CONTACT UNKNOWN` 表示缺少有效接触信息。触觉来自手套，Hand2 Beta2 不提供触觉。
- 六维力：每侧约 0.2 s 输出一行，`F[N]` 为 Fx/Fy/Fz，`T[N·m]` 为 Tx/Ty/Tz，`raw` 为原始值；所选侧 3 s 无新帧或通道不匹配时报错退出。旧入口 `read_tianji_right_force.py` 仍可用，默认只读右臂。

## 遥操作与回位

除快速开始中的联合遥操作外，还有以下入口：

```bash
python scripts/teleop_quest_tianji.py --arms-only  # 只控制机械臂
python scripts/teleop_wuji_hand2.py --side both    # 只用手套控制 Hand2
python scripts/jog_tianji.py --side left           # 键盘点动
python scripts/home_tianji.py                      # 天机回 ready_pose
python scripts/clear_tianji_errors.py              # 只清错，不使能、不运动
python scripts/home_wuji_hand2.py --side both      # Hand2 回零，每侧默认 3 s
```

所有运动命令都要在交互终端按回车确认。天机遥操作和点动确认后先清错、回初始位姿，再等待接合；独立回位命令确认后同样先清错，回位后退出。

| 操作           | 按键或手势                                                                           |
| -------------- | ------------------------------------------------------------------------------------ |
| 接合／脱离     | Enter（天机入口可配置）；等待接合或回位中按下则取消。手部遥操作中 Enter 只开始／恢复 |
| 暂停、取消等待 | Space                                                                                |
| 停止跟随并回位 | H（可配置）；手势：暂停后任一手先呈非张开，再双手张开保持 1 s                        |
| 退出           | Q 或 Ctrl+C                                                                          |
| 手势开始／恢复 | 双手比 V 保持 0.3 s                                                                  |
| 手势暂停       | 任一手摇滚手势保持 0.3 s                                                             |
| 点动平移       | W/S、A/D、R/F：沿原生基座 X/Y/Z 正负方向，默认每键 5 mm                              |
| 点动旋转       | I/K、J/L、U/O：绕原生基座 X/Y/Z 正负方向，默认每键 2°                               |

- **按键配置**：天机配置 `controls` 的 `toggle_engagement_key`（默认 `"enter"`，也可为单个字母或数字）和 `ready_pose_key`（默认 `h`）不能用 Q/C/S/X；改成其他键后 Enter 仍可开始／恢复。按键需终端获得焦点。
- **手势**：仅联合模式可用，由 `gesture_engagement_enabled` 统一开关，当前配置为关闭（配置省略该项时为开启）；键盘始终可用。联合模式下接合、脱离同时作用于双臂和双手。
- **接合与暂停**：设备未就绪时按接合键会等就绪后自动接合。追踪丢失、关键反馈无效或控制器错误会暂停，排除后须重新接合；暂停时天机保持实测关节，重新接合以当前机器人和手柄位姿为基准。
- **回位（H）**：先停止双臂和双手跟随，再清错并移动所选机械臂到 `ready_pose`，灵巧手保持暂停；到位后保持脱离。回位中按 Space、接合键、摇滚手势、Q 或 Ctrl+C 中止；独立回位命令用 Ctrl+C 中止。
- **Hand2 回零**：双侧先左后右，每侧到位并去使能后继续，失败或中止则不做下一侧；单侧回零后保持零角，按 Q 退出。

### Quest 参考系与左右对应

`--side` 指机器人侧；Wuji 左手套对应左 Hand2、右手套对应右 Hand2。

`quest.coordinate_frame` 当前配置为 `world`：使用 Quest LOCAL 世界坐标，重新定位可能改变原点。`headset`（配置省略该项时的默认值）的原点和水平朝向跟随头显（忽略俯仰、侧倾），移动头显或转头也会改变手柄相对位姿。查看器始终显示 LOCAL 位姿。参考系的前、左、上映射到机器人相同物理方向，位姿以接合时为基准。

### 运行日志

`teleop_quest_tianji.py` 每次启动写一份 JSONL 运行日志，并在终端打印路径（默认 `logs/teleop_quest_tianji_<时间>_<进程>.jsonl`，相对启动目录）。日志包含启动参数与运行环境、警告和错误、录制故障和退出时的进程状态；每次暂停写入原因、设备诊断和故障前约 2 s 的控制周期快照。正常运行时不逐秒记录状态。排查暂停时请保留该文件。

## 手套标定

左右手分别标定，`NAME` 换成唯一用户名（不存在则创建，存在则更新）；完成后把用户名填入 Wuji 配置的 `sdk_user_name`。

```bash
python scripts/calibrate_wuji_glove.py --kind joints --side left --user-name NAME
python scripts/calibrate_wuji_glove.py --kind tactile --side left --user-name NAME
```

按终端引导完成动作，参考官方[关节标定](https://docs.wuji.tech/docs/en/wuji-studio/latest/calibration/)和[触觉标定](https://docs.wuji.tech/docs/en/wuji-studio/latest/tactile-calibration/)图示。触觉标定全程不得接触，要求 24×31 传感器数据。

## 常见问题

- **Quest 无追踪**：按手柄按键唤醒，放在头显摄像头视野内，关闭系统菜单；详见 [Quest 客户端](quest_app/README.md#连接与追踪)。
- **天机不能开始运动**：先排除控制器或伺服故障；启动清错失败时不会运动，可用 `clear_tianji_errors.py` 单独查看清错结果。
- **退出时提示停机未确认**：按下实体急停并检查设备。正常退出会下伺服并等待反馈确认，通信故障或强杀进程时可能无法完成。
- **分析 IK 报错**：`python scripts/analyze_tianji_ik.py failure.txt`，输入须含 `[IK诊断]` 或对应 JSON，离线运行。

## 测试

```bash
conda run -n bimanual-teleop python -m unittest discover -s tests -q
```

测试不连接设备；实机验收步骤见[开发参考](docs/development.md#测试与验收)。
