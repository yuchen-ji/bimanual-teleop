# 双臂机器人遥操作

Quest 手柄控制天机机械臂，Wuji Glove 控制 Hand2。

模块指南：[Quest 客户端](quest_app/README.md) · [官方天机 SDK](bimanual_teleop/vendor/tianji/README.md)。协议和接口见[开发参考](docs/development.md)。

## 安装

在项目根目录执行，Python 统一使用 Conda 环境 `bimanual-teleop`：

```bash
PIP_USER=false conda env create -f environment.yml
conda activate bimanual-teleop
```

已有环境使用 `conda env update -n bimanual-teleop -f environment.yml`，随后重新激活环境。环境禁用用户级 Python 包，避免版本冲突。

运行主机支持 Linux x86_64，需要 ADB。天机官方 Python SDK 和预编译库已随项目提供，无需编译。Quest 开启开发者模式，连接 USB，并在头显中授权调试：

```bash
adb devices -l
adb install -r quest_app/artifacts/quest-capture-debug.apk
```

迁移时复制完整项目并创建上述 Conda 环境即可；天机 SDK 不需要额外下载。只有自行构建 Quest APK 才需要其构建工具，见 [Quest 指南](quest_app/README.md#构建)。

## 配置

| 文件 | 需要设置的内容 |
| --- | --- |
| [天机配置](configs/tianji_teleop.yaml) | `controller_ip`、运动参数 `profile`、回位目标 `ready_pose`、参考系 `quest.coordinate_frame` |
| [Wuji 配置](configs/wuji_teleop.yaml) | 左右设备地址 `devices`、已标定用户名 `sdk_user_name`、Hand2 反馈频率 `feedback_hz`；空用户名使用 SDK 默认用户 |

参数单位和数组顺序见 YAML 注释，修改后重启程序。联合遥操作读取两份配置，可用 `--tianji-config PATH`、`--wuji-config PATH` 指定其他文件。手套查看和遥操作支持 `--user-name NAME` 临时选择已有用户。

## 查看设备

```bash
python scripts/view_quest.py
python scripts/view_wuji_glove.py --side left
python scripts/home_tianji.py --inspect
python scripts/read_tianji_force.py
```

查看命令分别运行，同一设备一次只运行一个入口。手套右侧使用 `--side right`；关闭窗口退出。压力颜色表示相对值，不是牛顿；`CONTACT UNKNOWN` 表示缺少有效接触信息，仍可显示压力。触觉来自手套，Hand2 Beta2 不提供触觉反馈。

腕部六维力默认同时读取双臂，可加 `--side left` 或 `--side right` 只读单侧（`--side both` 为双侧）。每侧约每 0.2 秒输出一次，行首标明 `left`／`right`，`F[N]` 为 Fx、Fy、Fz，`T[N·m]` 为 Tx、Ty、Tz，`raw` 为原始值；Ctrl+C 退出。任一所选侧连续 3 秒没有新帧或反馈通道不匹配时，报错并退出。原命令 `python scripts/read_tianji_right_force.py` 保留，默认只读右臂，也支持 `--side`。

所有天机入口默认使用随包官方 SDK；可用 `--sdk-root PATH` 指定配套的同版本官方目录。旧 `--library` 已移除。

## 遥操作与回位

```bash
# 双臂双手联合遥操作
python scripts/teleop_quest_tianji.py

# 只控制机械臂；单臂再加 --side left 或 --side right
python scripts/teleop_quest_tianji.py --arms-only

# 手套控制 Hand2：left、right 或 both
python scripts/teleop_wuji_hand2.py --side both

# 键盘点动左臂
python scripts/jog_tianji.py --side left

# 天机回配置中的 ready_pose；可加 --side 选择单臂
python scripts/home_tianji.py

# 只清错，不使能或运动；可加 --side
python scripts/clear_tianji_errors.py

# Hand2 依次回零，每侧默认 3 秒
python scripts/home_wuji_hand2.py --side both
```

运动命令需要交互终端。检查运动范围、释放实体急停并做好急停准备，按回车确认。天机遥操作和点动先清错、回初始位姿，再等待接合；手部遥操作连接就绪后等待接合；独立回位命令确认后直接回位。

| 操作 | 按键或手势 |
| --- | --- |
| 手部遥操作开始／恢复 | Enter |
| 天机遥操作接合／脱离 | Enter（可配置）；等待接合、接合中或回位中按下则取消 |
| 暂停／取消等待 | Space |
| 退出 | Q 或 Ctrl+C |
| 联合模式开始／恢复 | 双手同时比 V 保持 0.3 秒 |
| 联合模式暂停 | 任一手摇滚手势保持 0.3 秒 |
| 停止跟随并回位，到位后脱离 | H；联合模式已暂停时也可双手张开保持 1 秒（需开启手势控制） |
| 点动平移 | W/S、A/D、R/F：原生基座 X/Y/Z 正负方向，默认每键 5 mm |
| 点动旋转 | I/K、J/L、U/O：绕原生基座 X/Y/Z 正负方向，默认每键 2° |

在 [天机配置](configs/tianji_teleop.yaml) 的 `controls` 中，`toggle_engagement_key` 设置接合／脱离键（默认 `"enter"`，即回车，也可填单个字母或数字），`ready_pose_key` 设置停止跟随并回位的按键（默认 H）；终端需获得焦点。启动前的回车仍用于确认运动。`gesture_engagement_enabled: false` 关闭所有手势控制，包括比 V 接合、摇滚手势暂停和双手张开回位；键盘控制仍可用。联合模式的接合／脱离同步控制双臂与双手。

天机遥操作使用默认配置时，设备未就绪按 Enter 会等待就绪后接合，Space 或再次按 Enter 可取消；已接合时再次按 Enter 脱离。若切换键改为其他按键，Enter 保留开始/恢复功能，所配置的按键负责切换和取消。追踪丢失、关键反馈无效或控制器错误会暂停；排除原因后须重新接合。天机暂停时保持实测关节目标，重新接合使用当前机器人和手柄位姿建立基准。

接合时按 H 会先停止双臂与双手跟随，再清错并运动到 ready pose；未接合或已暂停时按 H 直接回位。无需再次按回车，到位后保持脱离，不自动恢复跟随。录制时按 H 会先结束当前录制。回位只移动所选机械臂，灵巧手保持暂停。手势回位需要暂停后任一手先明确呈非张开姿态，再双手张开；持续保持只触发一次。回位期间 Space、配置的接合/脱离键（默认 Enter）、摇滚手势（需开启手势控制）、Q 或 Ctrl+C 可中止。独立天机回位使用 Ctrl+C 中止。

Hand2 双侧回零按先左后右执行，每侧到位并去使能后继续，完成后退出；失败或中止不继续下一侧。单侧回零到位后保持零角，Q 退出。

两个遥操作命令均支持：

- `--viewer`：独立窗口显示启动时连接的全部 RealSense 彩色画面；无相机或设备占用时提示并继续。关闭窗口不影响遥操作，快捷键仍在终端输入。
- `-v` / `--verbose`：显示调试日志、跟随受限提示及完整故障诊断。默认仅输出关键状态和故障；`NO_COLOR=1` 关闭颜色。

双臂 Quest 遥操作每次启动都会创建详细的结构化运行日志，终端会打印绝对路径。默认位置为
`logs/teleop_quest_tianji_<时间>_<进程>.jsonl`，也可用 `--log-file PATH` 指定尚不存在的文件。
日志采用逐行 JSON，包含配置与进程信息、每秒设备状态、Wuji 原始新鲜度期限、录制进程状态、
主循环调度及处理器时间。正常控制周期先保存在内存中；发生暂停时才把故障前约 2 秒的逐周期
映射、左右臂求解、驱动提交、目标期限及看门狗停机快照一次性写入，避免日志写盘干扰 5 ms 控制循环。

### Quest 参考系与左右对应

`quest.coordinate_frame` 支持 `headset`（默认）和 `world`。`headset` 的原点和水平朝向跟随头显，忽略俯仰、侧倾；`world` 使用 Quest LOCAL 世界坐标，重新定位可能改变原点。查看器始终显示 LOCAL 位姿。

**左手柄控制右臂，右手柄控制左臂。** `--side` 指机器人侧；Wuji 始终左手套对应左 Hand2、右手套对应右 Hand2。参考系的前、左、上映射到机器人相同物理方向，位置和旋转以接合姿态为基准。在 `headset` 模式下，移动头显或改变其 yaw 也会改变手柄相对位姿。

## 数据采集

```bash
# 双臂双手遥操作，同时启用原始数据采集和共享相机预览
python scripts/teleop_quest_tianji.py --record --viewer

# 采集后分别导出两种动作空间；输出路径不得已存在
python scripts/convert_recording.py --input recordings/<session> --output datasets/episodes_eef.zarr --action-space eef
python scripts/convert_recording.py --input recordings/<session> --output datasets/episodes_joint.zarr --action-space joint
```

接合后按 **C** 开始、**S** 保存、**X** 作废当前条；手动暂停保存，故障暂停标记不完整。三路 RGB 为 640×480、30 Hz，只有主 D435 默认采深度；低维状态默认记录 200 Hz，控制目标沿用机械臂 200 Hz、手部 120 Hz。相机序列号及输出位置见 [采集配置](configs/recording.yaml)。

原始数据保持各流真实时间戳，离线统一到主 RGB 帧时间，导出 DP Zarr。字段、时间语义、恢复操作和训练接口见[数据采集指南](docs/data_collection.md)。已有环境补装依赖：`PIP_USER=false python -m pip install -e '.[recording]'`。

## 手套标定

左右分别标定，将 `NAME` 换成唯一用户名；首次创建用户，后续更新同名模型。完成后把用户名填入 Wuji 配置的 `sdk_user_name`。

```bash
python scripts/calibrate_wuji_glove.py --kind joints --side left --user-name NAME
python scripts/calibrate_wuji_glove.py --kind tactile --side left --user-name NAME
```

按终端引导完成动作。参考官方[关节标定图示](https://docs.wuji.tech/docs/en/wuji-studio/latest/calibration/)和[触觉标定图示](https://docs.wuji.tech/docs/en/wuji-studio/latest/tactile-calibration/)。触觉标定全程保持无接触，要求 24×31 传感器数据。

## 常见问题

- **Quest 无追踪**：拿起手柄按键唤醒，并放在头显摄像头可见范围；关闭系统菜单。Quest 3S 真正休眠后可能需要短按实体电源键，详见 [Quest 指南](quest_app/README.md#连接与追踪)。
- **天机不能开始运动**：先排除具体控制器或伺服故障；启动清错仍失败时不会运动。可用 `clear_tianji_errors.py` 单独检查清错结果。
- **退出停机未确认**：按实体急停并检查设备。正常退出会请求所控机械臂下伺服并等待新反馈；通信故障或强制结束进程可能使该流程无法完成。
- **离线分析 IK 报错**：运行 `python scripts/analyze_tianji_ik.py failure.txt`，输入须含 `[IK诊断]` 或对应 JSON；它不分析 `[控制诊断]`，也不连接设备。

## 测试

```bash
conda run -n bimanual-teleop python -m unittest discover -s tests -q
```

测试使用模拟设备、离线运动学和本机回环。实机验收按只读查看、单侧小幅运动、暂停与断流、双侧联合的顺序进行。
