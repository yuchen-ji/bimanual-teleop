# 验证记录

## 模型清理与 Git 资源复现（2026-10-14）

删除仿真目录中 227 个未使用或重复文件，目录磁盘占用从约 378 MiB 降至 173 MiB。保留 467 个必需源文件和运行资源，并加入 Git 跟踪；资源说明和 CAD / 网格的字节保留规则一并跟踪。采集数据、镜像和运行缓存继续忽略。

- `python bimanual_teleop/sim/run.py resources --check-git`：无缺失、无未引用、无未跟踪的必需资源。
- 从 Git 暂存区导出完整项目，在独立目录运行 `python bimanual_teleop/sim/run.py assets`：384 个派生文件全部与当前版本逐字节一致，无需聊天附件或项目外 CAD。
- 清理后实际运行 `python bimanual_teleop/sim/run.py --headless --steps 120`：正常完成，54 关节，退出状态 closed、tick=120、error=null。
- 项目测试 787 项通过；仿真测试共 31 项，30 项通过、1 项因主机缺少 Torch 跳过。

复现与清理报告保存在本机 `.runtime/resource-reproduce-report.json`、`.runtime/asset-cleanup-report.json`，测试日志为 `.runtime/resource-*-tests.log`；这些验证输出不进入 Git。日常启动命令和录制格式保持一致。

## 当前导轨任务 CAD（2026-10-14）

使用“导轨任务模型”中的长导轨、长导轨台、桌子、转运箱和腕部组件重新装配。实际 Isaac 验收报告为 [cad-validation/report.json](../data/cad-validation/report.json)，`passed=true`，属于验证输出，不是学习示范。

- 场景中有两侧各五层、共十个转运箱；机器人保持 54 个可动关节。
- 默认导轨落稳在长导轨台上，支撑高度误差约 0.59 mm；孔销对齐后在 1 mm 高度内释放，误差约 0.55 mm。20 mm 自由跌落不保证自行插入，未作为新 CAD 的通过条件。
- 左右传感器输入固定关节均能读到有限反力；相机刚性随动检查的最大位置误差小于 1 µm。
- 三路 640×480 RGB 均通过冻结物理时间渲染和移动可见方块检查，排除了重复旧帧。
- 完整可见导轨台约 112 万面；碰撞网格为 24 万面，24,244 个表面采样点的最大距离误差约 0.079 mm。长导轨的 299 个凸碰撞体总体积与源网格相符，定位孔保持空心。
- 项目回归测试 787 项通过；仿真测试共 30 项，29 项通过、1 项因主机缺少 Torch 而跳过；SDK 隔离导入测试另复查 3 项通过。

本轮三相机测得约 5.0 帧/墙钟秒、0.167 倍实时。240 / 120 / 30 Hz 仍是仿真时间频率，尚未达到三图像实时 30 FPS。默认查看模式不创建三路采集渲染产品。

本轮原图为 `data/cad-validation/settled_camera_0.png`、`settled_camera_1.png`、`settled_camera_2.png`，模型来源、装配顺序和质量假设见 [模型说明](task-models.md)。以下为替换前环境的历史记录，不替代当前 CAD 的验收结果。

## 2026-10-14：单命令原生界面与本地记录

日常入口改为 `python bimanual_teleop/sim/run.py`。在用户 Remmina / XRDP `:10` 中实际打开 Isaac 原生窗口，完整机器人、桌面、箱子与导轨台可见，54 关节状态有限且 tick 持续增加；默认不运行 server / client / 网页服务，也不创建三相机 RGB 渲染产品。[原生视图报告](../data/live/native-view-validation.json) · [原生窗口截图](../data/live/native-local-20261014.png)。

`--record --trajectory ... --steps 1440` 在同一进程中完成实际 PhysX → 三路 RGB → NVENC → MP4 / Zarr / B 点 → 离线导出：[本地链路报告](../data/live/native-record-validation.json) `passed: true`。**180 图像帧、1440 个物理状态、实际原生窗口 B 键确认在第 18 帧**；完整数据 180 帧，子任务 18 / 162 帧，action 为 180×52。三视频均可完整解码；实际 TCP 与独立 FK 最大位置差约 0.89 µm，执行 tick 不晚于采集 tick。轨迹明确标记 test_data，正常导出拒绝，验证导出显式允许测试数据；这不是完整人工任务示范。

仿真测试 **24 项，23 通过、1 跳过**；项目回归测试 **812 项通过**。日志见 `data/local-tests-20261014.log`、`data/local-regression-20261014.log`。跳过的仍是缺少 Torch 的实际训练消费者检查。

删除重复的 Three.js 全景实现、其资源安装入口和 8081 转发；外部设备分支保留为可选。补齐容器本地记录依赖，并将 pip 临时目录移到主机运行缓存，避免容器私有 64 MiB /tmp 在下载 PyAV 时耗尽。三相机同步与原始 / 导出格式继续复用原实现。

这次验证证明当前 Remmina 会话能显示和运行原生场景。2026-10-09 的黑屏截图与 NGX 报错不能作为 XRDP 不兼容或原生界面不可用的证据。下面保留历史检查结果；旧浏览器入口已移除。

## 历史记录：2026-10-09

日期：2026-10-09。新增代码、配置、文档、模型与运行输出均位于 `bimanual_teleop/sim`。真机入口、驱动、配置、记录模块以及 diffusion_policy 均未修改；根目录 `AGENTS.md` 的已有用户修改保留。

## 软件与原有项目

- 原有项目 `python scripts/run_tests.py discover -s tests -q`：**787 项通过**，输出见 `sim/data/regression-tests.log`。
- 仿真 `python scripts/run_tests.py discover -s bimanual_teleop/sim/tests -v`：**22 项中 21 项通过、1 项跳过**。跳过的是实际 TianjiRailImageDataset 读取检查，因为当前主机 Conda 未安装训练运行时 Torch；未运行策略训练。
- 无第三方依赖、无 SDK、无设备访问的包导入与注解解析检查通过。
- 导轨网格闭合、表面方向一致、体积与面积匹配；70 个碰撞凸体没有填满两个通孔。54 关节 FK／Jacobian 与独立有限差分一致，左右肩原点及手指朝外安装方向通过校验。
- 四时间戳时钟同步、过期输入拒绝、隐藏设备故障、原点变化、重复事件与共享内存边界通过。输入采用只传最新值的队列，跳过中间样本不会误判为设备故障；真实过期与客户端故障计数变化仍锁定暂停，需 Enter 重新接管。
- 网络测试覆盖 B 点重试去重、暂停时拒绝 B、恢复记录、异步保存及继续控制。整任务窗口可以跨 B，子任务窗口按离线范围切断；实际暂停／缺帧即使合并范围后仍切断窗口。未审核标签和测试数据默认拒绝导出。

## 官方引擎与真实物理

用户目录内 Apptainer 1.5.4 运行 NVIDIA 官方 Isaac Sim 5.1.0 镜像，识别 RTX PRO 6000 Blackwell、97887 MiB、驱动 580.95.05。容器额外包及 Kit 缓存写入 4 GiB overlay，项目挂载为只读，图像共享内存映射与主机一致。

[物理报告](../data/isaac-validation/report.json) 中 `passed: true`，检查包括：

- 14 臂关节与 40 手关节有限、在限制内。物理末端与 URDF FK 的位置偏差小于 1 µm，角度偏差小于 2 µrad。
- 料箱内导轨落稳，箱底高度误差约 **17 µm**。
- 三相机返回 640×480 uint8 RGB。渲染不推进物理时间；移动可见测试方块后三路像素位置均发生预期变化，排除了重复静止图像。
- 物理运动腕部关节前后，渲染相机的世界变换均与独立 FK 相符；两个腕部相机确实随手安装链接运动。
- 两孔对齐后，释放导轨的支撑高度误差约 **29 µm**；对称偏航 **8°** 时被定位销挡住，最终高出支撑高度约 **9 mm**。几何检查与真实接触检查共同验证通孔。

截图位于 `sim/data/isaac-validation/camera_0.png`、`camera_1.png`、`camera_2.png`；定位试验为 `fixture_probe_aligned.png` 与 `fixture_probe_misaligned.png`。

复现：

```bash
bimanual_teleop/sim/docker/exec-apptainer.sh /isaac-sim/python.sh /workspace/bimanual-teleop/bimanual_teleop/sim/run.py validate_isaac
```

## 引擎到学习格式的链路

[端到端报告](../data/isaac-smoke-d8885202/report.json) 中 `passed: true`。实际 Isaac 引擎 → 共享内存 → 主机桥接 → 三路 NVENC H.264 → 保存已跑通：**60 图像帧、475 个物理状态、B 点位于第 30 帧**。随后解码三视频并导出完整任务 60 帧，以及两个 30 帧的测试子任务；验证了 **52D action**、三相机数组、采样边界、执行 tick 和同一源示范 UUID。该录制使用合成输入，明确标记为测试数据，正常导出不会把它当成人工示范。

复现：

```bash
python bimanual_teleop/sim/run.py smoke
```

## 历史浏览器查看入口（已移除）

2026-10-09 曾通过 8090 浏览器页面加载约 110 万三角面的完整 CAD，并验证其跟随实际物理状态。2026-10-14 原生界面验证通过后，这份重复显示实现已删除，不再作为启动入口。

2026-10-14 复查更正：原生界面日志 `data/live/engine-gui.log` 第 431 行记录 `Simulation App Startup Complete`，第 504 行记录 `Isaac scene ready; GUI viewport; 54 joints, three cameras`。此前“停留于渲染初始化”的表述证据不足；截图只反映当时未显示场景，不能证明后续始终黑屏。`Failed to create NGX context` 在成功的 headless 日志中也出现，不能单独认定为原生显示问题的原因。尚未通过空场景、显示会话与容器图形链路的隔离测试定位原因。

[浏览器验证报告](../data/live/scene-viewer-validation.json) `passed: true`：54 个关节实时跟随 Isaac；两个 TCP 的浏览器世界矩阵与独立 Python FK 一致，导轨位置／四元数与物理状态一致；鼠标旋转和“回到全景”按钮均验证通过，没有 JavaScript 错误。[场景截图](../data/live/scene-3d.png)。这是查看界面，物理仍由 Isaac 执行，三路采集相机保持原有时序与入口。

新增 `GET /pose` 只提供实际关节、导轨和 tick；查看器不提供机械臂控制接口。修复了引擎重连时把提前到达的心跳误当作共享内存初始化消息的问题。仿真模块 22 项测试仍为 21 通过、1 跳过。

## 性能与待完成的人工验收

配置的物理 **240 Hz**、控制 **120 Hz**、相机 **30 Hz** 是仿真时间频率。共享 GPU 环境多次物理验证实测 **5.5–7.4 帧／墙钟秒、0.18–0.25 倍实时**；最后一次约 5.52 FPS、0.184 倍实时。图像／状态仍在同一仿真 tick 对齐，但**尚未达到实时 30 FPS**。正式长任务采集前需要优化运行速度并测量人工操控延迟；没有中断服务器上的其他 GPU 作业。

Wuji SDK 登录、两只手套、Quest ADB 串流与脚踏板需要在操作员电脑完成实际联调。合成输入不能证明这些设备已实测连接。完整的双手抓持、单手承重、两孔插接释放和移入成品箱还需人工遥操验收，不能用定位落体试验代替完整任务成功。

导轨质量 0.8 kg、孔径 10 mm、相机 90×25×25 mm 外壳、70° 视场与安装外参是待实测校准初值；D435i SLDPRT 尚未转换为精确相机 CAD。缺失／零质量坐标链接使用清单中记录的 1 g 近似。已有六维数据是固定手安装关节反力，包含重力和惯性；深度、实际关节扭矩与手指触觉尚未实现。
