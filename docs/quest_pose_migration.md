# Quest 三路位姿采集与可视化迁移到新电脑

在 Ubuntu 新电脑上建立独立工程，采集 Quest 头显和左右 Touch 手柄的位姿，通过 USB ADB 传输，在电脑的同一个交互式 3D 窗口显示位置、朝向和追踪状态。

本文是可直接交给新电脑上 AI 的实施规范，不要求复制原仓库、原 APK 或原电脑缓存。Android 端、Python 端和自动测试由 AI 按本文重新建立。下文的 `quest_pose` 命令和新包名属于目标工程，须在实现完成后使用。

## 1 给新电脑上的 AI 的任务

将本文作为附件，并发送以下指令：

```text
请在一个空目录中，按照这份文档实现完整的 Quest 三路位姿采集与可视化工程。
目标系统是 Ubuntu 22.04/24.04 x86_64，设备是 Quest 3 或 Quest 3S 及左右 Touch 手柄。
实现 Android 原生 OpenXR 采集端、USB ADB JSON 接收端、Python 三路 3D 可视化和自动测试。
第三方依赖从本文列出的公开来源获取，不依赖原项目、原 APK 或其他电脑上的缓存。
Python 的运行、安装和测试统一在 Conda 环境 bimanual-teleop 中进行。
保持指定版本、协议、坐标变换、追踪判定、参考空间切换和退出清理规则。
交付可安装 APK、源码、环境文件、操作说明和验证记录，不要只生成另一个计划。
可以自主完成文件创建、依赖配置、构建和无设备测试。
需要头显操作时说明具体步骤；没有真实设备时，将实机项目标记为待验收。
```

### 功能边界

- 采集头显 VIEW 位姿和左右手柄 grip 位姿，三路均相对 LOCAL 空间。
- 请求 90 Hz；实际采样频率和运行时刷新率分别报告。
- 三路可视化约 30 Hz，显示位置、XYZW 四元数、坐标轴和追踪状态。
- 处理手柄未唤醒、追踪丢失、系统菜单、重新定位、收流停顿、USB 断开和退出。
- 不引入机器人、手套、相机、录制、滤波、Unity、ROS 或网络传输模块。

## 2 新电脑的环境和工程结构

新电脑需要本地图形桌面、Conda、JDK、Android SDK，以及能传输数据的 USB 线。设备开启开发者模式，并在头显中批准新电脑的 USB 调试授权。

### 固定版本

| 部分 | 版本或配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04/24.04 x86_64 |
| Python | 3.10，Conda 环境 `bimanual-teleop` |
| Python 依赖 | NumPy 1.26.4、Matplotlib 3.10.9、Tk |
| Java | JDK 17 |
| Gradle | 8.5 |
| Android Gradle Plugin | 8.1.4 |
| Android SDK | compileSdk 32、targetSdk 32、minSdk 26 |
| Build Tools | 33.0.1 |
| Android NDK | 27.0.12077973 |
| Android CMake | 3.22.1 |
| 原生配置 | C++20、`c++_shared`、仅 `arm64-v8a` |
| OpenXR loader | `org.khronos.openxr:openxr_loader_for_android:1.1.53` |

先按固定版本搭建。Android command-line tools 和 platform-tools 不属于上述固定构建版本；安装时记录其版本。若新版 `sdkmanager` 要求更新的 Java，可用它所要求的 Java 安装 SDK，但运行 Gradle 构建时必须切回 JDK 17。

### 独立工程结构

```text
quest-pose/
  README.md
  environment.yml
  pyproject.toml
  quest_pose/
    __init__.py
    __main__.py
    protocol.py
    source.py
    viewer.py
  quest_app/
    settings.gradle
    build.gradle
    gradle.properties
    CMakeLists.txt
    gradlew
    gradlew.bat
    gradle/wrapper/
    src/main/AndroidManifest.xml
    src/main/java/org/questpose/capture/MainActivity.java
    src/main/cpp/quest_capture.cpp
    src/main/assets/notices/
  tests/
  artifacts/
    quest-pose-debug.apk
    SHA256SUMS
    VERIFICATION.md
```

`artifacts` 是构建后生成的交付目录，不预置原项目的 APK。文档和构建脚本不能写死原电脑的用户名、SDK 路径或缓存路径。

### Python 环境

生成以下最小 `environment.yml`：

```yaml
name: bimanual-teleop
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - tk
  - pip:
      - numpy==1.26.4
      - matplotlib==3.10.9
      - -e .
variables:
  PIP_USER: "false"
  PYTHONNOUSERSITE: "1"
```

`pyproject.toml` 使用 setuptools，仅发现 `quest_pose*` 包；运行依赖仅为上述 NumPy 和 Matplotlib。不要添加原遥操作工程的依赖。安装和运行：

```bash
# 在新工程根目录执行
PIP_USER=false conda env create -f environment.yml
conda activate bimanual-teleop
python -m quest_pose --help
```

若环境已存在，先确认其用途，再用 `conda env update -n bimanual-teleop -f environment.yml` 更新，不删除既有环境。禁止使用系统 Python、`pip --user` 或 `sudo pip` 安装本工程依赖。

### Android SDK 和 USB 权限

从 [Android 官方下载页](https://developer.android.com/studio#command-tools)安装 Android SDK command-line tools，或使用 Android Studio 的 SDK Manager。`ANDROID_HOME` 指向新电脑实际的 SDK 目录；`JAVA_HOME` 指向 JDK 17；把 SDK 的 `platform-tools` 和 command-line tools 的 `bin` 加入 PATH。

安装所需组件，并按提示接受许可证：

```bash
sdkmanager --licenses
sdkmanager 'platform-tools' 'platforms;android-32' \
  'build-tools;33.0.1' 'ndk;27.0.12077973' 'cmake;3.22.1'
java -version
adb version
adb devices -l
```

Ubuntu 若出现 `no permissions`，检查 `plugdev` 组和 USB udev 规则。可安装 `android-sdk-platform-tools-common`；需要时将当前用户加入 `plugdev`，重新登录后再连接设备。不要以 root 长期运行查看器。[Android 官方设备连接说明](https://developer.android.com/studio/run/device)

## 3 Quest 端的实现要求

### 工程身份和依赖来源

统一使用：

```text
Android package / namespace: org.questpose.capture
Activity: org.questpose.capture/.MainActivity
Native library: questcapture
logcat tag: QuestCapture
rootProject.name: QuestPoseCapture
```

新包名允许与旧采集程序共存，避免新电脑的 debug 签名与旧 APK 冲突。Python 端和安装、诊断命令必须使用同一新包名。

Meta OpenXR SDK 固定为以下源码，不获取浮动分支：

```text
commit: bbed2f20e38a5df7113630771c83cb8279e4fc26
URL: https://codeload.github.com/meta-quest/Meta-OpenXR-SDK/tar.gz/bbed2f20e38a5df7113630771c83cb8279e4fc26
SHA-256: 0edf48d28b2e87e826d398d839d9cbeb3d0386b0a29a1b3bbfabda4d3e2255bf
```

Gradle wrapper 固定 8.5，下载地址为 `https://services.gradle.org/distributions/gradle-8.5-bin.zip`，`distributionSha256Sum` 为：

```text
9d926787066a081739e8200858338b4a69e837c3a821a33aca9db09dd4a41026
```

获取并校验官方 Gradle 分发包后生成完整 wrapper，交付 wrapper JAR 和脚本。Gradle 仓库使用 Google Maven、Maven Central 和 Gradle Plugin Portal；开启 Prefab，依赖指定的 OpenXR loader；配置 NDK、ABI、CMake 和共享 C++ runtime。

CMake 的核心配置为：

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(questcapture LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
    meta_openxr_sdk
    URL https://codeload.github.com/meta-quest/Meta-OpenXR-SDK/tar.gz/bbed2f20e38a5df7113630771c83cb8279e4fc26
    URL_HASH SHA256=0edf48d28b2e87e826d398d839d9cbeb3d0386b0a29a1b3bbfabda4d3e2255bf
)
FetchContent_MakeAvailable(meta_openxr_sdk)
add_subdirectory(
    ${meta_openxr_sdk_SOURCE_DIR}/Samples/3rdParty
    ${meta_openxr_sdk_BINARY_DIR}/3rdParty
)
add_subdirectory(
    ${meta_openxr_sdk_SOURCE_DIR}/Samples/SampleXrFramework
    ${meta_openxr_sdk_BINARY_DIR}/SampleXrFramework
)
add_library(questcapture SHARED src/main/cpp/quest_capture.cpp)
target_compile_definitions(questcapture PRIVATE XR_USE_TIMESPEC)
target_link_libraries(questcapture PRIVATE samplexrframework)
target_link_options(questcapture PRIVATE "-uANativeActivity_onCreate")
```

保留 Meta SDK、OpenXR loader、KTX 及其他打包依赖的许可证和 notices，随源码和 APK 一并交付。

### Activity 和 Manifest

Java Activity 继承 `NativeActivity`，从 Intent 读取 `session_id`，要求为 32 位十六进制 UUID；没有参数时生成 UUID 并移除连字符。必须在 `super.onCreate()` 前保存 session，再提供 `getSessionId()` 供 JNI 读取。

Manifest 必须包含：

- OpenGL ES 3.1 和 VR headtracking 为必需特性。
- `oculus.software.handtracking` 为可选，声明 `com.oculus.permission.HAND_TRACKING`，沿用允许手柄未唤醒时启动等待的配置。
- `com.oculus.supportedDevices=all`。
- Activity 为 `singleTask`、横屏、全屏、`exported=true`，正确声明配置变更。
- 原生库元数据 `android.app.lib_name=questcapture`。
- MAIN、LAUNCHER 和 `org.khronos.openxr.intent.category.IMMERSIVE_HMD`。

可选手部追踪声明只用于启动兼容；不得输出手部追踪结果作为手柄位姿。

### OpenXR 生命周期与位姿查询

采集类继承 Meta SDK 的 `OVRFW::XrApp`。沿用框架的 OpenXR 初始化、Android 生命周期、动作同步和 XR 帧提交。参考[固定版本框架源码](https://raw.githubusercontent.com/meta-quest/Meta-OpenXR-SDK/bbed2f20e38a5df7113630771c83cb8279e4fc26/Samples/SampleXrFramework/Src/XrApp.cpp)。

- 在基础扩展列表上添加 `XR_KHR_convert_timespec_time`、`XR_FB_display_refresh_rate`，加载所需函数；必要扩展或函数不可用时明确报错并终止。
- 清空初始场景 URI，设置黑色背景和 `0.25` 渲染分辨率比例，保留正常 XR 帧循环。
- 三路共同参考空间是 `LocalSpace`。头显查询 `HeadSpace` 即 VIEW 空间，手柄查询左右 `ControllerGripSpace`。
- 保留 Touch interaction profile，移除 `/interaction_profiles/khr/simple_controller` 回退绑定。
- Session 初始化时请求 90 Hz，读取实际刷新率并输出事件；请求或查询失败时输出错误事件并终止。

框架动作同步后，每次 `Update()` 按以下顺序采集：

```text
读取一次 CLOCK_MONOTONIC → query_ns
xrConvertTimespecTimeToTimeKHR → xr_time
处理截至 xr_time 已生效的 LOCAL 原点变化
读取左右 grip pose action 的 isActive
用同一个 xr_time，分别执行三次 xrLocateSpace：
  HeadSpace → LocalSpace
  LeftControllerGripSpace → LocalSpace
  RightControllerGripSpace → LocalSpace
输出一条包含三路结果的 frame JSON
```

三路共用查询时间。不要分别取时，也不要用渲染预测时间替换本协议的查询时间。[空间查询规范](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrLocateSpace.html)、[时间转换规范](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrConvertTimespecTimeToTimeKHR.html)

保留每路原始 flags，位置和朝向分别判断是否有效。缺失部分输出 `null`；单路查询失败时将该路 flags 清零，其他路继续输出。

LOCAL 原点变化事件到来时，立即分配并报告新 origin，将 `(changeTime, origin)` 加入排序队列；只有查询时间达到生效时间才切换 frame 的 origin。origin 初始值为 0，不根据编号大小判断生效先后。

## 4 两端通信协议和坐标约定

### JSON Lines 消息

用 `__android_log_write(ANDROID_LOG_INFO, "QuestCapture", ...)` 输出，每条消息占一行，不附加文本前缀。使用固定小数点格式，不输出 NaN 或 Infinity，单条消息限制在 2 KiB 内。

frame 示例：

```json
{
  "v": 1,
  "type": "frame",
  "session": "0123456789abcdef0123456789abcdef",
  "seq": 0,
  "origin": 0,
  "query_ns": 1000000000,
  "xr_time": 2000000000,
  "send_ns": 1001000000,
  "state": 5,
  "refresh_hz": 90.0,
  "head": {"p": [0.0, 1.5, 0.0], "q": [0.0, 0.0, 0.0, 1.0], "flags": 15, "active": null},
  "left": {"p": [-0.3, 1.0, -0.5], "q": [0.0, 0.0, 0.0, 1.0], "flags": 15, "active": true},
  "right": {"p": [0.3, 1.0, -0.5], "q": [0.0, 0.0, 0.0, 1.0], "flags": 15, "active": true}
}
```

示例为了阅读而分行；实际输出必须压缩为单行 JSON。

| 字段 | 约定 |
| --- | --- |
| `v`、`type` | 版本固定为整数 1；消息类型为 frame 或 event |
| `session` | 本次启动的 UUID hex，两端必须一致 |
| `seq` | 初始为 0，每输出一帧递增一次 |
| `origin` | LOCAL 空间代次，初始为 0 |
| `query_ns`、`send_ns` | 设备 CLOCK_MONOTONIC 纳秒；send 不早于 query |
| `xr_time` | 转换后的 OpenXR 查询时刻，保留整数 |
| `state` | OpenXR session state；5 为 FOCUSED |
| `refresh_hz` | 实际运行时刷新率 |
| `p` | 米，XYZ 三分量或 null |
| `q` | XYZW 四元数或 null |
| `active` | 头显必须为 null，手柄必须为布尔值 |

flags 位定义：1 为朝向有效，2 为位置有效，4 为朝向 tracked，8 为位置 tracked。`valid` 要求位 1、2 均存在；`tracked` 要求位 4、8 均存在。不能把 valid 当成完整 tracked。

event 示例：

```json
{
  "v": 1,
  "type": "event",
  "session": "0123456789abcdef0123456789abcdef",
  "event": "session_state",
  "device_ns": 1000000000,
  "details": {"state": 5, "time": 2000000000}
}
```

| event | details 必需字段 |
| --- | --- |
| `session_state` | `state`、`time` |
| `refresh_rate` | `from_hz`、`to_hz` |
| `error` | `operation`、有符号整数 `result` |
| `reference_space_change` | `origin`、`change_time`、`pose_valid`、`pose_in_previous_space` |

`pose_in_previous_space` 含原始 OpenXR 坐标的 p 和 q；`pose_valid=false` 时二者均为 null。

### 坐标换基与时间

Android 输出原始 OpenXR 右手坐标：X 向右、Y 向上、Z 向后。Python 接收器将三路位置和旋转统一转换到 X 向前、Y 向左、Z 向上的 FLU 右手坐标：

```text
p_flu = (-p_z, -p_x, p_y)
q_flu = (-q_z, -q_x, q_y, q_w)

B = [[ 0, 0,-1],
     [-1, 0, 0],
     [ 0, 1, 0]]

R_flu = B · R_openxr · Bᵀ
```

父、子坐标轴同时换基，可视化层不再转换。三路共享 `(session, origin)` 标识的 LOCAL 参考空间；LOCAL 不保证位于地面或朝向地理北方，重新定位后的坐标不能直接与前一代拼接。

电脑记录首次读到该帧的 `received_monotonic_ns=time.monotonic_ns()`。设备时钟、XrTime 和电脑时钟分别保留，不用设备时间与电脑时间相减报告延迟。query 是位姿查询时刻，不是光学传感器曝光时刻。

## 5 Python 接收器和三路可视化

### 接口和生命周期

定义不可变 `QuestPose`、`QuestFrame` 数据类型。QuestPose 保留位置、XYZW 四元数、flags、active 及各项有效性属性；QuestFrame 保留 frame 协议字段和主机接收时间。

```text
QuestSource(serial=None, keep_awake=True)
  start()
  get_latest() -> QuestFrame | None
  health() -> ready 与原因
  close()

QuestPoseView
  update(frame, health)
```

模块导入不连接设备、不启动 ADB、不创建窗口。CLI 为 `python -m quest_pose`，支持 `--serial SERIAL`、`--no-keep-awake`、`--help`。

启动顺序：

1. 用 `adb devices -l` 选择已授权的 USB Quest，排除网络设备；多台必须指定 serial。
2. 检查采集 APK 已安装；检查同包名是否已有进程，已有会话则报错，不停止它。
3. 创建新 UUID session，重置接收状态。
4. 默认发送 KEYCODE_WAKEUP，再发送 `com.oculus.vrpowermanager.prox_close` 广播。
5. 启动 logcat 子进程及后台接收线程，确认进程未提前退出。
6. 启动 Activity 并传入 session_id；检查返回码和 `am start` 的错误输出。

核心 ADB 命令：

```bash
adb -s SERIAL logcat -v raw -T 1 QuestCapture:I '*:S'
adb -s SERIAL shell am start -n org.questpose.capture/.MainActivity \
  --es session_id SESSION_ID
```

先读日志再启动 APK，避免漏掉初始化事件。raw 格式输出消息正文，标签过滤只接收协议来源。[logcat 官方说明](https://developer.android.com/tools/logcat)

Python 使用 subprocess 参数列表，不拼接 shell 命令。短时 ADB 调用设置 10 秒超时。

### 解码和异常状态

后台逐行读日志，首次读到时记录主机时间；严格验证版本、字段类型、数组长度和数值。整数不能接受布尔值，数值必须有限，四元数模长平方与 1 的差不超过 0.001；无效位置或朝向必须是 null。

- 空行和 logcat 分隔行跳过，其他 session 的消息忽略。
- seq 和 query_ns 必须严格递增；重复或倒序帧丢弃并报告。
- 缺帧报告缺失序号范围，继续处理后续正常帧。
- 损坏消息丢弃，不生成替代位姿；正常新帧恢复可恢复的数据状态。
- 参考空间事件到来时清空 latest；按 xr_time 检查新 origin 的生效时间。事件缺失或时间不匹配时保留诊断。
- USB EOF、运行时错误为终止性状态，保留具体原因，不静默重连；用户重启程序创建新 session。

只缓存最新帧，不建立 GUI 回放队列。所有跨线程共享状态用锁保护。重复 get_latest 不修改接收时间、不产生新样本。

health.ready 要求接收器运行、没有未解决错误或待处理原点事件、1 秒内有新帧、会话为 FOCUSED、三路 valid/tracked、手柄 active=true、实际刷新率约 90 Hz（容差 0.01 Hz）。全局未就绪不应遮蔽其他独立可用对象；显示阻断原因须另行明确。

### 显示规则

GUI 主线程每 33 ms 读取最新帧，复用现有绘图对象。三路均显示位置点、HEAD/LEFT/RIGHT 标签和 0.18 m 长的 XYZ 正轴，轴色分别为红、绿、蓝，各对象标记色可区分。

窗口显示三路位置、XYZW 四元数、追踪状态，以及全局 session state、origin、seq、实际 XR 刷新率和接收状态。坐标标签为 `X forward (m)`、`Y left (m)`、`Z up (m)`，三轴等比例。

将已换基四元数归一化后生成旋转矩阵 R，每根轴绘制 `p → p + 0.18·R[:,i]`。

| 条件 | 行为 |
| --- | --- |
| valid 且 tracked | 实线显示 |
| valid 但未完整 tracked | 淡色虚线，标注 valid, NOT tracked |
| 某一路无效 | 隐藏该路，其他路继续显示 |
| 手柄 active=false | 隐藏该手柄；头显不检查 active=true |
| 无数据、收流超 250 ms、失焦、原点切换中或终止性错误 | 隐藏全部位姿并显示原因 |

250 ms 和 1 秒阈值都是主机收流停顿，不是源数据年龄。失焦及终止性事件应立即反映在显示状态中，不能仅依赖最后一帧的旧 state。

支持鼠标旋转和缩放、F 适配全部可见对象。只在首帧、参考空间变化或按 F 时调整视野范围，保留用户视角。

### 退出清理

关闭窗口、Ctrl+C 和启动失败进入统一 finally 清理：停止 GUI timer，终止或必要时 kill logcat，回收线程和管道，仅停止本程序拥有的 APK 会话，再恢复休眠检测。

请求电源覆盖前记录清理标志，使广播超时也能尝试恢复。close 幂等；恢复失败保留标志以允许重试，并输出手动恢复命令。已有外部会话不得被唤醒、停止或修改电源状态。

正常退出码为 0，启动失败为 1；默认输出可读原因，不输出冗长异常堆栈。

## 6 构建 安装与日常运行

以下操作在目标工程实现后进行。设置好 JAVA_HOME 和 ANDROID_HOME，在 Android 工程目录执行：

```bash
./gradlew --no-daemon --console=plain assembleDebug lintDebug
```

将实际构建的 debug APK 复制到工程根目录的 `artifacts/quest-pose-debug.apk`。执行 `apksigner verify --verbose` 验证签名，检查包内原生库仅为 arm64-v8a，并生成 SHA256SUMS。README 必须记录实际构建产物路径，不能猜测文件名。

在工程根目录安装并运行：

```bash
adb devices -l
adb -s SERIAL install -r artifacts/quest-pose-debug.apk
conda activate bimanual-teleop
python -m quest_pose --serial SERIAL
```

SERIAL 替换为设备序列号。只有一台已授权 USB Quest 时可省略 --serial。唤醒头显和手柄，保持手柄在头显摄像头可见范围，关闭系统菜单、权限或边界确认窗口；关闭电脑窗口退出。

调试签名随电脑变化：首次用独立新包名通常不会与旧应用冲突；以后换电脑重建同一包名时，可能出现 `INSTALL_FAILED_UPDATE_INCOMPATIBLE`。需使用相同签名密钥，或在明确接受应用数据丢失后卸载该采集应用再安装；不要自动卸载。

若 APK 残留或 USB 重连后休眠行为异常，确认当前没有正在使用的采集会话，再执行：

```bash
adb -s SERIAL shell am force-stop org.questpose.capture
adb -s SERIAL shell am broadcast -a com.oculus.vrpowermanager.automation_disable
```

## 7 测试与迁移验收

实施顺序为：协议与坐标测试 → Android 输出 → ADB 接收生命周期 → 合成帧可视化 → 实机三路追踪与恢复 → 交付。

### 无设备测试

使用 unittest，图形测试采用 Agg 后端。测试导入、解码和图形几何时不能触碰真实设备。

```bash
conda run -n bimanual-teleop python -m unittest discover -s tests -q
```

必须覆盖：

- 编译真实采集 C++ 配合 XR 测试替身，验证三路共用查询时刻、seq、flags、active、刷新率失败和原点生效边界；其 JSON 输出交给真实 Python 解码器。
- `[1,2,3] → [-3,-1,2]`，单位四元数和非零旋转的换基，绘制轴方向及无重复换基。
- 部分位置或朝向失效、valid/tracked 区分、头显 active=null、单路失效不遮蔽其他路。
- 错误版本、缺字段、截断 JSON、非有限数值、错误数组长度、非单位四元数。
- 旧 session、重复或倒序帧、缺帧、参考事件缺失、不同生效时间和原点编号顺序。
- 250 ms 隐藏、1 秒收流异常、失焦、原点切换、数据恢复、终止性错误不自动恢复。
- ADB 多设备和权限状态、已有会话保护、启动失败、USB EOF、关闭窗口、幂等清理和电源恢复。
- get_latest 不重新计时；接收持续运行时 GUI 不积压旧帧。

### 构建验收

- [ ] 在干净工程目录下载公开依赖并构建成功，SDK 和 Gradle 校验值一致。
- [ ] assembleDebug 和 lintDebug 完成，lint 错误为零，警告记录具体原因。
- [ ] APK 签名验证成功，包名正确，原生库仅为 arm64-v8a，许可证齐全。
- [ ] Python 安装后可脱离源码目录导入和查看 --help。
- [ ] 无设备时明确说明原因，返回失败退出码，无长异常堆栈。

### 实机验收

记录设备型号、Horizon OS 版本、USB 直连或集线器、新电脑系统和工具版本。没有设备时以下项目保持待验收：

- [ ] 三路位姿和 XYZ 正方向与实际动作一致，左右对应正确。
- [ ] 正常聚焦的稳定 60 秒片段，源帧平均频率为 88–92 Hz；分别统计源查询间隔、主机到达间隔和缺失序号。
- [ ] 固定手柄后移动头显，手柄保持 LOCAL 位姿，不产生头显相对运动。
- [ ] 单路遮挡时其他路继续显示，失效和 valid 但未 tracked 的表现正确。
- [ ] 手柄未唤醒时可启动等待，不把手部输入当成手柄。
- [ ] 系统菜单、休眠、重新定位的状态和 origin 可追溯，恢复后显示新数据。
- [ ] USB 拔出时隐藏位姿并报告原因；窗口退出无接收进程残留。
- [ ] 退出后恢复休眠检测；失败时给出可执行的恢复命令。
- [ ] 连续运行 30 分钟，内存及缓存不随运行时间持续增长。

统计时令 `Δt = (last.query_ns - first.query_ns) / 1e9` 秒、N 为收到的不同正常帧数。接收样本频率为 `(N - 1) / Δt`；源序号跨度频率为 `(last.seq - first.seq) / Δt`，同时报告缺失序号数、查询和到达间隔的最大值。不能仅凭刷新率字段或零序号缺口宣布稳定 90 Hz；采集停顿后序号也可能连续。

当前来源实现的 38 项相关自动测试已通过，仅作为协议和行为基线；不代表目标独立工程、新电脑或三路显示扩展已通过验收。

## 8 常见问题与交付清单

| 现象 | 检查和处理 |
| --- | --- |
| ADB 无设备 | 确认数据线、USB 接口、开发者模式、授权和 udev 权限 |
| unauthorized | 在头显内批准新电脑的 USB 调试 |
| offline 或间歇断流 | 检查线缆和集线器，优先直连；重启采集程序建立新 session |
| APK 启动但无追踪 | 关闭菜单和边界窗口，确认应用聚焦、手柄唤醒并可见 |
| Quest 3S 亮屏但仍失焦 | 真休眠后可能需短按实体电源键解除传感器锁 |
| 只有一只手柄显示 | 查看该侧 active 和 flags，不将无效位姿补成零值 |
| 窗口不弹出 | 确认本地图形桌面、Conda Tk 和交互式 Matplotlib backend |
| 运动方向错误 | 检查 FLU 换基是否正确且只执行一次，四元数顺序是否为 XYZW |
| 重新定位后坐标跳变 | 检查 origin 和生效事件，允许 LOCAL 原点变化，不强行拼接 |
| SDK 下载或构建失败 | 检查公开依赖访问、校验值、JDK 17 和固定 SDK 组件，不静默升级 |

Quest 3S 的 ADB 唤醒广播不能保证解除传感器锁。不得用周期广播掩盖停顿，或把亮屏、ADB 成功返回当作追踪恢复。

最终交付清单：

- [ ] 完整 Android 和 Python 源码、测试及最小环境文件。
- [ ] 完整 Gradle wrapper 和固定依赖来源及校验值。
- [ ] 已签名 APK、SHA256SUMS、许可证和第三方 notices。
- [ ] README：安装、构建、运行、参数、退出和恢复命令。
- [ ] VERIFICATION：实际测试数、构建工具版本、签名与 ABI 检查、lint 警告、实机结果和待验收项目。

复制到新电脑的最小材料是本文；实施完成后迁移的是整个独立工程和交付目录，而不是原遥操作项目的环境或缓存。
