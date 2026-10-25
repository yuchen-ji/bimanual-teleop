# 导轨抓握与预插入仿真

当前初始状态为左手已握紧导轨：无运动学支撑、无固定连接，碰撞体和摩擦参数保持原样。仅用于仿真。

## 当前抓握场景

在远程桌面的终端启动，先关闭已有 Isaac 实例：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
BIMANUAL_SIM_GUI=1 bash bimanual_teleop/sim/docker/exec-apptainer.sh /isaac-sim/python.sh \
  /workspace/bimanual-teleop/workspace/experiments/rail_pose_preinsert/adapt_contact_sim.py \
  --output /workspace/bimanual-teleop/workspace/outputs/rail_pose_preinsert/adapt_recorded_20261025/native
```

- **Restart task**：恢复保存的抓握姿态，支撑和固定连接保持关闭。
- **Play / Pause / Resume physics**：播放或暂停无支撑保持；初始化后默认暂停。
- 滑移超过 2 mm / 3°、手指接触超限或机器人碰到环境时暂停，不自动恢复支撑。
- 当前入口的初始状态由下面的 JSON 文件控制，调整姿态后应同时保存实测关节角和驱动目标。

| 文件 | 用途 |
| --- | --- |
| `hand_profiles/left_grasp_initial_state.json` | 启动和重启使用的实测姿态、夹持目标及导轨位姿 |
| `hand_profiles/recorded_left_tight_grasp.json` | 原始人类示范的同步关节参考、基座变换及来源 |
| `hand_profiles/left_recorded_adapted_grasp.json` | 适配后的抓握姿态、相对变换和验证结果 |
| `adapt_contact_sim.py`、`adapt_contact_control.py` | 场景入口、状态恢复、接触轨迹和运行检查 |
| `extract_recorded_grasp.py`、`adapt_recorded_grasp.py` | 提取示范并拟合抓握 |
| `squeeze_recorded_grasp.py`、`balance_recorded_grasp.py` | 微量闭合、预紧和指根卸载/拇指夹持调整 |

实测关节角与驱动目标不同：前者用于恢复物理姿态，后者包含夹持预紧量，不能直接将驱动目标当作初始关节位置。

已通过无支撑保持 5.1 秒、末端上移 5 mm / 旋转 1° 的随动测试，以及恢复初始状态后的约 5 秒播放测试。末端方向相对示范约差 5.75°，个别指关节最大约差 42°；拇指主要由中段接触。原始可视网格与 PhysX 简化碰撞面有细小差异。这些结果仅说明当前仿真抓握通过，不代表实机或完整插入验收。

结果目录为 `workspace/outputs/rail_pose_preinsert/adapt_recorded_20261025/`：`validation.json` 是抓握汇总；`native/initial_grasp_playback_result.json` 是重启验证；`native/result.json` 是当前状态。保留成功测试的状态、轨迹、碰撞核对结果及当前运行日志。

## 完整预插入实验

`run.py` 保留递料、抓握、位姿估计和搬运流程，在孔轴外 20 mm 处保持，不执行插入。它使用 `config.yaml`，与上面的已握紧初始状态入口分开：

```bash
python workspace/experiments/rail_pose_preinsert/run.py --mode step --until preinsert --perception foundationpose
```

**Apply config / Restart task** 读取配置并复位；场景结构变化时重建。**Apply config / Rebuild scene** 强制重建。FoundationPose 服务在重试间复用；后台断开时先恢复服务再重试。详细参数见 `run.py --help` 与 `config.yaml`。

`contact_fixed` 是几何贴合后建立固定连接的诊断模式，不能作为无支撑抓稳证据。四指预闭合（`--until four_fingers`）同样只用于接触诊断。完整预插入流程仍需单独验收。

## 检查

```bash
python scripts/run_tests.py discover -s workspace/experiments/rail_pose_preinsert/tests
```
