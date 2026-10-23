# 左手持轨搬运至预插入位姿

独立于原 `rail_grasp_insert` 双手接力实验。本实验只执行：正确就位目标示教 → 模拟人工递料 → 左手闭合 → 初始位姿估计 → 固定抓握关系计算 → 持物碰撞规划 → 搬运到沿孔轴外移 **20 mm** 的预插入位姿 → 保持。没有下降插入、松手或撤离。

主机复用 `/home/jiyuchen/project/WYS/diffusion_policy/workspace/rail_pose_insert/` 的坐标链、官方离线 FK/IK、FCL 碰撞检查、RRT 与五次曲线。Isaac 通过文件请求主机计算，不导入或连接天机、Hand2、RealSense 实机驱动。三个输入 JSON 全部由仿真生成，写在本次输出的 `inputs/`，标记 `test_data=true`，真机入口会拒绝。

## 验证模式

| 参数 | 验证内容 | 不能据此认定的结论 |
|---|---|---|
| `--perception oracle`（默认） | 以实际仿真位姿作观测，检查坐标转换、持轨规划和执行 | FoundationPose 已可靠识别 |
| `--perception foundationpose` | 头部渲染 RGB、米制深度，去畸变后真实 GPU 位姿估计 | 真实反光、深度噪声与遮挡效果已覆盖 |
| `--grasp fixed` | 显式选择的理想固定抓握诊断；固定物理约束表达“不滑动” | 手指已依靠摩擦抓稳，或手轨不存在穿模 |
| `--grasp physics`（默认） | 启用手轨碰撞、关闭固定约束，检查各指接触和手轨漂移，失败保留证据 | 真机手部控制与摩擦已标定 |

模拟人工递料只进行一次明确记录的导轨状态放置，在闭合期间由运动学支撑模拟人手扶持。之后取消支撑。物理模式要求拇指和四根支持手指分别达到接触力门槛、合力方向对向并持续稳定。固定模式使用 USD 固定关节，不逐帧改写导轨位姿；为表达理想刚体，固定模式会过滤手轨接触，所以不能验收抓持。

默认模板已撤下旧单臂旋转和人为 30 mm 偏移。参考 episode 000034 在约 9.246 s 的实际五指状态，按四指根部排列和掌面方向构造中部包握；四指弯曲承托、拇指对向。各关节适配限制在参考值 ±0.4 rad，来源和改动保存在 `grasp_template.json`。记录中没有已标定的手轨变换，因此这属于几何重建，不能称为精确复现真实手腕朝向。

**当前模板仍未通过物理抓稳验收。** 指尖点拟合不能约束整个指节，固定约束预览已查出五指穿入薄壁。所有模式在抓握复核和规划前都检查实际状态的手部实体网格采样；采样穿入超过 0.2 mm 会停止。无穿入样点也不等于完整的无相交证明，物理接触、滑移和外部碰撞检查仍须通过。闭合期间允许加载到稳定接触，形成测量参考后恢复 5 mm / 0.035 rad 的手轨漂移限制，不会持续重置参考。

仿真仍是有努力限制的关节位置驱动，并不是天机笛卡尔阻抗控制器的等效复现。实验单独使用 `arm_stiffness=25000`、`arm_damping=700`，不改默认场景；首轮 5000/300 的驱动存在约 4.5 mm 静态法兰偏差，不能满足 2 mm 到位要求。真实驱动增益、重力补偿、手套、接触摩擦、工具质量及法兰安装关系仍需标定。

## 运行

以下使用主机 `bimanual-teleop` Conda 环境，入口自动启动已有 Isaac Sim 5.1 官方容器。默认桌子、箱子、导轨台及头部相机布局在每次启动时读取并保存快照。一次只运行一个使用共享 overlay 的 Isaac 实例。

先做不启动 Isaac 的当前场景规划预检：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python workspace/experiments/rail_pose_preinsert/run.py --preflight --mode auto
```

在 Remmina 桌面终端逐步查看：

```bash
python workspace/experiments/rail_pose_preinsert/run.py --mode step
```

场景自动完成目标示教。点击 Simulation 面板的 **Next stage** 执行闭合；测量和规划完成后，再点一次执行搬运。**Run automatically** 自动推进；**Pause task / Resume** 只暂停任务推进，物理仍运行；暂停全部物理使用 Isaac 时间轴。到位后保持，不会插入。右臂全程保持初始目标。

无窗口运行物理抓握和搬运流程（抓握失败会在搬运前停止）：

```bash
python workspace/experiments/rail_pose_preinsert/run.py --headless --mode auto
```

以渲染 RGB-D 测试 FoundationPose（主机需有同级 FoundationPose 项目及现有 `foundationpose` 环境）：

```bash
python workspace/experiments/rail_pose_preinsert/run.py \
  --headless --mode auto --perception foundationpose
```

仿真语义分割提供**可见导轨掩码**，替代当前真机流程中的人工掩码；位姿由 FoundationPose 估计，不从真值初始化或修正。真值仅用于独立最终误差验收。该模式暂时假定分割正确，不能验收自动分割算法。

先只诊断抓握，不进入规划和搬运：

```bash
python workspace/experiments/rail_pose_preinsert/run.py \
  --headless --mode auto --until grasp
```

显式加 `--grasp fixed` 可查看理想约束诊断。其手轨碰撞被过滤，不能作为抓稳证据；实际网格采样发现穿模仍会拒绝完成预览。

`--record` 额外录制三路视频；默认在各次测量保存三路 PNG。`--output` 必须是本项目 `workspace/outputs/` 下尚不存在的目录。`--config` 可使用独立 YAML 覆盖实验参数；不改默认仿真配置。`--display :10` 可指定自己的 X11 桌面。

## 结果

启动时打印输出绝对路径，默认位于：

```text
/home/jiyuchen/project/WYS/bimanual-teleop/workspace/outputs/rail_pose_preinsert/<本次运行>/
```

| 文件 | 内容 |
|---|---|
| `result.json` | 最终状态、失败原因、感知/抓握模式、实际终点误差 |
| `inputs/calibration.json` | 仿真头部光学系到 SDK 左臂基座的外参 |
| `inputs/target.json` | 正确就位时的原始 CAD 目标位姿 |
| `inputs/scene.json` | 当前桌子与两个完整箱堆的保守包络；导轨台由目标锚定 |
| `plan.json` | 原真机预插入规划器输出，包含抓握变换、关节路径和碰撞记录 |
| `initial_state.json` | 图像时刻对应的实测关节与历史 |
| `teach/measure/verify_observation.json` | 三阶段输入的观测与来源 |
| `verification.json` | 上游预插入视觉/名义余量检查 |
| `perception_accuracy.json` | 与同时刻仿真真值比较的中心、CAD 原点及方向误差，仅用于诊断，不修正估计 |
| `hand_profiles.json` | 保留原 episode 路径、示教时刻、关节顺序和手型命令 |
| `grasp_template.json` | 五指参考、几何构造、关节适配和指尖拟合误差；不是接触证书 |
| `grasp_geometry.json` | 复核时实际手部网格对导轨实体的采样穿入量；超限停止 |
| `collision_approximations.json` | 实际导入 USD 碰撞近似、缩放、材质、摩擦及手轨碰撞是否被过滤 |
| `captures/trial_000/` | 三路图像及实际状态；视觉模式另有深度、掩码和位姿叠图 |
| `trace.jsonl`、`events.jsonl` | 仿真反馈、接触、力、阶段、递料和固定约束启停 |
| `runtime.log`、`foundationpose.log` | Isaac 与可选 GPU 感知日志 |
| `effective.yaml`、`runtime.json`、`metadata.json` | 场景、控制假设、基座配准及代码/资源摘要 |

`preflight_passed` 只表示离线预检通过。`simulation_preinsert_passed` 需要实际物理法兰到位（2 mm、0.5°，稳定 0.5 s）、导轨复测通过、抓握漂移及接触检查通过，并以场景真值独立检查实际导轨终点（2 mm、0.5°）。固定抓握模式始终报告 `physical_grasp_validated=false`，所有模式始终报告 `real_grasp_validated=false`。

`--until grasp` 下，物理模式完成仅记录 `simulation_grasp_passed`；固定模式完成仅记录 `grasp_preview_complete`，不会将预览记作预插入成功。

路径检查沿用上游的离散关节采样，最大 0.005 rad、外部余量 2 mm；不保证连续扫掠体，也不能覆盖真机未建模障碍。桌子及箱堆保守实心包络不允许规划穿过空腔，适用于本次接料后搬运。

## 检查

```bash
python scripts/run_tests.py discover -s workspace/experiments/rail_pose_preinsert/tests -v
```

## 历史旧模板验证记录（2026-10-22）

下表使用旧旋转和 30 mm 偏移（保留在 `legacy_geometry.yaml`）。这些搬运结果不能替新模板证明抓持或到位。

| 输出目录（本项目 `workspace/outputs/rail_pose_preinsert/` 下） | 结果 |
|---|---|
| `20261022_154435_c7fc3d` | 当前场景离线坐标链、IK、完整持轨碰撞规划通过 |
| `oracle_fixed_01` | 5000/300 位置驱动带载偏差约 4.5 mm，到位失败，保留作为调参依据 |
| `oracle_fixed_02` | 真值感知＋固定抓握物理搬运通过；实际法兰误差 1.29 mm，实际导轨中心误差 1.79 mm，方向误差约 0.175°；没有插入 |
| `foundationpose_fixed_01` | 渲染 RGB-D → 独立 GPU 推理链路运行成功；抓握后位姿不一致，规划前停止，未执行搬运 |
| `oracle_physics_01` | 移除递料支撑后抓持漂移越界（约 3.0 mm、2.43°），在抓持检查阶段停止，未执行搬运 |
| `preflight_final` | 最终代码的离线规划及测试 JSON 生成通过；真机加载器明确拒绝这些仿真外参 |
| `oracle_fixed_final` | 额外复跑因共享 overlay 已被另一导轨实验占用而未启动，不计入物理通过次数 |

FoundationPose 的目标估计出现约 180° 方向歧义：中心误差仅约 0.8～2.9 mm，但原始 CAD 原点在导轨一端，方向反转使原点差约 48 cm。握持后的估计中心误差约 7 cm，三帧也不一致。图像显示当前递料姿态中手臂遮挡了较多导轨，尚未满足“头部能清楚看到两端”的理想条件；这些失败不能直接等同于该理想真机初态的表现。下一步需调整递料姿态和手型，保证两端可见，再处理估计方向一致性。

动态抓持失败说明当前几何基线使用的 30 mm 偏移和手型尚不能作为物理中部抓握策略。没有放宽位姿一致性、到位或漂移门槛来报告成功。

新增实验测试 7 项通过；通用仿真回归 50 项执行、49 项通过、1 项按原条件跳过。检查覆盖 CAD/光学/基座坐标、轨迹采样与上游一致、方向反转不被中心误差掩盖、实际到位、保持不插入及抓持失败停止。上述物理结果来自当前独立实验，不引用旧双手插入成功记录。

`oracle_fixed_02/feedback_audit.json` 对已保存的 128 条搬运阶段反馈补充核对：右臂漂移为 0，手型最大漂移约 0.0028 rad，关节跟踪最大偏差约 0.0028 rad，采样记录中的机器人／持轨外部接触力均为 0。该记录是离线审查，不是一次新的物理运行或连续时间碰撞证明。Ruff 和空白检查通过。

## 自然包握与碰撞复核

- `human_wrap_preview_01`：修正朝向后的固定约束预览；腕部画面呈现四指包握，但独立实体采样确认五指穿入导轨，最大已测穿入约 0.951 mm。旧版运行当时尚无这一拒绝门槛，故其 `grasp_preview_complete` 只保留作诊断，不能作为有效抓握。
- `human_wrap_physics_02`：启用手轨碰撞后五指均有接触，但撤掉递料支撑后滑移约 16 mm / 15.1°，在抓稳前停止。失败状态网格采样最大穿入约 0.180 mm，显示物理接触与固定约束预览不同。
- 导轨显示网格约 480 × 45.82 × 25.71 mm；299 个凸碰撞块体积相对误差约 9.4e-7。体积一致仅是资产检查，不能替代接触几何和烘焙结果验证。
- 手部原始显示网格与 URDF 碰撞网格相同；导入后使用凸包近似，部分凹部会被填充。个别原始 STL 不封闭，不能用其体积比认定碰撞近似精度。
- 已在本实验绑定可编辑的手部刚体材质，并遍历实例代理验证实际碰撞体继承配置摩擦；逐形状记录随运行保存。未修改其他导轨实验或通用场景。
- `human_wrap_physics_04`：确认实际碰撞形状比例为 1、接触偏移 0.2 mm、静止偏移 0、手部摩擦 0.8，手轨碰撞没有被过滤；手部近似仍为凸包。五指接触后仍滑移约 18.1 mm / 15.1°，在递料支撑移除后停止，没有进入规划或搬运。参数确实生效不等于握姿已成立。
- `human_wrap_preflight_guard_01`：已用独立预检验证穿模拒绝生效，旧的指尖拟合通过不能绕过实体网格检查。本轮实验测试 8 项通过，Ruff 通过；未把这些软件检查记作抓握成功。
