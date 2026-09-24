# 数据采集快速使用说明

## 1. 开始前检查

确认双臂、双手、Quest 和三台 RealSense 周围无人员与障碍物，实体急停已释放并可随时触发。

采集使用三路 NVENC 硬件编码。先确认 NVIDIA 驱动可用：

```bash
nvidia-smi -L
```

如果该命令失败，`--record` 会在机械臂运动前退出，不会自动改用三路 CPU 编码。

相机序列号、深度开关、输出目录和帧池长度 `frame_capacity` 位于 `configs/recording.yaml`。`frame_capacity` 按主机内存选择，CPU 核数不用另配；改法见该文件注释和 [README 的配置说明](../README.md#配置)。还应检查 `recordings/` 所在磁盘具有足够空间。

## 2. 启动

在项目根目录执行：

```bash
conda activate bimanual-teleop
python scripts/teleop_quest_tianji.py --record --viewer
```

程序会先完成 NVIDIA 编码预检和三台相机预检，然后才进入实机运动确认与初始回位。

## 3. 采集按键

1. 按 Enter 接合双臂双手遥操作。接合成功后按 `configs/recording.yaml` 的 `start_delay_s` 自动开始一条采集；默认是立即开始。
2. Enter、Space、暂停手势、回位或设备故障只会脱离遥操作，并暂停这一条。再次接合后继续同一条，时间从暂停时刻接上，不把脱离的那段算进采集时间。
3. 按 S 结束并保存原始条目，遥操作继续。保存之后如果仍保持接合，不会自动开下一条；需要先脱离再接合。
4. 按 X 作废当前条。
5. 按 Q 保存当前条并退出。Ctrl+C 或进程退出也会结束当前条。

S、X、Q 和等待时间都在 `configs/recording.yaml` 的 `controls` 与 `start_delay_s` 中。

保存成功后条目状态是 `captured`，表示原始数据完整，但尚未生成 `raw.zarr`。

## 4. 离线整理

退出遥操作后执行：

```bash
python scripts/finalize_recording.py --input recordings/<session>
```

整理过程会校验计数和时间序列、计算双臂正运动学、压缩深度与低维数据，并生成现有格式：

```text
recordings/<session>/episode_000000/
  episode.json
  camera_0.mp4
  camera_1.mp4
  camera_2.mp4
  raw.zarr/
  raw_spool/
```

只有 `episode.json` 中状态为 `complete` 的条目可以转换为训练数据。`raw_spool/` 用于复核和重新整理，确认最终数据后可按项目的数据保留策略归档。

## 5. 转换训练数据

末端动作空间：

```bash
python scripts/convert_recording.py \
  --input recordings/<session> \
  --output datasets/episodes_eef.zarr \
  --action-space eef
```

关节动作空间：

```bash
python scripts/convert_recording.py \
  --input recordings/<session> \
  --output datasets/episodes_joint.zarr \
  --action-space joint
```

## 6. 状态含义

- `capturing`：正在采集，不能转换。
- `captured`：原始数据已保存，等待离线整理。
- `finalizing`：正在离线整理。
- `complete`：整理完成，可以转换。
- `failed`：存在采集或写入故障，不能转换。
- `discarded`：操作员作废，不能转换。

## 7. 故障恢复

采集、编码或写盘失败时，当前条会标记为不完整，但遥操作不会因此自动停止：

1. 保持安全操作并主动按 Enter 脱离遥操作。
2. 根据终端和运行日志排除相机、显卡驱动或磁盘问题。
3. 在脱离状态按 C 恢复采集进程。这个按键是 `controls.recover_key`。
4. 等待相机重新就绪，再次接合。新的一条会自动开始。

设备故障、安全保护和控制看门狗仍会立即停止运动。

每次启动都会打印 `logs/teleop_quest_tianji_*.jsonl` 的绝对路径。排查故障时同时保留该日志和对应的整个条目目录，不要只保存终端截图。

预览使用独立的最新帧共享内存。预览允许跳帧，关闭或卡顿不会占用正式录制帧，也不会影响编码与写入。
