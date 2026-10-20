# 头部相机与真实导轨视频对齐

默认头部相机 `sim_camera_0` 根据 `rail_task_episode/episode_000010/camera_0.mp4` 的最后一帧（索引 1268，FoundationPose 帧号 `00022920`）反算，并使用该帧的桌面深度平面约束俯仰和高度。头部相机不再使用机器人 CAD 默认光心或自动朝向导轨台的近似视角。腕部相机保持原有设置。

## 当前参数

参数在 `configs/default.yaml` 的 `cameras.head` 中。世界坐标为 +X 从机器人指向桌子远端、+Y 向机器人左侧、+Z 向上。相机位姿采用 OpenCV 光学坐标：+X 向右、+Y 向下、+Z 向前；四元数顺序为 XYZW。

| 参数 | 值 |
| --- | --- |
| 世界位置，米 | 约 `[0.10298, 0.05751, 1.43141]` |
| 光轴俯角 | 约 48.7° |
| 图像尺寸 | 640 × 480 |
| fx / fy，像素 | 388.1229248046875 / 387.6435852050781 |
| cx / cy，像素 | 332.01959228515625 / 260.2157897949219 |
| 镜头畸变 | 录制时的 D436 系数，顺序 k1、k2、p1、p2、k3 |

使用 Isaac Sim 5.1 的原生 `OmniLensDistortionOpenCvPinholeAPI` 设置内参和畸变，同时填写 USD 焦距、光圈与主点偏移。这样同时保留真实的 fx/fy 差异及偏离图像中心的主点，避免只修改视场角带来的画面偏移。设置 `distortion_coefficients: []` 可输出对应的去畸变针孔图像。

启动命令保持为 `python -m bimanual_teleop.sim`。点击 **Simulation → View head camera** 切换到头部相机，默认以 1280 × 960 渲染桌面视口；点击 **Frame entire scene** 回到全景。也可在 Viewport 的相机菜单中选择末尾为 `Head_depth_Link/sim_camera_0` 的相机。录制输出固定采用配置中的 640 × 480，与当前桌面窗口大小无关。

## 主相机画面清晰度

此前“View head camera”把桌面视口也限制在 640 × 480，窗口放大时只会放大低分辨率画面。此前 PNG 还有明显的随机渲染颗粒；这类颗粒不是木纹或相机失焦，不能通过修改相机位置解决。实际检查中 USD 相机 `fStop=0`，景深和运动模糊均关闭。

当前 `cameras.preview_scale: 2` 将主相机桌面预览提升到 1280 × 960 原生渲染。原始光学内参仍以 640 × 480 为基准，RTX 原生镜头模型按目标分辨率缩放投影，因此视野和画面比例保持一致；不是把 640 × 480 的图片插值拉大。录制的三路 RGB 仍为 640 × 480。若需要降低桌面预览负载，将 `preview_scale` 设为 1；增加到 2 会使该视口的像素数变为四倍。

`rendering.py` 统一使用 TAA，将实时直接光照采样提高到每像素 8 次、反射采样到 4 次、间接漫反射采样到 4 次，并显式关闭景深和运动模糊。这里调整的是实时渲染器；`SimulationApp.samples_per_pixel_per_frame` 只对 PathTracing 有效，不能用于提高当前 RaytracedLighting 的采样。配置和当前 `/rtx` 设置同时写入，避免新建渲染产品或重置场景时恢复低采样默认值。

当前容器的 NGX 初始化失败，DLAA 不可用；不能只把抗锯齿选项改成 4 就认为 DLAA 已生效。实际效果以渲染后的参数和 PNG 为准。远程桌面的缩放或有损图像压缩可能进一步影响显示；核对时优先查看原尺寸 PNG。桌面和箱子已加入本地木纹与塑料细纹贴图；地砖、办公室物体和现场独有的磨损等细节仍未重建。

## 计算与参考数据

`configs/head_camera_reference.json` 保存原始 FoundationPose 矩阵、内参、畸变、参考帧、输入位姿文件哈希和计算时的场景位置。运行仿真不需要 FoundationPose 项目或原始视频。

初始计算链为 `T_world_camera = T_world_rail × T_rail_originalCAD × inverse(T_camera_originalCAD)`。仿真导轨已旋转并居中，因此必须使用派生模型清单中的 `cad_rotation` 和 `origin_in_rotated_cad_m`。`camera_calibration.py` 实现该计算，以及从 OpenCV 到 USD 相机坐标的转换。

细长导轨单独估计的短轴朝向不够稳定。最终使用去畸变 RGB 对齐的米制深度，对桌面拟合平面 `normal_camera · point_camera + offset = 0`：从可见木质桌面的矩形区域初始化 RANSAC，再扩展到同平面内点。当前共 29,016 个内点，点到平面的均方根残差约 1.41 mm；该残差是深度平面拟合指标，不是相机标定精度。桌面法向约束世界竖直方向，导轨长轴约束水平朝向，保留导轨中心的图像射线，并通过已知导轨高度求解相机平移。原始估计、最终矩阵、深度来源与哈希均保存在参考 JSON 中。

计算时导轨台原点为 `[0.46, 0, 0.73]`，导轨中心为 `[0.458758353, 0.006474946, 0.821278683]`，导轨姿态为 `[0, 0, 90.064975509]` 度。相机以该场景为参考反算后固定在机器人上；以后移动导轨不会自动拖动相机。

这是以导轨已落座、首尾约定正确为前提的视觉对齐结果，未经过真机基座外参测量。它优先匹配导轨及导轨台的投影；机械臂姿态和办公室背景尚未复现。桌面黑垫、材质颜色及室内光照已按视频目视近似，外观调整参数见 `configs/default.yaml` 的 `scene.appearance`。

## 验证

在官方 Isaac Python 运行环境中运行 `python -m bimanual_teleop.sim.validate_head_camera`。结果位于 `data/head-camera-alignment/`，不产生学习示范：

- `head_camera.png`：当前默认姿态下的头部相机图像。
- `head_camera_preview.png`：按 `preview_scale` 生成的主相机预览，默认 1280 × 960。
- `report.json`：实际 USD 世界位姿和三个 RTX 投影探针的像素误差。
- `reference_raw.png`、`reference.png`：本次核对所用的原视频末帧及去畸变末帧，属于本地验证数据。

验证通过前，不能仅凭配置矩阵推断渲染器已经正确应用主点和畸变。探针检查会在画面中心及两侧分别渲染已知位置的标记，并比较实际像素位置与 OpenCV 模型的预期位置；同时检查采集分辨率和预览分辨率下的投影。报告也记录实际生效的渲染参数，而非只记录请求参数。

参考：[Isaac Sim 5.1 SimulationApp 渲染参数](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html)。
