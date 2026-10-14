# 仿真模型资源

本目录由 Git 跟踪，包含当前导轨任务的全部模型输入与可直接运行的派生模型。克隆仓库即可取得模型，无需聊天附件或另一个 CAD 目录。原始 CAD 由用户提供；机械臂与手保留仿真实际使用的子集，来源摘要在 `source/provenance.json`。

| 目录 | 内容 |
| --- | --- |
| `source/robot/` | 去掉旧夹爪的天机 URDF，以及它引用的 19 个 STL |
| `source/hand2/` | 左右 Hand2 安装版 URDF，以及它们引用的 54 个 STL |
| `source/rail_task/` | 六个在用 CAD：长导轨、长导轨台、桌子、转运箱、力传感器、相机及支架 |
| `generated/robot.urdf`、`generated/meshes/` | 装配后的 54 关节机器人和导入所需 STL |
| `generated/task/` | 桌子、箱子、导轨台、工具网格及 CAD 转换清单 |
| `generated/rail/` | 长导轨可见网格和 299 个保留孔的凸碰撞体 |
| `generated/manifest.json` | 源文件摘要、模型身份、装配与物理参数 |

已删除旧夹爪、短导轨、短支架、仅作参考的整场景、未使用的 USD / STEP / URDF 变体、旧相机 CAD、旧导轨碰撞体和重复网格。原来的用户附件与项目外 CAD 目录不属于本次清理范围。

`source/` 与 `generated/` 同时跟踪，以支持修改模型和直接运行。源 STL 与运行 STL 中相同的文件内容由 Git 自动共用 blob。模型中最大单文件约 48 MiB，当前直接使用普通 Git，无需 Git LFS。

检查资源完整性和 Git 跟踪：

```bash
python -m bimanual_teleop.sim.resources --check-git
```

修改配置或源模型后，重新生成（主机使用 `bimanual-teleop` Conda）：

```bash
PIP_USER=false python -m pip install -r bimanual_teleop/sim/requirements-assets.txt
bimanual_teleop/sim/docker/install-cad-reader.sh
python -m bimanual_teleop.sim.assets
python -m bimanual_teleop.sim.resources
```

检查命令列出缺失或未引用文件；仅显式添加 `--prune` 才删除未引用资源。新加模型需要先加入转换清单或 URDF 引用，再做清理。Isaac 镜像、可写层、编译缓存和采集数据仍不跟踪，由 `docker/bootstrap-apptainer.sh` 和运行程序生成。
