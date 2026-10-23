# 仿真模块约定

- 通用仿真环境代码、资源与标定留在本目录；具体任务实验的代码、配置、说明和测试放在项目根目录 `workspace/experiments/`，输出放在 `workspace/outputs/`。不得修改真机入口、配置或采集格式。
- 主机 Python 使用 `bimanual-teleop` Conda 环境。用户已授权 Isaac Sim 5.1 官方容器使用其 Python 3.11；此例外仅用于仿真运行。
- 可以读取并复用现有 Quest/Wuji 输入适配器和纯函数；服务器不得创建天机、Hand2 或 RealSense 实机驱动。
- 仿真数据及运行缓存不进入 Git。所有派生数据保留原始示范标识。
- 采集只记录 B 键分段点；任务名称与分组仅在离线处理阶段指定。
