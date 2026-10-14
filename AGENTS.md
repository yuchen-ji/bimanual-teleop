# Python 环境

- 本项目的 Python 运行、安装和测试统一使用 Conda 环境 `bimanual-teleop`。
- 首次创建：在项目根目录运行 `PIP_USER=false conda env create -f environment.yml`。
- 运行前执行 `conda activate bimanual-teleop`；非交互命令可用 `conda run -n bimanual-teleop ...`。
- Python 依赖只安装在此环境中，禁止使用系统 Python、`pip --user` 或 `sudo pip` 安装本项目依赖。



# 对话背景信息设定

输出对话请按照以下要求：

2. Mermaid 图在对话框中展示 png格式。不要每次加载对话渲染那么长时间。
3. 计划的markdow文件，输出到  “项目根目录下”/plan/ 目录下面。
4. 表述要清晰，术语要统一。
