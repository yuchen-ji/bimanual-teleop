# `difabs fail` 的只读静态分析（2026-10-23）

后续进展：使用受限的底层读取接口取得 389 组静止反馈，重建并核对重力模型后，定位当前只有左 J7 持续超限；用户同时说明已卸下灵巧手。参见 [左 J7 力矩失配与负载配置](tianji_j7_torque_mismatch_20261023.md)。下文保留静态分析阶段的证据范围；历史 8 次故障的逐次失败索引仍未恢复。

## 已确认的结论

从控制器安装文件 `FXSI.ko` 中还原的正常有限数值检查条件为：

```text
abs(关节传感器反馈扭矩[i] - 模型静态重力补偿扭矩[i])
    > 0.2 * R.A{arm}.L{i}.BASIC.TorqueMax
```

满足条件即输出 `difabs fail` 并返回失败，外层输出 `Abnormal Torque2`，最终触发错误 6。检查对象是机器人关节扭矩反馈；不是腕部新增六维力传感器的原始读数。模型项包含控制器采用的机器人动力学、基座重力和当前工具参数。

程序从 J1 开始依次检查到配置的自由度，遇到第一个失败关节便返回。错误分支只记录函数名和 `arm`，不记录 `i`、比较数值或阈值。因此现有日志与磁盘程序能够确认检查逻辑和阈值来源，仍不能唯一确定此前 8 次失败的具体关节，更不能直接认定某个传感器硬件损坏。

## 程序来源与只读边界

通过同一固定版本 SDK 的独立 `RecvFile` / TCP 10240 文件通道，在主机 2026-10-23 15:48–15:51 +08:00 下载了：

1. `/proc/self/exe`：文件服务本身的 ELF，用于识别服务；不是运动控制模块。
2. `/etc/rc.local`：仅默认空启动脚本。
3. `/home/FUSION/Tmp/ctrl_package.tar`：控制器留存安装包。
4. `/home/FUSION/bin/FXSI.ko`：已安装的实时控制内核模块。
5. `/home/FUSION/bin/FXPSI`：已安装的用户态参数解析程序。
6. `/home/FUSION/Config/cfg/robot.ini`：持久化配置快照。

安装包中的模块和参数程序分别与已安装文件逐字节相同：

| 文件 | SHA-256 |
| --- | --- |
| `FXSI.ko` | `95a44d980d5a61a2d385e5db7944f37411965bfec04778440b43e93d2d12d2c0` |
| `FXPSI` | `84cbcd69dd9cb2fb4047637aa49947f30912b556ed4b8765929639edecb50ac8` |
| `robot.ini` | `c1579026284dc7e13de4a147f5a243054baf7ee33d3f096de5635851c980f77d` |

`FXPSI` 中包含版本常量 `100343009`，与先前故障日志中的启动版本一致。`FXSI.ko` 保留了符号表，但没有调试类型信息。

六次下载各自的 `strace` 网络记录均显示只创建一个 TCP socket，连接 `192.168.1.190:10240`，没有 UDP socket。没有创建机器人控制连接，未清错、使能、切换模式、松闸、修改参数、上传文件或重启。安装包只提取普通文件作静态检查，未运行其脚本、程序或内核模块；未附加调试器或读取控制器内核内存。

本机厂家 SDK 源码未包含 `___FX_SyncFFDTorque` 的实现。本次结论来自控制器安装二进制的符号、重定位、反汇编和配置解析路径交叉验证。另查阅的官方仓库发布列表和工具 ZIP 没有作为该版本检查逻辑的依据。

## 检查逻辑的证据链

以下 `FXSI.ko` 地址为 ELF `.text` 节内地址；这是可重定位内核模块，不是运行时绝对地址。

### 反馈比较项

- `___FX_SyncFFDTorque` 起点为 `.text+0x0f00`，大小 `0x3f4`。
- `0x10d5–0x10dc` 取 `SysInfo.V[5000 + 1000*arm + 56 + i]`。
- 该槽位在 `FX_RtTask` 的 `0x4574–0x457a` 由轴反馈结构 `axis+0x1c0` 写入。
- `_FX_MacDatUpdate` 的 `0xad3e–0xad4a`、`0xae04–0xae0a`、`0xae97–0xae9d` 将 `FX_ServoGetSensorTorque` 取得的原始关节传感器扭矩经比例和方向处理后传递到该反馈字段。外部 SDK 将相应反馈称作 `m_FB_Joint_SToq`，单位 N·m。

### 模型比较项

- `0x0fd6` 调用 `FX_DCTRL_JImp_CalGrav`，将第一输出写入静态数组 `init_torque.34139`，即 `.bss+0x3c0`。
- 随后复制到 `grv_torque.34140`，即 `.bss+0x360`。
- `CalGrav` 的 `0x1fb12–0x1fb22` 准备零速度/零加速度数组；`0x1fbdd–0x1fc0b` 的第一次 `StdNE` 调用计算静态重力模型项。第二次 `StdNE` 使用实际速度和加速度，输出到另一个数组，不是此处 `difabs` 比较的第一输出。
- `0x10df` 从关节传感器反馈中减去 `init_torque[i]`，`0x10e9` 调用 `FX_FABS`。
- 阻尼扭矩在通过该比较之后才在 `0x1062–0x1080` 加入初始化计算，因此不能将这个残差解释为阻抗刚度项或阻尼项超限。

### 阈值及比较方向

- `0x110a` 读取 `.rodata.cst8+0x10` 的双精度常量；原始字节 `9a 99 99 99 99 99 c9 3f` 解码为 **0.2**。
- `0x1112` 乘以配置结构地址 `cfg + arm*0xc0c + i*0x12c + 0x138` 的 double。
- 在已安装 `FXPSI::LoadRobotIni` 中，`0x80749c3–0x8074a6c` 将 BASIC 段的 `TorqueMax` 写入相同地址；`0x120+0x18=0x138`。另一处 MOTOR 段也有同名键，但不是该比较读取的字段。
- `0x111b` 比较绝对差与阈值，`0x111f` 的 `jbe` 在差值小于等于阈值时继续；大于阈值时进入 `0x1125–0x114a` 的错误返回路径。
- `.bss+0x3a0` 的静态循环变量为 `i.34134`。错误分支没有将其写入日志或 ArmError。

等价的相关逻辑如下；省略了其他初始化和换算步骤：

```c
calculate_static_gravity_torque(arm, joint_position, init_torque);
for (i = 0; i < configured_dof; ++i) {
    double diff = abs(joint_sensor_torque[i] - init_torque[i]);
    if (check == 1 && diff > 0.2 * basic_torque_max[i]) {
        log_error("%s: arm%d: difabs fail", function_name, arm);
        return false;
    }
    // The original function then converts this joint's initialization torques.
}
```

`__FX_RequestTransToSensorMode` 先调用 `FX_MacCheckEncoderDiff`，通过后才到达扭矩同步检查。因此这些记录中的 `difabs fail` 出现在编码器进入检查通过之后；这不证明实体零位及所有动力学输入必然正确。

## 按本次磁盘配置计算的阈值

两臂相应 BASIC.TorqueMax 一致：

| 关节 | BASIC.TorqueMax（N·m） | `difabs` 阈值（N·m） |
| --- | ---: | ---: |
| J1、J2 | 108 | 21.6 |
| J3、J4 | 35 | 7.0 |
| J5、J6、J7 | 18 | 3.6 |

这是从现存 `robot.ini` 计算的值，未通过控制连接读取实时参数，不能声称故障瞬间或当前内存中的配置必然与该文件一致。

## 仍不能从现有文件确定的内容

- 缺少故障同周期的七路关节传感器扭矩、`init_torque/grv_torque` 和失败循环索引 `i`；正常日志也没有保存这些量。不能从 `Torque2`、腕部之前的上转现象或旧的 J1 编码器问题推定失败关节。
- 未读取运行内核内存，未将磁盘模块与当前内存映像作字节比对。可以确认安装包与已安装程序一致，以及程序版本常量与故障启动日志一致。
- 这个残差超限可能涉及负载/质心、基座重力、机器人模型或角度输入、关节扭矩比例/方向/零偏及外界受力；仅靠分支判断不能区分。

要锁定故障关节，需要厂家提供故障时 `i`、关节传感器扭矩和 `grv_torque` 的已有诊断数据或只读诊断入口。应核查这些比较项的一致性，不应通过增大 TorqueMax、修改 0.2 常量、盲调零偏或清编码器零点绕过检查。本次未复现故障，未作实机运动验证，故障未解除。

## 本地证据与复核

完整证据保存在项目 `logs/tianji_difabs_analysis_20261023/`：

- `installed_FXSI.ko`、`installed_FXPSI`、`controller_package.tar`、`robot.ini` 及各自抓取报告、网络记录。
- `installed_FXSI.ko.disasm.txt`、`installed_FXPSI.disasm.txt`：完整离线反汇编。
- `binary_logic_evidence.txt`：相关函数、配置加载片段及指令地址说明。
- `static_analysis.json`：哈希、公式、阈值表、网络审计及限制。
- `analyze_binary.py`：在 Conda `bimanual-teleop` 下复核程序哈希、关键指令字节、0.2 常量、配置字段映射和六次 TCP-only 下载记录；已运行通过。
