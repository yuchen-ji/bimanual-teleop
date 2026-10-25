"""Manual main/wrist camera capture and offline hand-eye calibration."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from bimanual_teleop.devices.tianji.config import DEFAULT_CONFIG
from bimanual_teleop.devices.tianji.sdk import add_sdk_argument
from .storage import ROOT


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="kind", required=True)
    wrist = commands.add_parser("wrist", help="腕部手眼标定：键盘位置控制、C 手动保存")
    actions = wrist.add_subparsers(dest="action", required=True)
    capture = actions.add_parser("collect", help="键盘点动所选机械臂，C 手动记录一组")
    capture.add_argument("--side", choices=("left", "right"), default="left",
                         help="每次控制并采集一侧，默认 left")
    capture.add_argument("--tianji-config", type=Path, default=DEFAULT_CONFIG)
    capture.add_argument("--robot-ip", help="控制器 IP，默认读取天机配置")
    add_sdk_argument(capture)
    capture.add_argument("--model", type=Path)
    capture.add_argument("--left-serial", help="默认使用采集配置 camera_1")
    capture.add_argument("--right-serial", help="默认使用采集配置 camera_2")
    capture.add_argument("--same-row-mm", type=float, default=100.)
    capture.add_argument("--step-mm", type=float, default=5., help="每键平移量，默认 5 mm")
    capture.add_argument("--step-deg", type=float, default=2., help="每键旋转量，默认 2°")
    capture.add_argument("--transition-s", type=float, default=.25)
    capture.add_argument("--samples", type=int, default=25, help="目标组数，可按 Q 提前结束")
    capture.add_argument("--no-preview", action="store_true")
    capture.add_argument("--output", type=Path, default=ROOT / "data" / datetime.now().strftime("wrist_%Y%m%d_%H%M%S"))
    solve = actions.add_parser("solve", help="离线求解法兰到腕部相机的手眼外参")
    solve.add_argument("capture", type=Path)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--model", type=Path)
    add_sdk_argument(solve)
    solve.add_argument("--limits", type=Path)
    export = actions.add_parser("export", help="导出通过验证的腕部相机外参")
    export.add_argument("result", type=Path)
    export.add_argument("--output", type=Path, required=True)
    external = commands.add_parser("external", help="主视角眼在手外标定：固定相机、末端固定板")
    actions = external.add_subparsers(dest="action", required=True)
    capture = actions.add_parser("collect", help="键盘点动机器人，C 手动记录一组")
    capture.add_argument("--side", choices=("left", "right"), default="left")
    capture.add_argument("--tianji-config", type=Path, default=DEFAULT_CONFIG)
    capture.add_argument("--robot-ip", help="覆盖天机配置中的控制器 IP")
    add_sdk_argument(capture)
    capture.add_argument("--model", type=Path)
    capture.add_argument("--external-serial", help="主视角相机序列号，默认读取 configs/recording.yaml 的 camera_0")
    capture.add_argument("--external-resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    capture.add_argument("--same-row-mm", type=float, default=100., help="同一行相邻圆心距离，默认 100 mm")
    capture.add_argument("--step-mm", type=float, default=5.)
    capture.add_argument("--step-deg", type=float, default=2.)
    capture.add_argument("--transition-s", type=float, default=.25)
    capture.add_argument("--samples", type=int, default=25, help="目标组数；Q 提前结束并保留数据，离线求解至少需要 15 组")
    capture.add_argument("--no-preview", action="store_true", help="关闭预览，按键仍在交互终端输入")
    capture.add_argument("--output", type=Path, default=ROOT / "data" / datetime.now().strftime("external_%Y%m%d_%H%M%S"))
    solve = actions.add_parser("solve", help="联合求解基座到相机、法兰到标定板变换")
    solve.add_argument("capture", type=Path)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--model", type=Path, help="与采样时同一份运动学模型")
    add_sdk_argument(solve)
    solve.add_argument("--limits", type=Path, help="覆盖验收阈值的 JSON 对象")
    export = actions.add_parser("export", help="导出通过验证的实测结果")
    export.add_argument("result", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--world-from-base", type=Path, help="可选 T_world_base（4×4 JSON），用于同时导出世界坐标外参")
    detect = commands.add_parser("detect", help="检查现有 4×5 非对称圆点板检测与编号")
    detect.add_argument("image", type=Path)
    detect.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.kind == "detect":
            _detect(args.image, args.output)
        elif args.action == "collect":
            from bimanual_teleop.common.config import positive_number
            from bimanual_teleop.recording.config import load_config
            if args.samples <= 0:
                command.error("采样组数必须为正整数")
            numeric = ("same_row_mm", "step_mm", "step_deg", "transition_s")
            if args.kind == "external":
                from .external_collect import collect
                if min(args.external_resolution) <= 0:
                    command.error("图像宽高必须为正整数")
                if args.external_serial is None:
                    args.external_serial = load_config().cameras[0]
            else:
                from .collect import collect
                cameras = load_config().cameras
                args.left_serial = args.left_serial or cameras[1]
                args.right_serial = args.right_serial or cameras[2]
            for name in numeric:
                positive_number(getattr(args, name), "--" + name.replace("_", "-"))
            collect(args)
        elif args.action == "solve":
            from .external import Limits
            if args.kind == "external":
                from .external_workflow import solve_session
            else:
                from .wrist_workflow import solve_session

            limits = Limits(**json.loads(args.limits.read_text())) if args.limits else Limits()
            report = solve_session(args.capture, args.output, limits, model_path=args.model, sdk_root=args.sdk_root)
            print(json.dumps({"passed": report["passed"], "errors": report["errors"]}, ensure_ascii=False, indent=2))
            return int(not report["passed"])
        elif args.action == "export":
            if args.kind == "external":
                from .external_workflow import export_result
                print(export_result(args.result, args.output, args.world_from_base))
            else:
                from .wrist_workflow import export_result
                print(export_result(args.result, args.output))
    except KeyboardInterrupt:
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(f"相机标定失败：{error}")
        return 1
    return 0


def _detect(image, output):
    import cv2
    import numpy as np
    from PIL import Image
    from .board import detect_board

    rgb = np.array(Image.open(image).convert("RGB"))
    centers = detect_board(rgb)
    if centers is None:
        raise ValueError("未检测到全部 20 个圆点")
    for index, point in enumerate(centers):
        pixel = tuple(np.round(point).astype(int))
        cv2.circle(rgb, pixel, 7, (255, 0, 0), 2)
        cv2.putText(rgb, str(index), pixel, cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
    print(f"Detected {len(centers)} circles: {output}")
