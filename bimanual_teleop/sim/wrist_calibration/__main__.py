"""Read-only local capture, offline hand-eye solve, then apply measured wrist optics."""

import argparse
from datetime import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Run only on the USB-connected local Linux PC")
    collect.add_argument("--robot-ip", default="192.168.1.190")
    collect.add_argument("--side", choices=("left", "right", "both"), default="both")
    collect.add_argument("--left-serial", default="408322072104")
    collect.add_argument("--right-serial", default="408322070938")
    collect.add_argument("--same-row-mm", type=float, default=100., help="Measured center spacing along a straight row; not the 50 mm stagger step")
    collect.add_argument("--samples", type=int, default=20)
    collect.add_argument("--no-preview", action="store_true")
    collect.add_argument("--output", default=str(Path(__file__).resolve().parents[1]/"data/wrist-handeye"/datetime.now().strftime("capture_%Y%m%d_%H%M%S")))
    solve = commands.add_parser("solve", help="Offline server analysis; never opens robot or camera drivers")
    solve.add_argument("capture"); solve.add_argument("--output", required=True)
    apply = commands.add_parser("apply", help="Apply a successful measured result to the default simulation")
    apply.add_argument("result")
    detect = commands.add_parser("detect", help="Check this board in an existing RGB image")
    detect.add_argument("image"); detect.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        if args.samples < 15:
            parser.error("Capture at least 15 poses per wrist; 20 is recommended")
        from .collect import collect as run
        run(args)
    elif args.command == "solve":
        from .workflow import solve_session
        report = solve_session(args.capture, args.output)
        print(json.dumps({"passed": report["passed"], "errors": report["errors"], "wrists": {
            side: value["quality"] for side, value in report["calibrations"].items()}}, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    elif args.command == "apply":
        from .workflow import apply_result
        print(apply_result(args.result))
    else:
        import cv2
        import numpy as np
        from PIL import Image
        from .core import detect_board
        rgb = np.array(Image.open(args.image).convert("RGB")); centers = detect_board(rgb)
        if centers is None:
            raise SystemExit("All 20 circles were not detected")
        for index, point in enumerate(centers):
            cv2.circle(rgb, tuple(np.round(point).astype(int)), 7, (255, 0, 0), 2)
            cv2.putText(rgb, str(index), tuple(np.round(point+[10, -10]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2)
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output)
        print(f"Detected {len(centers)} circles: {output}")


if __name__ == "__main__":
    main()
