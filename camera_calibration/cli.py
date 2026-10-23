"""Camera calibration: local read-only capture, offline solve and explicit export."""

import argparse
import json
from datetime import datetime

from .storage import ROOT


def _capture_options(parser, samples):
    parser.add_argument("--robot-ip", default="192.168.1.190")
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / datetime.now().strftime("capture_%Y%m%d_%H%M%S")),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    wrist = commands.add_parser(
        "wrist", help="Calibrate wrist cameras against their flanges"
    )
    actions = wrist.add_subparsers(dest="action", required=True)
    capture = actions.add_parser("collect", help="Run on the USB-connected PC")
    _capture_options(capture, 20)
    capture.add_argument("--side", choices=("left", "right", "both"), default="both")
    capture.add_argument("--left-serial", default="408322072104")
    capture.add_argument("--right-serial", default="408322070938")
    capture.add_argument("--same-row-mm", type=float, default=100.0)
    solve = actions.add_parser(
        "solve", help="Offline solve with independent validation"
    )
    solve.add_argument("capture")
    solve.add_argument("--output", required=True)
    apply = actions.add_parser(
        "apply", help="Explicitly apply a passed wrist result to simulation"
    )
    apply.add_argument("result")

    external = commands.add_parser(
        "external",
        help="Calibrate the fixed main (third-person) D436 camera using a calibrated wrist",
    )
    actions = external.add_subparsers(dest="action", required=True)
    capture = actions.add_parser(
        "collect", help="Capture stationary paired images and measured joints"
    )
    _capture_options(capture, 25)
    capture.add_argument("--wrist-result", required=True)
    capture.add_argument("--side", choices=("left", "right"), default="left")
    capture.add_argument("--external-serial", default="348122072137")
    capture.add_argument(
        "--external-resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=[640, 480],
    )
    solve = actions.add_parser("solve", help="Offline robust two-camera refinement")
    solve.add_argument("capture")
    solve.add_argument("--wrist-result", required=True)
    solve.add_argument(
        "--limits",
        help="Optional JSON object overriding the documented acceptance limits",
    )
    solve.add_argument("--output", required=True)
    export = actions.add_parser(
        "export", help="Export a measured result after it passes validation"
    )
    export.add_argument("result")
    export.add_argument("--output", required=True)
    export.add_argument(
        "--simulation",
        action="store_true",
        help="Write an explicit simulation head-camera overlay",
    )

    detect = commands.add_parser(
        "detect", help="Check the 20-circle board in an RGB image"
    )
    detect.add_argument("image")
    detect.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if getattr(args, "action", None) == "collect":
        if args.samples < 15:
            parser.error("Capture at least 15 independent groups")
        if args.kind == "external" and min(args.external_resolution) <= 0:
            parser.error("External image dimensions must be positive")
        if args.kind == "wrist":
            from .collect import collect
        else:
            from .external_collect import collect
        print(collect(args))
    elif getattr(args, "action", None) == "solve":
        if args.kind == "wrist":
            from .workflow import solve_session

            report = solve_session(args.capture, args.output)
        else:
            from pathlib import Path

            from .external import Limits
            from .external_workflow import solve_session

            limits = (
                Limits(**json.loads(Path(args.limits).read_text()))
                if args.limits
                else Limits()
            )
            report = solve_session(args.capture, args.wrist_result, args.output, limits)
        print(
            json.dumps(
                {"passed": report["passed"], "errors": report["errors"]}, indent=2
            )
        )
        if not report["passed"]:
            raise SystemExit(1)
    elif getattr(args, "action", None) == "apply":
        from .simulation import apply_result

        print(apply_result(args.result))
    elif getattr(args, "action", None) == "export":
        from .external_workflow import export_result

        print(export_result(args.result, args.output, args.simulation))
    else:
        _detect(args.image, args.output)


def _detect(image, output):
    from pathlib import Path

    import cv2
    import numpy as np
    from PIL import Image

    from .board import detect_board

    rgb = np.array(Image.open(image).convert("RGB"))
    centers = detect_board(rgb)
    if centers is None:
        raise SystemExit("All 20 circles were not detected")
    for index, point in enumerate(centers):
        cv2.circle(rgb, tuple(np.round(point).astype(int)), 7, (255, 0, 0), 2)
        cv2.putText(
            rgb,
            str(index),
            tuple(np.round(point + [10, -10]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
    print(f"Detected {len(centers)} circles: {output}")


if __name__ == "__main__":
    main()
