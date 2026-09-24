"""将在线采集的原始会话整理为现有 MP4 和 raw.zarr 格式。"""

import argparse
from pathlib import Path
import sys

from bimanual_teleop.devices.tianji.sdk import add_sdk_argument
from bimanual_teleop.recording.finalize import finalize_recordings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="原始记录根目录、会话目录或单个条目")
    add_sdk_argument(parser)
    args = parser.parse_args(argv)
    try:
        report = finalize_recordings(args.input, sdk_root=args.sdk_root)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"整理失败：{error}", file=sys.stderr)
        return 1
    print(f"已完成 {report['complete']} 条，跳过 {report['skipped']} 条。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
