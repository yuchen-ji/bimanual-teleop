"""将在线采集的原始条目整理为现有 MP4 和 raw.zarr 格式。"""

import argparse
from pathlib import Path
import sys

from bimanual_teleop.common.console import EpisodeProgress
from bimanual_teleop.devices.tianji.sdk import add_sdk_argument
from bimanual_teleop.recording.finalize import finalize_recordings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="任意目录；递归整理其下所有层级的条目，作废条目会被删除")
    parser.add_argument("--spool-archive", type=Path, default=Path("raw_spools"),
                        help="整理成功后 raw_spool 移到 <此目录>/<task>/<episode>，条目内留下同名符号链接；"
                             "相对当前目录解析，默认 raw_spools，不得位于 --input 之内")
    parser.add_argument("--refinalize", action="store_true",
                        help="把已 complete 的条目改回 captured，并从 raw_spool 重新生成 raw.zarr；"
                             "符号链接失效时到 --spool-archive 中查找")
    add_sdk_argument(parser)
    args = parser.parse_args(argv)
    try:
        with EpisodeProgress("整理条目") as progress:
            report = finalize_recordings(args.input, sdk_root=args.sdk_root, spool_archive=args.spool_archive,
                                         refinalize=args.refinalize, progress=progress)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"整理失败：{error}", file=sys.stderr)
        return 1
    for path in report["deleted"]:
        print(f"已删除作废条目：{path}")
    for path, error in report["errors"]:
        print(f"整理失败：{path}：{error}", file=sys.stderr)
    print(f"已完成 {report['complete']} 条，删除作废 {report['discarded']} 条，"
          f"跳过 {report['skipped']} 条，失败 {report['failed']} 条；"
          f"本次归档 raw_spool {len(report['archived'])} 条到 {args.spool_archive.resolve()}。")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
