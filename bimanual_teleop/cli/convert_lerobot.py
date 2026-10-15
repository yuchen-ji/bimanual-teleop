"""将已整理的 Tianji 记录离线导出为 LeRobotDataset v3.0。"""

import argparse
import json
from pathlib import Path
import sys

from bimanual_teleop.common.console import EpisodeProgress
from bimanual_teleop.cli.convert_recording import summary_lines
from bimanual_teleop.recording.lerobot_export import convert_lerobot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="原始记录根目录、session 或单条 episode")
    parser.add_argument("--output", type=Path, help="新建 LeRobot 数据集目录，必须在输入目录外，禁止覆盖")
    parser.add_argument("--repo-id", default="local/tianji", help="数据集标识；仅本地导出，不自动上传")
    parser.add_argument("--action-space", required=True, choices=("joint", "eef"),
                        help="joint: 双臂关节＋双手 54 维；eef: 双臂末端目标＋双手 52 维")
    parser.add_argument("--task", help="覆盖任务句子；旧数据没有 dataset.json 时必须指定")
    parser.add_argument("--fps", type=int, help="覆盖转换配置中的 target_fps，默认 30 Hz")
    parser.add_argument("--conversion-config", type=Path, help="沿用 DP 转换配置，要求 mode: repair")
    parser.add_argument("--quality-config", type=Path, help="质量阈值 YAML")
    parser.add_argument("--no-raw", action="store_true", help="不附带原始 Zarr；此时导出不包含原始深度")
    parser.add_argument("--allow-mixed-metadata", action="store_true", help="允许合并标定、模型或控制参数不一致的记录")
    parser.add_argument("--max-frames-per-shard", type=int, default=10000, help="每个分片的目标帧数上限；单条 episode 不切开")
    parser.add_argument("--encoder-threads", type=int, default=2, help="每路 H.264 编码线程数")
    parser.add_argument("--dry-run", action="store_true", help="仅规划对齐与质量检查，不解码视频、不写数据")
    parser.add_argument("--report", type=Path, help="另存转换报告 JSON")
    parser.add_argument("--verbose", action="store_true", help="逐条打印详细质量信息")
    args = parser.parse_args(argv)
    if args.output is None and not args.dry_run:
        parser.error("未使用 --dry-run 时必须指定 --output")
    try:
        with EpisodeProgress("对齐与导出条目") as progress:
            report = convert_lerobot(
                args.input, args.output, repo_id=args.repo_id, action_space=args.action_space,
                task=args.task, fps=args.fps, conversion_config=args.conversion_config,
                quality_config=args.quality_config, include_raw=not args.no_raw,
                allow_mixed_metadata=args.allow_mixed_metadata, dry_run=args.dry_run,
                max_frames_per_shard=args.max_frames_per_shard, encoder_threads=args.encoder_threads,
                progress=progress)
        if args.report is not None:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"LeRobot 转换失败：{error}", file=sys.stderr)
        return 1
    print("\n".join(summary_lines(report, verbose=args.verbose)))
    if any("error" in item for item in report["episodes"]):
        print("试运行发现错误；未写入数据集。")
        return 1
    prefix = "试运行，将导出" if args.dry_run else "已导出"
    print(f"{prefix} LeRobot {report['lerobot_version']}：{report['output_episodes']} 条原始演示 → "
          f"{report['lerobot_episodes']} 条连续 episode，{report['output_frames']} 帧，{report['fps']} Hz。")
    if not args.dry_run:
        print(f"数据集：{args.output.resolve()}；原始 Zarr 附件：{'保留' if report['include_raw'] else '未包含'}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
