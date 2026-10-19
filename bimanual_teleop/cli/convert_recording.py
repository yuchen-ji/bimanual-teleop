"""将遥操作记录导出为 Diffusion Policy 数据集；默认修复模式，逐条汇报丢弃、修复和质量等级。"""

import argparse
import json
from pathlib import Path
import sys

from bimanual_teleop.common.console import EpisodeProgress
from bimanual_teleop.recording.convert import convert_recordings
from bimanual_teleop.recording.quality import FAIL, WARN, summarize


_SIDES = {"left": "左", "right": "右"}
_STATES = {"robot_joint": "臂关节", "robot_eef_pose": "臂末端位姿", "wrench": "臂力传感器", "hand_joint": "手关节"}
_SKIPPED = {"review_reject": "审片剔除", "error": "出错"}


def reason_label(name):
    """报告中剔除原因键名的中文说明；不认识的键名原样返回。"""
    for suffix, text in (("_depth_unmatched", " 深度缺帧"), ("_unmatched", " 缺帧")):
        if name.endswith(suffix):
            return name[:-len(suffix)] + text
    side, _, rest = name.partition("_")
    if side in _SIDES and rest.endswith("_invalid_or_gap"):
        state = rest[:-len("_invalid_or_gap")]
        return f"{_SIDES[side]}{_STATES.get(state, state)}缺失或间隔过大"
    if side in _SIDES and rest.endswith("_command_invalid_or_stale"):
        return f"{_SIDES[side]}{'臂' if rest.startswith('arm') else '手'}指令缺失或过期"
    return name


def _ratio(part, whole):
    return f"{part / whole:.1%}" if whole else "-"


def _reasons(reasons, limit=3):
    items = sorted(reasons.items(), key=lambda item: -item[1])
    text = "，".join(f"{reason_label(name)} {count}" for name, count in items[:limit])
    return text + ("，…" if len(items) > limit else "")


def _dropped(item):
    edge, interior, short = item["edge_trimmed_frames"], item["interior_invalid_frames"], item["short_segments"]
    if not (edge["start"] or edge["end"] or interior or short["segments"]):
        return "无"
    parts = [f"首尾 {edge['start']}+{edge['end']} 帧"
             + (f"（{_reasons(item.get('edge_trimmed_reasons', {}))}）" if edge["start"] or edge["end"] else ""),
             f"中间 {interior} 帧" + (f"（{_reasons(item['interior_invalid_reasons'])}）" if interior else "")]
    if short["segments"]:
        parts.append(f"短片段 {short['segments']} 段 / {short['frames']} 帧")
    return "；".join(parts)


def _repairs(item):
    result = []
    reused = [f"{entry['camera']} {entry['reused_frames']} 帧（图像最旧 {entry['max_image_age_ms']:.0f} ms）"
              for entry in item.get("camera_repairs", ()) if entry["reused_frames"]]
    if reused:
        result.append("缺帧沿用前一张图：" + "，".join(reused))
    state = item.get("state_repairs")
    if state and state["frames"]:
        result.append(f"放宽插值 {state['frames']} 帧（前后样本相隔超过 {state['beyond_ms']:.0f} ms，"
                      f"最大 {state['max_gap_ms']:.0f} ms）")
    for pause in item.get("pauses", ()):
        result.append(f"主相机第 {pause['after_main_row']} 帧前暂停 {pause['removed_wait_ns'] / 1e9:.1f} s，"
                      + ("按审片结论接上" if pause["merged"] else "在此切开"))
    return result


def _details(item):
    lines = [f"      时长 {item['duration_s']:.1f} s"]
    for camera, stats in item["cameras"].items():
        line = f"      {camera}：{stats['frames']} 帧"
        if "fps" in stats:
            line += (f"，{stats['fps']:.2f} fps，丢帧 {stats['dropped_frames']}，断档 {stats['gaps']}，"
                     f"最大间隔 {stats['max_gap_ms']:.0f} ms")
        if stats["pauses"]:
            line += f"，暂停 {stats['pauses']}（暂停后首帧为本相机原始帧 {stats['pause_after_rows']}）"
        lines.append(line)
    for camera, stats in item["cross_camera"].items():
        if stats:
            lines.append(f"      {camera} 对 camera_0：偏差 p50 {stats['p50_offset_ms']:.1f} / p99 "
                         f"{stats['p99_offset_ms']:.1f} / 最大 {stats['max_offset_ms']:.1f} ms，"
                         f"超出匹配容差 {stats['unmatched_frames']} 帧")
    for name, stats in item["streams"].items():
        if "hz" in stats:
            lines.append(f"      {name}：{stats['hz']:.0f} Hz，最大间隔 {stats['max_gap_ms']:.0f} ms")
    return lines


def episode_lines(item, *, verbose=False):
    """一条记录的等级、丢弃、修复和超标项；verbose 时附每路统计。"""
    head = f"  [{item['level']}] {item['source_episode']}：" if "level" in item else f"  {item['source_episode']}："
    if item.get("skipped") == "review_reject":
        note = item.get("review", {}).get("note")
        return [head + "审片剔除，整条跳过" + (f"（{note}）" if note else "")]
    if "reference_frames" not in item:
        if "error" in item:
            return [head + f"无法转换：{item['error']}"]
        reason = f"（{item['reason']}）" if item.get("reason") else ""
        hint = "，需先整理" if item.get("level") == WARN else ""
        return [head + f"状态 {item['status']}，整条跳过{reason}{hint}"]
    lines = [head + f"参考 {item['reference_frames']} 帧，写入 {item['output_frames']} 帧"
                    f"（{_ratio(item['output_frames'], item['reference_frames'])}），{item['segments']} 个片段",
             f"      丢弃：{_dropped(item)}"]
    if repairs := _repairs(item):
        lines.append("      修复：" + "；".join(repairs))
    if "error" in item:
        lines.append(f"      正式转换会中止：{item['error']}")
    problems = [entry["message"] for entry in item["checks"] if entry["level"] in (WARN, FAIL)]
    if problems:
        lines.append("      问题：" + "；".join(problems))
    return lines + (_details(item) if verbose else [])


def summary_lines(report, *, verbose=False):
    """逐条和总计的等级、丢弃与修复；首尾裁剪与中间剔除分开统计。"""
    summary = summarize(report)
    status = "，".join(f"{name} {count}" for name, count in summary["status"].items())
    mode = {"repair": "，修复模式", "strict": "，严格模式"}.get((report.get("conversion_config") or {}).get("mode"), "")
    lines = [f"条目 {summary['episodes']} 条（{status}）{mode}："]
    for item in report["episodes"]:
        lines += episode_lines(item, verbose=verbose)
    if summary["levels"]:
        lines.append("等级：" + "，".join(f"{level} {count}" for level, count in summary["levels"].items()))
    if summary["problems"]:
        top = list(summary["problems"].items())[:10]
        lines.append("问题分布：" + "，".join(f"{name} {count} 条" for name, count in top)
                     + ("，…" if len(summary["problems"]) > len(top) else ""))
    lines.append(f"合计：参考 {summary['reference_frames']} 帧，有效 {summary['valid_frames']} 帧，"
                 f"写入 {summary['output_frames']} 帧（利用率 {_ratio(summary['output_frames'], summary['reference_frames'])}），"
                 f"{summary['output_segments']} 个连续片段")
    dropped = []
    if summary["skipped"]:
        dropped.append(f"整条跳过 {sum(summary['skipped'].values())} 条（"
                       + "，".join(f"{_SKIPPED.get(name, name)} {count}" for name, count in summary["skipped"].items())
                       + "）")
    dropped.append(f"首尾 {summary['edge_trimmed_frames']} 帧，中间 {summary['interior_invalid_frames']} 帧，"
                   f"短片段 {summary['short_segments']} 段 / {summary['short_segment_frames']} 帧")
    lines.append("丢弃：" + "；".join(dropped))
    if summary["interior_invalid_reasons"]:
        lines.append(f"中间剔除原因：{_reasons(summary['interior_invalid_reasons'], limit=8)}")
    if repairs := summary["repairs"]:
        reused = "，".join(f"{camera} {count} 帧" for camera, count in repairs["camera_reused_frames"].items() if count)
        lines.append(f"修复：缺帧沿用前一张图 {reused or '0 帧'}；放宽插值 {repairs['state_frames']} 帧；"
                     f"暂停在此切开 {repairs['pauses_split']} 处，按审片结论接上 {repairs['pauses_merged']} 处")
    if summary["without_output"]:
        lines.append(f"没有写入任何帧的条目：{'，'.join(summary['without_output'])}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="原始记录根目录、任务目录或单个 episode")
    parser.add_argument("--output", type=Path, help="新建的 .zarr 数据集，禁止覆盖；--dry-run 时可省略")
    parser.add_argument("--action-space", required=True, choices=("eef", "joint"),
                        help="eef: 双臂目标末端+双手（52维）；joint: 双臂目标关节+双手（54维）")
    parser.add_argument("--include-depth", action="store_true",
                        help="导出主相机深度并检查深度匹配；默认只导出 RGB，不因深度缺帧剔除样本")
    parser.add_argument("--conversion-config", type=Path,
                        help="转换 YAML；默认 configs/recording_conversion.yaml（修复模式），写 mode: strict 改用严格模式")
    parser.add_argument("--quality-config", type=Path,
                        help="质量等级阈值 YAML；默认 configs/recording_quality.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="只做对齐、有效性规划和质量检查并打印汇总，不解码视频、不写数据集；"
                             "列出所有会让正式转换中止的错误")
    parser.add_argument("--allow-mixed-metadata", action="store_true",
                        help="允许相机内参、运动学模型、控制参数等元数据不一致的条目合并进同一数据集")
    parser.add_argument("--verbose", action="store_true", help="逐条列出每台相机、跨相机和每路低维流的统计")
    parser.add_argument("--report", type=Path, help="另存完整质量报告 JSON")
    args = parser.parse_args(argv)
    if args.output is None and not args.dry_run:
        parser.error("未使用 --dry-run 时必须指定 --output")
    try:
        options = {key: value for key, value in (("conversion_config", args.conversion_config),
                                                 ("quality_config", args.quality_config)) if value is not None}
        with EpisodeProgress("规划条目" if args.dry_run else "转换条目") as progress:
            report = convert_recordings(args.input, args.output,
                                        action_space=args.action_space, include_depth=args.include_depth,
                                        allow_mixed_metadata=args.allow_mixed_metadata, dry_run=args.dry_run,
                                        progress=progress, **options)
        if args.report is not None:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"转换失败：{error}", file=sys.stderr)
        return 1
    print("\n".join(summary_lines(report, verbose=args.verbose)))
    errors = [item for item in report["episodes"] if "error" in item]
    if errors:
        print(f"试运行发现 {len(errors)} 条错误，正式转换会在第一条处中止；未写入数据集。")
        return 1
    if args.dry_run:
        print(f"试运行：将导出 {report['output_episodes']} 条原始演示、{report['output_segments']} 个连续片段、"
              f"{report['output_frames']} 帧；未写入数据集。")
    else:
        print(f"已导出 {report['output_episodes']} 条原始演示、{report['output_segments']} 个连续片段、"
              f"{report['output_frames']} 帧：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
