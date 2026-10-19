"""逐条审片：生成三路拼接预览视频并播放，按键记录保留或剔除，结论写入条目旁的 review.json。"""

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from bimanual_teleop.common.console import EpisodeProgress
from bimanual_teleop.recording.episodes import COMPLETE, episode_label, find_episodes, read_manifest
from bimanual_teleop.recording.policy import load_policy
from bimanual_teleop.recording.review import (KEEP, PREVIEW, REJECT, detect_pauses, make_preview, read_review,
                                              write_review)
from bimanual_teleop.recording.schema import CAMERA_FPS

_DEFAULT_REASONS = {True: "审片确认接缝两侧动作连续", False: "审片决定保留暂停边界"}


def player_command(player, path, title):
    """Command that plays ``path``; None when no player is available."""
    if player:
        return shlex.split(player) + [str(path)]
    if shutil.which("ffplay"):
        return ["ffplay", "-autoexit", "-loglevel", "error", "-window_title", title, str(path)]
    if shutil.which("xdg-open"):
        return ["xdg-open", str(path)]
    return None


def _clock(row):
    seconds = row / CAMERA_FPS
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def _ask(prompt, choices):
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer
        print(f"请输入 {' / '.join(choices)}")


class _Player:
    def __init__(self, command):
        self.command, self.process = command, None

    def play(self):
        self.stop()
        if self.command is not None:
            self.process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(2.)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def _review_pauses(pauses, existing):
    reviews = dict(existing)
    for pause in pauses:
        row = pause["after_main_row"]
        previous = reviews.get(str(row))
        hint = f"，现为{'合并' if previous['merge'] else '保留边界'}" if previous else ""
        choice = _ask(f"  接缝 {row}（预览 {_clock(row)}，暂停 {pause['removed_wait_ms'] / 1e3:.1f} s{hint}）："
                      "合并 m / 保留边界 k / 暂不决定 s：", ("m", "k", "s"))
        if choice == "s":
            continue
        merge = choice == "m"
        reason = input("  理由（回车用默认）：").strip() or _DEFAULT_REASONS[merge]
        reviews[str(row)] = {"merge": merge, "reason": reason}
    return reviews


def review_episode(episode, label, args, position):
    """Returns 'keep', 'reject', 'skip' or 'quit'."""
    import zarr

    descriptor = read_manifest(episode)
    review = read_review(episode) or {}
    duration = (descriptor["end_ns"] - descriptor["start_ns"]) / 1e9
    header = f"[{position}] {label}：{duration:.1f} s"
    if "decision" in review:
        header += f"，已审：{review['decision']}" + (f"（{review['note']}）" if review.get("note") else "")
    print(header)
    preview = episode / PREVIEW
    if args.refresh_preview or not preview.is_file():
        with EpisodeProgress("生成预览") as progress:
            progress.episode(label)
            make_preview(episode, preview, scale=args.scale, progress=progress)
    try:
        pauses = detect_pauses(zarr.open_group(str(episode / "raw.zarr"), mode="r"), descriptor,
                               args.pause_offset_tolerance_ms)
    except (KeyError, ValueError) as error:
        print(f"  暂停识别失败，跳过接缝复核：{error}")
        pauses = []
    for pause in pauses:
        print(f"  暂停接缝：主相机原始帧 {pause['after_main_row']} 起，预览 {_clock(pause['after_main_row'])}，"
              f"暂停 {pause['removed_wait_ms'] / 1e3:.1f} s")
    player = _Player(None if args.no_play else player_command(args.player, preview, label))
    if not args.no_play and player.command is None:
        print(f"  未找到播放器，请手动打开：{preview}")
    try:
        player.play()
        while (answer := _ask("保留 y / 剔除 n / 重播 r / 跳过 s / 退出 q：", ("y", "n", "r", "s", "q"))) == "r":
            player.play()
    finally:
        player.stop()
    if answer in ("s", "q"):
        return {"s": "skip", "q": "quit"}[answer]
    decision = KEEP if answer == "y" else REJECT
    note = input("备注（回车不改）：").strip()
    changes = {"decision": decision}
    if note:
        changes["note"] = note
    if decision == KEEP and pauses:
        changes["pause_reviews"] = _review_pauses(pauses, review.get("pause_reviews", {})) or None
    write_review(episode, **changes)
    print(f"  已记录：{decision}")
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="原始记录根目录、任务目录或单个 episode")
    parser.add_argument("--all", action="store_true", help="也重审已有 keep/reject 结论的条目")
    parser.add_argument("--no-play", action="store_true", help="只生成预览并提问，不启动播放器")
    parser.add_argument("--player", help="播放命令，预览路径追加在末尾；默认 ffplay，找不到时用 xdg-open")
    parser.add_argument("--scale", type=float, default=.5, help="预览中每路画面相对 640×480 的缩放，默认 0.5")
    parser.add_argument("--refresh-preview", action="store_true", help="重新生成已存在的 preview.mp4")
    parser.add_argument("--pause-offset-tolerance-ms", type=float,
                        help="暂停识别阈值，ms；默认取 configs/recording_conversion.yaml 中的值，"
                             "导出另用配置时应与其一致")
    args = parser.parse_args(argv)
    if not 0 < args.scale <= 1:
        parser.error("--scale 必须在 (0, 1] 内")
    counts = {"keep": 0, "reject": 0, "skip": 0}
    try:
        if args.pause_offset_tolerance_ms is None:
            args.pause_offset_tolerance_ms = load_policy().pause_offset_tolerance_ms
        root = args.input.expanduser().resolve()
        episodes = [episode for episode in find_episodes(root) if read_manifest(episode).get("status") == COMPLETE]
        pending = [episode for episode in episodes
                   if args.all or "decision" not in (read_review(episode) or {})]
        print(f"complete 条目 {len(episodes)} 条，待审 {len(pending)} 条。")
        for index, episode in enumerate(pending, 1):
            result = review_episode(episode, episode_label(episode, root), args, f"{index}/{len(pending)}")
            if result == "quit":
                break
            counts[result] += 1
    except (EOFError, KeyboardInterrupt):
        print()
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"审片失败：{error}", file=sys.stderr)
        return 1
    print(f"本次保留 {counts['keep']} 条，剔除 {counts['reject']} 条，跳过 {counts['skip']} 条；"
          "结论保存在各条目的 review.json。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
