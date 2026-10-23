"""Read recorded joints and synchronized video without constructing device drivers."""

from pathlib import Path
import hashlib
import json

import numpy as np

from bimanual_teleop.recording.schema import STREAM_FIELDS
from bimanual_teleop.sim.config import load_config, sim_path
from bimanual_teleop.sim.kinematics import RobotModel, pose_vector


KEY_TIMES = {"ready": 0., "left_open": 4., "left_grasp": 7., "left_carry": 12.,
             "right_open": 12., "two_hands": 16., "right_grasp": 24., "seated": 32., "release": 40.}


def read_stream(episode, stream):
    width = sum(n for _, n in STREAM_FIELDS[stream])
    dtype = np.dtype([("time_ns", "<i8"), ("sequence", "<i8"), ("values", "<f8", (width,))])
    path = Path(episode)/"raw_spool/streams"/(stream.replace("/", "__")+".bin")
    if path.stat().st_size % dtype.itemsize:
        raise ValueError(f"Incomplete numeric stream: {path}")
    data = np.fromfile(path, dtype=dtype)
    if not len(data) or not np.isfinite(data["values"]).all() or np.any(np.diff(data["time_ns"]) < 0):
        raise ValueError(f"Invalid numeric stream: {path}")
    return data


def extract_reference(episode, model):
    episode = Path(episode)
    manifest_bytes = (episode/"episode.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    camera = np.fromfile(episode/"raw_spool/cameras/camera_0_rgb.bin",
                         dtype=[("time_ns", "<i8"), ("sequence", "<i8"), ("source_ms", "<f8")])
    streams = {f"{kind}/{side}": read_stream(episode, f"{kind}/{side}")
               for kind in ("arms", "hands") for side in ("left", "right")}
    samples = {}
    for label, seconds in KEY_TIMES.items():
        stamp = int(camera[0]["time_ns"])+round(seconds*1e9)
        q = np.zeros(54)
        for stream, offset, width in (("arms/left", 0, 7), ("arms/right", 7, 7),
                                     ("hands/left", 14, 20), ("hands/right", 34, 20)):
            data = streams[stream]
            selected = np.abs(data["time_ns"]-stamp) <= 75_000_000
            if not selected.any():
                selected[np.argmin(abs(data["time_ns"]-stamp))] = True
            q[offset:offset+width] = np.median(data["values"][selected, :width], axis=0)
        samples[label] = {"video_time_s": seconds, "joint_positions_rad": q.tolist(),
                          "flange_world": {side: pose_vector(model.flange(side, q, world=True)).tolist()
                                           for side in ("left", "right")}}
    provenance = {"source_episode": str(episode), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "source_model_sha256": manifest["metadata"].get("model_sha256"),
                  "source_demo_id": manifest.get("source_demo_id", manifest.get("uuid")),
                  "camera_clock_origin_ns": int(camera[0]["time_ns"]),
                  "joint_order": list(model.names), "angles": "rad", "sampling_window_s": .15}
    return {"provenance": provenance, "samples": samples}


def analyze(episode, output):
    import cv2
    from PIL import Image, ImageDraw
    config = load_config()
    model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
    report = extract_reference(episode, model)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/"reference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    times = (4, 7, 9, 12, 16, 20, 24, 28)
    sheet = Image.new("RGB", (1280, 1584), "white"); draw = ImageDraw.Draw(sheet)
    for camera in range(3):
        cap = cv2.VideoCapture(str(Path(episode)/f"camera_{camera}.mp4"))
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera_{camera} video")
        for index, seconds in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds*1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Missing video frame at {seconds}s")
            x, y = index % 4*320, (camera*2+index//4)*264
            sheet.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320, 240)), (x, y+24))
            draw.text((x+6, y+5), f"camera_{camera}, {seconds}s", fill="black")
        cap.release()
    sheet.save(output/"three_views.png")
    rows = ["# rail_place3 关节与视频对照", "", "参考 episode_000010；实测关节按视频时间对齐，取 150 ms 中值。", "",
            "视频中两手多在端部持物；本任务要求右手改抓中部，因此使用手型与臂部姿态作参考，重新求解末端位置。", "",
            "![三路关键帧](three_views.png)", "", "| 参考阶段 | 视频时间 / s | 左臂 J1…J7 / deg | 右臂 J1…J7 / deg |", "|---|---|---|---|"]
    for label, sample in report["samples"].items():
        q = np.asarray(sample["joint_positions_rad"])
        rows.append(f"| {label} | {sample['video_time_s']} | {np.rad2deg(q[:7]).round(1).tolist()} | {np.rad2deg(q[7:14]).round(1).tolist()} |")
    rows += ["", "完整两手 20 关节角和末端位姿见 reference.json；手指顺序为拇指、食指、中指、无名指、小指，每指 4 关节。", "",
             "示范中的左手四指共同包握承托，右手握稳后左手可调整；旧任务的三指夹底板、其余两指大幅收拢不作为新任务手型。"]
    (output/"README.md").write_text("\n".join(rows)+"\n")
    return report
