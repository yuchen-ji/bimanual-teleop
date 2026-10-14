"""Generate and validate editable subtask schemes; raw markers remain immutable."""

from __future__ import annotations

import argparse
from pathlib import Path

from bimanual_teleop.sim._dependencies import yaml

from .recording import discover, read_boundaries


def intervals(count, cuts):
    points = [0, *sorted(set(int(c) for c in cuts if 0 < int(c) < count)), count]
    return list(zip(points[:-1], points[1:]))


def make_scheme(directory):
    episodes = []
    for path, document in discover(directory):
        if document["status"] != "complete":
            continue
        cuts = [p["frame_index"] for p in read_boundaries(path)]
        parts = intervals(document["frames"], cuts)
        episodes.append({"source_demo_id": document["source_demo_id"], "source_path": str(path.resolve()),
                         "frame_count": document["frames"], "cuts": sorted(set(cuts)),
                         "ranges": [{"start": start, "end": end, "task": None} for start, end in parts]})
    return {"schema_version": 1, "name": "scheme_01", "episodes": episodes}


def load_scheme(path):
    scheme = yaml.safe_load(Path(path).read_text())
    if not isinstance(scheme, dict) or scheme.get("schema_version") != 1:
        raise ValueError("Unsupported subtask scheme")
    if not isinstance(scheme.get("name"), str) or not scheme["name"]:
        raise ValueError("Scheme needs a name")
    seen = set()
    for episode in scheme["episodes"]:
        source = episode["source_demo_id"]
        if source in seen:
            raise ValueError("Repeated source demonstration in scheme")
        seen.add(source)
        last = 0
        for span in episode["ranges"]:
            start, end, task = span["start"], span["end"], span["task"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= episode["frame_count"]:
                raise ValueError("Subtask ranges must use valid half-open frame indices")
            if start < last:
                raise ValueError("Subtask ranges must be ordered and non-overlapping")
            if task is not None and (not isinstance(task, str) or not task.strip() or task in (".", "..") or "/" in task or "\\" in task):
                raise ValueError("Task names must be nonempty simple directory names")
            last = end
    return scheme


def main():
    parser = argparse.ArgumentParser(description="Create an unnamed subtask scheme from B-key markers")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Scheme output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(make_scheme(args.input), allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Written {output}; set ranges[].task explicitly before subtask export")


if __name__ == "__main__":
    main()
