"""Small, fixed recording contract for this three-camera bimanual rig."""

from dataclasses import dataclass
from pathlib import Path
import math

from bimanual_teleop.common.config import load_yaml_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recording.yaml"


_CONTROL_KEYS = ("save_key", "discard_key", "quit_key", "recover_key")


def _control_key(value, name):
    if not isinstance(value, str) or len(value) != 1 or value.lower() in (" ", "\n", "\r"):
        raise ValueError(f"recording.controls.{name} must be one character")
    return value.lower()


@dataclass(frozen=True)
class RecordingConfig:
    cameras: tuple[str, str, str]
    main_depth: bool = True
    state_hz: float = 200.
    output_dir: str = "recordings"
    start_delay_s: float = 0.
    frame_capacity: int = 256
    save_key: str = "s"
    discard_key: str = "x"
    quit_key: str = "q"
    recover_key: str = "c"

    def __post_init__(self):
        if len(self.cameras) != 3 or len(set(self.cameras)) != 3 or any(
                not isinstance(s, str) or not s.strip() for s in self.cameras):
            raise ValueError("recording.cameras must contain three distinct serial strings")
        if type(self.main_depth) is not bool:
            raise ValueError("recording.main_depth must be true or false")
        if isinstance(self.state_hz, bool) or not math.isfinite(self.state_hz) or not 0 < self.state_hz <= 1000:
            raise ValueError("recording.state_hz must be in (0, 1000]")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("recording.output_dir must be a nonempty path")
        if (isinstance(self.start_delay_s, bool) or not isinstance(self.start_delay_s, (int, float))
                or not math.isfinite(self.start_delay_s) or not 0 <= self.start_delay_s <= 60):
            raise ValueError("recording.start_delay_s must be in [0, 60]")
        if (isinstance(self.frame_capacity, bool) or not isinstance(self.frame_capacity, int)
                or not 16 <= self.frame_capacity <= 2048):
            raise ValueError("recording.frame_capacity must be an integer in [16, 2048]")
        keys = [self.save_key, self.discard_key, self.quit_key, self.recover_key]
        if any(not isinstance(key, str) or len(key) != 1 or key in (" ", "\n", "\r") for key in keys):
            raise ValueError("recording control keys must be one character each")
        if len(set(keys)) != len(keys):
            raise ValueError("recording control keys must be distinct")


def load_config(path=DEFAULT_CONFIG):
    values = dict(load_yaml_config(path))
    controls = values.pop("controls", {})
    if not isinstance(controls, dict):
        raise ValueError("recording.controls must be a mapping")
    unknown = set(values) - {
        "cameras", "main_depth", "state_hz", "output_dir", "start_delay_s", "frame_capacity",
    }
    if unknown:
        raise ValueError(f"Unknown recording settings: {sorted(unknown)}")
    unknown_controls = set(controls) - set(_CONTROL_KEYS)
    if unknown_controls:
        raise ValueError(f"Unknown recording controls: {sorted(unknown_controls)}")
    if "cameras" not in values:
        raise ValueError("recording.cameras is required")
    for name in _CONTROL_KEYS:
        if name in controls:
            values[name] = _control_key(controls[name], name)
    return RecordingConfig(**{**values, "cameras": tuple(values["cameras"])})
