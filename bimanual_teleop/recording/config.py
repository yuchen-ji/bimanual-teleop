"""Small, fixed recording contract for this three-camera bimanual rig."""

from dataclasses import dataclass, field
from pathlib import Path
import math

from bimanual_teleop.common.config import load_yaml_config

from .schema import CAMERAS, SIDES

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "recording.yaml"


_CONTROL_KEYS = ("save_key", "discard_key", "quit_key", "recover_key")
DEFAULT_TASK = "Pick and place the car seat rails"
DEFAULT_CAMERA_ROLES = dict(zip(CAMERAS, ("overhead", "left_arm", "right_wrist")))


def _strings(value, name, *, allowed=None, required=True):
    if (not isinstance(value, (list, tuple)) or (required and not value)
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(set(value)) != len(value)
            or (allowed is not None and any(item not in allowed for item in value))):
        raise ValueError(f"{name} must contain distinct nonempty strings"
                         + (f" from {allowed}" if allowed is not None else ""))
    return tuple(value)


@dataclass(frozen=True)
class CollectionDevice:
    model: str
    role: str
    sides: tuple[str, ...]
    components: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("model", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"recording.dataset.collection_devices.{name} must be a nonempty string")
        for name, allowed, required in (("sides", SIDES, True), ("components", None, False)):
            value = _strings(getattr(self, name), f"recording.dataset.collection_devices.{name}",
                             allowed=allowed, required=required)
            object.__setattr__(self, name, value)


def _default_devices():
    return (CollectionDevice("Wuji-glove", "finger_motion_input", SIDES),
            CollectionDevice("Quest-3s", "wrist_pose_input", SIDES,
                             ("headset", "left_controller", "right_controller")))


@dataclass(frozen=True)
class DatasetConfig:
    task: str = DEFAULT_TASK
    camera_roles: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERA_ROLES))
    collection_devices: tuple[CollectionDevice, ...] = field(default_factory=_default_devices)

    def __post_init__(self):
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("recording.dataset.task must be a nonempty string")
        if (not isinstance(self.camera_roles, dict) or set(self.camera_roles) != set(CAMERAS)
                or any(not isinstance(role, str) or not role.strip()
                       for role in self.camera_roles.values())):
            raise ValueError(f"recording.dataset.camera_roles must name all three cameras: {CAMERAS}")
        object.__setattr__(self, "camera_roles", dict(self.camera_roles))
        if (not isinstance(self.collection_devices, (list, tuple)) or not self.collection_devices
                or any(not isinstance(device, CollectionDevice) for device in self.collection_devices)):
            raise ValueError("recording.dataset.collection_devices must be a nonempty device list")
        object.__setattr__(self, "collection_devices", tuple(self.collection_devices))


def _dataset_config(values):
    if not isinstance(values, dict):
        raise ValueError("recording.dataset must be a mapping")
    unknown = set(values) - {"task", "camera_roles", "collection_devices"}
    if unknown:
        raise ValueError(f"Unknown recording dataset settings: {sorted(unknown)}")
    values = dict(values)
    if "collection_devices" in values:
        devices = values["collection_devices"]
        if not isinstance(devices, list):
            raise ValueError("recording.dataset.collection_devices must be a device list")
        parsed = []
        for device in devices:
            required = {"model", "role", "sides"}
            if (not isinstance(device, dict) or not required <= set(device)
                    or set(device) - required - {"components"}):
                raise ValueError("Each collection device needs model, role, sides and optional components")
            parsed.append(CollectionDevice(**device))
        values["collection_devices"] = tuple(parsed)
    return DatasetConfig(**values)


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
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self):
        if not isinstance(self.dataset, DatasetConfig):
            raise ValueError("recording.dataset must be a DatasetConfig")
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
    dataset = _dataset_config(values.pop("dataset", {}))
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
    return RecordingConfig(**{**values, "cameras": tuple(values["cameras"]), "dataset": dataset})
