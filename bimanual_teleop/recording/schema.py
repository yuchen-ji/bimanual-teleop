"""Fixed raw recording layout shared by capture, finalization, review and conversion."""

SIDES = ("left", "right")
STATE_STREAMS = tuple(f"{kind}/{side}" for kind in ("arms", "hands") for side in SIDES)
COMMAND_STREAMS = tuple(f"{kind}/{side}" for kind in ("arm_commands", "hand_commands") for side in SIDES)
STREAMS = STATE_STREAMS + COMMAND_STREAMS

# Online record and spool layout; values are float64 in this order.
STREAM_FIELDS = {
    "arms/left": (("joint_pos", 7), ("wrench", 6)),
    "arms/right": (("joint_pos", 7), ("wrench", 6)),
    "hands/left": (("joint_pos", 20),),
    "hands/right": (("joint_pos", 20),),
    "arm_commands/left": (("joint_pos", 7), ("eef_pose", 7)),
    "arm_commands/right": (("joint_pos", 7), ("eef_pose", 7)),
    "hand_commands/left": (("joint_pos", 20),),
    "hand_commands/right": (("joint_pos", 20),),
}
# raw.zarr adds the flange pose computed by finalization from measured joints.
RAW_FIELDS = {stream: dict(fields) | ({"eef_pose": 7} if stream.startswith("arms/") else {})
              for stream, fields in STREAM_FIELDS.items()}

# Main view, robot left arm, robot right arm; serials are bound in recording.yaml.
CAMERAS = ("camera_0", "camera_1", "camera_2")
# Correct the upside-down left-arm mounting before preview and recording.
RGB_ROTATION_DEG = {"camera_0": 0, "camera_1": 180, "camera_2": 0}
MAIN_CAMERA = CAMERAS[0]
CAMERA_FPS = 30
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
RGB_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
DEPTH_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
DEPTH_STREAM = f"cameras/{MAIN_CAMERA}/depth"


def rgb_stream(camera):
    return f"cameras/{camera}/rgb"


RGB_STREAMS = tuple(rgb_stream(camera) for camera in CAMERAS)
