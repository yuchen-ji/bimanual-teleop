"""Experiment-owned Isaac desktop layout. Imported by GUI runs only."""
from __future__ import annotations

import asyncio
import copy

import numpy as np

DEFAULTS = {
    "enabled": True, "maximize": True, "resolution": [1280, 960],
    "main_eye_m": [-.8, -1.25, 1.8], "main_target_m": [.45, 0., .78],
    "inspector_width_fraction": .13, "tools_width_fraction": .115,
    "content_height_fraction": .22, "camera_column_fraction": .60, "wrist_height_fraction": .45,
}
VIEW_TITLES = {"main": "Viewport", "head": "Rail - Head camera",
               "left": "Rail - Left wrist", "right": "Rail - Right wrist"}


def prepare_extensions():
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    # The stock inspector constructs SDK Camera wrappers with default square
    # intrinsics and resets camera child prims. Do not let it touch task sensors.
    if manager.is_extension_enabled("isaacsim.util.camera_inspector"):
        manager.set_extension_enabled_immediate("isaacsim.util.camera_inspector", False)
    for extension in ("omni.kit.window.stage", "omni.kit.window.property",
                      "omni.kit.window.content_browser", "omni.kit.window.console"):
        if (not manager.is_extension_enabled(extension)
                and not manager.set_extension_enabled_immediate(extension, True)):
            raise RuntimeError("Could not enable layout extension: " + extension)


def layout_settings(supplied=None):
    if supplied is not None and not isinstance(supplied, dict):
        raise ValueError("desktop_layout must be a mapping")
    settings = {**copy.deepcopy(DEFAULTS), **(supplied or {})}
    if set(settings) != set(DEFAULTS):
        raise ValueError("Unknown desktop_layout setting")
    for key in ("enabled", "maximize"):
        if type(settings[key]) is not bool:
            raise ValueError(f"desktop_layout.{key} must be boolean")
    size = settings["resolution"]
    if (not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size)
            or size[0]*3 != size[1]*4):
        raise ValueError("desktop_layout.resolution must be positive integer pixels in 4:3")
    for key in ("main_eye_m", "main_target_m"):
        vector = np.asarray(settings[key], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"desktop_layout.{key} requires three finite metres")
        settings[key] = vector.tolist()
    if np.linalg.norm(np.subtract(settings["main_eye_m"], settings["main_target_m"])) < 1e-6:
        raise ValueError("Desktop camera eye and target must differ")
    for key in ("inspector_width_fraction", "tools_width_fraction", "content_height_fraction",
                "camera_column_fraction", "wrist_height_fraction"):
        if not isinstance(settings[key], (int, float)) or not .05 <= settings[key] <= .8:
            raise ValueError(f"desktop_layout.{key} must be between 0.05 and 0.8")
    return settings


class DesktopLayout:
    def __init__(self, settings, camera_paths, output):
        self.settings = layout_settings(settings)
        self.camera_paths, self.output = camera_paths, output
        self.windows = {}
        self.pending = None
        self.inspector = self.inspector_subscription = None

    def restore(self):
        if self.pending is None or self.pending.done():
            self.pending = asyncio.ensure_future(self._restore())

    async def _restore(self):
        import omni.appwindow
        import omni.kit.app
        from common import write_json
        from omni import ui
        from omni.kit.viewport.utility import create_viewport_window
        from omni.kit.viewport.window import get_viewport_window_instances

        app = omni.kit.app.get_app()
        try:
            if self.settings["maximize"]:
                omni.appwindow.get_default_app_window().maximize_window()
            for _ in range(5):
                await app.next_update_async()
            existing = {window.title: window for window in get_viewport_window_instances()}
            for title, window in existing.items():
                if title not in VIEW_TITLES.values():
                    window.visible = False
            for role, title in VIEW_TITLES.items():
                window = self.windows.get(role) or existing.get(title) or create_viewport_window(title)
                if window is None:
                    raise RuntimeError("Could not create viewport: " + title)
                self.windows[role] = window
                window.visible = True
                api = window.viewport_api
                api.camera_path = "/OmniverseKit_Persp" if role == "main" else self.camera_paths[("head", "left", "right").index(role)]
                api.fill_frame = False
                api.resolution = tuple(self.settings["resolution"])
            self.build_inspector()
            handles = {}
            for title in ("Camera Inspector", "Stage", "Simulation", "Property", "Content", "Console", "DockSpace"):
                handle = ui.Workspace.get_window(title)
                if handle is None:
                    raise RuntimeError("Layout window unavailable: " + title)
                handles[title] = handle
                if title != "DockSpace":
                    handle.visible = True
            # Let deferred docking callbacks finish before applying the layout.
            for _ in range(5):
                await app.next_update_async()
            main = ui.Workspace.get_window(VIEW_TITLES["main"])
            main.dock_in(handles["DockSpace"], ui.DockPosition.SAME)
            await app.next_update_async()
            handles["Stage"].dock_in(main, ui.DockPosition.RIGHT, self.settings["tools_width_fraction"])
            render_settings = ui.Workspace.get_window("Render Settings")
            if render_settings is not None:
                render_settings.visible = False
            handles["Simulation"].dock_in(handles["Stage"], ui.DockPosition.BOTTOM, .52)
            handles["Property"].dock_in(handles["Simulation"], ui.DockPosition.SAME)
            handles["Content"].dock_in(main, ui.DockPosition.BOTTOM, self.settings["content_height_fraction"])
            handles["Console"].dock_in(handles["Content"], ui.DockPosition.SAME)
            handles["Camera Inspector"].dock_in(main, ui.DockPosition.LEFT, self.settings["inspector_width_fraction"])
            head = ui.Workspace.get_window(VIEW_TITLES["head"])
            left = ui.Workspace.get_window(VIEW_TITLES["left"])
            right = ui.Workspace.get_window(VIEW_TITLES["right"])
            head.dock_in(main, ui.DockPosition.RIGHT, self.settings["camera_column_fraction"])
            left.dock_in(head, ui.DockPosition.BOTTOM, self.settings["wrist_height_fraction"])
            right.dock_in(left, ui.DockPosition.RIGHT, .5)
            # Docking is deferred; select tabs only after the new groups exist.
            for _ in range(5):
                await app.next_update_async()
            for title in ("Content", "Stage", "Simulation", "Camera Inspector"):
                handles[title].focus()
            self.frame_scene()
            for _ in range(5):
                await app.next_update_async()
            write_json(self.output / "desktop_layout.json", {
                "status": "applied", "settings": self.settings,
                "views": {role: {"title": window.title, "camera_path": str(window.viewport_api.camera_path),
                                  "resolution": list(window.viewport_api.resolution), "fill_frame": window.viewport_api.fill_frame}
                          for role, window in self.windows.items()},
                "workspace": ui.Workspace.dump_workspace(),
            })
        except Exception as error:  # noqa: BLE001 -- A desktop failure must not stop the physics task.
            import traceback
            traceback.print_exc()
            write_json(self.output / "desktop_layout.json", {"status": "failed", "error": str(error)})

    def frame_scene(self):
        from isaacsim.core.utils.viewports import set_camera_view
        window = self.windows.get("main")
        if window is not None:
            window.viewport_api.camera_path = "/OmniverseKit_Persp"
            window.focus()
            set_camera_view(eye=np.asarray(self.settings["main_eye_m"]), target=np.asarray(self.settings["main_target_m"]),
                            camera_prim_path="/OmniverseKit_Persp", viewport_api=window.viewport_api)

    def view_head_camera(self):
        window = self.windows.get("head")
        if window is not None:
            window.viewport_api.camera_path = self.camera_paths[0]
            window.focus()

    def build_inspector(self):
        if self.inspector is not None:
            self.inspector.visible = True
            return
        import omni.kit.app
        import omni.usd
        from omni import ui
        from pxr import Gf, UsdGeom
        self.inspector = ui.Window("Camera Inspector", width=360, height=600)
        with self.inspector.frame, ui.ScrollingFrame(), ui.VStack(spacing=8, height=0):
            ui.Label("Camera Inspector", height=24)
            ui.Label("Read-only | USD camera axes | metres | WXYZ", word_wrap=True, height=35)
            selector = ui.ComboBox(1, "Overview", "Head camera", "Left wrist", "Right wrist")
            pose_label = ui.Label("", word_wrap=True, height=330)
            def focus_selected():
                role = tuple(VIEW_TITLES)[selector.model.get_item_value_model().as_int]
                self.windows[role].focus()
            ui.Button("View selected camera", clicked_fn=focus_selected)
        def update(_event):
            role = tuple(VIEW_TITLES)[selector.model.get_item_value_model().as_int]
            path = self.windows[role].viewport_api.camera_path
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            prim = stage.GetPrimAtPath(path)
            if not prim:
                return
            world = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            local = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
            def describe(name, transform):
                xyz = ", ".join(f"{v:.5f}" for v in transform.GetTranslation())
                q = transform.GetRotation().GetQuat()
                quat = ", ".join(f"{v:.5f}" for v in (q.GetReal(), *q.GetImaginary()))
                return f"{name} position (XYZ, m)\n{xyz}\n{name} quaternion (WXYZ)\n{quat}"
            pose_label.text = f"{path}\n\n{describe('World', world)}\n\n{describe('Local', local)}\n\nViewport: 4:3"
        self.inspector_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(update)

    def close(self):
        self.inspector_subscription = None
        if self.pending is not None and not self.pending.done():
            self.pending.cancel()
        if self.inspector is not None:
            self.inspector.destroy()
            self.inspector = None
