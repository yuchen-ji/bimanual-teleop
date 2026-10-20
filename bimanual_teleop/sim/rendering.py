"""Shared native-resolution camera rendering settings; import after SimulationApp starts."""


def configure_camera_rendering():
    import carb
    import omni.replicator.core as rep

    # Isaac 5.1: 1 = TAA, 2 = FXAA. TAA also handles stochastic lighting noise.
    # This avoids depending on NGX/DLAA availability inside the container.
    values = {
        "/rtx/post/aa/op": 1,
        "/rtx/directLighting/sampledLighting/samplesPerPixel": 8,
        "/rtx/reflections/sampledLighting/samplesPerPixel": 4,
        "/rtx/indirectDiffuse/fetchSampleCount": 4,
        "/rtx/post/motionblur/enabled": False,
        "/rtx/post/dof/enabled": False,
    }
    settings = carb.settings.get_settings()
    rep.settings.set_render_rtx_realtime(antialiasing="TAA")
    for path, value in values.items():
        # New render products and stage resets can re-read the defaults.
        settings.set(path.replace("/rtx/", "/rtx-defaults/", 1), value)
        settings.set(path, value)
    return {path: settings.get(path) for path in values}
