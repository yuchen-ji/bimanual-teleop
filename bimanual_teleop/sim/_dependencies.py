"""Load optional simulation dependencies only when simulation code runs.

Importing the package must remain safe on an operator PC without the simulator,
vendor SDKs, numerical packages, or access to devices.
"""

from importlib import import_module


class DeferredImport:
    def __init__(self, module: str, attribute: str | None = None):
        self.module = module
        self.attribute = attribute
        self.loaded = None

    def _load(self):
        if self.loaded is None:
            module = import_module(self.module)
            self.loaded = getattr(module, self.attribute) if self.attribute else module
        return self.loaded

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)


np = DeferredImport("numpy")
yaml = DeferredImport("yaml")
zmq = DeferredImport("zmq")
msgpack = DeferredImport("msgpack")
zarr = DeferredImport("zarr")
trimesh = DeferredImport("trimesh")
Rotation = DeferredImport("scipy.spatial.transform", "Rotation")
Blosc = DeferredImport("numcodecs", "Blosc")
Image = DeferredImport("PIL.Image")
