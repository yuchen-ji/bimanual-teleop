import unittest

from bimanual_teleop.sim.scene_lifecycle import (
    RenderResources,
    SceneReplacement,
    run_scenes,
)


class LifecycleTests(unittest.TestCase):
    def test_multiple_scene_replacements_keep_the_application(self):
        app = type("App", (), {"is_running": lambda self: True})()
        calls = []

        def execute(instance, config, factory):
            self.assertIs(instance, app)
            self.assertEqual(factory, config["version"])
            calls.append(config["version"])
            if config["version"] < 3:
                return SceneReplacement({"version": config["version"]+1}, config["version"]+1)

        clears = []
        run_scenes(app, {"version": 0}, 0, execute, lambda: clears.append(True))
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(len(clears), 3)

    def test_shared_product_is_destroyed_once_after_all_annotators_detach(self):
        order = []
        resource = RenderResources()
        product = type("Product", (), {"path": "/head", "destroy": lambda self: order.append("destroy")})()
        resource.product(product)
        resource.product(product)
        for name in ("rgb", "depth", "mask"):
            resource.annotator(type("Annotator", (), {"detach": lambda self, n=name: order.append(n)})())
        resource.close()
        resource.close()
        self.assertEqual(order, ["rgb", "depth", "mask", "destroy"])

    def test_one_detach_failure_does_not_skip_remaining_cleanup(self):
        resource = RenderResources()
        def fail(_):
            raise ValueError("detach failed")
        resource.annotator(type("Annotator", (), {"detach": fail})())
        removed = []
        resource.product(type("Product", (), {"path": "/head", "destroy": lambda self: removed.append(True)})())
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            resource.close()
        self.assertEqual(removed, [True])
