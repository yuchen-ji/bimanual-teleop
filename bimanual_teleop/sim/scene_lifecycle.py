"""Scene replacement requests and explicit render-resource ownership."""
from dataclasses import dataclass


@dataclass
class SceneReplacement:
    config: dict
    factory: object


class RenderResources:
    def __init__(self):
        self.products = {}
        self.annotators = []

    def product(self, product):
        self.products[str(product.path)] = product
        return product

    def annotator(self, annotator):
        self.annotators.append(annotator)
        return annotator

    def close(self):
        errors = []
        for annotator in self.annotators:
            try:
                annotator.detach()
            except Exception as error:  # noqa: BLE001 -- Release the other resources too.
                errors.append(error)
        self.annotators.clear()
        for product in self.products.values():
            try:
                product.destroy()
            except Exception as error:  # noqa: BLE001 -- Report failures after all cleanup attempts.
                errors.append(error)
        self.products.clear()
        if errors:
            raise RuntimeError(f"Scene render-resource cleanup failed: {errors}")


def run_scenes(app, config, factory, execute, clear_stage):
    """Never construct/close the application when replacing a scene."""
    while app.is_running():
        replacement = execute(app, config, factory)
        if replacement is None:
            return
        clear_stage()
        config, factory = replacement.config, replacement.factory
