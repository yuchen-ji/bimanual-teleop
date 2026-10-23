"""Run pip's script with the selected Python interpreter, including Isaac Python."""

import importlib.util
import runpy
from pathlib import Path


def main():
    specification = importlib.util.find_spec("pip")
    if specification is None or specification.origin is None:
        raise SystemExit("pip is unavailable in this Python environment")
    runpy.run_path(
        str(Path(specification.origin).with_name("__main__.py")), run_name="__main__"
    )


if __name__ == "__main__":
    main()
