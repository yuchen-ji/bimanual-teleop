"""SDK-free import and transport checks for the public contracts."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


SOURCE_DIR = Path(__file__).resolve().parents[1]

# Run with isolated Python and no site packages. Keep the guard active during
# annotation resolution and serialization, not just the initial imports.
ISOLATED_SETUP = """
# Load stdlib copy before the guard: it probes org.python.core on CPython too.
import copy
import importlib
import importlib.abc
import inspect
from pathlib import Path
import pickle
import pkgutil
import sys
import typing

source_dir = Path(sys.argv[1])
sys.path.insert(0, str(source_dir))
forbidden_attempts = []

class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition('.')[0]
        if root != 'bimanual_teleop' and root not in sys.stdlib_module_names:
            forbidden_attempts.append('import ' + fullname)
            raise ImportError('Contracts must not import external packages: ' + fullname)

def reject_device_access(event, args):
    forbidden = (
        event.startswith('socket.')
        # ctypes imports the existing CPython process handle with PyDLL(None).
        # Loading any named shared library (including a vendor SDK) is forbidden.
        or (event == 'ctypes.dlopen' and args[0] is not None)
        or event in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}
    )
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = args[0].decode() if isinstance(args[0], bytes) else args[0]
        forbidden = forbidden or path.startswith('/dev/')
    if forbidden:
        forbidden_attempts.append(event)
        raise RuntimeError('Contracts must not access hardware or launch processes: ' + event)

sys.meta_path.insert(0, ImportGuard())
sys.addaudithook(reject_device_access)

modules = [importlib.import_module('bimanual_teleop')]
# Discover importable source packages, excluding test code. Runtime caches,
# recordings and vendored SDK files are not public contract modules.
for entry in pkgutil.walk_packages([str(source_dir / 'bimanual_teleop')], 'bimanual_teleop.'):
    if 'tests' in entry.name.split('.'):
        continue
    modules.append(importlib.import_module(entry.name))
"""


class ContractTests(unittest.TestCase):
    def run_isolated(self, code: str) -> None:
        script = ISOLATED_SETUP + "\n" + textwrap.dedent(code)
        # Catch forbidden attempts even if future package code catches the
        # guard's exception and silently falls back to another path.
        script += "\nassert not forbidden_attempts, forbidden_attempts\n"
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", script, str(SOURCE_DIR)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_modules_import_without_sdks_or_device_access(self) -> None:
        self.run_isolated("assert modules, 'No package modules were discovered'")

    def test_annotations_resolve_on_records_and_inherited_protocol_methods(self) -> None:
        self.run_isolated("""
            resolved_count = 0
            for module in modules:
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    if cls.__module__ != module.__name__:
                        continue
                    resolved_count += len(typing.get_type_hints(cls))
                    # getmembers includes inherited generic protocol methods.
                    for _, member in inspect.getmembers(cls):
                        if isinstance(member, property):
                            member = member.fget
                        if inspect.isfunction(member):
                            resolved_count += len(typing.get_type_hints(member))
            assert resolved_count, 'No annotations were checked'
        """)

    def test_plain_records_support_python_transport_with_missing_data(self) -> None:
        # This verifies ordinary Python transport only; no untrusted pickle is loaded.
        self.run_isolated("""
            from bimanual_teleop.types import (
                HandSkeleton, JointState, OperatorInput, Sample,
                SampleHeader, SampleRef, SourceTime,
            )

            raw_ref = SampleRef('glove.left.emf', 'session-a/device-epoch-2', 31)
            skeleton_ref = SampleRef('glove.left.skeleton', 'session-a/device-epoch-2', 9)
            hands = {'left': Sample(
                    SampleHeader(skeleton_ref, 5000001, True),
                    HandSkeleton(
                        frame='left_glove',
                        joint_names=('wrist',),
                        positions_m=((0.01, -0.02, 0.03),),
                        confidences=(0.8,),
                        source_refs=(raw_ref,),
                    ),
                )}
            operator = OperatorInput(wrists={})
            feedback = Sample(
                SampleHeader(
                    ref=SampleRef('hand.left.joints', 'session-a/device-epoch-1', 101),
                    received_monotonic_ns=5000003,
                    valid=False,
                    source_time=SourceTime(34567, 'us', 'left_hand', 'device_reported'),
                    source_sequence=4,
                ),
                JointState(
                    joint_names=('thumb_S1', 'thumb_S2'),
                    position_rad=(0.3, None),
                    motor_current_a=(0.4, None),
                ),
            )
            records = (operator, hands, feedback)
            restored = pickle.loads(pickle.dumps(records))
            assert restored == records, 'Transport lost missing values or raw provenance'
        """)


if __name__ == "__main__":
    unittest.main()
