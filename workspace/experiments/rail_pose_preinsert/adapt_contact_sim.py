"""Native simulation for contact trials fitted from a recorded hand posture."""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]


def main():
    import yaml
    from adapt_contact_control import AdaptContactController

    from bimanual_teleop.sim.engine import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = yaml.safe_load((args.output/'effective.yaml').read_text())
    task = data.pop('experiment')
    runtime = json.loads((args.output/'runtime.json').read_text())
    factory = partial(AdaptContactController, task=task, output=args.output, runtime=runtime,
                      mode='step', headless=False, perception='oracle', grasp='physics',
                      adaptation_output=args.output)
    run(data, '', gui=True, standalone=True, action_source_factory=factory, keep_gui_on_error=True)


if __name__ == '__main__':
    main()
