import os, sys
import argparse
from types import SimpleNamespace

from fdfat.cfg import get_cfg
from fdfat.main import do_train

def entrypoint():
    default_cfg = get_cfg()

    parser = argparse.ArgumentParser(description='train loop.')
    for k, v in default_cfg:
        parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    args = parser.parse_args()

    do_train(SimpleNamespace(**vars(args)))