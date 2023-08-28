# import os
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import argparse
from types import SimpleNamespace

from fdfat.utils.logger import LOGGER
from fdfat.cfg import get_cfg

def parse_bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def entrypoint():
    default_cfg = get_cfg()

    parser = argparse.ArgumentParser(description='train loop.')
    for k, v in default_cfg:
        t = type(v)
        t = parse_bool if t == bool else (t if v is not None else str)
        parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()

    print(args)

    args = SimpleNamespace(**vars(args))
    
    if args.task == "train":
        from fdfat.engine import trainer
        engine = trainer.TrainEngine(args)
        engine.prepare()
        engine.do_train()
    elif args.task == "val":
        from fdfat.engine import validator
        engine = validator.ValEngine(args)
        engine.prepare()
        engine.do_validate()
    elif args.task == "predict":
        from fdfat.engine import predictor
        engine = predictor.PredictEngine(args)
        if args.input is None:
            raise ValueError("Input is empty")
        lmk, rendered = engine.predict(args.input, render=True)
        rendered.save("predict.jpg")
    elif args.task == "export":
        from fdfat.engine import exporter
        engine = exporter.ExportEngine(args)
        engine.export()
    elif args.task == "track":
        from fdfat.tracking.facial_sort import FacialSORT
        sort = FacialSORT(args)
        sort.run()
    else:
        raise AttributeError(f"task '{args.task}' is not supported")
