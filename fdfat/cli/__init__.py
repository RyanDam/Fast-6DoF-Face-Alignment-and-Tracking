import argparse
from types import SimpleNamespace

from fdfat.utils.logger import LOGGER
from fdfat.cfg import get_cfg
# from fdfat.main import TrainEngine, ValEngine, TestEngine, 
from fdfat.engine import trainer, validator, tester, exporter

def entrypoint():
    default_cfg = get_cfg()

    parser = argparse.ArgumentParser(description='train loop.')
    for k, v in default_cfg:
        parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    args = parser.parse_args()
    args = SimpleNamespace(**vars(args))
    
    if args.task == "train":
        engine = trainer.TrainEngine(args)
        engine.prepare()
        engine.do_train()
    elif args.task == "val":
        engine = validator.ValEngine(args)
        engine.prepare()
        engine.do_validate()
    elif args.task == "predict":
        engine = tester.TestEngine(args)
        if args.input is None:
            raise ValueError("Input is empty")
        lmk, rendered = engine.predict(args.input, render=True)
        rendered.save("predict.jpg")
    elif args.task == "export":
        engine = exporter.ExportEngine(args)
        engine.export()
