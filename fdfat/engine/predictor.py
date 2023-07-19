import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Union
from types import SimpleNamespace

from fdfat import __version__
from fdfat.utils.logger import LOGGER
from fdfat.engine.base import BaseEngine
from fdfat.utils.model_utils import preprocess
from fdfat.utils.utils import render_lmk

class PredictEngine(BaseEngine):

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):
        super().__init__(cfg_)
        self.load_model()

        self.target_checkpoint_path = self.cfgs.checkpoint if self.cfgs.checkpoint is not None else self.save_best
        self.load_checkpoint(self.target_checkpoint_path)

        self.net.eval()

    def predict(self, input, render=False):

        if isinstance(input, str):
            input = Image.open(input)

        preprocessed = preprocess(input, self.cfgs.imgsz)
        preprocessed = torch.from_numpy(preprocessed.astype(np.float32)).to(self.cfgs.device)
        with torch.no_grad():
            if self.cfgs.test_warmup:
                for _ in range(5):
                    _ = self.net(preprocessed)
            start = time.time()
            y = self.net(preprocessed)
            y = y.detach().cpu().numpy()
            end = time.time()

        LOGGER.info(f"Predicted in {int((end-start)*1000):d}ms")

        lmk = y[0][:70*2].reshape((70,2))

        if render:
            rendered = render_lmk(input.copy(), (lmk+0.5)*input.size[0], point_size=1)
            return lmk, rendered

        return lmk