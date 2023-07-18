import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Union
from types import SimpleNamespace

from fdfat import __version__
from fdfat.utils.logger import LOGGER
from fdfat.engine.base import BaseEngine
from fdfat.utils.model_utils import preprocess, get_latest_opset
from fdfat.utils.utils import render_lmk

class ExportEngine(BaseEngine):

    def __init__(self, cfg_: Union[str, Path, Dict, SimpleNamespace]):
        super().__init__(cfg_)
        self.load_model() # prevent miss behaviour with batchnorm not in eval mode

        self.target_checkpoint_path = self.cfgs.checkpoint if self.cfgs.checkpoint is not None else self.save_best
        self.load_checkpoint(self.target_checkpoint_path)

        self.net.eval()

    def export(self):
        import torch

        if self.cfgs.export_format == "torchscript":
            target_saved_path = self.save_wdir / 'torchscript.pt'

            temp_input = torch.rand(1, 3, self.cfgs.imgsz, self.cfgs.imgsz, requires_grad=True)

            traced_net = torch.jit.trace(self.net, temp_input)
            
            traced_net.save(target_saved_path)

            LOGGER.info(f"Model saved to: {target_saved_path}")

        elif self.cfgs.export_format in ["onnx", "saved_model", "tflite"]:
            import torch.onnx
            import onnx

            onnx_saved_path = self.save_wdir / 'model.onnx'

            temp_input = torch.rand(1, 3, self.cfgs.imgsz, self.cfgs.imgsz, requires_grad=True)

            output = self.net(temp_input)

            # Export the model
            torch.onnx.export(self.net,               # model being run
                            temp_input,                         # model input (or a tuple for multiple inputs)
                            onnx_saved_path,   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            opset_version=get_latest_opset(),
                            input_names = ['input'],   # the model's input names
                            output_names = ['output']) # the model's output names

            if self.cfgs.export_simplify:
                import onnxsim
                model_onnx = onnx.load(onnx_saved_path)  # load onnx model
                LOGGER.info(f'Simplifying with onnxsim {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'Simplified ONNX model could not be validated'
                onnx.save(model_onnx, onnx_saved_path)

            LOGGER.info(f"Model saved to: {onnx_saved_path}")

            if self.cfgs.export_format in ["saved_model", "tflite"]:
                import onnx2tf
                
                tf_saved_path = self.save_wdir / 'saved_model'

                onnx2tf.convert(
                    input_onnx_file_path=onnx_saved_path,
                    output_folder_path=tf_saved_path,
                    not_use_onnxsim=True,
                    non_verbose=True
                )

                LOGGER.info(f"Model saved to: {tf_saved_path}")