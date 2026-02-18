# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import json
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import (
    ARM64,
    IS_JETSON,
    LINUX,
    LOGGER,
    PYTHON_VERSION,
    ROOT,
    YAML,
    is_jetson,
)
from ultralytics.utils.checks import (
    check_executorch_requirements,
    check_requirements,
    check_suffix,
    check_version,
    check_yaml,
    is_rockchip,
)
from ultralytics.utils.downloads import attempt_download_asset, is_url
from ultralytics.utils.nms import non_max_suppression


def check_class_names(names: list | dict) -> dict[int, str]:
    """Check class names and convert to dict format if needed.

    Args:
        names (list | dict): Class names as list or dict format.

    Returns:
        (dict): Class names in dict format with integer keys and string values.

    Raises:
        KeyError: If class indices are invalid for the dataset size.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    """Apply default class names to an input YAML file or return numerical class names.

    Args:
        data (str | Path, optional): Path to YAML file containing class names.

    Returns:
        (dict): Dictionary mapping class indices to class names.
    """
    if data:
        try:
            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """Minimal AutoBackend for local PyTorch `.pt` models.

    This minimal wrapper supports loading local `.pt` checkpoint files or
    in-memory `torch.nn.Module` instances and running inference. It does not
    attempt to support other backend formats.
    """

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo26n.pt",
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """Initialize the AutoBackend for inference.

        Args:
            model (str | torch.nn.Module): Path to the model weights file or a module instance.
            device (torch.device): Device to run the model on.
            dnn (bool): Use OpenCV DNN module for ONNX inference.
            data (str | Path, optional): Path to the additional data.yaml file containing class names.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization.
            verbose (bool): Enable verbose logging.
        """
        super().__init__()
        nn_module = isinstance(model, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            mnn,
            ncnn,
            imx,
            rknn,
            pte,
            axelera,
            triton,
        ) = self._model_type("" if nn_module else model)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu or rknn  # BHWC formats (vs torch BCHW)
        stride, ch = 32, 3  # default stride and channels
        end2end, dynamic = False, False
        metadata, task = None, None

        # Set device
        cuda = isinstance(device, torch.device) and torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):  # GPU dataloader formats
            device = torch.device("cpu")
            cuda = False

        # Download if not local
        w = attempt_download_asset(model) if pt else model  # weights path

        # PyTorch (in-memory or file)
        if nn_module or pt:
            if nn_module:
                pt = True
                if fuse:
                    if IS_JETSON and is_jetson(jetpack=5):
                        # Jetson Jetpack5 requires device before fuse https://github.com/ultralytics/ultralytics/pull/21028
                        model = model.to(device)
                    model = model.fuse(verbose=verbose)
                model = model.to(device)
            else:  # pt file
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=device, fuse=fuse)  # load model, ckpt

            # Common PyTorch model processing
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            ch = model.yaml.get("channels", 3)
            for p in model.parameters():
                p.requires_grad = False
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            end2end = getattr(model, "end2end", False)

        # TorchScript
        elif jit:
            import torchvision  # noqa - https://github.com/ultralytics/ultralytics/pull/19747

            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = YAML.load(metadata)
            
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch", "channels"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
                    metadata[k] = ast.literal_eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
            kpt_names = metadata.get("kpt_names")
            end2end = metadata.get("end2end", False) or metadata.get("args", {}).get("nms", False)
            dynamic = metadata.get("args", {}).get("dynamic", dynamic)
            ch = metadata.get("channels", 3)
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"Metadata not found for 'model={w}'")

        # Check names
        if "names" not in locals():  # names missing
            names = default_class_names(data)
        names = check_class_names(names)

        self.__dict__.update(locals())  # assign all variables to self

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run inference on input tensor and return model outputs.

        Supports the minimal options used by the project. Returns a tensor or
        list of tensors depending on model output.
        """
        _b, _ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)

        # TorchScript
        elif self.jit:
            y = self.model(im)

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert numpy array to torch tensor on the model device."""
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """Warm up model with a dummy forward pass on the current device."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup model
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)  # 16 boxes works best empirically
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)  # warmup NMS

    @staticmethod
    def _model_type(p: str = "path/to/model.pt") -> list[bool]:
        """Return model-type flags. Minimal: only local `.pt` supported."""
        # Expected order unpacked by AutoBackend.__init__ (18 flags + triton):
        # pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu,
        # tfjs, paddle, mnn, ncnn, imx, rknn, pte, axelera, triton
        if not isinstance(p, str) or p == "":
            return [False] * 18 + [False]

        # Disallow URLs and non-.pt formats in this minimal build
        if is_url(p) or not p.lower().endswith(".pt"):
            raise TypeError("Only local PyTorch .pt models are supported in this minimal build.")

        # Only `.pt` supported: set `pt` flag True, others False, triton False
        types = [False] * 18
        types[0] = True
        return [*types, False]
