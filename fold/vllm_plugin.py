"""vLLM plugin that applies FOLD to the drafter.

vLLM loads general plugins in every process. When FOLD_CONFIG is set (a JSON
object with the keyword arguments of fold.attention.install), the GPU model
runner is wrapped so that the drafter gets full KV storage when it is loaded
and FOLD is installed right before CUDA graph capture.

    FOLD_CONFIG='{"window": 2048, "alpha": 2.0}'
"""
import json
import logging
import os

import torch

logger = logging.getLogger("vllm.fold")


def register():
    raw = os.environ.get("FOLD_CONFIG")
    if not raw:
        return
    config = json.loads(raw)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_fold", False):
        return
    from .attention import install, use_full_draft_storage

    load_model, capture_model = GPUModelRunner.load_model, GPUModelRunner.capture_model

    def load_model_with_fold(self, *args, **kwargs):
        load_model(self, *args, **kwargs)
        if getattr(self.speculator, "model", None) is not None:
            use_full_draft_storage(self.speculator)

    def capture_model_with_fold(self, *args, **kwargs):
        if getattr(self.speculator, "model", None) is not None:
            with torch.inference_mode():
                report = install(self.speculator, **config)
            logger.info("FOLD installed: %s", report)
        return capture_model(self, *args, **kwargs)

    GPUModelRunner.load_model = load_model_with_fold
    GPUModelRunner.capture_model = capture_model_with_fold
    GPUModelRunner._fold = True
