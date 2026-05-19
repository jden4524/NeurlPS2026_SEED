# seed/model_adapters/base_adapter.py
import abc
import torch
from typing import List, Tuple, Optional, Any

class BaseVisionLanguageAdapter(abc.ABC):
    """
    Base adapter class for low-level intervention on multimodal large models.
    Responsible for decoupling the differences between model architectures (e.g., Qwen-VL) when extracting visual features and injecting into the low-level Hidden State.
    """

    def __init__(self, model: torch.nn.Module):
        """
        :param model: An instantiated PyTorch model
        """
        self.model = model
        # Used to store registered hook handles to prevent memory leaks
        self._active_hooks: List[torch.utils.hooks.RemovableHandle] = []

    @abc.abstractmethod
    def extract_global_vision_feature(self, inputs: dict) -> torch.Tensor:
        """
        [Must implement] Perform a forward pass to obtain A and V by hijacking Attention,
        and compute the weighted compressed vector e_vis that represents the core semantics of the whole image.

        :param inputs: The input dictionary for the model's forward pass
        :return: e_vis (Shape: [Dim])
        """
        pass

    @abc.abstractmethod
    def register_injection_hook(self):
        """
        [Must implement] Register a Forward Pre-Hook at the bottom of the LLM (typically the first Transformer layer).
        When the trigger condition is met, inject e_vis into the current Hidden State.
        """
        pass

    @abc.abstractmethod
    def trigger_injection(self, e_vis: torch.Tensor, alpha: float, steps: int):
        """
        Activate the injection state, called by the Generator when the entropy criterion is met.
        """
        pass

    @abc.abstractmethod
    def step_injection(self):
        """
        Called after each decoding step ends, decrements the remaining injection steps.
        """
        pass

    def remove_hooks(self):
        """Clean up all registered Hooks to prevent OOM"""
        for handle in self._active_hooks:
            handle.remove()
        self._active_hooks.clear()

    def __del__(self):
        self.remove_hooks()