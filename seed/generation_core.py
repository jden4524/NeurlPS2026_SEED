# seed/generation_core.py
import torch
import torch.nn.functional as F
from typing import Dict, Any
from transformers import LogitsProcessor, LogitsProcessorList

# ==========================================
# 🧠 Brain monitor: Entropy Monitor
# ==========================================
class EntropyMonitorLogitsProcessor(LogitsProcessor):
    """
    A non-intrusive eavesdropper mounted inside model.generate().
    It computes the entropy between every two generated tokens and orchestrates the Adapter to inject features.
    """
    def __init__(self, adapter, config):
        self.adapter = adapter
        self.config = config

        # Statistics panel
        self.step = 0
        self.injections_triggered = 0
        self.injection_details = []

        # Memory-safe design: avoid the CPU compute explosion caused by calling sum(list) every step
        self.entropy_history = []
        self.running_sum_entropy = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1. Safely compute information entropy
        probs = F.softmax(scores, dim=-1)
        entropy_tensor = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        # 💥 [Memory safety lock 1]: .item() must be called!
        # Force-detach the Tensor on the GPU from the computation graph and convert it to a plain Python float.
        # Otherwise self.entropy_history will become a bottomless pit of VRAM leaks!
        current_entropy = entropy_tensor[0].item()

        # 2. Maintain the running average (O(1) complexity)
        self.entropy_history.append(current_entropy)
        self.running_sum_entropy += current_entropy
        running_avg_entropy = self.running_sum_entropy / len(self.entropy_history)

        # 3. Advance the Adapter's injector state
        self.adapter.step_injection()

        # 4. Brain decision mechanism
        in_window = self.config.window_start <= self.step <= self.config.window_end
        is_converging = current_entropy < running_avg_entropy
        above_threshold = current_entropy > self.config.entropy_threshold
        can_inject = (not self.adapter.injection_active) and (self.adapter.e_vis is not None)

        if in_window and is_converging and above_threshold and can_inject:
            # Trigger injection: tell the Adapter to pull the switch on the next Token Forward
            self.adapter.trigger_injection(
                e_vis=self.adapter.e_vis,
                alpha=self.config.injection_alpha,
                steps=self.config.injection_steps
            )
            self.injections_triggered += 1
            self.injection_details.append({
                "trigger_step": self.step,
                "entropy": round(current_entropy, 4),
                "avg_entropy": round(running_avg_entropy, 4)
            })

        self.step += 1

        # A LogitsProcessor must return the original scores
        return scores


# ==========================================
# 🚀 Scheduling center: SEED Generator
# ==========================================
class SEEDGenerator:
    def __init__(self, model, processor, adapter, config):
        self.model = model
        self.processor = processor
        self.adapter = adapter
        self.config = config

    def generate(self, inputs: Dict[str, Any], max_new_tokens: int = 100, return_stats: bool = False, **kwargs):
        """A lightweight launcher that fully embraces the official generate"""

        # 1. Pre-battle preparation: extract features and mount the injection hook
        self.adapter.extract_global_vision_feature(inputs)
        self.adapter.register_injection_hook()

        # 2. Initialize our entropy-monitoring brain
        entropy_processor = EntropyMonitorLogitsProcessor(self.adapter, self.config)
        logits_processor = LogitsProcessorList([entropy_processor])

        try:
            # 3. Hand control back to the official generate
            # Enjoy a perfect KV Cache, mRoPE positional encoding, and FlashAttention optimizations!
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    logits_processor=logits_processor,
                    **kwargs
                )
        finally:
            # 💥 [Memory safety lock 2]: absolute cleanup guarantee!
            # Whether generation completes successfully or is interrupted by an unexpected error,
            # the finally block guarantees that the Hook inserted at the bottom of the LLM is forcibly removed, preventing pollution of the global model state.
            self.adapter.remove_hooks()

        # 4. Extract statistics
        stats = {
            "injections_triggered": entropy_processor.injections_triggered,
            "injection_details": entropy_processor.injection_details,
            "final_entropy_avg": (entropy_processor.running_sum_entropy / len(entropy_processor.entropy_history)) if entropy_processor.entropy_history else 0
        }
        
        if return_stats:
            return output_ids, stats
        return output_ids