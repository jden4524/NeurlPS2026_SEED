# seed/model_adapters/mllama_adapter.py

import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Any
from .base_adapter import BaseVisionLanguageAdapter

class MllamaAdapter(BaseVisionLanguageAdapter):
    """
    SEED adapter designed specifically for Llama 3.2 Vision (Mllama architecture).
    Uses a Qwen-style attention-weighted mechanism (Attention-Weighted Pooling) to extract features.
    """
    def __init__(self, model, processor):
        super().__init__(model)
        self.processor = processor
        self.original_sdpa = F.scaled_dot_product_attention
        
        self.e_vis = None
        self.injection_active = False
        self.injection_alpha = 0.0
        self.remaining_injection_steps = 0
        
        self.attention_cache = {}
        self.is_inside_target = [False]
        
        # 1. Locate the text model's bottom layer (used for Decode-stage injection)
        #    Compatible with old and new transformers structures:
        #      Old (<4.49): model.language_model (MllamaForCausalLM) -> .model.layers
        #      New (>=4.49): model.model.language_model (MllamaTextModel) -> .layers
        text_model = self._locate_text_model()
        if text_model is None:
            raise RuntimeError("Unable to locate the Mllama text model (language_model)")
        self.text_layer0 = text_model.layers[0]

        # 2. Locate the cross-attention layers of the text model (used to intercept the visual feature sequence)
        self.cross_attn_layers = [
            layer for layer in text_model.layers
            if hasattr(layer, "cross_attn") and layer.cross_attn is not None
        ]
        if not self.cross_attn_layers:
            raise RuntimeError("Unable to find a cross-attention layer in the language model!")

        # 3. 🌟 Locate the last layer of the vision tower (used to intercept the attention weight matrix)
        #    Compatible with old and new structures:
        #      Old: model.vision_model.encoder.layers
        #      New: model.model.vision_model.transformer.layers
        vision_layers = self._locate_vision_layers()
        if vision_layers is None:
            raise RuntimeError("Unable to locate the Mllama vision tower's self_attn module")
        self.vision_target_attn_module = vision_layers[-1].self_attn

    def _locate_text_model(self):
        """Return the MllamaTextModel that hosts .layers; compatible with multiple transformers version layouts."""
        m = self.model
        candidates = [
            getattr(getattr(m, "language_model", None), "model", None),  # Old: language_model.model
            getattr(getattr(m, "model", None), "language_model", None),  # New: model.language_model
            getattr(m, "language_model", None),                          # Rare case: directly attached to the root
        ]
        for cand in candidates:
            if cand is not None and hasattr(cand, "layers"):
                return cand
        return None

    def _locate_vision_layers(self):
        """Return the layers list of the vision encoder; compatible with encoder.layers / transformer.layers."""
        m = self.model
        vision_roots = [
            getattr(getattr(m, "model", None), "vision_model", None),  # New structure
            getattr(m, "vision_model", None),                          # Old structure
        ]
        for vm in vision_roots:
            if vm is None:
                continue
            for sub_name in ("transformer", "encoder"):
                sub = getattr(vm, sub_name, None)
                if sub is not None and hasattr(sub, "layers") and len(sub.layers) > 0:
                    return sub.layers
        return None

    def extract_global_vision_feature(self, inputs: dict) -> None:
        """Features are not in input_ids; leave empty and wait for the Hook to trigger"""
        self.e_vis = None
        return None

    def register_injection_hook(self):
        if not self.cross_attn_layers:
            return 

        first_cross_layer = self.cross_attn_layers[0]

        # ==========================================
        # 🕵️‍♂️ Task 1: Vision-tower SDPA hijacking (faithfully replicating Qwen)
        # ==========================================
        def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            if self.is_inside_target[0]:
                scale = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
                attn_weight = (query @ key.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn_weight + attn_mask
                self.attention_cache["vision_last_layer"] = torch.softmax(attn_weight, dim=-1).detach()
            return self.original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, **kwargs)

        def vis_pre(m, a, k): self.is_inside_target[0] = True; return a, k
        def vis_post(m, i, o): self.is_inside_target[0] = False

        # ==========================================
        # 🕵️‍♂️ Task 2: Intercept the cross-attention layer & weighted fusion to produce e_vis
        # ==========================================
        def cross_attn_pre_hook(module, args, kwargs):
            cross_states = kwargs.get("cross_attention_states", None)

            # Trigger during Prefill stage and only if not yet extracted
            if cross_states is not None and self.e_vis is None:
                # The cross_states passed by Mllama is pure visual features, shape [batch, num_vis_tokens, dim]
                all_vis = cross_states[0]

                # Start executing Qwen's weighted-sum logic!
                if "vision_last_layer" in self.attention_cache:
                    raw_attn = self.attention_cache["vision_last_layer"]
                    if raw_attn.dim() == 3: raw_attn = raw_attn.unsqueeze(0)

                    A_ref = raw_attn.sum(dim=-2).mean(dim=1)[0]
                    if A_ref.shape[0] != all_vis.shape[0]:
                        A_ref = F.interpolate(A_ref.view(1, 1, -1), size=all_vis.shape[0], mode='linear', align_corners=False).view(-1)

                    weights = A_ref / A_ref.sum()
                    # 💡 Attention weights are applied across thousands of vision tokens, condensing them into a single global feature
                    self.e_vis = (all_vis * weights.unsqueeze(-1)).sum(dim=0).detach()
                else:
                    # Extreme edge-case fallback: degrade to mean pooling
                    self.e_vis = all_vis.mean(dim=0).detach()

            return args, kwargs

        # ==========================================
        # 🕵️‍♂️ Task 3: Perform intervention injection at LLM Layer 0
        # ==========================================
        def text_layer0_pre_hook(module, args, kwargs):
            hidden_states = args[0]
            seq_len = hidden_states.shape[1]

            if seq_len == 1: # Only trigger during the Decode (token-emission) stage
                if self.injection_active and self.remaining_injection_steps > 0 and self.e_vis is not None:
                    modified_hs = hidden_states.clone()

                    # Dimension alignment: e_vis is 1D [dim], needs to be reshaped to [batch, 1, dim]
                    e_vis_aligned = self.e_vis.view(1, 1, -1).expand(modified_hs.shape[0], 1, -1)
                    e_vis_aligned = e_vis_aligned.to(modified_hs.device).to(modified_hs.dtype)

                    # ⚠️ Key point: Mllama's cross_attention_states is a raw vector used for K/V projection,
                    # not within the magnitude of the LLM's residual stream; adding it directly would break the hidden state distribution and produce garbled output.
                    # Here we align e_vis's RMS to the current hidden_state's RMS so that alpha maintains
                    # the same "injection strength" semantics as in Qwen.
                    hs_f = hidden_states.float()
                    ev_f = e_vis_aligned.float()
                    ref_rms  = hs_f.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
                    evis_rms = ev_f.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
                    e_vis_aligned = (ev_f * (ref_rms / evis_rms)).to(modified_hs.dtype)

                    # 💡 Forcefully inject and modify the current token's hidden state
                    modified_hs += self.injection_alpha * e_vis_aligned
                    return (modified_hs,) + args[1:], kwargs

            return args, kwargs

        # 🚀 Final mounting: the ultimate Hook combination spanning three major components
        h_vis_pre = self.vision_target_attn_module.register_forward_pre_hook(vis_pre, with_kwargs=True)
        h_vis_post = self.vision_target_attn_module.register_forward_hook(vis_post)
        h_cross = first_cross_layer.register_forward_pre_hook(cross_attn_pre_hook, with_kwargs=True)
        h_text = self.text_layer0.register_forward_pre_hook(text_layer0_pre_hook, with_kwargs=True)
        
        self._active_hooks.extend([h_vis_pre, h_vis_post, h_cross, h_text])
        
        # Replace the underlying computation engine
        torch.nn.functional.scaled_dot_product_attention = hooked_sdpa

    def trigger_injection(self, e_vis, alpha: float, steps: int):
        self.injection_alpha = alpha
        self.remaining_injection_steps = steps
        self.injection_active = True
        
    def step_injection(self):
        if self.injection_active:
            self.remaining_injection_steps -= 1
            if self.remaining_injection_steps <= 0:
                self.injection_active = False

    def remove_hooks(self):
        super().remove_hooks()
        # Clear state and restore the official SDPA to ensure the next evaluation is clean and uncontaminated
        self.e_vis = None
        self.attention_cache.clear()
        torch.nn.functional.scaled_dot_product_attention = self.original_sdpa