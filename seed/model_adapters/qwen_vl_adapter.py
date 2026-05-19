# seed/model_adapters/qwen_vl_adapter.py
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple
from .base_adapter import BaseVisionLanguageAdapter

class QwenVLAdapter(BaseVisionLanguageAdapter):
    def __init__(self, model, processor):
        super().__init__(model)
        self.processor = processor
        self.original_sdpa = F.scaled_dot_product_attention
        
        self.vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        # State variables
        self.vis_indices = []
        self.e_vis = None
        self.injection_active = False
        self.injection_alpha = 0.0
        self.remaining_injection_steps = 0
        
        self.attention_cache = {}
        self.is_inside_target = [False]

    def extract_global_vision_feature(self, inputs: dict) -> torch.Tensor:
        """Parse the index positions of vision Tokens"""
        ids_list = inputs["input_ids"][0].tolist() if inputs["input_ids"].dim() == 2 else inputs["input_ids"].tolist()
        start_positions = [i for i, x in enumerate(ids_list) if x == self.vision_start_id]
        end_positions = [i for i, x in enumerate(ids_list) if x == self.vision_end_id]
        self.vis_indices = [(s + 1, e) for s, e in zip(start_positions, end_positions)]
        return None

    def register_injection_hook(self):
        """Mount the core Layer 0 hook: Prefill extraction + Decode injection"""
        try:
            layer0 = self.model.model.language_model.layers[0] if hasattr(self.model, "model") and hasattr(self.model.model, "language_model") else self.model.language_model.model.layers[0]
            visual_core = self.model.model.visual if hasattr(self.model, "model") else self.model.visual
            target_attn_module = visual_core.blocks[-1].attn
        except AttributeError:
            raise RuntimeError("Unable to locate LLM Layer 0 or the vision tower.")

        # 1. Local SDPA hijacking (used only at the moment of Prefill)
        def hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            if self.is_inside_target[0]:
                scale = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
                attn_weight = (query @ key.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn_weight + attn_mask
                self.attention_cache["vision_last_layer"] = torch.softmax(attn_weight, dim=-1).detach()
            return self.original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, **kwargs)

        # Pre/post hooks for the vision tower
        def vis_pre(m, a, k): self.is_inside_target[0] = True; return a, k
        def vis_post(m, i, o): self.is_inside_target[0] = False

        # 2. Core LLM Layer 0 hook
        def first_layer_pre_hook(module, args, kwargs):
            hidden_states = args[0]
            seq_len = hidden_states.shape[1]

            if seq_len > 1:
                # ================= Prefill: compute e_vis =================
                vis_embs = [hidden_states[0, s:e, :] for s, e in self.vis_indices if s < e <= seq_len]
                
                if vis_embs and "vision_last_layer" in self.attention_cache:
                    all_vis = torch.cat(vis_embs, dim=0)
                    raw_attn = self.attention_cache["vision_last_layer"]
                    if raw_attn.dim() == 3: raw_attn = raw_attn.unsqueeze(0)
                    
                    A_ref = raw_attn.sum(dim=-2).mean(dim=1)[0]
                    if A_ref.shape[0] != all_vis.shape[0]:
                        A_ref = F.interpolate(A_ref.view(1, 1, -1), size=all_vis.shape[0], mode='linear', align_corners=False).view(-1)
                    
                    weights = A_ref / A_ref.sum()
                    self.e_vis = (all_vis * weights.unsqueeze(-1)).sum(dim=0).detach()
                    # print("🎯 Visual feature extraction succeeded!")

            else:
                # ================= Decode: injection intervention =================
                if self.injection_active and self.remaining_injection_steps > 0 and self.e_vis is not None:
                    modified_hs = hidden_states.clone()
                    modified_hs[:, -1, :] += self.injection_alpha * self.e_vis.to(modified_hs.device).to(modified_hs.dtype)
                    # print(f"💉 Injection intervention triggered! Pre-intervention value: {hidden_states[:, -1, :].mean().item()}, post-intervention value: {modified_hs[:, -1, :].mean().item()}")
                    return (modified_hs,) + args[1:], kwargs

            return args, kwargs

        # 3. Perform the mounting
        h1 = target_attn_module.register_forward_pre_hook(vis_pre, with_kwargs=True)
        h2 = target_attn_module.register_forward_hook(vis_post)
        h3 = layer0.register_forward_pre_hook(first_layer_pre_hook, with_kwargs=True)
        self._active_hooks.extend([h1, h2, h3])
        
        # Global modification (will be restored on remove_hooks)
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
        # The most important step: fully restore the official SDPA so other evaluations/generations are not affected at all
        torch.nn.functional.scaled_dot_product_attention = self.original_sdpa