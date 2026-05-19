# seed/config.py
from dataclasses import dataclass

@dataclass
class SEEDConfig:
    # ==========================================
    # 🎯 Trigger Conditions
    # ==========================================

    # Trigger window start: how many generation steps must pass before intervention is allowed (protects the earliest context-building stage)
    window_start: int = 1

    # Trigger window end: stop intervening after this many generation steps (protects the late conclusion-drawing stage of long-range reasoning)
    window_end: int = 40

    # Absolute entropy threshold: even if entropy is decreasing (converging), it must exceed this value to trigger, indicating that the model "lacks confidence"
    entropy_threshold: float = 1.0

    # ==========================================
    # 💉 Injection Parameters
    # ==========================================

    # Injection strength (Alpha): scaling coefficient when the visual feature e_vis is added to the Hidden State
    injection_alpha: float = 0.5

    # Number of steps (K): after triggering, inject the visual feature at the first layer's input for this many consecutive steps
    injection_steps: int = 3