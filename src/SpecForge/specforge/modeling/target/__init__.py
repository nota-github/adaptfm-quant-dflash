try:
    # eagle3_target_model is heavily coupled to sglang. Keep it optional so the
    # dflash hf backend (which does not need it) can be imported without sglang
    # installed. sglang hard-pins transformers==4.57.1, incompatible with newer
    # target models (e.g. Qwen3.5) that require transformers>=5.x.
    from .eagle3_target_model import (
        CustomEagle3TargetModel,
        Eagle3TargetModel,
        HFEagle3TargetModel,
        SGLangEagle3TargetModel,
        get_eagle3_target_model,
    )
except ImportError:
    CustomEagle3TargetModel = None
    Eagle3TargetModel = None
    HFEagle3TargetModel = None
    SGLangEagle3TargetModel = None
    get_eagle3_target_model = None
from .target_head import TargetHead

__all__ = [
    "Eagle3TargetModel",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "get_eagle3_target_model",
    "TargetHead",
]
