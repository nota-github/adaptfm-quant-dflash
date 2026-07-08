# from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
try:
    # Optional: eagle3 target backend is sglang-coupled. Allow import without
    # sglang for the dflash hf path (sglang pins transformers==4.57.1, which is
    # incompatible with newer targets such as Qwen3.5 needing transformers>=5.x).
    from .target.eagle3_target_model import (
        CustomEagle3TargetModel,
        HFEagle3TargetModel,
        SGLangEagle3TargetModel,
        get_eagle3_target_model,
    )
except ImportError:
    CustomEagle3TargetModel = None
    HFEagle3TargetModel = None
    SGLangEagle3TargetModel = None
    get_eagle3_target_model = None

__all__ = [
    "LlamaForCausalLMEagle3",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "get_eagle3_target_model",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
