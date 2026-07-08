__all__ = [
    "DFlashDraftModel",
    "extract_context_feature",
    "sample",
]


def __getattr__(name):
    if name in {"DFlashDraftModel", "extract_context_feature", "sample"}:
        from .model import DFlashDraftModel, extract_context_feature, sample

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
