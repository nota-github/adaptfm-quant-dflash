"""QAD (Quantization-Aware Distillation) toolbox.

Modules:
- quant_config: cyankiwi-mask INT4 g32 sym modelopt config + verifier
- data:        Jackrong tokenize + pack (think + no_think views)
- smoke:       single-step memory/API smoke
- train:       Composite-KL training loop (teacher BF16 + student INT4 fake-quant)
- export:      Two-stage compressed-tensors export (Stage A: bake; Stage B: oneshot)
"""
