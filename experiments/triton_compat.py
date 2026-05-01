"""
triton_compat.py — Compatibility shim for Triton 3.6 + older torch._inductor.

Triton 3.6 (installed for the lm-eval MXFP4 path on 120B) removed the
`AttrsDescriptor` class entirely. Older torch._inductor still imports it
unconditionally — see torch/_inductor/runtime/hints.py — and crashes with
`ImportError` even though our training path never actually calls torch.compile.

The trigger chain that hits this is:
    transformers.Trainer.__init__
        → accelerator.unwrap_model
            → from deepspeed import DeepSpeedEngine
                → @compiler.compile() decorator at deepspeed module top level
                    → torch.compile → torch._dynamo → torch._inductor → triton

This shim injects a stub `AttrsDescriptor` into both legacy locations so the
import succeeds. The stub is never actually used because our code does not
trigger compilation; it only needs to exist as a symbol. Import this module
before any torch / transformers / deepspeed import.
"""

import triton.backends.compiler as _tbc
import triton.compiler.compiler as _tcc


class _AttrsDescriptorStub:
    """Sentinel; never instantiated in our code paths."""

    @classmethod
    def from_dict(cls, _d):
        return cls()

    def __init__(self, *_args, **_kwargs):
        self.property_values = {"tt.divisibility": 16, "tt.equal_to": 1}

    __name__ = "AttrsDescriptor"


for _mod in (_tbc, _tcc):
    if not hasattr(_mod, "AttrsDescriptor"):
        _mod.AttrsDescriptor = _AttrsDescriptorStub
