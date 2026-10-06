import contextlib

import torch

_STATE = {"segs": None, "window": 1, "reset": None}
_MASK_CACHE = {}


# ----------------------------------------------------------------- masks
def segment_mask(segs, device, dtype, window=1):
    """Additive [n, n] mask: causal, and query i may attend key j only if
    seg(j) > seg(i) - window. segs must be non-decreasing. The mask is cached
    so that all attention layers (and checkpoint recomputes) of one forward
    share a single matrix."""
    segs = segs.to(device)
    n = segs.shape[0]
    cuts = [0] + (torch.nonzero(segs[1:] != segs[:-1]).squeeze(1) + 1).tolist() + [n]
    key = (n, tuple(cuts), window, str(device), dtype)
    m = _MASK_CACHE.get(key)
    if m is not None:
        return m
    _MASK_CACHE.clear()
    lo = torch.finfo(dtype).min
    m = torch.full((n, n), lo, dtype=dtype, device=device).triu_(1)
    for k in range(window, len(cuts) - 1):       # block k sees blocks k-window+1 .. k
        m[cuts[k]:cuts[k + 1], :cuts[k - window + 1]] = lo
    _MASK_CACHE[key] = m
    return m


def _decode_mask(segs, k_len, device, dtype, window=1):
    """[1, 1, 1, k_len] mask for one decode step. The generated token (and all
    earlier generated tokens) belong to the last prompt segment."""
    sg = segs.to(device)
    m = torch.zeros(k_len, dtype=dtype, device=device)
    blocked = (int(sg[-1]) - sg) >= window
    m[:len(sg)][blocked] = torch.finfo(dtype).min
    return m[None, None, None, :]


def _restricted_mask(hidden_states, past_key_values, layer_idx, attention_mask):
    segs = _STATE["segs"]
    if segs is None:
        return attention_mask
    q_len = hidden_states.shape[1]
    if segs.shape[-1] == q_len:                                    # prefill
        assert hidden_states.shape[0] == 1, "segment mask supports batch 1 only"
        return segment_mask(segs, hidden_states.device, hidden_states.dtype,
                            _STATE["window"])[None, None]
    if q_len == 1 and past_key_values is not None:                 # decode step
        # per-layer length: earlier layers have already appended this token
        k_len = int(past_key_values.get_seq_length(layer_idx)) + 1
        return _decode_mask(segs, k_len, hidden_states.device, hidden_states.dtype,
                            _STATE["window"])
    return attention_mask


# ----------------------------------------------------------------- patches
def _patch_attention(cls, mask_pos):
    """Wrap cls.forward so its attention_mask argument is replaced while the
    recurrent-only restriction is active. mask_pos: positional index of
    attention_mask in the original signature (after self)."""
    if getattr(cls, "_pathway_patched", False):
        return
    orig = cls.forward

    def forward(self, *args, **kwargs):
        if _STATE["segs"] is not None:
            args = list(args)
            hs = args[0] if args else kwargs["hidden_states"]
            pkv = kwargs.get("past_key_values",
                             args[mask_pos + 1] if len(args) > mask_pos + 1 else None)
            if len(args) > mask_pos:
                args[mask_pos] = _restricted_mask(hs, pkv, getattr(self, "layer_idx", None), args[mask_pos])
            else:
                kwargs["attention_mask"] = _restricted_mask(
                    hs, pkv, getattr(self, "layer_idx", None), kwargs.get("attention_mask"))
        return orig(self, *args, **kwargs)

    cls.forward = forward
    cls._pathway_patched = True


def _patch_recurrent(cls):
    """Wrap the recurrent mixer so that, while the attention-only restriction is
    active, the ORIGINAL forward runs once per contiguous segment. Every call
    starts from a zero state and a zero-padded conv, so the reset is exact
    without kernel support for sequence boundaries. Only the last segment gets
    the cache, so decoding continues from the current segment's own state."""
    if getattr(cls, "_pathway_patched", False):
        return
    orig = cls.forward

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        segs = _STATE["reset"]
        n = hidden_states.shape[1]
        if segs is None or n == 1 or segs.shape[-1] != n:
            return orig(self, hidden_states, cache_params, attention_mask, **kwargs)
        assert hidden_states.shape[0] == 1, "recurrent reset supports batch 1 only"
        sg = segs.tolist()
        cuts = [0] + [i for i in range(1, n) if sg[i] != sg[i - 1]] + [n]
        outs = []
        for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            last = j == len(cuts) - 2
            am = attention_mask[:, a:b] if (attention_mask is not None and attention_mask.dim() == 2) else None
            outs.append(orig(self, hidden_states[:, a:b], cache_params if last else None, am, **kwargs))
        return torch.cat(outs, dim=1)

    cls.forward = forward
    cls._pathway_patched = True


def _model_classes():
    """(attention class, attention_mask position, recurrent class) per family."""
    out = []
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as q35
        # Qwen3_5Attention.forward(hidden_states, position_embeddings, attention_mask, past_key_values, ...)
        out.append((q35.Qwen3_5Attention, 2, q35.Qwen3_5GatedDeltaNet))
    except ImportError:
        pass
    try:
        import transformers.models.nemotron_h.modeling_nemotron_h as nh
        # NemotronHAttention.forward(hidden_states, attention_mask, past_key_values, ...)
        out.append((nh.NemotronHAttention, 1, nh.NemotronHMamba2Mixer))
    except ImportError:
        pass
    return out


class Pathways:
    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        mods = list(base.modules())
        found = []
        for attn, pos, rec in _model_classes():
            if any(isinstance(m, attn) for m in mods) and any(isinstance(m, rec) for m in mods):
                _patch_attention(attn, pos)
                _patch_recurrent(rec)
                found.append(attn.__name__)
        if not found:
            raise ValueError("no supported hybrid architecture (Qwen3.5 / Nemotron-H) found in model")
        self.families = found

    @contextlib.contextmanager
    def recurrent_only(self, segs, window=1):
        _STATE["segs"], _STATE["window"] = segs, window
        try:
            yield
        finally:
            _STATE["segs"], _STATE["window"] = None, 1

    @contextlib.contextmanager
    def attention_only(self, segs):
        _STATE["reset"] = segs
        try:
            yield
        finally:
            _STATE["reset"] = None

    def restrict(self, mode, segs, window=1):
        """mode: 'both' (no restriction) | 'recurrent' | 'attention'."""
        if mode == "recurrent":
            return self.recurrent_only(segs, window)
        if mode == "attention":
            return self.attention_only(segs)
        if mode == "both":
            return contextlib.nullcontext()
        raise ValueError(mode)
