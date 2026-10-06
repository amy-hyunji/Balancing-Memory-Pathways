import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

MODELS = {
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "nemotron-h-4b": "nvidia/Nemotron-H-4B-Instruct-128K",
}

# LoRA on all layers: attention, recurrent mixer and MLP projections
# (PEFT does not allow out_proj / conv1d on Nemotron-H's Mamba2 mixers)
LORA_TARGETS = {
    "qwen3_5": ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                "gate_proj", "up_proj", "down_proj"],
    "nemotron_h": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "up_proj", "down_proj"],
}


def load(name, device="cuda", dtype=torch.bfloat16):
    """Returns (model, tokenizer, family). Attention runs on the sdpa path
    (custom 4D masks). Qwen3.5 ships as a multimodal *ForConditionalGeneration,
    so the declared architecture is loaded rather than AutoModelForCausalLM."""
    name = MODELS.get(name, name)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(name)
    cls = getattr(transformers, (cfg.architectures or ["AutoModelForCausalLM"])[0])
    model = cls.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").to(device)
    family = "nemotron_h" if cfg.model_type.startswith("nemotron_h") else "qwen3_5"
    return model, tok, family
