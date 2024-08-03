from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import torch.nn as nn

torch.set_grad_enabled(False)  # avoid blowing up mem

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

model = None
tokenizer = None
sae = None

def lazy_load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model = HookedTransformer.from_pretrained("gemma-2b", device = 'cuda:0')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

def load_sae(target_layer):
    sae, cfg_dict, _ = SAE.from_pretrained(
        release = "gemma-2b-res-jb",
        sae_id = f"blocks.{target_layer}.hook_resid_post",
        device = "cuda:0"
    )
    return sae

def process_prompt(prompt, target_layer=6):
    # Lazy load the model, tokenizer, and sae
    lazy_load_model_and_tokenizer()
    sae = load_sae(target_layer)

    sv_logits, cache = model.run_with_cache(prompt, prepend_bos=True)
    hook_point = sae.cfg.hook_name
    sv_feature_acts = sae.encode(cache[hook_point])

    sae_out = sae.decode(sv_feature_acts)

    topk_values, topk_indices = torch.topk(sv_feature_acts, 3)

    # Flatten the values and indices
    topk_values = topk_values.flatten()
    topk_indices = topk_indices.flatten()

    return topk_values, topk_indices


def steer_generate(prefix, layers):
    return "not implemented"

