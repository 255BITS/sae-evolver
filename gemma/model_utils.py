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

def lazy_load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model = HookedTransformer.from_pretrained("gemma-2b", device = 'cuda:0')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

sae_cache = {}
def load_sae(target_layer):
    print("LOAD SAE", target_layer)
    global sae_cache
    if target_layer in sae_cache:
        print("CACHE HIT", target_layer)
        return sae_cache[target_layer]
    sae, cfg_dict, _ = SAE.from_pretrained(
        release = "gemma-2b-res-jb",
        sae_id = f"blocks.{target_layer}.hook_resid_post",
        device = "cuda:0"
    )
    sae_cache[target_layer]=sae
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
    model.reset_hooks()
    editing_hooks = []
    position = 5#len(model.to_tokens(prefix))-1
    for target_layer, value in layers.items():
        editing_hooks += [(f"blocks.{target_layer}.hook_resid_post", steering_hook(value, target_layer, position))]
    print(editing_hooks,"HOOK")
    sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)
    res = hooked_generate([prefix], editing_hooks, seed=None, **sampling_kwargs)
  
    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    return res_str

def steering_hook(value, target_layer, position):
    def _steering_hook(resid_pre, hook):
        if resid_pre.shape[1] == 1:
            return

        for idx, coeff in value.items():
            sae = load_sae(target_layer)
            steering_vector = sae.W_dec[idx]
            print("POSITION", position, resid_pre.shape)
            # using our steering vector and applying the coefficient
            print("SETTING COEFF", coeff, idx)
            resid_pre[:, :position - 1, :] += coeff * steering_vector
    return _steering_hook

def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=False,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs)
    return result

