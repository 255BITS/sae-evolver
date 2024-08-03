from transformers import AutoModelForCausalLM, AutoTokenizer
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

def gather_residual_activations(model, target_layer, inputs):
    target_act = None
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs
    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act

model = None
tokenizer = None
sae = None

def lazy_load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
            device_map='auto',
        ).cuda()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

def lazy_load_sae():
    global sae
    if sae is None:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_20/width_16k/average_l0_71/params.npz",
            force_download=False,
        )
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).to("cuda") for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.cuda()

def process_prompt(prompt, target_layer=20):
    # Lazy load the model, tokenizer, and sae
    lazy_load_model_and_tokenizer()
    lazy_load_sae()

    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).cuda()

    # Gather residual activations from the specified layer
    target_act = gather_residual_activations(model, target_layer, inputs)

    # Encode the activations using JumpReLUSAE encoder
    sae_acts = sae.encode(target_act.to(torch.float32))

    # Get the max(-1) values and indices
    values, indices = sae_acts.max(-1)

    return values, indices

