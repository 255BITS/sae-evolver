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

model = None
tokenizer = None

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



def lazy_load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
            device_map='auto',
        ).cuda()

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")


def sae_params():
    return {
            10: {"width": 16, "average": 77},
            15: {"width": 16, "average": 78},
            24: {"width": 16, "average": 73},
            18: {"width": 16, "average": 74},
            20: {"width": 16, "average": 71}
    }

sae_cache = {}
def load_sae(target_layer):
    if target_layer in sae_cache:
        return sae_cache[target_layer]

    params = sae_params()[target_layer]
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename=f"layer_{target_layer}/width_{params['width']}k/average_l0_{params['average']}/params.npz",
        force_download=False,
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to("cuda") for k, v in params.items()}
    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae.cuda()
    sae_cache[target_layer] = sae
    return sae

def process_prompt(prompt, target_layer=6):
    # Lazy load the model, tokenizer, and sae
    lazy_load_model_and_tokenizer()
    sae = load_sae(target_layer)


    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).cuda()

    target_act = gather_residual_activations(model, target_layer, inputs)

    sae_acts = sae.encode(target_act.to(torch.float32))

    values, indices = sae_acts.max(-1)

    return values, indices

def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x

def steer_generate(prefix, layers):
    lazy_load_model_and_tokenizer()
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to("cuda")
    handles = []
    def _steer_sae(target_layer):
        sae = load_sae(target_layer)
        def steer_sae(mod, inputs, outputs):
            original_tensor = untuple_tensor(outputs)
            for idx, coeff in value.items():

                #for idx, coeff in value.items():
                steering_vector = sae.W_dec[idx]
                original_tensor[None] = original_tensor + coeff * steering_vector
            return outputs
        return steer_sae
    for target_layer, value in layers.items():
        handle = model.model.layers[target_layer].register_forward_hook(_steer_sae(target_layer))
        handles.append(handle)
    result = model.generate(**inputs, do_sample=True, temperature=1.0, max_new_tokens=128)
    decoded_text = tokenizer.decode(result[0])

    for handle in handles:
        handle.remove()
    return decoded_text

