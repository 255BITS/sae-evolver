import sys
import os
import random
import yaml
import torch
import argparse
from gemma.model_utils import steer_generate
from sae_evolution import load_candidate

def main():
    parser = argparse.ArgumentParser(description="Generate prompts using the specified Gemma model and SAE repository.")
    parser.add_argument("candidate_yaml", type=str, help="Path to the candidate YAML file")
    parser.add_argument("example_yaml", type=str, help="Path to the example YAML file")
    parser.add_argument("--gemma-model", type=str, default="google/gemma-2-2b", help="Which Gemma model to use")
    parser.add_argument("--sae", type=str, default="google/gemma-scope-2b-pt-res", help="Which Gemmascope SAE repo to use")
    
    args = parser.parse_args()

    prompt_format = yaml.safe_load(open(args.example_yaml, "r").read())["prompt_format"]

    data = load_candidate(args.candidate_yaml)
    while True:
        seed = random.randint(0, 10000)
        user_input = input("Enter your input: ")

        prompt = prompt_format.replace("USER_INPUT", user_input)

        torch.cuda.manual_seed_all(seed)
        output = steer_generate(prompt, {}, model_name=args.gemma_model, sae_repo_id=args.sae)
        print("Generated output without steering:")
        print("---")
        print(output)
        print("---")

        torch.cuda.manual_seed_all(seed)
        output = steer_generate(prompt, data.layers, model_name=args.gemma_model, sae_repo_id=args.sae)

        print("Generated output with steering:")
        print("---")
        print(output)

if __name__ == "__main__":
    main()

