import sys
import os
import random
import yaml
import torch
from gemma.model_utils import steer_generate
from sae_evolution import load_candidate

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_prompt.py <candidate_yamle> <example_yaml>")
        sys.exit(1)

    example_yaml = sys.argv[2]
    candidate_yaml = sys.argv[1]
    prompt_format = yaml.safe_load(open(example_yaml, "r").read())["prompt_format"]

    data = load_candidate(candidate_yaml)
    while(True):
        seed = random.randint(0, 2**32 - 1)
        user_input = input("Enter your input: ")

        prompt = prompt_format.replace("USER_INPUT", user_input)

        torch.cuda.manual_seed_all(seed)
        output = steer_generate(prompt, {})
        print("Generated output without steering:")
        print("---")
        print(output)
        print("---")

        torch.cuda.manual_seed_all(seed)
        output = steer_generate(prompt, data.layers)
        
        print("Generated output with steering:")
        print("---")
        print(output)

if __name__ == "__main__":
    main()

