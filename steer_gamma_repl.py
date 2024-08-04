import sys
import os
import yaml
from gemma.model_utils import steer_generate
from sae_evolution import load_candidate

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_prompt.py <candidate_yamle> <example_yaml>")
        sys.exit(1)

    example_file = sys.argv[2]
    candidate_yaml = sys.argv[1]
    prompt_format = yaml.safe_load(example_yaml)["prompt_format"]

    data = load_candidate(candidate_yaml)
    while(True):
        user_input = input("Enter your input: ")

        prompt = prompt_format.replace("USER_INPUT", user_input)
        output = steer_generate(prompt, {})
        print("Generated output without steering:")
        print("---")
        print(output)
        print("---")

        output = steer_generate(prompt, data.layers)
        
        print("Generated output with steering:")
        print("---")
        print(output)

if __name__ == "__main__":
    main()

