import sys
import os
import yaml
from gemma.model_utils import steer_generate
from sae_evolution import load_candidate

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_prompt.py <yaml_file>")
        sys.exit(1)

    yaml_file = sys.argv[1]

    data = load_candidate(yaml_file)
    while(True):
        user_input = input("Enter your input: ")

        prompt = "User: "+user_input+"\nGPT:" 
        #output = steer_generate(prompt, {})
        #print("Generated output without steering:")
        #print('--', data.layers)
        #print(output)

        output = steer_generate(prompt, data.layers)
        
        print("Generated output with steering:")
        print(output)

if __name__ == "__main__":
    main()

