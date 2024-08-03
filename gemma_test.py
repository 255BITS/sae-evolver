from gemma.model_utils import process_prompt

def main():
    prompt = "Would you be able to travel through time using a wormhole?"
    values, indices = process_prompt(prompt)
    print("Values:", values)
    print("Indices:", indices)

if __name__ == "__main__":
    main()

