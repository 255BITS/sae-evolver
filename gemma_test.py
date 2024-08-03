from gemma.model_utils import process_prompt, steer_generate

def main():
    prompt = "Would you be able to travel through time using a wormhole?"
    values, indices = process_prompt(prompt, 6)
    print("Values:", values)
    print("Indices:", indices)

    print(steer_generate("What is the most iconic structure known to man?", {6: {10200: 300.0}}))
    print(steer_generate("What is on your mind?", {6: {10200: 300.0}}))

if __name__ == "__main__":
    main()

