import sys
import os
import groq

# Open the metaprompt.txt file and read its contents
with open('metaprompt.txt', 'r') as f:
    prompt = f.read()

# Replace USER_INPUT with the first command-line argument
if len(sys.argv) > 1:
    prompt = prompt.replace('USER_INPUT', sys.argv[1])
else:
    print("Error: no input provided")
    sys.exit(1)

print(prompt)
print("___")

# Get the API key from an environment variable
api_key = os.getenv('GROQ_API_KEY')

if api_key is None:
    print("Error: GROQ_API_KEY environment variable is not set")
    sys.exit(1)

# Initialize the Groq client with the API key
client = groq.Client(api_key=api_key)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.1-70b-versatile",
)
print(chat_completion.choices[0].message.content)

