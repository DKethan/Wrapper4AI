from Wrapper4AI.wrap import connect

client = connect(provider="openai", model="gpt-4o", api_key="")

# Get a joke from the model
joke_response = client.chat("Tell me a joke.")
print(joke_response)

print("\n---\n")

# Prepare a prompt for code generation
user_prompt = "give me a code to create a simple calculator in python"

chat_history = [
    {"role": "system", "content": "You are a state-of-the-art Python developer. You need to answer the question."},
    {"role": "user", "content": user_prompt}
]

# Get a response using chat history
code_response = client.chat_with_history(chat_history)
print(code_response)
