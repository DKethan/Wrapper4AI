from Wrapper4AI.wrap import connect

client = connect(provider="openai", model="gpt-4o", api_key="")

response = client.chat("Tell me a joke.")
print(response)

print("\n---\n")

history = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "A programming language."},
    {"role": "user", "content": "Who created it?"}
]
print(client.chat_with_history(history))
