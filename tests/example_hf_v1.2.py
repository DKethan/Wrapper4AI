from Wrapper4AI.wrap import connect

client = connect("huggingface_inference", "microsoft/Phi-3-mini-4k-instruct", "")

response = client.chat("Tell me a joke.")
print(response)

print("\n---\n")

x = "Who created it?"

history = [
    {"role": "system", "content": "What is Python?"},
    {"role": "user", "content": x}
]
print(client.chat_with_history(history))
