from Wrapper4AI.wrap import connect

client = connect("openai", "")

print(client.chat(user_prompt="tell me a joke"))