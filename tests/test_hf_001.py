from Wrapper4AI.wrap import connect

client = connect("huggingface", "microsoft/Phi-3-mini-4k-instruct", "")

print(client.chat(user_prompt="tell me a joke  ?"))