from Wrapper4AI.wrap import connect
from UI4AI import run_chat

client = connect("huggingface", "microsoft/Phi-3-mini-4k-instruct", "")

run_chat(
    generate_response=client.chat_with_history,
    generate_title=client.generate_title,
    count_tokens=client.count_tokens,
    max_history_tokens=3000
)