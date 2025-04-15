import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from UI4AI import run_chat
from Wrapper4AI.wrap import connect

client = connect("openai", "gpt-4o", "")

run_chat(
    generate_response=client.chat_with_history,
    generate_title=client.generate_title,
    count_tokens=client.count_tokens,
    max_history_tokens=3000
)