from typing import List, Dict

from .responder import GenerativeResponder
from .config import Config

class Client:
    def __init__(self, provider: str, model: str, api_key: str):
        Config.set_api_key(provider, model, api_key)
        self.responder = GenerativeResponder(provider=provider, model=model)

    def chat(self, user_prompt: str) -> str:
        return self.responder.generate_response(user_prompt=user_prompt)

    def chat_with_history(self, messages: List[Dict]) -> str:
        return self.responder.generate_response(messages)

    def generate_title(self, prompt: str) -> str:
        return self.responder.generate_title(prompt)

    def truncate_history(self, messages: List[Dict]) -> List[Dict]:
        return self.responder.truncate_history(messages)

    def count_tokens(self, messages: List[Dict]) -> int:
        return self.responder.count_tokens(messages)

    def show_history(self):
        self.responder.print_history()

    def clear_history(self):
        self.responder.reset_history()

def connect(provider: str, model: str, api_key: str) -> Client:
    return Client(provider, model, api_key)
