from ..base_handler import BaseHandler


class BedrockHandler(BaseHandler):
    """Handler for Bedrock models."""

    def __init__(self, model: str = "default-model"):
        super().__init__(model)

    def _generate_response(self, messages):
        raise NotImplementedError("_generate_response is not implemented.")

    def _generate_title(self, prompt):
        raise NotImplementedError("_generate_title is not implemented.")

    def _truncate_history(self, messages):
        raise NotImplementedError("_truncate_history is not implemented.")

    def _count_tokens(self, messages):
        raise NotImplementedError("_count_tokens is not implemented.")