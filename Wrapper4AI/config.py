class Config:
    _api_keys = {}
    _default_provider = "openai"
    _default_model = "gpt-4o"
    _max_history_tokens = 3000

    @classmethod
    def set_api_key(cls, provider: str, model: str, api_key: str):
        cls._api_keys[(provider.lower(), model)] = api_key

    @classmethod
    def get_api_key(cls, provider: str, model: str = None):
        return cls._api_keys.get((provider.lower(), model or cls.get_default_model()))

    @classmethod
    def get_default_provider(cls) -> str:
        return cls._default_provider

    @classmethod
    def get_default_model(cls) -> str:
        return cls._default_model

    @classmethod
    def get_max_history_tokens(cls) -> int:
        return cls._max_history_tokens
