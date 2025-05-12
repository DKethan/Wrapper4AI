from setuptools import setup, find_packages

setup(
    name="Wrapper4AI",
    version="0.1.2",
    author="Kethan Dosapati",
    description="A lightweight multi-provider wrapper for LLM chat with history and token management.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DKethan/Wrapper4AI/tree/dev-01",
    packages=find_packages(),
    install_requires=[
        "openai",
        "tiktoken",
        "requests",
        "huggingface_hub",
        "boto3",
        "anthropic"
    ],
    python_requires='>=3.7',
)
