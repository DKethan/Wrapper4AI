from setuptools import setup, find_packages

setup(
    name="Wrapper4AI",
    version="0.1.0",
    author="Kethan Dosapati",
    description="Streamlit UI for LLM chat apps",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DKethan/Wrap4AI/tree/dev-01",
    packages=find_packages(),
    install_requires=["openai", "tiktoken"],
    python_requires='>=3.7',
)
