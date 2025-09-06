from setuptools import setup, find_packages

setup(
    name="argument-mining-api",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "openai>=1.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "peft>=0.4.0",
        "python-dotenv>=0.19.0",
    ],
    python_requires=">=3.9",
)