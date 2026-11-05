from setuptools import setup, find_packages

setup(
    name="altsteer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch==2.9.1",
        "transformers==4.57.6",
        "accelerate==1.12.0",
        "numpy==2.2.6",
        "tqdm==4.67.1",
        "openai==2.15.0",
        "python-dotenv==1.2.1",
    ],
)
