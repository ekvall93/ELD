from setuptools import setup, find_packages

setup(
    name="ELD",
    version="0.0.1",
    description="Spatial landmark detection for tissue images and registration",
    author="Markus Ekvall",
    author_email="marekv@kth.se",
    packages=find_packages(),
    install_requires=[
        "flit_core >=3.4,<4",
        "numpy"
    ],
    extras_require={
        "doc": [
            "sphinx~=4.2.0",
            "myst-parser",
            "furo"
        ]
    }
)