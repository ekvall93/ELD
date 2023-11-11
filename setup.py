from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ELD",
    version="0.0.1",
    description="Spatial landmark detection for tissue images and registration",
    author="Markus Ekvall",
    author_email="marekv@kth.se",
    packages=find_packages(),
    install_requires=required,  # Use the list read from the requirements.txt file
    extras_require={
        "doc": [
            "sphinx~=4.2.0",
            "myst-parser",
            "furo",
            "nbsphinx"
        ]
    },
    entry_points={
        'console_scripts': [
            'eld-train=ELD.train:main',  # "eld-train" is the command to call your script
        ],
    }
)