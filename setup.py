from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="atc-hallucination",
    version="1.0.0",
    author="ATC Hallucination Detection Project",
    description="Multi-Agent Reinforcement Learning framework for Air Traffic Control with hallucination detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Somnathab3/ATC_Hallucination",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Transportation",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "atc-train=train:main",
            "atc-test=test_shifts:main",
            "atc-intrashift-test=src.testing.intrashift_tester:main",
            "atc-visualize=visualize_air_traffic:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["scenarios/*.json"],
    },
)