from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ssvep-bci",
    version="1.0.0",
    author="BCIH Team",
    author_email="your-email@example.com",
    description="SSVEP-based Brain-Computer Interface for real-time EEG control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ssvep-bci",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "ssvep-bci=analysis:main",
            "brainbit-bridge=BrainBitLSL:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 