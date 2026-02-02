#!/usr/bin/env python3
"""
Setup.py for ECTC Gateway
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("../README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ectc-gateway",
    version="1.0.0",
    description="ECTC Battery-Free Sensor Network Gateway",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ECTC Project Team",
    author_email="support@ectc-project.org",
    url="https://github.com/ectc/ectc-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "monitoring": [
            "prometheus-client>=0.10",
            "grafana-client>=3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ectc-gateway=ectc_gateway.main:main",
            "ectc-diagnostics=tools.diagnostics:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ectc_gateway": [
            "models/*.tflite",
            "models/*.pkl",
            "config/*.yaml",
        ],
    },
    zip_safe=False,
)
