"""Fallback shim so `pip install -e .` works on old pip/setuptools toolchains
(PEP 621 metadata in pyproject.toml requires setuptools>=61; the sandboxed
system Python ships 58). Metadata here mirrors pyproject.toml exactly."""

from setuptools import setup

setup(
    name="orbit-eval",
    version="0.8.3",
    description=(
        "Honest statistics for robot-policy evaluation: CRN-paired checkpoint "
        "comparisons vs retrain-level method comparisons, shipping the "
        "measured ORBIT noise atlas."
    ),
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=["orbit_eval", "orbit_eval.gate"],
    install_requires=[],
    entry_points={"console_scripts": [
        "orbit-eval=orbit_eval.cli:main",
        "orbit-shipgate=orbit_eval.gate.cli:main",
    ]},
)
