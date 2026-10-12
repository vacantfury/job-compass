"""Load the exact NumPy solution files included by the ML interview pages."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


SOLUTIONS = (
    Path(__file__).resolve().parents[2]
    / "docs/en/interview-prepare/machine-learning/solutions"
)


@pytest.fixture
def load_ml_solution():
    def load(filename: str) -> ModuleType:
        path = SOLUTIONS / filename
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
