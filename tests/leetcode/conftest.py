"""Load the exact Python files imported by the interview-preparation pages."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


SOLUTIONS = (
    Path(__file__).resolve().parents[2]
    / "docs/en/interview-prepare/leetcode/solutions"
)


@pytest.fixture
def load_solution():
    def load(filename: str) -> ModuleType:
        path = SOLUTIONS / filename
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
