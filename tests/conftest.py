"""Load helpers only after isolating their filesystem and external effects."""

import importlib
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType

import pytest
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    def blocked(*args, **kwargs):
        raise AssertionError("External I/O must be explicitly mocked")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    monkeypatch.setattr(os, "system", blocked)

    # These modules call setup_logging during import.
    logging_utils = importlib.import_module("scripts.utils.logging_utils")
    monkeypatch.setattr(logging_utils, "PROJECT_ROOT", tmp_path)

    # Never import a local config.py, even if one exists in the checkout.
    config = ModuleType("scripts.utils.config")
    config.REMOTE_SERVER_1030_HOST = "example.invalid"
    config.REMOTE_SERVER_1030_PORT = 2222
    config.REMOTE_SERVER_1030_USER = "test-user"
    config.SSH_PRIVATE_KEY_PATH_1030 = str(tmp_path / "unused-key")
    config.REMOTE_1030_VISA_SOLUTION_COMPANY_JSON_DIR = "/test-data"
    monkeypatch.setitem(sys.modules, config.__name__, config)


@pytest.fixture
def load_script(tmp_path, monkeypatch, isolated_runtime):
    def load(relative_path):
        path = REPO_ROOT / "scripts" / relative_path
        spec = importlib.util.spec_from_file_location(
            "test_target_" + path.stem.replace("-", "_"), path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ("PROJECT_ROOT", "REPO_ROOT"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, tmp_path)
        if hasattr(module, "DOCS_DIR_PATH"):
            monkeypatch.setattr(module, "DOCS_DIR_PATH", tmp_path / "docs")
        return module

    return load


@pytest.fixture
def write_file(tmp_path):
    def write(relative_path, content):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    return write
