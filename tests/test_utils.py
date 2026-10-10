"""Path, JSON and logging contracts without touching checkout data."""

import json
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_project_root_resolves_symlinks(load_script, tmp_path, monkeypatch, depth):
    module = load_script("utils/path_utils.py")
    source = tmp_path / "repo" / "scripts" / "utils" / "path_utils.py"
    source.parent.mkdir(parents=True)
    source.touch()
    alias = tmp_path / "alias.py"
    alias.symlink_to(source)
    monkeypatch.setattr(module, "__file__", str(alias))
    assert module.get_project_root(depth) == source.parents[depth]


@pytest.mark.parametrize("value", [{"items": [1, True, None]}, ["first", "second"], 42, None])
@pytest.mark.parametrize("as_string", [False, True])
def test_read_json(load_script, write_file, value, as_string):
    module = load_script("utils/json_utils.py")
    path = write_file("records.json", json.dumps(value))
    assert module.read_json_file(str(path) if as_string else path) == value


@pytest.mark.parametrize("problem", ["missing", "invalid", "directory", "encoding"])
def test_read_json_reports_errors(load_script, tmp_path, caplog, problem):
    module = load_script("utils/json_utils.py")
    path = tmp_path / "record.json"
    if problem == "invalid":
        path.write_text("{broken", encoding="utf-8")
    elif problem == "directory":
        path.mkdir()
    elif problem == "encoding":
        path.write_bytes(b"\xff")
    assert module.read_json_file(path) is None
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_logging_uses_configured_root(load_script, tmp_path, monkeypatch):
    module = load_script("utils/logging_utils.py")
    configure = Mock()
    monkeypatch.setattr(module.logging, "basicConfig", configure)
    module.setup_logging()
    module.setup_logging()
    assert (tmp_path / "logs").is_dir()
    assert configure.call_args.kwargs == {
        "level": module.logging.DEBUG,
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "filename": str(tmp_path / "logs" / "logbook.log"),
        "filemode": "w",
    }
