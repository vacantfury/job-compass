"""Inspect command construction only; no SSH, key reads or child processes."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def rsync(load_script):
    return load_script("utils/rsync_downloader.py")


def test_sync_command_defaults(rsync, tmp_path, monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout="", returncode=0))
    monkeypatch.setattr(subprocess, "run", run)
    destination = tmp_path / "output" / "data"
    assert rsync.RsyncDownloader().sync_folder("/public-data", destination)
    assert destination.is_dir()
    run.assert_called_once_with([
        "rsync", "-az", "-v", "--progress", "--delete",
        "--exclude", "._*", "--exclude", ".DS_Store", "--exclude", ".AppleDouble", "--exclude", ".LSOverride",
        "-e", f"ssh -p 2222 -i {tmp_path / 'unused-key'}",
        "test-user@example.invalid:/public-data/", destination,
    ], check=True, capture_output=True, text=True)
    assert not (tmp_path / "unused-key").exists()


def test_sync_options_do_not_mutate_exclusions(rsync, tmp_path, monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout="", returncode=0))
    monkeypatch.setattr(subprocess, "run", run)
    downloader = rsync.RsyncDownloader()
    defaults = downloader.default_excludes.copy()
    excludes = ["*.tmp"]
    options = ["--dry-run"]
    assert downloader.sync_folder("/public-data", tmp_path / "out", delete=False, verbose=False, excludes=excludes, additional_options=options)
    command = run.call_args.args[0]
    assert "--delete" not in command
    assert "-v" not in command
    assert "--progress" not in command
    assert "--dry-run" in command
    assert command[command.index("*.tmp") - 1] == "--exclude"
    assert downloader.default_excludes == defaults
    assert excludes == ["*.tmp"]
    assert options == ["--dry-run"]


@pytest.mark.parametrize("error", [subprocess.CalledProcessError(23, "rsync", stderr="Synthetic failure"), FileNotFoundError("rsync unavailable")])
def test_sync_failures_return_false(rsync, tmp_path, monkeypatch, capsys, error):
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))
    assert not rsync.RsyncDownloader().sync_folder("/public-data", tmp_path / "out", verbose=False)
    assert capsys.readouterr().out


def test_sync_wrapper_forwards_options(rsync, tmp_path, monkeypatch):
    sync = Mock(return_value=False)
    monkeypatch.setattr(rsync.RsyncDownloader, "sync_folder", sync)
    assert rsync.sync_remote_folder("/public-data", tmp_path / "out", delete=False) is False
    sync.assert_called_once_with("/public-data", tmp_path / "out", delete=False)


@pytest.mark.parametrize("success", [False, True])
def test_fetch_routes_to_expected_local_directory(load_script, tmp_path, monkeypatch, success):
    module = load_script("pull_visa_solution_company_json.py")
    sync = Mock(return_value=success)
    monkeypatch.setattr(module, "sync_remote_folder", sync)
    assert module.fetch_visa_solution_company_json() is success
    sync.assert_called_once_with("/test-data", tmp_path / "docs/assets/json/visa-solution-company")
