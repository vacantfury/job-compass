"""Date synchronization with temporary documents and synthetic Git output."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(params=["created", "updated"])
def date_script(request, load_script):
    created = request.param == "created"
    return SimpleNamespace(
        module=load_script("sync_git_created_date.py" if created else "sync_git_last_updated.py"),
        field_name="createdDate" if created else "lastUpdated",
        getter="get_file_created_date" if created else "get_file_last_updated_date",
        created=created,
    )


@pytest.mark.parametrize("output", ["", "  \n", "2026-01-03 10:20:30 -0800\n2025-12-02 04:05:06 +0000\n"])
def test_git_dates_use_expected_history(date_script, tmp_path, monkeypatch, output):
    path = str(tmp_path / "page with spaces.md")
    check = Mock(return_value=output)
    monkeypatch.setattr(subprocess, "check_output", check)
    value = getattr(date_script.module, date_script.getter)(path)
    expected = ("2025-12-02" if date_script.created else "2026-01-03") if output.strip() else None
    assert value == expected
    history_args = ["--follow", "--diff-filter=A"] if date_script.created else ["-1"]
    check.assert_called_once_with(["git", "log", *history_args, "--format=%ai", "--", path], text=True)


def test_git_failure_returns_none(date_script, tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", Mock(side_effect=subprocess.CalledProcessError(1, ["git", "log"])))
    assert getattr(date_script.module, date_script.getter)(str(tmp_path / "page.md")) is None


@pytest.mark.parametrize("has_frontmatter", [False, True])
def test_add_date_preserves_body_and_is_idempotent(date_script, write_file, has_frontmatter):
    body = "# Heading\n\nBody\n\n---\nA horizontal rule\n"
    prefix = "---\ntitle: Guide\n---\n\n" if has_frontmatter else ""
    path = write_file("page.md", prefix + body)
    date_script.module.update_frontmatter(path, "2026-01-02")
    content = path.read_text(encoding="utf-8")
    fields = "title: Guide\n" if has_frontmatter else ""
    assert content == f"---\n{fields}{date_script.field_name}: 2026-01-02\n---\n\n{body}"
    date_script.module.update_frontmatter(path, "2026-01-02")
    assert path.read_text(encoding="utf-8") == content


def test_existing_dates_preserve_creation_and_refresh_update(date_script, write_file):
    path = write_file("page.md", f"---\ntitle: Guide\n{date_script.field_name}: 2000-01-01\ndescription: Keep\n---\n\nBody\n{date_script.field_name}: body example\n")
    date_script.module.update_frontmatter(path, "2026-01-02")
    expected_date = "2000-01-01" if date_script.created else "2026-01-02"
    assert path.read_text(encoding="utf-8") == f"---\ntitle: Guide\n{date_script.field_name}: {expected_date}\ndescription: Keep\n---\n\nBody\n{date_script.field_name}: body example\n"


def test_walk_only_updates_markdown_with_history(date_script, write_file, monkeypatch):
    tracked = write_file("docs/nested/tracked.md", "Tracked")
    untracked = write_file("docs/untracked.md", "Untracked")
    other = write_file("docs/data.json", "{}")

    def get_date(path):
        assert str(path).endswith(".md")
        return "2026-01-02" if path == str(tracked) else None

    monkeypatch.setattr(date_script.module, date_script.getter, get_date)
    date_script.module.main()
    assert f"{date_script.field_name}: 2026-01-02" in tracked.read_text(encoding="utf-8")
    assert untracked.read_text(encoding="utf-8") == "Untracked"
    assert other.read_text(encoding="utf-8") == "{}"


@pytest.mark.xfail(raises=AssertionError, reason="A valid empty frontmatter block is not recognized, so no date is inserted.")
def test_empty_frontmatter_receives_date(date_script, write_file):
    path = write_file("page.md", "---\n---\n\nBody")
    date_script.module.update_frontmatter(path, "2026-01-02")
    assert path.read_text(encoding="utf-8") == f"---\n{date_script.field_name}: 2026-01-02\n---\n\nBody"


@pytest.mark.xfail(raises=AssertionError, reason="Date field detection matches YAML comments instead of a top-level key.")
def test_commented_date_does_not_replace_real_field(date_script, write_file):
    path = write_file("page.md", f"---\n# {date_script.field_name}: example\ntitle: Guide\n---\nBody")
    date_script.module.update_frontmatter(path, "2026-01-02")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert f"{date_script.field_name}: 2026-01-02" in lines
    assert f"# {date_script.field_name}: example" in lines


@pytest.mark.xfail(raises=AssertionError, reason="The lastUpdated regex consumes the next YAML field when the date is empty.")
def test_blank_last_updated_preserves_next_field(load_script, write_file):
    module = load_script("sync_git_last_updated.py")
    path = write_file("page.md", "---\nlastUpdated:\ntitle: Keep\n---\nBody")
    module.update_frontmatter(path, "2026-01-02")
    assert path.read_text(encoding="utf-8") == "---\nlastUpdated: 2026-01-02\ntitle: Keep\n---\nBody"


@pytest.fixture(params=["created", "updated"])
def action_script(request, load_script):
    created = request.param == "created"
    return SimpleNamespace(
        module=load_script("git-actions/add_created_date_to_md.py" if created else "git-actions/add_last_updated_to_md.py"),
        field_name="createdDate" if created else "lastUpdated",
        ensure="ensure_created_date" if created else "ensure_last_updated",
        add="add_created_date" if created else "add_last_updated",
        get_files="get_added_md_files" if created else "get_modified_md_files",
        created=created,
    )


@pytest.mark.parametrize("existing", [False, True])
def test_incremental_dates_add_or_refresh_only_top_level_fields(action_script, existing):
    field = f"{action_script.field_name}: 2000-01-01\n" if existing else ""
    content = f"---\ntitle: Guide\n{field}nested:\n  {action_script.field_name}: example\n---\nBody\n"
    result = getattr(action_script.module, action_script.ensure)(content, "2026-01-02")
    expected_date = "2000-01-01" if existing and action_script.created else "2026-01-02"
    assert f"\n{action_script.field_name}: {expected_date}\n" in result
    assert f"\n  {action_script.field_name}: example\n" in result
    assert result.endswith("---\nBody\n")
    assert getattr(action_script.module, action_script.ensure)(result, "2026-01-02") == result


def test_incremental_dates_create_frontmatter(action_script):
    ensure = getattr(action_script.module, action_script.ensure)
    assert ensure("Body\n", "2026-01-02") == f"---\n{action_script.field_name}: 2026-01-02\n---\n\nBody\n"


@pytest.mark.parametrize("output", ["", "a.md\nfolder/page with spaces.md\ndata.json\nupper.MD\n"])
def test_incremental_git_file_filter(action_script, monkeypatch, output):
    run = Mock(return_value=SimpleNamespace(stdout=output, returncode=0))
    monkeypatch.setattr(subprocess, "run", run)
    assert getattr(action_script.module, action_script.get_files)() == (["a.md", "folder/page with spaces.md"] if output else [])
    mode = "A" if action_script.created else "AM"
    run.assert_called_once_with(["git", "diff", f"--diff-filter={mode}", "--name-only", "origin/main...HEAD"], capture_output=True, text=True)


def test_incremental_commit_date(action_script, monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout="2026-01-02\n", returncode=0))
    monkeypatch.setattr(subprocess, "run", run)
    assert action_script.module.get_commit_date() == "2026-01-02"
    run.assert_called_once_with(["git", "log", "-1", "--format=%cd", "--date=short"], capture_output=True, text=True)


def test_incremental_main_uses_only_selected_files(action_script, write_file, monkeypatch):
    selected = write_file("selected.md", "Body")
    untouched = write_file("untouched.md", "Unchanged")
    monkeypatch.setattr(action_script.module, action_script.get_files, lambda: [selected])
    monkeypatch.setattr(action_script.module, "get_commit_date", lambda: "2026-01-02")
    action_script.module.main()
    assert selected.read_text(encoding="utf-8") == f"---\n{action_script.field_name}: 2026-01-02\n---\n\nBody"
    assert untouched.read_text(encoding="utf-8") == "Unchanged"


def test_incremental_add_skips_write_when_current(action_script, write_file, monkeypatch):
    selected = write_file("selected.md", f"---\n{action_script.field_name}: 2026-01-02\n---\nBody")
    with monkeypatch.context() as scoped:
        import builtins

        real_open = builtins.open

        def read_only(path, mode="r", *args, **kwargs):
            assert mode == "r"
            return real_open(path, mode, *args, **kwargs)

        scoped.setattr(builtins, "open", read_only)
        getattr(action_script.module, action_script.add)(selected, "2026-01-02")


def test_incremental_main_stops_when_no_files(action_script, monkeypatch):
    monkeypatch.setattr(action_script.module, action_script.get_files, lambda: [])
    date = Mock(side_effect=AssertionError("No Git lookup needed"))
    monkeypatch.setattr(action_script.module, "get_commit_date", date)
    action_script.module.main()
    date.assert_not_called()


@pytest.mark.xfail(raises=AssertionError, reason="The closing-frontmatter regex strips leading whitespace from the Markdown body.")
def test_incremental_dates_preserve_indented_body(action_script):
    content = "---\ntitle: Guide\n---\n\n    code_example()\n"
    result = getattr(action_script.module, action_script.ensure)(content, "2026-01-02")
    assert result.endswith("---\n\n    code_example()\n")


@pytest.mark.xfail(raises=AssertionError, reason="A second lastUpdated pass removes the blank line inserted by the first pass.")
def test_incremental_last_updated_is_idempotent_for_new_frontmatter(load_script, write_file):
    module = load_script("git-actions/add_last_updated_to_md.py")
    path = write_file("page.md", "Body")
    module.add_last_updated(path, "2026-01-02")
    first = path.read_text(encoding="utf-8")
    module.add_last_updated(path, "2026-01-02")
    assert path.read_text(encoding="utf-8") == first
