"""Overview parsing, ordering, rendering and safe directory handling."""

from datetime import date
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest


@pytest.fixture
def overview(load_script):
    return load_script("generate_folder_overview.py")


def test_simple_frontmatter_limits_fields_and_preserves_colons(overview):
    assert overview.parse_simple_frontmatter(
        '# Comment\ntitle: "A: guide"\ndescription: \'Details\'\norder: -2\n'
        "ignored: value\nno delimiter\nempty:\n"
    ) == {"title": "A: guide", "description": "Details", "order": "-2"}


@pytest.mark.parametrize("value", ["", "|", ">"])
def test_simple_frontmatter_skips_missing_or_multiline_values(overview, value):
    assert overview.parse_simple_frontmatter(f"description: {value}\n  continuation") == {}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("---\ntitle: Guide\norder: 3\ndescription: >-\n  First\n  second\n---\nBody", {"title": "Guide", "order": 3, "description": "First second"}),
        ("---\n[one, two]\n---\nBody", {}),
        ("---\nnull\n---\nBody", {}),
        ("---\ntitle: [broken\n---\nBody", {}),
        ("# Body\n---\ntitle: Later\n---\n", {}),
        ("---\ntitle: Unclosed\n", {}),
        ("---\n\n---\nBody", {}),
        ("---\r\ntitle: Guide\r\n---\r\nBody", {"title": "Guide"}),
    ],
)
def test_extract_frontmatter(overview, write_file, content, expected):
    assert overview.extract_frontmatter(write_file("page.md", content)) == expected


def test_extract_missing_and_yaml_fallback(overview, tmp_path, write_file, monkeypatch):
    assert overview.extract_frontmatter(tmp_path / "missing.md") == {}
    monkeypatch.setattr(overview, "yaml", None)
    path = write_file("page.md", "---\ntitle: 'Guide'\norder: 2\n---\nBody")
    assert overview.extract_frontmatter(path) == {"title": "Guide", "order": "2"}


def test_extract_preserves_folded_scalar_final_newline(overview, write_file):
    path = write_file("page.md", "---\ndescription: >\n  First\n  second\n---\nBody")
    assert overview.extract_frontmatter(path)["description"] == "First second\n"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 0), ("", 0), (3, 3), (-2, -2), ("4", 4), ("2.9", 2), ("-2.9", -2),
     ("1e2", 100), (1.9, 1), ("bad", 0), ([], 0), ({}, 0), ("nan", 0)],
)
def test_normalize_order(overview, value, expected):
    assert overview.normalize_order(value) == expected


@pytest.mark.parametrize("value", ["inf", float("inf")])
def test_normalize_infinite_order(overview, value):
    assert overview.normalize_order(value) == 0


def test_build_items_filters_entries_and_sorts_all_tiebreakers(overview, write_file, tmp_path):
    for name in ("overview.md", "index.md", "CLAUDE.md", "AGENTS.md", "GEMINI.md", "data.json"):
        write_file(name, "Ignored")
    write_file("hidden/article.md", "No index, no folder entry")
    write_file("folder/index.md", "---\ntitle: Z Folder\n---\nBody")
    write_file("first.md", "---\ntitle: First\norder: -1\ndescription: Intro\n---\nBody")
    write_file("z.md", "---\ntitle: alpha\n---\nBody")
    write_file("a.md", "---\ntitle: Alpha\n---\nBody")
    write_file("fallback.md", "Plain body")
    write_file("last.md", "---\ntitle: 42\norder: '2.9'\n---\nBody")

    items = overview.build_overview_items(tmp_path, "en")
    assert [item["href"] for item in items] == [
        "./first.md", "./folder/index.md", "./a.md", "./z.md", "./fallback.md", "./last.md"
    ]
    assert items[0] == {"title": "First", "description": "Intro", "href": "./first.md", "order": -1, "kind": "article"}
    assert items[1]["kind"] == "folder"
    assert items[4]["title"] == "fallback"
    assert items[4]["description"] == "This file mainly introduces content related to fallback."
    assert items[-1]["title"] == "42"
    assert items[-1]["order"] == 2


def test_folder_title_fallback(overview, write_file, tmp_path):
    write_file("guide/index.md", "---\ntitle: null\ndescription: ''\n---\nBody")
    assert overview.build_overview_items(tmp_path, "en") == [{
        "title": "guide", "description": "This file mainly introduces content related to guide.",
        "href": "./guide/index.md", "order": 0, "kind": "folder",
    }]


@pytest.mark.parametrize(("relative", "locale"), [("docs/en", "en"), ("docs/en/guide", "en"), ("docs/english", "zhHans"), ("docs/zhHans/guide", "zhHans"), ("other/en", "zhHans")])
def test_locale_detection(overview, tmp_path, relative, locale):
    assert overview.detect_locale(tmp_path / relative) == locale


def test_render_roundtrip_escapes_closing_tags(overview):
    items = [{"title": '</script><script>alert("test")</script>', "description": "A\nB", "href": "./a.md", "order": 0, "kind": "article"}]
    content = overview.render_overview_content(items, "2026-01-02", "en")
    assert content.startswith("---\ntitle: Directory Overview\n")
    assert "lastUpdated: 2026-01-02\n" in content
    assert content.count("</script>") == 1
    assert "<\\/script>" in content
    assert '<DirectoryOverview :items="overviewItems" />' in content
    assert overview.parse_existing_items(content) == items


def test_unknown_locale_uses_default(overview):
    assert overview.render_overview_content([], "2026-01-02", "unknown") == overview.render_overview_content([], "2026-01-02", overview.DEFAULT_LOCALE)


@pytest.mark.parametrize("content", ["no script", "const overviewItems = broken\n</script>", 'const overviewItems = {"title": "a"}\n</script>'])
def test_parse_existing_items_rejects_bad_data(overview, content):
    assert overview.parse_existing_items(content) is None


def test_read_existing_overview_handles_missing_and_unreadable(overview, tmp_path):
    assert overview.read_existing_overview(tmp_path / "missing.md") is None
    assert overview.read_existing_overview(tmp_path) is None


def test_currency_ignores_date_but_checks_items_and_locale(overview, write_file):
    path = write_file("overview.md", overview.render_overview_content([], "2000-01-01", "en"))
    assert overview.existing_overview_is_current(path, [], "en")
    assert not overview.existing_overview_is_current(path, [{"title": "new"}], "en")
    assert not overview.existing_overview_is_current(path, [], "zhHans")


def test_generation_is_idempotent_and_updates_changed_items(overview, write_file, tmp_path, monkeypatch):
    directory = tmp_path / "docs" / "en" / "guide"
    article = write_file("docs/en/guide/page.md", "---\ntitle: Guide\n---\nBody")

    class FixedDate:
        @staticmethod
        def today():
            return date(2026, 1, 2)

    monkeypatch.setattr(overview, "date", FixedDate)
    assert overview.generate_overview_for_directory(directory)
    output = directory / "overview.md"
    assert "lastUpdated: 2026-01-02" in output.read_text(encoding="utf-8")
    assert overview.parse_existing_items(output.read_text(encoding="utf-8"))[0]["title"] == "Guide"
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "write_text", Mock(side_effect=AssertionError("Unchanged overview must not be rewritten")))
        assert overview.generate_overview_for_directory(directory)
    article.write_text("---\ntitle: Revised\n---\nBody", encoding="utf-8")
    assert overview.generate_overview_for_directory(directory)
    assert overview.parse_existing_items(output.read_text(encoding="utf-8"))[0]["title"] == "Revised"


def test_dry_run_does_not_write(overview, write_file, tmp_path, capsys):
    write_file("docs/en/page.md", "Body")
    assert overview.generate_overview_for_directory(tmp_path / "docs/en", dry_run=True)
    assert not (tmp_path / "docs/en/overview.md").exists()
    assert "DRY RUN: docs/en (en) -> 1 items" in capsys.readouterr().out


def test_invalid_generation_targets(overview, tmp_path, write_file):
    assert not overview.generate_overview_for_directory(tmp_path / "missing")
    assert not overview.generate_overview_for_directory(write_file("file.md", "Body"))


def test_directory_resolution_and_relative_display(overview, tmp_path):
    directory = tmp_path / "docs" / "en"
    directory.mkdir(parents=True)
    assert overview.resolve_directory_arg("docs/en") == directory
    assert overview.resolve_directory_arg(str(directory)) == directory
    assert overview.relative_to_repo(directory) == "docs/en"
    assert overview.relative_to_repo(tmp_path.parent) == tmp_path.parent.as_posix()


@pytest.mark.parametrize("argument", ["../outside", "missing", "file.md"])
def test_directory_resolution_rejects_invalid_paths(overview, write_file, argument):
    write_file("file.md", "Body")
    with pytest.raises(ValueError):
        overview.resolve_directory_arg(argument)


def test_directory_resolution_rejects_symlink_escape(overview, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(overview, "REPO_ROOT", root)
    with pytest.raises(ValueError):
        overview.resolve_directory_arg("link")


def test_discovery_only_targets_existing_locale_overviews(overview, write_file, tmp_path):
    for name in ("docs/en/z/overview.md", "docs/en/a/overview.md", "docs/zhHans/guide/overview.md", "docs/en/no-overview/page.md", "docs/other/overview.md"):
        write_file(name, "Existing")
    (tmp_path / "docs/en/fake/overview.md").mkdir(parents=True)
    assert overview.discover_overview_directories() == [tmp_path / name for name in ("docs/en/a", "docs/en/z", "docs/zhHans/guide")]


@pytest.mark.parametrize(("args", "expected"), [([], 0), (["--all"], 0), (["missing"], 1)])
def test_main_empty_or_invalid_targets(overview, monkeypatch, args, expected):
    monkeypatch.setattr(sys, "argv", ["overview", *args])
    assert overview.main() == expected


def test_cli_rejects_all_with_explicit_directory(overview, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["overview", "--all", "docs/en"])
    with pytest.raises(SystemExit) as error:
        overview.parse_args()
    assert error.value.code == 2


def test_cli_all_dry_run_preserves_existing_files(overview, write_file, monkeypatch):
    path = write_file("docs/en/guide/overview.md", "Old")
    write_file("docs/en/guide/page.md", "Body")
    monkeypatch.setattr(sys, "argv", ["overview", "--all", "--dry-run"])
    assert overview.main() == 0
    assert path.read_text(encoding="utf-8") == "Old"


def test_main_processes_remaining_targets_after_failure(overview, tmp_path, monkeypatch):
    paths = [tmp_path / "first", tmp_path / "second"]
    monkeypatch.setattr(sys, "argv", ["overview", "--all"])
    monkeypatch.setattr(overview, "discover_overview_directories", lambda: paths)
    generate = Mock(side_effect=[False, True])
    monkeypatch.setattr(overview, "generate_overview_for_directory", generate)
    assert overview.main() == 1
    assert [call.args for call in generate.call_args_list] == [(paths[0], False), (paths[1], False)]
