"""PR records and notification formatting, with no outbound calls."""

import copy
import json
import sys

import pytest


@pytest.fixture
def pr_parser(load_script, write_file, monkeypatch):
    module = load_script("parse_pr_payload.py")
    template = write_file("templates/pr_record.json", json.dumps({"category": "guide"}))
    monkeypatch.setattr(module, "PR_RECORD_TEMPLATE_PATH", template)
    return module


@pytest.fixture
def pr_payload():
    return {
        "number": 17,
        "title": "Improve the guide",
        "assignee": {"login": "actual-author"},
        "user": {"login": "actual-author"},
        "created_at": "2026-01-01T12:00:00Z",
        "merged_at": "2026-01-03T12:00:00Z",
        "html_url": "https://example.invalid/pull/17",
    }


@pytest.mark.parametrize("reviewers", [None, [], [{"login": "reviewer"}, {"login": "second"}]])
def test_pr_record_mapping(pr_parser, pr_payload, reviewers):
    if reviewers is not None:
        pr_payload["requested_reviewers"] = reviewers
    original = copy.deepcopy(pr_payload)
    assert pr_parser.parse_pr_payload(pr_payload) == {
        "category": "guide",
        "id": 17,
        "title": "Improve the guide",
        "author": "actual-author",
        "createAt": "2026-01-01T12:00:00Z",
        "mergedAt": "2026-01-03T12:00:00Z",
        "link": "https://example.invalid/pull/17",
        "reviewer": "reviewer" if reviewers else "",
    }
    assert pr_payload == original


def test_pr_records_do_not_share_template_state(pr_parser, pr_payload):
    first = pr_parser.parse_pr_payload(pr_payload)
    first["category"] = "changed"
    pr_payload["number"] = 18
    second = pr_parser.parse_pr_payload(pr_payload)
    assert second["id"] == 18
    assert second["category"] == "guide"
    assert first["id"] == 17


def test_pr_cli_outputs_json(pr_parser, pr_payload, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["parse_pr_payload.py", "--pr", json.dumps(pr_payload)])
    pr_parser.main()
    assert json.loads(capsys.readouterr().out) == pr_parser.parse_pr_payload(pr_payload)


def test_unassigned_pr_uses_author(pr_parser, pr_payload):
    pr_payload["assignee"] = None
    assert pr_parser.parse_pr_payload(pr_payload)["author"] == "actual-author"


def test_pr_author_is_not_assignee(pr_parser, pr_payload):
    pr_payload["assignee"] = {"login": "assigned-editor"}
    assert pr_parser.parse_pr_payload(pr_payload)["author"] == "actual-author"


@pytest.fixture
def notifier(load_script):
    return load_script("send_discord_push_notification.py")


def test_aggregate_uses_latest_status_without_mutation(notifier):
    commits = [
        {"added": ["a.md", "b.md"], "modified": ["c.md"]},
        {"modified": ["a.md"], "removed": ["b.md"]},
        {"added": ["b.md"]},
        {},
    ]
    original = copy.deepcopy(commits)
    assert notifier.aggregate_changed_files(commits) == {"a.md": "M", "b.md": "A", "c.md": "M"}
    assert commits == original
    assert notifier.aggregate_changed_files([]) == {}


def test_format_file_list_empty_and_sorted(notifier):
    assert notifier.format_file_list({}) == ("_no file changes_", 0)
    assert notifier.format_file_list({"z.md": "D", "a.md": "A"}) == (
        "```\nA a.md\nD z.md\n```", 2
    )


@pytest.mark.parametrize("extra", [0, 1, 5])
def test_file_count_limit(notifier, extra):
    count = notifier.FILE_LIST_LIMIT + extra
    files = {f"page-{i:02}.md": "M" for i in range(count)}
    value, total = notifier.format_file_list(files)
    assert total == count
    assert value.count("M page-") == notifier.FILE_LIST_LIMIT
    assert (f"and {extra} more" in value) == bool(extra)


def test_file_character_limit(notifier):
    value, total = notifier.format_file_list({"x" * 1500 + ".md": "A"})
    assert total == 1
    assert len(value) == notifier.FIELD_VALUE_CHAR_LIMIT
    assert value.startswith("```\nA ")
    assert value.endswith("...```")


@pytest.mark.parametrize("event", [{}, {"commits": None, "head_commit": None}])
def test_payload_defaults(notifier, event):
    payload = notifier.build_payload(event, "team/docs", "main", "editor", "https://example.invalid")
    embed = payload["embeds"][0]
    assert embed["title"] == "Markdown update: 0 commits pushed to team/docs"
    assert embed["description"] == "(no commit message)"
    assert embed["color"] == notifier.COLOR_GREEN
    assert embed["url"] == "https://example.invalid/team/docs"
    assert embed["author"]["name"] == "editor"
    assert embed["fields"][2]["value"] == "[`?`](https://example.invalid/team/docs/commit/)"
    assert "timestamp" not in embed


def test_payload_full_event_and_truncation(notifier):
    event = {
        "commits": [{"removed": ["old.md"], "added": ["new.md"]}],
        "compare": "https://example.invalid/compare",
        "head_commit": {
            "id": "1234567890abcdef",
            "url": "https://example.invalid/commit",
            "author": {"name": "Test Author"},
            "message": "x" * (notifier.DESCRIPTION_CHAR_LIMIT + 1),
            "timestamp": "2026-01-01T00:00:00Z",
        },
    }
    original = copy.deepcopy(event)
    embed = notifier.build_payload(event, "team/docs", "topic", "editor", "https://example.invalid")["embeds"][0]
    assert embed["title"] == "Markdown update: 1 commit pushed to team/docs"
    assert embed["url"] == event["compare"]
    assert embed["color"] == notifier.COLOR_ORANGE
    assert embed["author"] == {
        "name": "Test Author", "url": "https://example.invalid/editor",
        "icon_url": "https://example.invalid/editor.png",
    }
    assert embed["description"] == "x" * (notifier.DESCRIPTION_CHAR_LIMIT - 3) + "..."
    assert embed["timestamp"] == event["head_commit"]["timestamp"]
    assert embed["fields"][0]["value"] == "topic"
    assert embed["fields"][2]["value"] == "[`1234567`](https://example.invalid/commit)"
    assert embed["fields"][3]["name"] == "Changed files (2)"
    assert event == original


def test_payload_after_sha_fallback_and_restored_file_color(notifier):
    event = {"after": "abcdef12345", "commits": [{"removed": ["a.md"]}, {"added": ["a.md"]}]}
    embed = notifier.build_payload(event, "team/docs", "main", "editor", "https://example.invalid")["embeds"][0]
    assert embed["color"] == notifier.COLOR_GREEN
    assert embed["fields"][2]["value"] == "[`abcdef1`](https://example.invalid/team/docs/commit/abcdef12345)"
