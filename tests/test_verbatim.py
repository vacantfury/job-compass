"""Verbatim checks retain body line numbers and reject hidden omissions."""

import sys

import pytest


@pytest.fixture
def verbatim(load_script):
    return load_script("check_verbatim.py")


@pytest.mark.parametrize(("text", "expected"), [(" **Hello** `world`\n\t", "Helloworld"), ("A, B!", "A,B!"), ("\u9762 \u8bd5", "\u9762\u8bd5"), ("", "")])
def test_normalize(verbatim, text, expected):
    assert verbatim.normalize(text) == expected


def test_body_lines_skip_wrappers_but_keep_code_and_line_numbers(verbatim):
    md = "\n".join([
        "---", "title: Guide", "---", "# Heading", "", "<InterviewDetail />",
        '<ReferenceSource url="https://example.invalid" />',
        "<ReferenceSource", '  url="https://example.invalid"', "/>",
        "## \u57fa\u672c\u4fe1\u606f", "Metadata", "### Child", "More metadata",
        "## Interview", "Actual body", "```python", "print('body')", "```",
        "::: warning \u514d\u8d23\u58f0\u660e", "Disclaimer", ":::", "Final body",
        "## \u9762\u8bd5\u7ed3\u679c\u53cd\u9988", "Outcome metadata", "## Next", "Last line",
    ])
    assert list(verbatim.body_lines(md)) == [(16, "Actual body"), (18, "print('body')"), (23, "Final body"), (27, "Last line")]


def test_unclosed_disclaimer_is_an_error(verbatim):
    with pytest.raises(ValueError):
        list(verbatim.body_lines("Body\n::: warning \u514d\u8d23\u58f0\u660e\nHidden"))


def test_other_warning_blocks_are_not_ignored(verbatim):
    assert list(verbatim.body_lines("::: warning Caution\nBody\n:::")) == [(1, "::: warning Caution"), (2, "Body"), (3, ":::")]


@pytest.mark.parametrize("marker", ["-", "*", "+", "1.", "2)", ">"])
def test_main_normalizes_links_lists_and_emphasis(verbatim, write_file, monkeypatch, capsys, marker):
    article = write_file("article.md", f"# Heading\n{marker} **Hello** [world](https://example.invalid)\n`Code`\n")
    source = write_file("source.txt", "Hello [world](https://example.invalid/source)\nCode")
    monkeypatch.setattr(sys, "argv", ["check", str(article), str(source)])
    assert verbatim.main() == 0
    assert capsys.readouterr().out == "OK: every body line appears verbatim in the source\n"


def test_main_reports_missing_body_with_original_line(verbatim, write_file, monkeypatch, capsys):
    article = write_file("article.md", "---\ntitle: Guide\n---\n# Heading\nOriginal\nInvented\n")
    source = write_file("source.txt", "Original")
    monkeypatch.setattr(sys, "argv", ["check", str(article), str(source)])
    assert verbatim.main() == 1
    assert capsys.readouterr().out == "1 body line(s) not found verbatim in the source:\n  6: Invented\n"


def test_main_rejects_unclosed_disclaimer(verbatim, write_file, monkeypatch, capsys):
    article = write_file("article.md", "Body\n::: warning \u514d\u8d23\u58f0\u660e\nHidden")
    source = write_file("source.txt", "Body")
    monkeypatch.setattr(sys, "argv", ["check", str(article), str(source)])
    assert verbatim.main() == 1
    assert capsys.readouterr().err


@pytest.mark.parametrize("args", [[], ["article.md"], ["article.md", "source.txt", "extra"]])
def test_main_usage_error(verbatim, monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["check", *args])
    assert verbatim.main() == 2
    assert capsys.readouterr().err


def test_unclosed_reference_cannot_pass_verbatim_check(verbatim, write_file, monkeypatch):
    article = write_file("article.md", '<ReferenceSource\n  url="https://example.invalid"\nInvented body')
    source = write_file("source.txt", "Original body")
    monkeypatch.setattr(sys, "argv", ["check", str(article), str(source)])
    assert verbatim.main() == 1
