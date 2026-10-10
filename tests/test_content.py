"""HTML extraction and subscriber metadata use temporary source files."""

import pytest
import yaml


@pytest.mark.parametrize(("html", "expected"), [
    ("", ""),
    ("<h1>Guide</h1><p>First <b>bold</b> end.</p>", "Guide\n\nFirst\n\nbold\n\nend."),
    ("<p>A &amp; B</p><p> C &lt; D </p>", "A & B\n\nC < D"),
    ("<!-- hidden --><script>hidden()</script><style>.hidden {}</style><p>Visible</p>", "Visible"),
    ("<p>First<br>Second", "First\n\nSecond"),
])
def test_html_to_text(load_script, html, expected):
    assert load_script("convert_html_to_markdown.py").html_to_text(html) == expected


def test_convert_html_file_uses_same_stem_and_preserves_source(load_script, write_file):
    module = load_script("convert_html_to_markdown.py")
    source = write_file("folder/page.v2.html", "<p>First</p><p>Second</p>")
    module.convert_html_file(str(source))
    assert source.with_suffix(".md").read_text(encoding="utf-8") == "First\n\nSecond"
    assert source.read_text(encoding="utf-8") == "<p>First</p><p>Second</p>"


@pytest.mark.parametrize(("frontmatter", "expected"), [("email: reader@example.invalid", "reader@example.invalid"), ("title: Guide", None), ("email: null", None)])
def test_extract_subscriber_email_and_filename(load_script, write_file, frontmatter, expected):
    module = load_script("notify_direct_hire_company_subscribers.py")
    source = write_file("nested/company.guide.md", f"---\n{frontmatter}\n---\nBody")
    assert module.extract_email_and_filename(str(source)) == (expected, "company.guide")


@pytest.mark.xfail(raises=AttributeError, reason="Empty frontmatter loads as None and crashes subscriber metadata extraction.")
def test_extract_subscriber_with_empty_frontmatter(load_script, write_file):
    module = load_script("notify_direct_hire_company_subscribers.py")
    source = write_file("guide.md", "---\n---\nBody")
    assert module.extract_email_and_filename(str(source)) == (None, "guide")


@pytest.mark.xfail(raises=yaml.YAMLError, reason="Splitting on every triple dash truncates valid quoted YAML values.")
def test_extract_subscriber_with_triple_dash_in_title(load_script, write_file):
    module = load_script("notify_direct_hire_company_subscribers.py")
    source = write_file("guide.md", '---\ntitle: "Guide --- details"\nemail: reader@example.invalid\n---\nBody')
    assert module.extract_email_and_filename(str(source)) == ("reader@example.invalid", "guide")
