#!/usr/bin/env python3
"""Test script for browser fetching Twitter and ChatGPT URLs."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from kash.kits.docs.utils.playwright_browser import (
    LoadState,
    check_playwright_installation,
    execute_browser_operation,
    setup_playwright,
)
from kash.utils.common.url import Url
from kash.utils.text_handling.markdownify_utils import markdownify_custom
from kash.web_content.canon_url import canonicalize_url
from kash.web_content.web_extract_readabilipy import extract_text_readabilipy


@dataclass
class TestCase:
    """Test case configuration."""

    name: str
    url: str
    expected_texts: list[str]
    output_prefix: str


# Test cases
TEST_CASES = [
    TestCase(
        name="Twitter URL",
        url="https://x.com/AP/status/1951109166803599360",
        expected_texts=[
            "BREAKING: El Salvador approves indefinite presidential reelection and extends presidential terms to six years."
        ],
        output_prefix="twitter",
    ),
    TestCase(
        name="ChatGPT URL",
        url="https://chatgpt.com/share/6889563f-6f0c-800e-9bdb-558cb0e72a02",
        expected_texts=[
            "Readability + Markdownify",
            "undetected-playwright",
            "Playwright just delivers the fully-rendered DOM",
        ],
        output_prefix="chatgpt",
    ),
]


async def setup_playwright_if_needed():
    """Ensure Playwright is installed."""
    is_installed, error_msg = check_playwright_installation()
    if not is_installed:
        print(f"Error: {error_msg}")
        print("Installing Playwright...")
        if not setup_playwright(confirm=False):
            raise RuntimeError("Failed to install Playwright")


async def fetch_and_extract(test_case: TestCase):
    """Fetch URL and extract markdown content."""
    print(f"\nTesting {test_case.name}: {test_case.url}")
    print("-" * 80)

    try:
        # Canonicalize and fetch
        canonical_url = canonicalize_url(Url(test_case.url))
        result = await execute_browser_operation(
            url=Url(canonical_url),
            operation="html",
            wait_for_load_state=LoadState.domcontentloaded,
            timeout=45000,
            viewport_width=1280,
            viewport_height=800,
            use_stealth=True,
            use_fingerprint=True,
        )

        if not result.content:
            print("❌ No content retrieved")
            return

        # Show operation details
        print(
            f"✅ Operation completed: {result.heuristic_name or 'no heuristic'}, {result.execution_time_ms}ms"
        )
        print(f"📄 Content: {len(result.content)} chars")

        # Extract markdown
        html_str = (
            result.content if isinstance(result.content, str) else result.content.decode("utf-8")
        )
        page_data = extract_text_readabilipy(Url(result.final_url), html_str)
        assert page_data.clean_html
        markdown_content = markdownify_custom(page_data.clean_html)

        print(f"📝 Markdown: {len(markdown_content)} chars")

        # Check for expected content
        found_texts = [text for text in test_case.expected_texts if text in markdown_content]

        if found_texts:
            print(f"✅ Found {len(found_texts)}/{len(test_case.expected_texts)} expected text(s)")
            for text in found_texts:
                print(f"   • {text[:80]}{'...' if len(text) > 80 else ''}")
        else:
            print("❌ Expected texts not found")
            print(f"   Looking for: {test_case.expected_texts}")

        # Save outputs
        html_file = Path(f"{test_case.output_prefix}_output.html")
        md_file = Path(f"{test_case.output_prefix}_output.md")
        html_file.write_text(html_str)
        md_file.write_text(markdown_content)
        print(f"💾 Saved: {html_file}, {md_file}")

        # Show content preview
        print("\n📖 Content preview:")
        print(markdown_content[:300] + ("..." if len(markdown_content) > 300 else ""))

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests."""
    print("Browser Fetch Test Script")
    print("=" * 80)

    # Setup
    await setup_playwright_if_needed()

    # Run tests
    for test_case in TEST_CASES:
        await fetch_and_extract(test_case)

    print(f"\n✅ Completed {len(TEST_CASES)} tests")


if __name__ == "__main__":
    asyncio.run(main())
