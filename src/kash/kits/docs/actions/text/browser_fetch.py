from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import is_url_resource
from kash.kits.docs.utils.playwright_browser import (
    LoadState,
    check_playwright_installation,
    execute_browser_operation,
    setup_playwright,
)
from kash.model import ActionInput, ActionResult, FileExt, Format, Param
from kash.model.items_model import ItemType
from kash.utils.common.format_utils import fmt_loc
from kash.utils.common.url import Url
from kash.utils.errors import InvalidInput
from kash.workspaces import current_ws

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Global constants for browser operation defaults
SCREENSHOT_FULL_PAGE = True
PDF_PRINT_BACKGROUND = True
USE_STEALTH = True
USE_FINGERPRINT = True
AUTO_INSTALL = True


@kash_action(
    precondition=is_url_resource,
    params=(
        Param(
            name="output_mode",
            description="Output mode: 'html' for HTML content, 'screenshot' for PNG/JPEG image, 'pdf' for PDF document",
            type=str,
            default_value="html",
        ),
        Param(
            name="wait_for_selector",
            description="CSS selector to wait for before considering page loaded (e.g., 'div.content')",
            type=str,
            default_value=None,
        ),
        Param(
            name="wait_for_load_state",
            description="Load state to wait for",
            type=LoadState,
            default_value=LoadState.networkidle,
        ),
        Param(
            name="timeout",
            description="Page load timeout in milliseconds",
            type=int,
            default_value=30000,
        ),
        Param(
            name="viewport_width",
            description="Browser viewport width in pixels",
            type=int,
            default_value=1280,
        ),
        Param(
            name="viewport_height",
            description="Browser viewport height in pixels",
            type=int,
            default_value=800,
        ),
        Param(
            name="format",
            description="Format for screenshot (png/jpeg) or PDF (A4/A3/Letter/etc)",
            type=str,
            default_value=None,
        ),
    ),
)
def browser_fetch(
    input: ActionInput,
    output_mode: str = "html",
    wait_for_selector: str | None = None,
    wait_for_load_state: LoadState = LoadState.networkidle,
    timeout: int = 30000,
    viewport_width: int = 1280,
    viewport_height: int = 800,
    format: str | None = None,
) -> ActionResult:
    """
    Fetch a URL using a headless browser (Playwright) with multiple output options.

    This action uses Playwright to interact with web pages, handling JavaScript-rendered
    content that regular HTTP requests cannot capture.

    Output modes:
    - 'html': Returns the HTML content of the page (default)
    - 'screenshot': Takes a screenshot and saves it as PNG or JPEG (full page)
    - 'pdf': Generates a PDF of the page (with background graphics)

    Prerequisites:
    - Playwright browsers must be installed: `uv run playwright install chromium`
    - This only needs to be done once per system

    The action will check for browser installation and optionally install automatically.

    Stealth Features:
    - use_stealth: Applies playwright-stealth evasions to bypass basic bot detection
    - use_fingerprint: Uses browserforge to generate realistic browser fingerprints
    - Both can be used together for maximum anti-detection capability

    Installation:
    - auto_install: Automatically install Playwright browsers if not found
    - confirm_install: Prompt for confirmation before installing (only if auto_install=True)

    URL-Specific Heuristics:
    The browser fetcher applies intelligent wait strategies based on the URL:
    - ChatGPT (chatgpt.com/share/, chatgpt.com/c/): Scrolls to load entire conversation
    - Twitter/X: Waits for tweet elements to appear
    - Other sites: Uses generic JS framework detection

    For ChatGPT conversations, it will automatically scroll through the entire
    conversation to ensure all messages are loaded before capturing content.
    """
    if not input.items:
        raise InvalidInput("No items provided")

    item = input.items[0]

    if not item.url:
        raise InvalidInput("Item must have a URL")

    # Validate output_mode
    valid_modes = ["html", "screenshot", "pdf"]
    if output_mode not in valid_modes:
        raise InvalidInput(
            f"Invalid output_mode '{output_mode}'. Must be one of: {', '.join(valid_modes)}"
        )

    # Check if Playwright browsers are installed
    is_installed, error_msg = check_playwright_installation()

    if not is_installed:
        if AUTO_INSTALL:
            # Try to set up Playwright with interactive prompt
            if not setup_playwright(confirm=False):
                raise InvalidInput(
                    "Playwright setup failed or was cancelled. "
                    "Please run manually: uv run playwright install chromium"
                )
        else:
            raise InvalidInput(error_msg)

    url = item.url

    ws = current_ws()

    # Set up format defaults and create appropriate output item
    if output_mode == "html":
        # For HTML mode, return content in the body
        output_item = item.derived_copy(
            type=ItemType.doc,
            format=Format.html,
            file_ext=FileExt.html,
        )
        target_path = None
    else:
        # For screenshot/PDF modes, create export items with proper format
        if output_mode == "screenshot":
            if format is None:
                format = "png"

            if format.lower() == "png":
                file_format = Format.png
                file_ext = FileExt.png
            elif format.lower() in ["jpg", "jpeg"]:
                file_format = Format.jpeg
                file_ext = FileExt.jpg
            else:
                raise InvalidInput(f"Unsupported screenshot format: {format}")

        elif output_mode == "pdf":
            if format is None:
                format = "letter"
            file_format = Format.pdf
            file_ext = FileExt.pdf
        else:
            raise InvalidInput(f"Unsupported output_mode for file creation: {output_mode}")

        # Create export item and get target path from workspace
        output_item = item.derived_copy(
            type=ItemType.export,
            format=file_format,
            file_ext=file_ext,
        )
        target_path = ws.assign_store_path(output_item)
        log.message("Will save %s to: %s", output_mode, fmt_loc(target_path))

    log.message(
        "Fetching %s with browser (mode: %s, wait_for: %s, load_state: %s)...",
        url,
        output_mode,
        wait_for_selector or "none",
        wait_for_load_state,
    )

    try:
        # Prepare operation-specific kwargs using global constants
        operation_kwargs = {}
        if output_mode == "screenshot":
            operation_kwargs.update(
                {
                    "full_page": SCREENSHOT_FULL_PAGE,
                    "format": format,
                }
            )
        elif output_mode == "pdf":
            operation_kwargs.update(
                {
                    "format": format,
                    "print_background": PDF_PRINT_BACKGROUND,
                }
            )

        # Run the async browser operation
        result = asyncio.run(
            execute_browser_operation(
                url=Url(url),
                operation=output_mode,  # pyright: ignore
                wait_for_selector=wait_for_selector,
                wait_for_load_state=wait_for_load_state,
                timeout=timeout,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                use_stealth=USE_STEALTH,
                use_fingerprint=USE_FINGERPRINT,
                output_path=target_path,
                **operation_kwargs,
            )
        )

        # Log comprehensive operation details
        log.message("Browser operation completed: %s", result)

        # Handle output based on mode
        if output_mode == "html":
            if result.content is None:
                raise InvalidInput("HTML content is None")
            if not isinstance(result.content, str):
                raise InvalidInput("HTML content must be string")
            log.message("Successfully fetched %d characters of HTML", len(result.content))

            # Set HTML content in body
            output_item.body = result.content
        else:
            # Screenshot or PDF mode - content saved to file
            if result.content is None:
                raise InvalidInput(f"{output_mode.capitalize()} content is None")
            if not isinstance(result.content, bytes):
                raise InvalidInput(f"{output_mode.capitalize()} content must be bytes")
            log.message(
                "Successfully generated %s: %s (%d bytes)",
                output_mode,
                target_path,
                len(result.content),
            )

            if target_path:
                output_item.mark_as_saved(target_path)

        # Update URL if there were redirects
        if result.final_url != url:
            log.message("Page redirected to: %s", result.final_url)
            output_item.url = Url(result.final_url)

        return ActionResult(items=[output_item])

    except TimeoutError:
        raise InvalidInput(f"Timeout loading page after {timeout}ms")
    except Exception as e:
        raise InvalidInput(f"Error during browser operation: {e}")
