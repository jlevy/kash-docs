from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from prettyfmt import abbrev_obj, fmt_timedelta
from typing_extensions import override

from kash.utils.common.url import Url

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

log = getLogger(__name__)


@dataclass(frozen=True)
class BrowserOperationResult:
    """
    Complete result from a browser operation.
    """

    final_url: str
    content: str | bytes | None
    operation: str
    format: str | None
    heuristic_used: URLHeuristic | None
    heuristic_name: str | None
    load_state_used: str
    timeout_used: int
    viewport_width: int
    viewport_height: int
    use_stealth: bool
    use_fingerprint: bool
    execution_time_ms: int
    output_path: Path | None
    wait_for_selector: str | None

    @override
    def __str__(self) -> str:
        """Clean string representation for logging."""
        return abbrev_obj(self)


@dataclass(frozen=True)
class URLHeuristic:
    """Configuration for URL-specific wait heuristics."""

    name: str
    strategy: Literal["scroll_to_load_all", "wait_for_selector"]
    selectors: list[str] = field(default_factory=list)
    scroll_pause: int = 1000
    max_scrolls: int = 50
    timeout: int = 10000

    @classmethod
    def for_scrolling_site(
        cls,
        name: str,
        selectors: list[str],
        scroll_pause: int = 1000,
        max_scrolls: int = 50,
    ) -> URLHeuristic:
        """Create a heuristic for sites that require scrolling to load content."""
        return cls(
            name=name,
            strategy="scroll_to_load_all",
            selectors=selectors,
            scroll_pause=scroll_pause,
            max_scrolls=max_scrolls,
        )

    @classmethod
    def for_selector_wait(
        cls,
        name: str,
        selectors: list[str],
        timeout: int = 10000,
    ) -> URLHeuristic:
        """Create a heuristic for sites that just need to wait for elements."""
        return cls(
            name=name,
            strategy="wait_for_selector",
            selectors=selectors,
            timeout=timeout,
        )


# Global configuration for advanced wait heuristics
WAIT_HEURISTICS = True

# URL-specific wait heuristics
# Each entry maps URL patterns to specific wait strategies
URL_HEURISTICS: dict[str, URLHeuristic] = {
    "https://chatgpt.com/share/": URLHeuristic(
        name="ChatGPT Share",
        strategy="scroll_to_load_all",
        selectors=[
            "[data-testid='conversation-turn']",
            ".text-base",
        ],
        scroll_pause=1000,
        max_scrolls=50,
    ),
    "https://chatgpt.com/c/": URLHeuristic(
        name="ChatGPT Conversation",
        strategy="scroll_to_load_all",
        selectors=[
            "[data-testid='conversation-turn']",
            ".text-base",
        ],
        scroll_pause=1000,
        max_scrolls=50,
    ),
    "https://twitter.com/": URLHeuristic(
        name="Twitter/X",
        strategy="wait_for_selector",
        selectors=["article", "[data-testid='tweetText']"],
        timeout=10000,
    ),
    "https://x.com/": URLHeuristic(
        name="Twitter/X",
        strategy="wait_for_selector",
        selectors=["article", "[data-testid='tweetText']"],
        timeout=10000,
    ),
}


class LoadState(StrEnum):
    """Valid load states for Playwright."""

    load = "load"
    domcontentloaded = "domcontentloaded"
    networkidle = "networkidle"
    commit = "commit"


def get_playwright_cache_dir() -> Path:
    """Get the Playwright cache directory for the current platform."""
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    elif platform.system() == "Linux":
        return Path.home() / ".cache" / "ms-playwright"
    elif platform.system() == "Windows":
        return Path.home() / "AppData" / "Local" / "ms-playwright"
    else:
        # Fallback to Linux pattern for unknown systems
        return Path.home() / ".cache" / "ms-playwright"


def check_playwright_installation() -> tuple[bool, str | None]:
    """
    Check if Playwright browsers are properly installed.
    Returns (is_installed, error_message).
    """
    try:
        # First check if playwright CLI is available
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "--version"], capture_output=True, text=True
        )

        if result.returncode != 0:
            return False, (
                "Playwright not properly installed in current Python environment. "
                "Please ensure the `playwright` package is installed."
            )

        # Check browser installation using playwright CLI
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run"],
            capture_output=True,
            text=True,
        )

        # If dry-run shows nothing to install, browsers are installed
        if "chromium" not in result.stdout.lower() or "already installed" in result.stdout.lower():
            return True, None

        # Also check cache directory as fallback
        cache_dir = get_playwright_cache_dir()
        chromium_dirs = list(cache_dir.glob("chromium-*")) if cache_dir.exists() else []

        if chromium_dirs:
            return True, None

        return False, (
            "Playwright browsers not installed. Typically you need to run "
            "`playwright install chromium` from the correct Python environment. "
            "This only needs to be done once."
        )

    except Exception as e:
        return False, f"Error checking Playwright installation: {e}"


def setup_playwright(confirm: bool = True, browsers: list[str] | None = None) -> bool:
    """
    Set up Playwright browsers in an idempotent way.

    Args:
        confirm: If True, prompt user for confirmation before installing.
                If False, install automatically without prompting.
        browsers: List of browsers to install (default: ["chromium"])

    Returns:
        True if Playwright is ready to use, False otherwise.
    """
    if browsers is None:
        browsers = ["chromium"]

    # Check if already installed
    is_installed, _ = check_playwright_installation()
    if is_installed:
        log.debug("Playwright browsers already installed")
        return True

    # Not installed - install automatically
    log.warning("Playwright browsers not found.")

    if confirm:
        # Prompt user for confirmation
        print("\nPlaywright requires browser binaries to be installed (one-time setup).")
        print(f"This will download ~200MB and install: {', '.join(browsers)}")
        print(f"Installation directory: {get_playwright_cache_dir()}")

        try:
            response = input("\nProceed with installation? [Y/n]: ").strip().lower()
            if response not in ("", "y", "yes"):
                log.warning("Installation cancelled by user.")
                return False
        except (EOFError, KeyboardInterrupt):
            print()  # New line after ^C
            log.warning("Installation cancelled.")
            return False

    # Install browsers (either confirmed by user or auto-install)
    return install_playwright_browsers(browsers)


def install_playwright_browsers(browsers: list[str] | None = None) -> bool:
    """
    Install Playwright browsers.

    Args:
        browsers: List of browsers to install (default: ["chromium"])

    Returns:
        True if successful, False otherwise.
    """
    if browsers is None:
        browsers = ["chromium"]

    try:
        log.warning("Installing Playwright browsers: %s...", ", ".join(browsers))

        # Run the installation command
        cmd = [sys.executable, "-m", "playwright", "install"] + browsers
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Also install system dependencies if needed
        if platform.system() == "Linux":
            log.warning("Installing system dependencies...")
            deps_cmd = [sys.executable, "-m", "playwright", "install-deps"] + browsers
            subprocess.run(deps_cmd, capture_output=True, text=True, check=True)

        log.warning("Playwright browsers installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        log.error("Failed to install Playwright browsers: %s", e.stderr)
        return False
    except Exception as e:
        log.error("Error installing Playwright browsers: %s", e)
        return False


async def create_stealth_context(
    browser: Browser,
    use_stealth: bool = True,
    use_fingerprint: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 800,
) -> BrowserContext:
    """
    Create a browser context with optional stealth features.

    Args:
        browser: The Playwright browser instance
        use_stealth: Whether to apply playwright-stealth evasions
        use_fingerprint: Whether to use browserforge fingerprinting
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height

    Returns:
        A configured browser context
    """
    # Create viewport configuration
    context_options = {"viewport": {"width": viewport_width, "height": viewport_height}}

    # Apply browserforge fingerprinting if requested
    if use_fingerprint:
        try:
            from browserforge.fingerprints import FingerprintGenerator
            from browserforge.injectors.playwright import AsyncNewContext

            # Generate a realistic fingerprint
            fingerprint_gen = FingerprintGenerator(
                browser=["chrome"],  # Match the browser we're using
                os=["windows", "macos", "linux"],
                device=["desktop"],
            )
            fingerprint = fingerprint_gen.generate()

            # Create context with injected fingerprint
            context = await AsyncNewContext(browser, fingerprint=fingerprint, **context_options)  # pyright: ignore
            log.debug("Applied browserforge fingerprint")
        except ImportError:
            log.warning("browserforge not available, creating standard context")
            context = await browser.new_context(**context_options)  # pyright: ignore
    else:
        context = await browser.new_context(**context_options)  # pyright: ignore

    return context


async def get_url_heuristic(url: Url) -> URLHeuristic | None:
    """
    Get the appropriate heuristic configuration for a given URL.

    Args:
        url: The URL to check

    Returns:
        Heuristic configuration dict or None if no match
    """
    for pattern, config in URL_HEURISTICS.items():
        if str(url).startswith(pattern):
            return config

    return None


async def apply_url_heuristics(page: Page, url: Url, timeout: int = 30000) -> URLHeuristic | None:
    """
    Apply URL-specific wait heuristics for known sites.

    Args:
        page: The Playwright page instance
        url: The URL being loaded
        timeout: Overall timeout in milliseconds

    Returns:
        The heuristic that was applied, or None if no heuristic matched
    """
    heuristic = await get_url_heuristic(url)
    if not heuristic:
        return None

    log.warning("Applying %s heuristics for URL", heuristic.name)

    if heuristic.strategy == "scroll_to_load_all":
        # For sites that lazy-load content on scroll (like ChatGPT)

        # Initial wait for at least one element
        if heuristic.selectors:
            try:
                await page.wait_for_selector(heuristic.selectors[0], timeout=timeout // 3)
            except Exception:
                log.warning("Initial selector not found, continuing anyway")

        # Scroll to load all content
        previous_height = 0
        scroll_count = 0

        while scroll_count < heuristic.max_scrolls:
            # Get current scroll height
            current_height = await page.evaluate("document.body.scrollHeight")

            if current_height == previous_height:
                # No new content loaded, we're done
                break

            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Wait for new content to load
            await page.wait_for_timeout(heuristic.scroll_pause)

            previous_height = current_height
            scroll_count += 1

        log.info("Scrolled %d times to load all content", scroll_count)

        # Scroll back to top for screenshots/content capture
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)

    elif heuristic.strategy == "wait_for_selector":
        # Simple selector-based waiting
        for selector in heuristic.selectors:
            try:
                await page.wait_for_selector(selector, timeout=heuristic.timeout)
                log.info("Found required selector: %s", selector)
                break
            except Exception:
                continue

    return heuristic


async def wait_for_js_frameworks(page: Page, timeout: int = 5000) -> None:
    """
    Apply heuristic wait strategies for JavaScript-heavy pages.

    This function tries to detect when common JS frameworks and dynamic content
    have finished loading, which is especially useful for SPAs (Single Page Applications)
    like ChatGPT, React apps, Next.js apps, etc.

    Args:
        page: The Playwright page instance
        timeout: Timeout for the wait function in milliseconds
    """
    try:
        # Wait a bit for dynamic content to start loading
        await page.wait_for_timeout(1000)

        # Try to wait for common JS framework indicators
        await page.wait_for_function(
            """() => {
                // Check if common JS frameworks are loaded
                return (
                    document.readyState === 'complete' &&
                    // React apps often have these
                    (document.querySelector('#root') || 
                     document.querySelector('#__next') ||
                     document.querySelector('.app') ||
                     document.querySelector('[data-reactroot]') ||
                     // Vue.js apps
                     document.querySelector('#app') ||
                     document.querySelector('[data-v-]') ||
                     // Angular apps
                     document.querySelector('app-root') ||
                     document.querySelector('[ng-version]') ||
                     // Generic content check - ensure meaningful content
                     document.body.innerText.length > 100)
                );
            }""",
            timeout=timeout,
        )
        log.debug("JS framework wait heuristics completed successfully")
    except Exception as e:
        log.debug("JS framework wait heuristics completed with: %s", e)


async def execute_browser_operation(
    url: Url,
    operation: Literal["html", "screenshot", "pdf"],
    wait_for_selector: str | None = None,
    wait_for_load_state: LoadState = LoadState.networkidle,
    timeout: int = 30000,
    viewport_width: int = 1280,
    viewport_height: int = 800,
    use_stealth: bool = True,
    use_fingerprint: bool = True,
    output_path: Path | None = None,
    **operation_kwargs: object,
) -> BrowserOperationResult:
    """
    Execute a browser operation (HTML fetch, screenshot, or PDF generation) using Playwright.

    Args:
        url: The URL to navigate to
        operation: Type of operation to perform
        wait_for_selector: Optional CSS selector to wait for before operation
        wait_for_load_state: Load state to wait for
        timeout: Page load timeout in milliseconds
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        use_stealth: Whether to apply playwright-stealth evasions
        use_fingerprint: Whether to use browserforge fingerprinting
        output_path: Path to save screenshot/PDF (required for those operations)
        **operation_kwargs: Additional arguments for the specific operation

    Returns:
        BrowserOperationResult with complete operation metadata and content
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        # Launch browser with specific args for better stealth
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
        ]

        browser: Browser = await p.chromium.launch(headless=True, args=launch_args)

        context = None
        try:
            # Create context with optional stealth features
            start_time = time.time()
            context = await create_stealth_context(
                browser,
                use_stealth=use_stealth,
                use_fingerprint=use_fingerprint,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
            )
            log.info("Created browser context in %s", fmt_timedelta(time.time() - start_time))

            page: Page = await context.new_page()
            log.info("Created page in %s", fmt_timedelta(time.time() - start_time))

            # Apply playwright-stealth if requested
            if use_stealth:
                try:
                    # Import and use playwright_stealth
                    from playwright_stealth.stealth import Stealth

                    stealth = Stealth()
                    await stealth.apply_stealth_async(page)
                except ImportError:
                    log.warning("playwright-stealth not available, proceeding without it")
                except Exception as e:
                    log.warning("Failed to apply playwright-stealth: %s", e)

            # Navigate to the page
            await page.goto(str(url), timeout=timeout)
            log.info("Navigated to page in %s", fmt_timedelta(time.time() - start_time))

            # Wait for specific conditions if requested
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, state="visible", timeout=timeout)
                log.info("Selector '%s' is visible", wait_for_selector)
            else:
                # Wait for the specified load state
                await page.wait_for_load_state(wait_for_load_state.value, timeout=timeout)  # pyright: ignore

            # Apply URL-specific heuristics first
            heuristic_used = None
            if WAIT_HEURISTICS:
                heuristic_used = await apply_url_heuristics(page, url, timeout)

            # Apply generic JS framework wait heuristics if enabled and using networkidle
            if WAIT_HEURISTICS and wait_for_load_state == LoadState.networkidle:
                await wait_for_js_frameworks(page)

            # Get the final URL after any redirects
            final_url = page.url

            # Execute the requested operation
            content: str | bytes | None
            format_value: str | None = None

            if operation == "html":
                content = await page.content()
                log.info("Got HTML content in %s", fmt_timedelta(time.time() - start_time))

            elif operation == "screenshot":
                if output_path is None:
                    raise ValueError("output_path is required for screenshot operation")

                # Extract screenshot-specific kwargs with defaults
                full_page = bool(operation_kwargs.get("full_page", True))
                format_value = str(operation_kwargs.get("format", "png"))

                content = await page.screenshot(
                    path=output_path,
                    full_page=full_page,
                    type=format_value,  # pyright: ignore
                )
                log.info("Captured screenshot in %s", fmt_timedelta(time.time() - start_time))

            elif operation == "pdf":
                if output_path is None:
                    raise ValueError("output_path is required for PDF operation")

                # Extract PDF-specific kwargs with defaults
                format_value = str(operation_kwargs.get("format", "A4"))
                print_background = bool(operation_kwargs.get("print_background", True))

                content = await page.pdf(
                    path=output_path,
                    format=format_value,  # pyright: ignore
                    print_background=print_background,
                )
                log.info("Generated PDF in %s", fmt_timedelta(time.time() - start_time))

            else:
                raise ValueError(f"Unsupported operation: {operation}")

            # Calculate execution time and return result
            execution_time_ms = int((time.time() - start_time) * 1000)

            return BrowserOperationResult(
                final_url=final_url,
                content=content,
                operation=operation,
                format=format_value,
                heuristic_used=heuristic_used,
                heuristic_name=heuristic_used.name if heuristic_used else None,
                load_state_used=wait_for_load_state.value,
                timeout_used=timeout,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
                use_stealth=use_stealth,
                use_fingerprint=use_fingerprint,
                execution_time_ms=execution_time_ms,
                output_path=output_path,
                wait_for_selector=wait_for_selector,
            )

        finally:
            if context is not None:
                await context.close()
            await browser.close()


## Tests


def test_get_playwright_cache_dir():
    """Test that we get a valid cache directory path."""
    cache_dir = get_playwright_cache_dir()
    assert isinstance(cache_dir, Path)
    assert cache_dir.is_absolute()


def test_check_playwright_installation():
    """Test the Playwright installation check."""
    is_installed, error_msg = check_playwright_installation()
    # This will vary by system, just ensure it returns the expected types
    assert isinstance(is_installed, bool)
    assert error_msg is None or isinstance(error_msg, str)
