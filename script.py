"""Download full Cursor docs as markdown from cursor.com/llms.txt."""

import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "cursor_markdown"
LLMS_INDEX_URL = "https://cursor.com/llms.txt"
BASE = "https://cursor.com"
WORKERS = 4
RETRIES = 3
TIMEOUT = 90
USER_AGENT = "langchain-learn-docs-fetch/1.0"
SKIP_EXISTING = True


def fetch_text(url: str) -> str:
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(attempt * 2)
    raise last_err  # type: ignore[misc]


def parse_md_urls(llms_text: str) -> list[str]:
    raw = re.findall(r"https://cursor\.com[^\s\)]*\.md", llms_text)
    seen: set[str] = set()
    urls: list[str] = []
    for url in raw:
        url = re.sub(r"https://cursor\.comhttps://cursor\.com", BASE, url)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def url_to_local_path(url: str) -> Path:
    path = urlparse(url).path.lstrip("/")
    return OUTPUT_DIR / path


def download_one(url: str) -> tuple[str, bool, str]:
    out = url_to_local_path(url)
    if SKIP_EXISTING and out.is_file() and out.stat().st_size > 0:
        return url, True, f"(cached) {out.relative_to(ROOT)}"
    try:
        text = fetch_text(url)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return url, True, str(out.relative_to(ROOT))
    except urllib.error.HTTPError as e:
        return url, False, f"HTTP {e.code}"
    except Exception as e:
        return url, False, str(e)


def main() -> None:
    print(f"Fetching index: {LLMS_INDEX_URL}")
    try:
        llms_text = fetch_text(LLMS_INDEX_URL)
    except Exception as e:
        raise SystemExit(f"Could not fetch {LLMS_INDEX_URL}: {e}") from e

    (OUTPUT_DIR / "llms.txt").write_text(llms_text, encoding="utf-8")
    urls = parse_md_urls(llms_text)
    if not urls:
        raise SystemExit("No .md URLs found in llms.txt")

    print(f"Downloading {len(urls)} pages -> {OUTPUT_DIR.relative_to(ROOT)}/")
    ok, failed, skipped = 0, [], 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, u): u for u in urls}
        for i, future in enumerate(as_completed(futures), 1):
            url, success, detail = future.result()
            if success:
                ok += 1
                if detail.startswith("(cached)"):
                    skipped += 1
                print(f"[{i}/{len(urls)}] {detail}")
            else:
                failed.append((url, detail))
                print(f"[{i}/{len(urls)}] FAIL {url}: {detail}", file=sys.stderr)

    fresh = ok - skipped
    print(
        f"\nDone: {ok}/{len(urls)} present "
        f"({fresh} new, {skipped} cached) under {OUTPUT_DIR.relative_to(ROOT)}"
    )
    if failed:
        print(f"Failed ({len(failed)}):", file=sys.stderr)
        for url, err in failed:
            print(f"  {url}: {err}", file=sys.stderr)
        # changelog.md is listed in llms.txt but returns 404 on cursor.com
        hard_fail = [f for f in failed if f[0] != f"{BASE}/changelog.md"]
        if hard_fail:
            sys.exit(1)


if __name__ == "__main__":
    main()
