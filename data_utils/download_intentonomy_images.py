import argparse
import json
# import os
import asyncio
import httpx
from pathlib import Path
from typing import List
from tqdm.asyncio import tqdm


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
    sleep: float = 0.2,
) -> bool:
    ensure_parent_dir(dest)

    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await client.get(url, timeout=20.0, follow_redirects=True)
                response.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(response.content)
                return True
            except Exception as e:
                if attempt >= retries:
                    print(f"Failed: {url} -> {dest} ({e})")
                    return False
                await asyncio.sleep(sleep * attempt)
    return False


def parse_images_from_json(json_path: Path) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("images", [])


async def main():
    parser = argparse.ArgumentParser(
        description="Download Intentonomy images using HTTPX Async"
    )
    parser.add_argument(
        "--json", nargs="+", required=True, help="Path(s) to JSON files"
    )
    parser.add_argument("--output-root", default="./data/sqhy_data/intent_resize")
    parser.add_argument("--skip-exist", action="store_true")
    parser.add_argument(
        "--concurrency", type=int, default=50, help="Number of concurrent downloads"
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    all_images = []
    for jp in args.json:
        path = Path(jp)
        if not path.exists():
            print(f"Error: {path} not found.")
            continue
        imgs = parse_images_from_json(path)
        all_images.extend([(img, path.name) for img in imgs])

    tasks_to_run = []
    skipped = 0

    for im, source_name in all_images:
        filename = im.get("filename")
        url = im.get("unsplash_url")
        if not filename or not url:
            continue

        dest = output_root / filename
        if args.skip_exist and dest.exists():
            skipped += 1
            continue
        tasks_to_run.append((url, dest))

    print(
        f"Total: {len(all_images)} | To download: {len(tasks_to_run)} | Skipped: {skipped}"
    )

    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=5, max_connections=args.concurrency
        )
    ) as client:
        tasks = [
            download_file(client, url, dest, semaphore) for url, dest in tasks_to_run
        ]

        results = await tqdm.gather(*tasks, desc="Downloading")

    success = sum(results)
    print(f"\nSummary: Success: {success} | Failed: {len(tasks_to_run) - success}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
