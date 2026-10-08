import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

PLAYLIST_ID = "PLwfmP9Dv35SxK9Wf56hvy0EluiLJhAiwK"
DOWNLOAD_DIR = Path(r"D:\Music\PhonePlaylist\Playlists")

API_KEY = os.environ.get("YOUTUBE_API_KEY")

PROCESSED_FILE = DOWNLOAD_DIR / "processed.json"
FAILED_FILE = DOWNLOAD_DIR / "failed.json"

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/playlistItems"

# ---------------------------------------------------------------------------


def get_playlist_videos(playlist_id: str, api_key: str) -> list[dict]:
    """Fetch every video in the playlist, paginating past the 50-per-call limit.

    No special handling is needed for playlists over 200/250 items — the
    API itself has no such cap. We just follow nextPageToken until it's
    no longer present.
    """
    videos = []
    page_token = None

    while True:
        params = {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(YOUTUBE_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            snippet = item["snippet"]
            video_id = snippet["resourceId"]["videoId"]
            title = snippet["title"]
            channel = snippet.get("videoOwnerChannelTitle", snippet.get("channelTitle", ""))

            # Deleted/private videos show up with these sentinel titles and
            # carry no usable metadata — skip them, nothing to download.
            if title in ("Private video", "Deleted video"):
                continue

            videos.append({"video_id": video_id, "title": title, "channel": channel})

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return videos


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic-ish write, avoids truncated files on crash


def sanitize_filename(name: str) -> str:
    """Strip characters that are illegal in Windows/macOS/Linux filenames."""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip().rstrip(".")
    return name or "untitled"


def download_video(video_id: str, title: str, out_dir: Path) -> tuple[bool, str]:
    """Download one video as mp3 via yt-dlp. Returns (success, error_message)."""
    safe_title = sanitize_filename(title)
    out_template = str(out_dir / f"{safe_title}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", out_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return True, ""
    else:
        # Keep just the last line or two of stderr — yt-dlp's full output is noisy.
        error_lines = [l for l in result.stderr.strip().splitlines() if l.strip()]
        error_msg = error_lines[-1] if error_lines else "Unknown yt-dlp error"
        return False, error_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading anything.",
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set the YOUTUBE_API_KEY environment variable first.")
        sys.exit(1)

    if PLAYLIST_ID == "PUT_YOUR_PLAYLIST_ID_HERE":
        print("ERROR: Edit PLAYLIST_ID at the top of this script first.")
        sys.exit(1)

    print(f"Fetching playlist {PLAYLIST_ID}...")
    try:
        current_videos = get_playlist_videos(PLAYLIST_ID, API_KEY)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch playlist: {e}")
        sys.exit(1)

    print(f"Playlist has {len(current_videos)} videos (private/deleted already excluded).")

    processed = load_json(PROCESSED_FILE)
    failed = load_json(FAILED_FILE)

    new_videos = [v for v in current_videos if v["video_id"] not in processed]

    if not new_videos:
        print("Nothing new to download.")
        return

    print(f"{len(new_videos)} new video(s) to download.")

    if args.dry_run:
        print("\n--- DRY RUN: would download ---")
        for v in new_videos:
            print(f"  [{v['video_id']}] {v['title']}")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for i, v in enumerate(new_videos, 1):
        print(f"[{i}/{len(new_videos)}] Downloading: {v['title']}")
        success, error_msg = download_video(v["video_id"], v["title"], DOWNLOAD_DIR)

        if success:
            processed[v["video_id"]] = {
                "title": v["title"],
                "channel": v["channel"],
            }
            save_json(PROCESSED_FILE, processed)  # save after every success
            # a video that previously failed and now succeeds should no
            # longer show up in failed.json
            failed.pop(v["video_id"], None)
        else:
            print(f"  FAILED: {error_msg}")
            failed[v["video_id"]] = {
                "title": v["title"],
                "channel": v["channel"],
                "error": error_msg,
            }
            save_json(FAILED_FILE, failed)

    succeeded = sum(1 for v in new_videos if v["video_id"] in processed)
    print(f"\nDone. {succeeded}/{len(new_videos)} downloaded successfully.")
    if failed:
        print(f"{len(failed)} total video(s) in failed.json — see that file for reasons.")


if __name__ == "__main__":
    main()