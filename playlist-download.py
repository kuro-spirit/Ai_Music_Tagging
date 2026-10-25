import os
import requests
import subprocess
import json
from pathlib import Path

URL = "https://www.googleapis.com/youtube/v3/playlistItems"
PARAMS = {
    "part": "snippet",
    "playlistId": "PLwfmP9Dv35SzjdwmCk__I-jzxSf7eynyO",
    "maxResults": 50,
    "key": os.environ.get("YOUTUBE_API_KEY")
}
DOWNLOAD_DIR = Path(r"D:\Music\PhonePlaylist\test")

def download_song(video_id: str) -> bool:
    result = subprocess.run([
        "D:\Projects\Ai_Music_Tagging\yt-dlp.exe",
        "-x",
        "--audio-format", "mp3",
        "-o", str(DOWNLOAD_DIR / "%(title)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}"
    ])

    return result.returncode == 0

def load_downloaded(file: str) -> dict:
    with open(file, "r", encoding="utf-8") as f:
        downloaded = json.load(f)
    return downloaded

def save_downloaded(file: str, data: dict) -> None:
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def download_playlist():
    page_token = None
    songs = load_downloaded(DOWNLOAD_DIR / "downloaded.json")

    while True:
        if page_token:
            PARAMS["pageToken"] = page_token

        response = requests.get(URL, PARAMS)
        data = response.json()

        for item in data["items"]:
            snippet = item["snippet"]
            song_id = snippet["resourceId"]["videoId"]

            if snippet["title"] in ("Private video", "Deleted video"):
                continue

            if song_id in songs:
                continue

            downloaded = download_song(song_id)

            if downloaded:
                songs[song_id] = {
                    "title": snippet["title"],
                    "channel": snippet.get("videoOwnerChannelTitle")
                }

                save_downloaded(DOWNLOAD_DIR / "downloaded.json", songs)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

if __name__ == "__main__":
    download_playlist()