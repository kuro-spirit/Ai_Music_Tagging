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

def download_song(video_id: str, title: str) -> bool:
    result = subprocess.run([
        "D:\Projects\Ai_Music_Tagging\yt-dlp.exe",
        "-x",
        "--audio-format", "mp3",
        "-o", str(DOWNLOAD_DIR / f"{title}.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}"
    ])

    return result.returncode == 0

def load_downloaded(file: str=DOWNLOAD_DIR / "downloaded.json") -> dict:
    with open(file, "r", encoding="utf-8") as f:
        downloaded = json.load(f)
    return downloaded

def save_downloaded(data: dict, file: str=DOWNLOAD_DIR / "downloaded.json") -> None:
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def call_ollama(title: str, model: str="qwen2.5:14b") -> str:
    prompt=f"""
            Format this youtube video title: {title}, making sure to remove all flavour text in the title.
            Return a JSON with keys artist and title.
            If the artist name isn't clear, or the name isn't in the correct language search online for the correct one.
            Keep the language of both the artist and title to the same as the song.
            """

    response = requests.post(
        url="http://localhost:11434/api/chat", 
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0}
        }
    )
    return response.json()["message"]["content"]

def download_playlist():
    page_token = None
    songs = load_downloaded()

    while True:
        if page_token:
            PARAMS["pageToken"] = page_token

        response = requests.get(URL, PARAMS)
        data = response.json()

        for item in data["items"]:
            snippet = item["snippet"]
            song_id = snippet["resourceId"]["videoId"]
            video_title = snippet["title"]
            channel_name = snippet.get("videoOwnerChannelTitle")

            result = json.loads(call_ollama(video_title))
            name = f"{result['artist']}--{result['title']}"

            if snippet["title"] in ("Private video", "Deleted video"):
                continue

            if song_id in songs:
                continue

            downloaded = download_song(song_id, name)

            if downloaded:
                songs[song_id] = {
                    "title": snippet["title"],
                    "channel": snippet.get("videoOwnerChannelTitle")
                }

                save_downloaded(songs)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

if __name__ == "__main__":
    download_playlist()
    Test_titles = [
        "郭采潔 Amber Kuo -《我的未來式》Official Lyric Video",
        "AMV Kimi no Na wa Original Zen Zen Zense",
        "我的少女時代 Our Times】Movie Theme Song - 田馥甄 Hebe Tien《小幸運 A Little Happiness》Official MV",
        "Joyce Chu 四葉草 Ft. Namewee 黃明志【MALAYSIA CHABOR 馬來西亞查某】 @RED People",
        "林峯-愛不疚 MV 鍾嘉欣與林峯版(80,90後電視回憶)"
    ]

    # for title in Test_titles: 
    #     print(call_ollama(title))