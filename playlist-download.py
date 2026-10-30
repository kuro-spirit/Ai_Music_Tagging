import os
import requests
import subprocess
import json
from pathlib import Path

URL = "https://www.googleapis.com/youtube/v3/playlistItems"
PARAMS = {
    "part": "snippet",
    "playlistId": "PLwfmP9Dv35Sz4tiuoJFxgO4Yoyh8-Lx-x",
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
            Task: Extract the main artist and song title from a YouTube video title.

            Output: A JSON object with key Artist and Title.

            Rules:
            - Extract the main artist and song title from the YouTube video title
            - Remove any extra flavour texts that isn't the artist name or song title (e.g. MV, Official, Lyric Video)
            - IF the artist cannot be found in the title, use your own knowledge to identify the correct artist
            - Give one artist ONLY
            - Keep the artist and title in the language of the song
            - For Chinese songs, use traditional script for HK, TW and Cantonese artists, and simplified for mainland artists.
            
            Examples:
            - Video Title: 光辉岁月 Beyond (歌词版) Sample Output: {{"Artist": "Beyond", "Title": "光輝歲月"}}
            - Video Title: Beyond 真的爱你 Zan Dik Ngoi Nei Sample Output: {{"Artist": "Beyond", "Title": "真的愛你"}}
            - Video Title: 我的驕傲 Sample Output: {{"Artist": "容祖兒", "Title": "我的驕傲"}}
            - Video Title: Eric周興哲《你，好不好？ How Have You Been?》Official Music Video《遺憾拼圖》片尾曲 Sample Output: {{"Artist": "周興哲", "Title": "你，好不好"}}
            - Video Title: Kaze Ni Fukarete Sample Output: {{"Artist": "福原 遥", "Title": "風に吹かれて"}}
            - Video Title: 黃小琥 Tiger Huang - 沒那麼簡單 [Official 官方完整版MV] Sample Output: {{"Artist": "黃小琥", "Title": "沒那麼簡單"}}
            - Video Title: 薛之谦 Joker Xue - 暧昧 歌词 Sample Output: {{"Artist": "薛之谦", "Title": "暧昧"}}
            
            Convert this video title: {title}
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
        # print(response.status_code, data)

        for item in data["items"]:
            snippet = item["snippet"]
            song_id = snippet["resourceId"]["videoId"]
            video_title = snippet["title"]
            channel_name = snippet.get("videoOwnerChannelTitle")

            result = json.loads(call_ollama(video_title))
            name = f"{result['Artist']}--{result['Title']}"

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