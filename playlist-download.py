import os
import requests
import subprocess
import json
import re
from mutagen.id3 import ID3, ID3NoHeaderError, TIT2, TPE1
from pathlib import Path

URL = "https://www.googleapis.com/youtube/v3/playlistItems"
PARAMS = {
    "part": "snippet",
    "playlistId": "PLwfmP9Dv35SxK9Wf56hvy0EluiLJhAiwK",
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

def clean_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*%]', "", name).rstrip(". ")

def tag_song(path: Path, title: str, artist: str):
    tags = ID3(path)

    tags.add(TIT2(encoding=3, text=title))
    tags.add(TPE1(encoding=3, text=artist))

    tags.save(path)

def call_ollama(title: str, model: str="qwen2.5:14b") -> str:
    prompt=f"""
            Task: Extract the main artist and song title from a YouTube video title.

            Output: A JSON object with the keys Artist and Title.

            Rules:
            - Extract the main artist and song title from the YouTube video title
            - Remove any extra flavour texts that isn't the artist name or song title (e.g. MV, Official, Lyric Video)
            - IF the artist cannot be found in the title, use your own knowledge to identify the correct artist
            - Give one artist ONLY
            - Convert only text that is already in Chinese characters.
            - If the video title is in English or romanized form, and the song has an original-language title, use the original title. Otherwise don't translate.
            - If a Chinese name appears in both Chinese and Latin forms, use the Chinese one.

            Script decision tree:
            1. What language is the song? (Cantonese, Mandarin, Japanese, English, other)
            2. Japanese: Use Japanese Script, keep Japanese Kanji as written, don't convert to Chinese HanZi
            3. Cantonese: Use traditional Chinese Script, never simplified, no matter the Artist's origin
            4. Mandarin: Traditional script for Hong Kong, Taiwan, Malaysia and Singapore artists; simplified for Mainland or Other artists.
            5. English: Use English
            6. Other: Keep as written for non-chinese songs
            
            Examples:
            - Video Title: If I could be a constellation Sample Output: {{"Artist": "結束バンド", "Title": "星座になれたら"}}
            - Video Title: Beyond 真的爱你 Zan Dik Ngoi Nei Sample Output: {{"Artist": "Beyond", "Title": "真的愛你"}}
            - Video Title: 我的驕傲 Sample Output: {{"Artist": "容祖兒", "Title": "我的驕傲"}}
            - Video Title: Eric周興哲《你，好不好？ How Have You Been?》Official Music Video《遺憾拼圖》片尾曲 Sample Output: {{"Artist": "周興哲", "Title": "你，好不好"}}
            - Video Title: Kaze Ni Fukarete Sample Output: {{"Artist": "福原遥", "Title": "風に吹かれて"}}
            - Video Title: 黃小琥 Tiger Huang - 沒那麼簡單 [Official 官方完整版MV] Sample Output: {{"Artist": "黃小琥", "Title": "沒那麼簡單"}}
            - Video Title: 薛之谦 Joker Xue - 暧昧 歌词 Sample Output: {{"Artist": "薛之谦", "Title": "暧昧"}}
            - Video Title: 王力宏 Leehom Wang - 唯一 Sample Output: {{"Artist": "王力宏", "Title": "唯一"}}
            - Video Title: 邓紫棋 (Deng Zi Qi / G.E.M) - 泡沫 (Pao Mo / Bubble) (Simplified Chinese / Pinyin Lyrics HD) Sample Output: {{"Artist": "鄧紫棋", "Title": "泡沫"}}
            
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

            if snippet["title"] in ("Private video", "Deleted video"):
                continue

            if song_id in songs:
                continue

            result = json.loads(call_ollama(video_title))
            name = clean_filename(f"{result['Artist']}--{result['Title']}")

            downloaded = download_song(song_id, name)

            if downloaded:
                songs[song_id] = {
                    "title": snippet["title"],
                    "channel": snippet.get("videoOwnerChannelTitle")
                }

                tag_song(DOWNLOAD_DIR / f"{name}.mp3", result['Title'], result['Artist'])
                save_downloaded(songs)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

if __name__ == "__main__":
    download_playlist()
    # print(clean_filename("Zadon--Ah Beng Meh?."))
    Test_titles = [
        "郭采潔 Amber Kuo -《我的未來式》Official Lyric Video",
        "AMV Kimi no Na wa Original Zen Zen Zense",
        "我的少女時代 Our Times】Movie Theme Song - 田馥甄 Hebe Tien《小幸運 A Little Happiness》Official MV",
        "Joyce Chu 四葉草 Ft. Namewee 黃明志【MALAYSIA CHABOR 馬來西亞查某】 @RED People",
        "林峯-愛不疚 MV 鍾嘉欣與林峯版(80,90後電視回憶)"
    ]

    # for title in Test_titles: 
    #     print(call_ollama(title))

    # path = r"D:\Music\PhonePlaylist\test\Beyond--真的愛你.mp3"

    # tag_song(path, "光輝歲月", "Beyond")