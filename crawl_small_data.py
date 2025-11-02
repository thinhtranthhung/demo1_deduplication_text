import requests
import time
import json

API_KEY = "2fad19d4-32c6-49e0-9e8d-16d4e584a834"
URL = "https://content.guardianapis.com/search"

articles = []

for page in range(1, 10):
    params = {
        "api-key": API_KEY,
        "page": page,
        "page-size": 200,
        "show-fields": "bodyText"
    }
    r = requests.get(URL, params=params)
    data = r.json()

    if "response" in data and "results" in data["response"]:
        for art in data["response"]["results"]:
            content = art.get("fields", {}).get("bodyText", "").strip()
            if content:
                articles.append({"content": content})

    print(f"✅ Page {page} done.")
    time.sleep(1)

# === Ghi ra file JSON (chỉ content) ===
with open("guardian_articles_ver2.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

# === Ghi ra file TXT (nội dung nối liên tục) ===
with open("guardian_articles_ver2.txt", "w", encoding="utf-8") as f:
    for art in articles:
        f.write(art["content"] + "\n\n" + "="*80 + "\n\n")

print(f"🎉 Crawl xong {len(articles)} bài từ The Guardian!")
print("📁 Saved to guardian_articles.json và guardian_articles.txt")
