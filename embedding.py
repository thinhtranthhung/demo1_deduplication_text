import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- 1. CẤU HÌNH ---
INPUT_JSON = 'guardian_articles_ver2.json'
CONTENT_KEY = 'content'
MODEL_NAME = 'all-MiniLM-L6-v2'
OUTPUT_TXT = 'embeddings_ver2.txt'
OUTPUT_JSON = 'embeddings_ver2.json'
BATCH_SIZE = 64

# --- 2. ĐẶT SEED CHO ỔN ĐỊNH ---
random.seed(42)
np.random.seed(42)

# --- 3. TẢI MÔ HÌNH ---
print(f"Đang tải mô hình {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("✅ Tải mô hình thành công.")

# --- 4. ĐỌC FILE JSON ---
print(f"Đang đọc file {INPUT_JSON}...")
try:
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    all_texts = [article.get(CONTENT_KEY, "") for article in articles_data]
    print(f"Đã đọc xong {len(all_texts)} bài báo.")
except Exception as e:
    print(f"❌ Lỗi khi đọc file JSON: {e}")
    exit()

# --- 5. MÃ HÓA & LƯU ---
print(f"🔁 Bắt đầu mã hóa {len(all_texts)} văn bản...")
all_embeddings = []

with open(OUTPUT_TXT, 'w', encoding='utf-8') as f_out:
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Đang xử lý các lô"):
        batch_texts = all_texts[i: i + BATCH_SIZE]
        embeddings = model.encode(batch_texts, convert_to_numpy=True)
        for vec in embeddings:
            f_out.write(' '.join(map(str, vec)) + '\n')
            all_embeddings.append(vec.tolist())

# --- 6. LƯU FILE JSON ---
print(f"💾 Đang lưu dữ liệu ra {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as jf:
    json.dump(all_embeddings, jf, ensure_ascii=False, indent=2)

print(f"\n🎉 Hoàn tất mã hóa {len(all_texts)} văn bản.")
print(f"📄 File TXT: {OUTPUT_TXT}")
print(f"📘 File JSON: {OUTPUT_JSON}")
