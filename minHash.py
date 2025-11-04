import json
import time
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import re

# CẤU HÌNH
INPUT_JSON = 'guardian_articles.json'
CONTENT_KEY = 'content'

NUM_PERM = 128
JACCARD_THRESHOLD = 0.5
BANDS = 32
BAND_WIDTH = NUM_PERM // BANDS
K_SHINGLES = 5

# ĐỌC DỮ LIỆU
print(f"Đang đọc file {INPUT_JSON}...")
try:
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    all_texts = [article.get(CONTENT_KEY, "") for article in articles_data]
    n_docs = len(all_texts)
    print(f"Đã đọc xong {n_docs} bài báo.")
except Exception as e:
    print(f"Lỗi khi đọc file JSON: {e}")
    exit()

# HÀM XỬ LÝ SHINGLES
def preprocess_and_shingle(text, k=5):
    text = re.sub(r'\s+', ' ', text.lower())
    return set([text[i:i + k] for i in range(len(text) - k + 1)])

# TẠO MINHASH + LSH
print(f"Đang tạo MinHash (perm={NUM_PERM}) cho {n_docs} văn bản...")
start = time.time()
minhashes = []
lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM, params=(BANDS, BAND_WIDTH))

for i, text in tqdm(enumerate(all_texts), total=n_docs, desc="Tạo MinHash"):
    shingles = preprocess_and_shingle(text, k=K_SHINGLES)
    m = MinHash(num_perm=NUM_PERM)
    for d in shingles:
        m.update(d.encode('utf8'))
    minhashes.append(m)
    lsh.insert(i, m)

print(f"Hoàn tất MinHash trong {time.time() - start:.2f}s.")

# TÌM CẶP TƯƠNG TỰ
print(f"Đang tìm các cặp 'ứng cử viên' (Jaccard ≥ {JACCARD_THRESHOLD})...")
start = time.time()
candidate_pairs = set()

for i in tqdm(range(n_docs), desc="Query LSH"):
    result = lsh.query(minhashes[i])
    for j in result:
        if i < j:
            candidate_pairs.add((i, j))

print(f"Tìm thấy {len(candidate_pairs)} cặp ứng cử viên.")

# KIỂM TRA CHI TIẾT
print(f"Đang kiểm tra chi tiết {len(candidate_pairs)} cặp...")
final_pairs = []
for (i, j) in tqdm(candidate_pairs, desc="Kiểm tra chi tiết"):
    jaccard = minhashes[i].jaccard(minhashes[j])
    if jaccard >= JACCARD_THRESHOLD:
        final_pairs.append((i, j, jaccard))

print(f"Hoàn tất tìm kiếm MinHash trong {time.time() - start:.2f}s.")
print(f"Tìm thấy {len(final_pairs)} cặp (Jaccard ≥ {JACCARD_THRESHOLD}).")

# LƯU KẾT QUẢ
output_file = 'check_by_minHash.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Tổng số cặp tương tự ≥ {JACCARD_THRESHOLD}: {len(final_pairs)}\n\n")
    final_pairs.sort(key=lambda x: x[2], reverse=True)
    for idx, (i, j, sim) in enumerate(final_pairs, 1):
        f.write(f"{idx}. Cặp ({i}, {j}) - Jaccard = {sim:.4f}\n")

print(f"\nĐã lưu kết quả vào '{output_file}'")
