import json
import time
import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import re

# ---------------------- CẤU HÌNH ----------------------
INPUT_JSON = 'guardian_articles_ver2.json'
CONTENT_KEY = 'content'

# Cấu hình MinHash
NUM_PERM = 128  # Số hàm băm, tương đương 128-bit

# --- FIX 3: Đặt lại ngưỡng Jaccard hợp lý ---
JACCARD_THRESHOLD = 0.6 #Ngưỡng Jaccard (0.2 là một điểm khởi đầu tốt)

# Cấu hình LSH (Banding)
# 128 hàm băm, chia làm 32 dải (bands), mỗi dải 4 hàng (rows)
# b * r = NUM_PERM (32 * 4 = 128)
BANDS = 32
ROWS = NUM_PERM // BANDS

# --- FIX 4: Chuyển sang Character Shingles (k=5) ---
# Dùng cụm 5 KÝ TỰ thay vì 2 TỪ. Cách này nhạy và chuẩn hơn.
K_SHINGLES = 5

# ---------------------- 1. ĐỌC DỮ LIỆU GỐC ----------------------
print(f"📂 Đang đọc file {INPUT_JSON}...")
try:
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    all_texts = [article.get(CONTENT_KEY, "") for article in articles_data]
    n_docs = len(all_texts)
    print(f"✅ Đã đọc xong {n_docs} bài báo.")
except Exception as e:
    print(f"❌ Lỗi khi đọc file JSON: {e}")
    exit()


# ---------------------- 2. PREPROCESS & TẠO SHINGLES ----------------------
# --- FIX 5: Sửa hàm shingle để dùng CỤM KÝ TỰ (Character k-grams) ---
def preprocess_and_shingle(text, k=5):  # k=5 là default
    """
    Chuẩn hóa text và tạo "shingles" (k-grams CỦA KÝ TỰ).
    Đây là cách làm chuẩn và nhạy hơn cho MinHash.
    """
    # Chỉ thay thế các khoảng trắng liền kề bằng 1 dấu cách
    text = re.sub(r'\s+', ' ', text.lower())
    # Tạo k-shingles (ví dụ: "this is" (k=3) -> ["thi", "his", "is ", "s i", " is"])
    return set([text[i:i + k] for i in range(len(text) - k + 1)])


print(f"⚙️  Đang tạo MinHash (perm={NUM_PERM}) cho {n_docs} văn bản...")
start = time.time()
minhashes = []

# --- SỬA LỖI TypeError: "can't multiply sequence by non-int" ---
# Chuyển params từ dict {'b':..., 'r':...} thành tuple (..., ...)
lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM, params=(BANDS, ROWS))

for i, text in tqdm(enumerate(all_texts), total=n_docs, desc="Tạo MinHash"):

    # --- SỬA LỖI NameError: 'shingles' is not defined ---
    # Phải gọi hàm shingle cho mỗi text BÊN TRONG vòng lặp
    shingles = preprocess_and_shingle(text, k=K_SHINGLES)

    m = MinHash(num_perm=NUM_PERM)

    # Xử lý trường hợp text rỗng (không có shingles)
    if not shingles:
        pass  # m sẽ là MinHash rỗng
    else:
        # Vòng lặp `for d in shingles:` bây giờ đã hợp lệ
        for d in shingles:
            m.update(d.encode('utf8'))

    minhashes.append(m)
    # Thêm vào LSH index
    lsh.insert(i, m)  # i chính là doc_id

print(f"✅ Hoàn tất MinHash trong {time.time() - start:.2f}s.")

# ---------------------- 3. TÌM KIẾM CẶP TƯƠNG TỰ (LSH) ----------------------
# LSH sẽ dùng JACCARD_THRESHOLD (đã set 0.2) để tìm ứng cử viên
print(f"🔍 Đang tìm các cặp 'ứng cử viên' (Jaccard ≥ {JACCARD_THRESHOLD})...")
start = time.time()
candidate_pairs = set()

# Dùng index.query() cho từng văn bản
for i in tqdm(range(n_docs), desc="Query LSH"):
    result = lsh.query(minhashes[i])
    # result chứa chính nó (i) và các văn bản khác (j)
    for j in result:
        if i < j:  # Chỉ lưu (i, j) chứ không lưu (j, i)
            candidate_pairs.add((i, j))

print(f"✅ Tìm thấy {len(candidate_pairs)} cặp ứng cử viên.")

# ---------------------- 4. KIỂM TRA LẠI (TÙY CHỌN, NHƯNG NÊN CÓ) ----------------------
# LSH có thể có sai sót nhỏ, ta kiểm tra lại Jaccard chính xác
print(f"🔍 Đang kiểm tra chi tiết {len(candidate_pairs)} cặp...")
final_pairs = []
for (i, j) in tqdm(candidate_pairs, desc="Kiểm tra chi tiết"):
    # datasketch ước tính jaccard rất nhanh
    jaccard = minhashes[i].jaccard(minhashes[j])

    # Lọc lại lần nữa với threshold (đề phòng LSH trả về kết quả < threshold)
    if jaccard >= JACCARD_THRESHOLD:
        final_pairs.append((i, j, jaccard))

print(f"⏱ Hoàn tất tìm kiếm MinHash trong {time.time() - start:.2f}s.")

# ---------------------- 5. LƯU KẾT QUẢ ----------------------
print(f"🎯 Tìm thấy {len(final_pairs)} cặp (Jaccard ≥ {JACCARD_THRESHOLD}).")
if final_pairs:
    final_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\n--- 10 cặp tương tự nhất (Jaccard) ---")
    for (i, j, sim) in final_pairs[:10]:
        print(f"Cặp ({i}, {j}) - Jaccard = {sim:.4f}")

# Chỉ lưu cột (i, j) để file visualize_clusters.py có thể đọc
# --- SỬA LỖI AttributeError: 'int3g' thành 'int32' ---
pairs_to_save = np.array([[p[0], p[1]] for p in final_pairs], dtype=np.int32)
np.save('minhash_similar_pairs.npy', pairs_to_save)
print(f"\n💾 Đã lưu {len(pairs_to_save)} cặp chỉ số vào 'minhash_similar_pairs.npy'")
print("🎉 Hoàn tất pipeline MinHash.")

# t la so 1
