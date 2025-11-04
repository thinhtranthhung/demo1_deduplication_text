import numpy as np
import faiss
import json
import time

EMBEDDING_FILE = 'embeddings.json'
TOP_K = 5
SIMILARITY_THRESHOLD = 0.9
ANN_THRESHOLD = 2000  # Ngưỡng để chuyển từ brute-force sang ANN

# ĐỌC DỮ LIỆU
print("Đang đọc vector embeddings từ JSON...")

try:
    with open(EMBEDDING_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        embeddings = np.array(data, dtype=np.float32)
except Exception as e:
    print(f"Lỗi khi đọc file JSON: {e}")
    exit()

if embeddings.ndim != 2:
    print("File phải chứa ma trận 2 chiều.")
    exit()

n_docs, dim = embeddings.shape
print(f"Đọc thành công {n_docs} vector, mỗi vector {dim} chiều.\n")

# CHUẨN HÓA
print("Chuẩn hóa về vector đơn vị")
faiss.normalize_L2(embeddings)

# XÂY DỰNG INDEX
if n_docs < ANN_THRESHOLD:
    print(f"Dữ liệu nhỏ (< {ANN_THRESHOLD}). Sử dụng Brute-force (IndexFlatIP).")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index đã thêm {index.ntotal} vector.\n")
else:
    nlist = max(32, min(int(n_docs / 100), int(np.sqrt(n_docs))))
    print(f"Dữ liệu lớn (≥ {ANN_THRESHOLD}). Sử dụng ANN (IndexIVFFlat, nlist={nlist})...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"Đang train index trên {n_docs} vector")
    start_time = time.time()
    index.train(embeddings)
    print(f"Train xong sau {time.time() - start_time:.2f}s.")

    print(f"Đang thêm {n_docs} vector vào index")
    start_time = time.time()
    index.add(embeddings)
    print(f"Thêm xong sau {time.time() - start_time:.2f}s. (ntotal={index.ntotal})\n")

    index.nprobe = min(20, nlist)
    print(f"(Đặt nprobe = {index.nprobe})")

# TÌM KIẾM
print(f"Đang tìm kiếm Top-{TOP_K} cặp tương tự...")

start_time = time.time()
distances, indices = index.search(embeddings, TOP_K)
end_time = time.time()
print(f"Hoàn tất sau {end_time - start_time:.2f}s.\n")

# LỌC CẶP GIỐNG
print(f"Lọc kết quả với similarity ≥ {SIMILARITY_THRESHOLD}...")

similar_pairs = set()
results = []

for i in range(n_docs):
    for rank in range(1, TOP_K):
        j = indices[i][rank]
        if j == -1 or i == j:
            continue

        sim = distances[i][rank]
        pair = tuple(sorted((i, j)))

        if sim >= SIMILARITY_THRESHOLD and pair not in similar_pairs:
            similar_pairs.add(pair)
            results.append((pair[0], pair[1], sim))


# GHI FILE
output_file = "check_by_faiss.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Tổng số cặp tương tự ≥ {SIMILARITY_THRESHOLD}: {len(results)}\n\n")

    results.sort(key=lambda x: x[2], reverse=True)
    for i, (a, b, sim) in enumerate(results, 1):
        f.write(f"{i}. Cặp ({a}, {b}) - Similarity = {sim:.4f}\n")

print(f"Đã ghi kết quả ra file: {output_file}")
