import numpy as np
import faiss
import json
import time

# ---------------------- CẤU HÌNH ----------------------
EMBEDDING_FILE = 'embeddings_ver2.txt'  # hoặc 'embeddings_ver2.json'
TOP_K = 5  # Số lượng kết quả tương tự cần lấy
SIMILARITY_THRESHOLD = 0.9  # Ngưỡng cosine similarity để coi là giống
USE_JSON = False  # True nếu muốn đọc từ JSON thay vì TXT
# Ngưỡng để chuyển sang dùng ANN. Dưới mức này, brute-force (Flat) nhanh hơn.
ANN_THRESHOLD = 2000

# ---------------------- ĐỌC DỮ LIỆU ----------------------
print("📂 Đang đọc vector embeddings ...")

try:
    if USE_JSON:
        with open(EMBEDDING_FILE, 'r', encoding='utf-8') as f:
            embeddings = np.array(json.load(f), dtype=np.float32)
    else:
        embeddings = np.loadtxt(EMBEDDING_FILE, dtype=np.float32)
except Exception as e:
    print(f"❌ Lỗi khi đọc file embeddings: {e}")
    exit()

if embeddings.ndim != 2:
    print("❌ File phải là ma trận 2 chiều (mỗi dòng 1 vector).")
    exit()

n_docs, dim = embeddings.shape
print(f"✅ Đọc thành công {n_docs} vector, mỗi vector {dim} chiều.\n")

# ---------------------- CHUẨN HÓA VECTOR ----------------------
print("⚙️  Chuẩn hóa vector về độ dài 1 (L2 normalization)...")
faiss.normalize_L2(embeddings)

# ---------------------- XÂY DỰNG INDEX ----------------------
# *** CẬP NHẬT LOGIC: Tự động chọn Index ***

if n_docs < ANN_THRESHOLD:
    # Nếu dữ liệu quá nhỏ, Brute-force (Flat) nhanh hơn và chính xác 100%
    print(f"🚀 Dữ liệu nhỏ (< {ANN_THRESHOLD}). Sử dụng IndexFlatIP (Brute-force).")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"✅ Index đã thêm {index.ntotal} vector.\n")
else:
    # Nếu dữ liệu đủ lớn, dùng ANN (IVFFlat) để tối ưu
    # Heuristic: Dùng nlist = 100 cho < 1M vector, hoặc ~ 4*sqrt(N)
    nlist = 100
    if n_docs < 100000:
        # Đảm bảo nlist * 39 < n_docs
        nlist = max(32, min(int(n_docs / 100), int(np.sqrt(n_docs))))
    else:
        nlist = max(100, int(np.sqrt(n_docs)))

    print(f"🚀 Dữ liệu lớn (≥ {ANN_THRESHOLD}). Sử dụng IndexIVFFlat (ANN) (nlist={nlist})...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"🏋️  Đang 'train' index trên {n_docs} vector...")
    start_time = time.time()
    index.train(embeddings)
    print(f"✅ Train hoàn tất sau {time.time() - start_time:.2f}s.")

    print(f"➕ Đang thêm {n_docs} vector vào index...")
    start_time = time.time()
    index.add(embeddings)
    print(f"✅ Thêm hoàn tất sau {time.time() - start_time:.2f}s. (ntotal={index.ntotal})\n")

    # Đặt nprobe cho IndexIVF
    index.nprobe = min(20, nlist)
    print(f"   (Đặt nprobe = {index.nprobe})")

# ---------------------- TÌM KIẾM ----------------------
print(f"🔍 Đang tìm kiếm (Top {TOP_K})...")

start_time = time.time()
distances, indices = index.search(embeddings, TOP_K)
end_time = time.time()
print(f"⏱ Hoàn tất sau {end_time - start_time:.2f}s.\n")

# ---------------------- LỌC KẾT QUẢ ----------------------
print(f"📊 Đang lọc kết quả với ngưỡng Similarity ≥ {SIMILARITY_THRESHOLD}...")
similar_pairs_set = set()  # Dùng set để lọc
similar_pairs_list = []  # Dùng list để lưu kết quả cuối

for i in range(n_docs):
    for rank in range(1, TOP_K):  # Bỏ chính nó (rank=0)
        j = indices[i][rank]
        if j == -1:  # Không tìm thấy
            continue

        sim = distances[i][rank]

        # Sắp xếp (i, j) để (5, 10) và (10, 5) là như nhau
        pair = tuple(sorted((i, j)))

        # *** SỬA LỖI: similar_pairs_T -> similar_pairs_set ***
        if sim >= SIMILARITY_THRESHOLD and pair not in similar_pairs_set:
            similar_pairs_set.add(pair)
            similar_pairs_list.append((pair[0], pair[1], sim))

# ---------------------- HIỂN THỊ KẾT QUẢ ----------------------
print(f"🎯 Tìm thấy {len(similar_pairs_list)} cặp duy nhất có độ tương đồng ≥ {SIMILARITY_THRESHOLD}")
if similar_pairs_list:
    # Sắp xếp theo độ tương đồng giảm dần
    similar_pairs_list.sort(key=lambda x: x[2], reverse=True)

    print("\n--- 10 cặp tương tự nhất ---")
    for (i, j, sim) in similar_pairs_list[:10]:
        print(f"Cặp ({i}, {j}) - Similarity = {sim:.4f}")

# ---------------------- (TÙY CHỌN) LƯU RA FILE ----------------------
pairs_to_save = np.array([[p[0], p[1]] for p in similar_pairs_list], dtype=np.int32)
np.save('faiss_similar_pairs.npy', pairs_to_save)
print(f"\n💾 Đã lưu {len(pairs_to_save)} cặp chỉ số vào 'faiss_similar_pairs.npy'")
print("🎉 Hoàn tất quá trình tìm kiếm bằng FAISS.")

