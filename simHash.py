import numpy as np
import time
from collections import defaultdict
import itertools


class SimHash:
    def __init__(self, dim, hash_bits=128, seed=42):
        if dim <= 0 or hash_bits <= 0:
            raise ValueError("Chiều vector và số bit băm phải >0.")
        if hash_bits != 128:
            raise ValueError("Code tối ưu cho 128-bit.")
        self.dim = dim
        self.hash_bits = hash_bits
        np.random.seed(seed)
        self.planes = np.random.randn(hash_bits, dim).astype(np.float32)
        print(f"Khởi tạo SimHash {hash_bits}-bit (seed={seed})")

    def hash(self, vectors: np.ndarray):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n, d = vectors.shape
        if d != self.dim:
            raise ValueError(f"Chiều vector ({d}) không khớp ({self.dim})")

        dots = np.dot(vectors, self.planes.T)
        bits = (dots > 0).astype(np.uint8)

        high = np.packbits(bits[:, :64], axis=1, bitorder='little').view(np.uint64).flatten()
        low = np.packbits(bits[:, 64:], axis=1, bitorder='little').view(np.uint64).flatten()
        return np.stack([high, low], axis=1)

    @staticmethod
    def hamming_distance(h1, h2):
        x_high = np.uint64(h1[0]) ^ np.uint64(h2[0])
        x_low = np.uint64(h1[1]) ^ np.uint64(h2[1])
        return bin(int(x_high)).count('1') + bin(int(x_low)).count('1')

# LSH HELPER
def get_band_hash(hash_high, hash_low, band_index, band_width):
    full_hash_int = (int(hash_high) << 64) | int(hash_low)
    shift_amount = band_index * band_width
    shifted_hash = full_hash_int >> shift_amount
    mask = (1 << band_width) - 1
    return shifted_hash & mask

BANDS = 8
BAND_WIDTH = 128 // BANDS
HAMMING_THRESHOLD = 15

# ĐỌC EMBEDDINGS
embeddings = np.loadtxt('embeddings.txt', dtype=np.float32)
n_docs, dim = embeddings.shape
print(f"Đọc {n_docs} vector, {dim} chiều.")

simhasher = SimHash(dim=dim, hash_bits=128, seed=42)
print("Đang tạo SimHash...")
start = time.time()
doc_hashes = simhasher.hash(embeddings)
print(f"Hoàn tất trong {time.time() - start:.2f}s")

# ==================== BUCKETING ====================
print(f"Băm {n_docs} hash vào {BANDS} dải...")
start = time.time()
hash_tables = [defaultdict(list) for _ in range(BANDS)]
for i in range(n_docs):
    hash_high, hash_low = doc_hashes[i]
    for j in range(BANDS):
        band_hash = get_band_hash(hash_high, hash_low, j, BAND_WIDTH)
        hash_tables[j][band_hash].append(i)
print(f"Hoàn tất băm trong {time.time() - start:.2f}s")

# THU THẬP CẶP ỨNG CỬ
print("Thu thập cặp ứng cử viên...")
start = time.time()
candidate_pairs = set()
for table in hash_tables:
    for bucket in table.values():
        if len(bucket) > 1:
            for pair in itertools.combinations(bucket, 2):
                candidate_pairs.add(tuple(sorted(pair)))
print(f"Tìm thấy {len(candidate_pairs)} cặp ứng cử viên trong {time.time() - start:.2f}s")

# ==================== KIỂM TRA CUỐI ====================
print(f"Kiểm tra {len(candidate_pairs)} cặp với ngưỡng {HAMMING_THRESHOLD}...")
start = time.time()
found_pairs = []
for i, j in candidate_pairs:
    dist = SimHash.hamming_distance(doc_hashes[i], doc_hashes[j])
    if dist <= HAMMING_THRESHOLD:
        found_pairs.append((i, j, dist))
end = time.time()
print(f"Tìm thấy {len(found_pairs)} cặp trùng lặp. {end - start:.2f}s")

# XUẤT FILE
found_pairs.sort(key=lambda x: x[2])
output_file = 'check_by_simHash.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Tổng số cặp Hamming ≤ {HAMMING_THRESHOLD}: {len(found_pairs)}\n\n")
    for idx, (i, j, d) in enumerate(found_pairs, 1):
        f.write(f"{idx}. Cặp ({i}, {j}) - Hamming = {d}\n")

print(f"Đã ghi {len(found_pairs)} cặp vào '{output_file}'")
