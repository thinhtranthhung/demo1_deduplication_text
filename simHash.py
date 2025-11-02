import numpy as np
import time
from collections import defaultdict
import itertools


# ==================== CLASS SIMHASH (TỪ FILE CỦA BẠN) ====================
# Giữ nguyên class SimHash của bạn để *tạo* hash
# (Mình chỉ sửa lại hàm hash để trả về 2x uint64 cho dễ xử lý)

class SimHash:
    """
    Bản cải tiến của SimHash:
    - 128-bit hash cho độ chính xác cao hơn.
    - Vector hóa tính toán nhanh hơn.
    - Kết quả ổn định nhờ seed cố định.
    """

    def __init__(self, dim, hash_bits=128, seed=42):
        if dim <= 0 or hash_bits <= 0:
            raise ValueError("Số chiều và số bit băm phải là số dương.")
        if hash_bits != 128:
            raise ValueError("Code này được tối ưu cho 128 bits (2 x 64bit).")
        self.dim = dim
        self.hash_bits = hash_bits
        np.random.seed(seed)
        self.planes = np.random.randn(hash_bits, dim).astype(np.float32)
        print(f"✅ Khởi tạo SimHash {hash_bits}-bit (seed={seed}).")

    def hash(self, vectors: np.ndarray):
        """
        Trả về mảng hash [n_samples, 2] (dạng high, low uint64)
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n, d = vectors.shape
        if d != self.dim:
            raise ValueError(f"Chiều vector ({d}) không khớp ({self.dim})")

        # Chuẩn hóa vector (có thể bỏ nếu embedding đã chuẩn hóa)
        # norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

        dots = np.dot(vectors, self.planes.T)  # [n_samples, hash_bits]
        bits = (dots > 0).astype(np.uint8)

        # Chuyển mảng bit thành uint128 (chia làm 2 uint64)
        # Dùng 'little' để packbits từ trái sang phải
        high = np.packbits(bits[:, :64], axis=1, bitorder='little').view(np.uint64).flatten()
        low = np.packbits(bits[:, 64:], axis=1, bitorder='little').view(np.uint64).flatten()

        return np.stack([high, low], axis=1)  # shape [n, 2]

    @staticmethod
    def hamming_distance(h1, h2):
        h1 = np.array(h1)
        h2 = np.array(h2)

        if h1.shape == (2,):  # 128-bit
            x_high = np.uint64(h1[0]) ^ np.uint64(h2[0])
            x_low = np.uint64(h1[1]) ^ np.uint64(h2[1])
            # Dùng .bit_count() của Python 3.10+ (nhanh hơn) nếu có thể
            # Hoặc dùng bin()
            return bin(int(x_high)).count('1') + bin(int(x_low)).count('1')
        else:  # 64-bit
            x = np.uint64(h1) ^ np.uint64(h2)
            return bin(int(x)).count('1')


# ==================== LSH BANDING HELPER ====================

def get_band_hash(hash_high, hash_low, band_index, band_width):
    """
    Trích xuất giá trị của 1 dải (band) từ 128-bit hash.
    Ví dụ: 128 bit, 8 dải -> band_width = 16.
    """
    # Chuyển thành 1 số nguyên 128-bit của Python
    full_hash_int = (int(hash_high) << 64) | int(hash_low)

    # Dịch bit sang phải để đưa dải cần lấy về đầu
    shift_amount = band_index * band_width
    shifted_hash = full_hash_int >> shift_amount

    # Tạo mặt nạ (mask) để lấy đúng band_width bits
    # (1 << band_width) - 1 tạo ra một số có band_width bit 1
    # ví dụ band_width=16 -> 0b1111111111111111
    mask = (1 << band_width) - 1

    return shifted_hash & mask


# ==================== CHƯƠNG TRÌNH CHÍNH (TỐI ƯU) ====================

# --- 1. CẤU HÌNH LSH ---
BANDS = 8  # Số lượng dải (b). 8, 16 là các giá trị phổ biến
BAND_WIDTH = 128 // BANDS  # 128 / 8 = 16 bits
HAMMING_THRESHOLD = 25  # Ngưỡng cuối cùng để kiểm tra

# --- 2. TẢI/TẠO HASH ---
HASH_FILE = 'simHash_ver3.npy'

try:
    print(f"🔹 Đang tải file '{HASH_FILE}'...")
    doc_hashes = np.load(HASH_FILE)
    print(f"✅ Tải thành công {len(doc_hashes)} mã SimHash.")
    if doc_hashes.shape[1] != 2:
        print(f"❌ File hash phải có 2 cột (high, low uint64).")
        exit()
except FileNotFoundError:
    print(f"⚠️ Không tìm thấy file '{HASH_FILE}'. Đang tạo lại...")

    print("📂 Đang đọc file 'embeddings_ver2.txt' ...")
    embeddings = np.loadtxt('embeddings_ver2.txt', dtype=np.float32)
    n_docs, dim = embeddings.shape
    print(f"✅ Đọc thành công {n_docs} vector, {dim} chiều.\n")

    simhasher = SimHash(dim=dim, hash_bits=128, seed=42)
    print("⚙️  Đang tạo SimHash cho toàn bộ văn bản...")
    start = time.time()
    doc_hashes = simhasher.hash(embeddings)
    print(f"✅ Hoàn tất tạo hash trong {time.time() - start:.2f}s.\n")
    np.save(HASH_FILE, doc_hashes)
    print(f"💾 Đã lưu file hash vào '{HASH_FILE}'.\n")

n_docs = len(doc_hashes)

# --- 3. BĂM VÀO XÔ (BUCKETING) ---
print(f"📊 Đang chia {n_docs} hash vào {BANDS} dải (mỗi dải {BAND_WIDTH} bit)...")
start = time.time()
# hash_tables là 1 list, mỗi phần tử là 1 dict (bảng băm)
hash_tables = [defaultdict(list) for _ in range(BANDS)]

for i in range(n_docs):
    hash_high, hash_low = doc_hashes[i]
    for j in range(BANDS):
        band_hash = get_band_hash(hash_high, hash_low, j, BAND_WIDTH)
        hash_tables[j][band_hash].append(i)  # Thêm doc_id (i) vào xô

print(f"✅ Hoàn tất băm vào xô trong {time.time() - start:.2f}s.")

# --- 4. THU THẬP "ỨNG CỬ VIÊN" ---
print("🤝 Đang thu thập các cặp 'ứng cử viên' (chung xô)...")
start = time.time()
candidate_pairs = set()
for table in hash_tables:
    for bucket in table.values():
        if len(bucket) > 1:
            # Nếu 1 xô có [10, 25, 99]
            # itertools.combinations(bucket, 2) sẽ tạo ra:
            # (10, 25), (10, 99), (25, 99)
            for pair in itertools.combinations(bucket, 2):
                candidate_pairs.add(tuple(sorted(pair)))

print(f"✅ Tìm thấy {len(candidate_pairs)} cặp ứng cử viên.")
print(f"⏱ Thời gian: {time.time() - start:.2f}s.")

# --- 5. KIỂM TRA LẦN CUỐI ---
print(f"🔍 Đang kiểm tra chi tiết {len(candidate_pairs)} cặp (Ngưỡng = {HAMMING_THRESHOLD})...")
start = time.time()
found_pairs = []
for (i, j) in candidate_pairs:
    dist = SimHash.hamming_distance(doc_hashes[i], doc_hashes[j])
    if dist <= HAMMING_THRESHOLD:
        found_pairs.append((i, j, dist))

end = time.time()
print(f"\n🎉 HOÀN TẤT (LSH Tối ưu) 🎉")
print(f"⏱ Thời gian kiểm tra cuối: {end - start:.2f}s.")
print(f"🎯 Tìm thấy {len(found_pairs)} cặp trùng lặp.")

if found_pairs:
    # Sắp xếp theo distance nhỏ nhất
    found_pairs.sort(key=lambda x: x[2])
    print("\n--- Một số cặp gần giống nhất ---")
    for (i, j, d) in found_pairs[:10]:
        print(f"Cặp ({i}, {j}) - Hamming = {d}")
else:
    print("❌ Không tìm thấy cặp nào. Hãy thử giảm số dải (BANDS) hoặc tăng THRESHOLD.")

