import json
import time
from pybloom_live import BloomFilter
from tqdm import tqdm
import re

# ---------------------- CẤU HÌNH ----------------------
INPUT_JSON = 'guardian_articles_ver2.json'
CONTENT_KEY = 'content'

# Cấu hình Bloom Filter
# Ước tính số lượng văn bản (ví dụ: 198)
ESTIMATED_ITEMS = 2000
# Tỷ lệ dương tính giả (0.1% = 0.001)
# Tỷ lệ càng nhỏ, bộ lọc càng tốn bộ nhớ. 0.001 là khá an toàn.
FALSE_POSITIVE_RATE = 0.001

# ---------------------- 1. ĐỌC DỮ LIỆU GỐC ----------------------
print(f"📂 Đang đọc file {INPUT_JSON}...")
try:
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    # Lấy cả ID (index) và text
    all_docs = [(i, article.get(CONTENT_KEY, "")) for i, article in enumerate(articles_data)]
    n_docs = len(all_docs)
    print(f"✅ Đã đọc xong {n_docs} bài báo.")
except Exception as e:
    print(f"❌ Lỗi khi đọc file JSON: {e}")
    exit()


# ---------------------- 2. CHUẨN HÓA VĂN BẢN ----------------------
def normalize_text(text):
    """
    Chuẩn hóa văn bản để kiểm tra trùng lặp Y HỆT.
    Xóa khoảng trắng, viết thường, xóa dấu câu.
    "Hello World!" và "hello world" sẽ được coi là Y HỆT.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())  # Xóa dấu câu, viết thường
    text = re.sub(r'\s+', '', text)  # Xóa tất cả khoảng trắng
    return text


# ---------------------- 3. KHỞI TẠO BLOOM FILTER ----------------------
print(f"🚀 Khởi tạo Bloom Filter cho ~{ESTIMATED_ITEMS} văn bản (lỗi {FALSE_POSITIVE_RATE * 100}%)...")
# capacity = số item dự kiến, error_rate = tỷ lệ lỗi mong muốn
bf = BloomFilter(capacity=ESTIMATED_ITEMS, error_rate=FALSE_POSITIVE_RATE)
print(f"✅ Khởi tạo thành công. Kích thước bộ lọc: {bf.num_bits // 8 / 1024:.2f} KB")

# ---------------------- 4. DUYỆT VÀ LỌC TRÙNG LẶP ----------------------
print("🔍 Bắt đầu lọc văn bản trùng lặp y hệt...")
start = time.time()

unique_doc_ids = []  # Danh sách ID các văn bản độc nhất (lần đầu thấy)
duplicate_doc_ids = []  # Danh sách ID các văn bản bị coi là trùng

for doc_id, text in tqdm(all_docs, desc="Đang duyệt văn bản"):
    # Chuẩn hóa văn bản trước khi kiểm tra
    normalized_text = normalize_text(text)

    # Nếu văn bản rỗng, bỏ qua
    if not normalized_text:
        continue

    # [Hỏi Bloom Filter] "Mày thấy văn bản này bao giờ chưa?"
    # Dùng .add() - nó sẽ trả về True nếu "có thể" đã tồn tại (dương tính)
    # và trả về False nếu "chắc chắn" chưa tồn tại (âm tính)
    if bf.add(normalized_text):
        # True -> Đã thấy rồi (hoặc dương tính giả)
        # Ta coi đây là văn bản trùng lặp
        duplicate_doc_ids.append(doc_id)
    else:
        # False -> Chắc chắn 100% chưa thấy
        # Đây là văn bản độc nhất. Ta thêm ID vào danh sách.
        unique_doc_ids.append(doc_id)

end = time.time()
print(f"⏱ Hoàn tất lọc trong {end - start:.4f}s.")

# ---------------------- 5. HIỂN THỊ KẾT QUẢ ----------------------
print("\n--- KẾT QUẢ BLOOM FILTER ---")
print(f"Tổng số văn bản đã xử lý: {len(unique_doc_ids) + len(duplicate_doc_ids)}")
print(f"  ➡️ Số văn bản ĐỘC NHẤT: {len(unique_doc_ids)}")
print(f"  ➡️ Số văn bản TRÙNG LẶP Y HỆT: {len(duplicate_doc_ids)}")

if duplicate_doc_ids:
    print("\n--- Một số văn bản bị lọc (ID) ---")
    print(duplicate_doc_ids[:10])

print("\n🎉 Hoàn tất pipeline Bloom Filter.")

