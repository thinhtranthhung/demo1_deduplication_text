import json
import time
from pybloom_live import BloomFilter
from tqdm import tqdm
import re

INPUT_JSON = 'guardian_articles.json'
CONTENT_KEY = 'content'
ESTIMATED_ITEMS = 2000
FALSE_POSITIVE_RATE = 0.001

# ĐỌC DỮ LIỆU GỐC
try:
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    # Lấy cả ID (index) và text
    all_docs = [(i, article.get(CONTENT_KEY, "")) for i, article in enumerate(articles_data)]
    n_docs = len(all_docs)
except Exception as e:
    print(f" Lỗi khi đọc file JSON: {e}")
    exit()


# CHUẨN HÓA VĂN BẢN
def normalize_text(text):

    text = re.sub(r'[^\w\s]', '', text.lower())  # Xóa dấu câu, viết thường
    text = re.sub(r'\s+', '', text)  # Xóa tất cả khoảng trắng
    return text


#  KHỞI TẠO BLOOM FILTER
print(f"Khởi tạo Bloom Filter cho ~{ESTIMATED_ITEMS} văn bản (lỗi {FALSE_POSITIVE_RATE * 100}%)...")

bf = BloomFilter(capacity=ESTIMATED_ITEMS, error_rate=FALSE_POSITIVE_RATE)
print(f"Khởi tạo thành công. Kích thước bộ lọc: {bf.num_bits // 8 / 1024:.2f} KB")

#  DUYỆT VÀ LỌC TRÙNG LẶP
print("Bắt đầu lọc văn bản trùng lặp y hệt...")
start = time.time()

unique_doc_ids = []
duplicate_doc_ids = []

for doc_id, text in tqdm(all_docs, desc="Đang duyệt văn bản"):
    # Chuẩn hóa văn bản trước khi kiểm tra
    normalized_text = normalize_text(text)
    if not normalized_text:
        continue
    if bf.add(normalized_text):
        duplicate_doc_ids.append(doc_id)
    else:
        unique_doc_ids.append(doc_id)

end = time.time()
print(f"Hoàn tất lọc trong {end - start:.4f}s.")

# HIỂN THỊ KẾT QUẢ
print("KẾT QUẢ BLOOM FILTER")
print(f"Tổng số văn bản đã xử lý: {len(unique_doc_ids) + len(duplicate_doc_ids)}")
print(f"Số văn bản ĐỘC NHẤT: {len(unique_doc_ids)}")
print(f"Số văn bản TRÙNG LẶP Y HỆT: {len(duplicate_doc_ids)}")

if duplicate_doc_ids:
    print("Một số văn bản bị lọc (ID) ")
    print(duplicate_doc_ids[:10])

print("\nHoàn tất Bloom Filter.")

