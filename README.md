# 🧠 Demo: Text Deduplication Project (`demo1_deduplication_text`)

Dự án này minh họa quy trình **phát hiện và loại bỏ văn bản trùng lặp** (deduplication) trong một tập dữ liệu lớn các bài báo.  
Mục tiêu là **phát hiện các đoạn text giống hệt hoặc gần giống nhau**, sử dụng nhiều kỹ thuật hiện đại như:

- 🌸 **Bloom Filter** – phát hiện trùng lặp *y hệt* (exact duplicates)
- 🔍 **Sentence Embedding + FAISS** – tìm kiếm *văn bản tương tự* bằng vector cosine similarity
- ⚡ **MinHash + LSH** – phát hiện *trùng gần đúng* dựa trên Jaccard similarity
- 🧩 **SimHash + LSH Banding** – phát hiện trùng lặp nhanh trên *biểu diễn bit 128 chiều*

---

## 📁 Cấu trúc thư mục

\`\`\`
demo1_deduplication_text/
├── minHash.py               # 🔹 Pipeline phát hiện trùng lặp bằng MinHash + LSH
├── bloom_filter.py          # 🔹 Pipeline phát hiện trùng lặp bằng Bloom Filter
├── embedding_faiss.py       # 🔹 Mã hóa văn bản và tìm tương đồng bằng FAISS
├── simhash_lsh.py           # 🔹 Phát hiện trùng bằng SimHash + Locality-Sensitive Hashing
│
├── guardian_articles_ver2.json   # 📄 Dữ liệu gốc (các bài báo)
├── embeddings_ver2.txt           # 📄 Vector embedding dạng text
├── embeddings_ver2.json          # 📄 Vector embedding dạng JSON
│
├── simHash_ver3.npy              # 💾 File hash SimHash 128-bit đã được lưu
├── faiss_similar_pairs.npy       # 💾 Kết quả cặp tương tự theo FAISS
├── minhash_similar_pairs.npy     # 💾 Kết quả cặp tương tự theo MinHash
│
└── README.md                     # 📘 File hướng dẫn (bạn đang đọc nè ❤️)
\`\`\`

---

## ⚙️ 1. Chuẩn bị môi trường

### Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt

🧩 2. Các bước pipeline
(1) 🌸 Bloom Filter — phát hiện trùng y hệt

File: bloom_filter.py

✅ Mục tiêu: phát hiện những bài báo hoàn toàn giống nhau (exact duplicates)
📦 Cách làm:

Đọc file JSON gốc (guardian_articles_ver2.json)

Chuẩn hóa text (xóa dấu câu, khoảng trắng, viết thường)

Dùng Bloom Filter để phát hiện văn bản đã xuất hiện

📘 Chạy: 

python bloom_filter.py


📊 Kết quả:

Danh sách ID văn bản độc nhất và trùng lặp

(2) 🔍 Embedding + FAISS — phát hiện tương đồng theo vector

File: embedding_faiss.py

✅ Mục tiêu: phát hiện những bài báo có nội dung tương tự nhau về ngữ nghĩa
📦 Cách làm:

Mã hóa toàn bộ bài báo thành vector bằng all-MiniLM-L6-v2 (SentenceTransformer)

Dùng FAISS để tìm các cặp vector có cosine similarity ≥ 0.9

📘 Chạy:

python embedding_faiss.py


📊 Kết quả:

File embeddings_ver2.txt / embeddings_ver2.json: lưu vector hóa

File faiss_similar_pairs.npy: chứa các cặp văn bản tương tự nhau

(3) ⚡ MinHash + LSH — phát hiện trùng gần đúng (Jaccard)

File: minHash.py

✅ Mục tiêu: phát hiện các bài báo có nhiều phần trùng nhau (không cần giống 100%)
📦 Cách làm:

Tạo character shingles (k-grams ký tự)

Sinh MinHash signatures (128 hàm băm)

Dùng Locality Sensitive Hashing (LSH) để nhóm các bài tương tự

📘 Chạy:

python minHash.py


📊 Kết quả:

In ra các cặp (i, j) có Jaccard ≥ 0.6

Lưu ra file minhash_similar_pairs.npy

(4) 🧩 SimHash + LSH Banding — phát hiện nhanh trùng bit

File: simhash_lsh.py

✅ Mục tiêu: phát hiện trùng lặp nhanh dựa trên Hamming distance của hash
📦 Cách làm:

Sinh SimHash 128-bit từ vector embedding

Chia thành nhiều dải (bands) và nhóm theo hash band giống nhau

Kiểm tra chi tiết bằng Hamming distance ≤ threshold

📘 Chạy:

python simhash_lsh.py


📊 Kết quả:

File simHash_ver3.npy: lưu hash

Danh sách cặp trùng lặp theo Hamming distance

❤️ Ghi chú

Nếu bạn chạy trên Google Colab hoặc Linux, hãy chắc chắn rằng:

Đã cài faiss-cpu (hoặc faiss-gpu nếu có CUDA)

Thư mục chứa dữ liệu (guardian_articles_ver2.json) nằm đúng chỗ

File JSON phải có khóa "content" cho mỗi bài báo