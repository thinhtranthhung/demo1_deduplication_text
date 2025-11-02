import json

# Tải file Guardian gốc
with open('guardian_articles_ver2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Chỉ số cần xem
i, j = (712, 937)


print(f"==== Document {i} ====")
print(data[i]['content'])  # in 1000 ký tự đầu tiên cho dễ xem

print(f"\n==== Document {j} ====")
print(data[j]['content'])
