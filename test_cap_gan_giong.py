import json
with open('guardian_articles.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

i, j = (712, 937)

print(f"==== Document {i} ====")
print(data[i]['content'])
print(f"\n==== Document {j} ====")
print(data[j]['content'])
