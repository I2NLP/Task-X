import json

print("正在合并 Binary 和 Span 的结果...")

# 1. 读取分类结果 (Binary)
binary_results = {}
with open('submission_binary.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        # 把 ID 和 预测标签 存起来
        binary_results[data['_id']] = data['conspiracy']

# 2. 读取提取结果 (Span) 并填入分类结果
final_data = []
with open('submission_span.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        doc_id = data['_id']
        
        # 核心修复：把 Binary 的答案填进去
        if doc_id in binary_results:
            data['conspiracy'] = binary_results[doc_id]
        else:
            print(f"警告: ID {doc_id} 在 Binary 结果中找不到！")
            
        final_data.append(data)

# 3. 保存最终文件
with open('submission_final.jsonl', 'w') as f:
    for item in final_data:
        f.write(json.dumps(item) + '\n')

print(f"✅ 合并完成！共处理 {len(final_data)} 条数据。")
print("结果已保存为 submission_final.jsonl")