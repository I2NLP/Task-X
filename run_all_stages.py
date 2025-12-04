import sys
import runpy
import time
import os
import json
import shutil
from datetime import timedelta

def log_time(task_name, start_time):
    elapsed = time.time() - start_time
    print(f"⏱️  [Timing] {task_name} 耗时: {str(timedelta(seconds=int(elapsed)))}", flush=True)

def warm_up():
    print("\n🔥 [Wrapper] 开始环境预热与依赖库加载...", flush=True)
    total_start = time.time()
    
    # 根据需要预加载的库
    libs = ["torch", "transformers", "datasets", "sklearn", "accelerate", "numpy"]
    
    for lib in libs:
        lib_start = time.time()
        print(f"   ... 正在加载 {lib}", end=" ", flush=True)
        try:
            __import__(lib)
            print(f"✅ (Done)", flush=True)
        except ImportError:
            print(f"❌ (Failed)", flush=True)
        log_time(f"加载 {lib}", lib_start)
        
    log_time("库加载总阶段", total_start)

def run_script(script_name):
    print(f"\n=======================================================")
    print(f"🚀 [Wrapper] 开始运行子脚本: {script_name}")
    print(f"=======================================================", flush=True)
    start_time = time.time()
    
    try:
        # 使用 runpy 在当前进程运行，共享已加载的库
        runpy.run_path(script_name, run_name="__main__")
        print(f"✅ [Wrapper] {script_name} 执行成功。", flush=True)
    except Exception as e:
        print(f"❌ [Wrapper] {script_name} 执行出错！", flush=True)
        raise e
    finally:
        log_time(f"运行 {script_name}", start_time)

def safe_rename(src, dst):
    """将源文件重命名/移动到目标路径"""
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst) # 确保没有旧文件干扰
        shutil.move(src, dst)
        print(f"💾 [Wrapper] 结果已重命名: {src} -> {dst}", flush=True)
    else:
        print(f"⚠️ [Wrapper] 警告: 未找到 {src}，跳过重命名。", flush=True)

def create_backup(src, backup_name):
    """显式创建一个备份副本"""
    if os.path.exists(src):
        shutil.copy(src, backup_name)
        print(f"🛡️  [Backup] 已创建额外备份: {backup_name}", flush=True)

def merge_results():
    print("\n🔗 [Wrapper] 开始合并 Binary 和 Span 的结果...", flush=True)
    start_time = time.time()
    
    binary_file = "submission_binary.jsonl"
    span_file = "submission_span.jsonl"
    final_file = "submission.jsonl"

    if not os.path.exists(binary_file) or not os.path.exists(span_file):
        print("❌ [Wrapper] 错误：缺少中间结果文件，无法合并！请检查之前的推理步骤是否成功。")
        return

    # 1. 读取二分类结果
    binary_data_map = {}
    with open(binary_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            binary_data_map[item['_id']] = item.get('conspiracy', 0) # 默认为 0 防止 null

    # 2. 读取 Span 结果并合并
    merged_count = 0
    final_data = []
    with open(span_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            doc_id = item['_id']
            
            # 将 Binary 的结果注入到 Span 的记录中
            if doc_id in binary_data_map:
                item['conspiracy'] = binary_data_map[doc_id]
            else:
                print(f"⚠️ ID {doc_id} 在 Binary 结果中未找到", flush=True)
            
            final_data.append(item)
            merged_count += 1

    # 3. 写入最终文件
    with open(final_file, 'w') as f:
        for item in final_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"✅ [Wrapper] 合并完成！生成文件: {final_file} (包含 {merged_count} 条数据)", flush=True)
    log_time("合并阶段", start_time)

if __name__ == "__main__":
    overall_start = time.time()
    
    # 1. 预热 (只痛一次)
    warm_up()
    
    # 2. 训练 (Binary)
    run_script("train_binary.py")
    
    # 3. 推理 (Binary)
    run_script("infer_binary.py")
    
    # --- 核心修改：先备份，再改名 ---
    if os.path.exists("submission.jsonl"):
        # 1. 创建永久备份
        create_backup("submission.jsonl", "backup_binary_result.jsonl") 
        # 2. 改名为流程需要的名字
        safe_rename("submission.jsonl", "submission_binary.jsonl")
    
    # 4. 训练 (Span)
    run_script("train_one_span.py")
    
    # 5. 推理 (Span)
    run_script("infer_one_span.py")
    
    # 同样给 Span 任务也做个备份
    if os.path.exists("submission.jsonl"):
        create_backup("submission.jsonl", "backup_span_result.jsonl")
        safe_rename("submission.jsonl", "submission_span.jsonl")
    
    # 6. 合并结果
    merge_results()
    
    log_time("整个任务流程", overall_start)