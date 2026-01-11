#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from openai import OpenAI
import time
import concurrent.futures
import random

# ================================
# 全局参数
# ================================
BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"
#MODEL = "Qwen/Qwen1.5-4B-Chat"
MODEL = "/home/joeyxzy/models/Qwen1.5-4B-Chat"

# 准备一组提示词，长度不一，模拟真实场景
PROMPTS = [
    "你好，请用一句话介绍自己。",
    "请背诵一首李白的《静夜思》。",
    "写一个 Python 的 Hello World 程序。",
    "将以下句子翻译成英文：今天天气真好，我想出去散步。",
    "解释一下什么是量子纠缠，用小学生能听懂的话。",
    "1+1等于几？",
    "给我讲一个冷笑话。",
    "列出太阳系的八大行星。",
]

def send_request(idx, prompt):
    """发送单个请求的函数"""
    print(f"🚀 [Req {idx}] 发送请求: {prompt[:10]}...")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    start_time = time.time()
    try:
        # 为了增加 Batching 的概率，我们让输出稍微长一点
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,  # 限制输出长度，让大家跑得差不多快
            temperature=0.7
        )
        content = resp.choices[0].message.content
        duration = time.time() - start_time
        print(f"✅ [Req {idx}] 完成 ({duration:.2f}s): {content[:10].replace('\n', ' ')}...")
        return idx, True
    except Exception as e:
        print(f"❌ [Req {idx}] 失败: {e}")
        return idx, False

def test_concurrent_batching():
    print("====================================")
    print(f"🚀 开始并发测试 (总请求数: {len(PROMPTS)})")
    print("====================================")

    # 使用线程池并发发送
    # max_workers=10 意味着同时有 10 个线程在发 HTTP 请求
    # 这远超 vLLM 的处理速度，会迫使请求在 vLLM 队列中堆积，从而触发 Batching
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, prompt in enumerate(PROMPTS):
            # 稍微加一点点随机延迟，模拟真实到达（可选，设为0就是瞬时爆发）
            # time.sleep(random.uniform(0, 0.05)) 
            futures.append(executor.submit(send_request, i, prompt))
        
        # 等待所有请求完成
        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    test_concurrent_batching()