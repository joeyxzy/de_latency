#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from openai import OpenAI
import requests
import json
import time

# ================================
# 全局参数
# ================================
BASE_URL = "http://localhost:8000/v1"   # 如果服务在其他机器，请修改
API_KEY = "EMPTY"                       # vLLM 不检查，但必须填写
MODEL = "Qwen/Qwen1.5-4B-Chat"


# ================================
# 1) ChatCompletion 普通请求
# ================================
def test_basic_chat():
    print("\n✅ 测试：普通 ChatCompletion 对话")
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
        )

        print("✅ 响应内容：")
        print(">>>", resp.choices[0].message.content)
    except Exception as e:
        print("❌ 普通对话测试失败：", e)


# ================================
# 2) ChatCompletion 流式输出
# ================================
def test_streaming_chat():
    print("\n✅ 测试：流式输出 ChatCompletion")
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "请写一首五言绝句"}],
            stream=True
        )

        print("✅ 流式输出：")
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)

        print("\n✅ 流式输出结束\n")

    except Exception as e:
        print("❌ 流式输出测试失败：", e)


# ================================
# 3) 使用 requests 测试
# ================================
def test_raw_request():
    print("\n✅ 测试：使用 requests 原始请求")
    try:
        url = f"{BASE_URL}/chat/completions"
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "你认识有人叫徐卓一么？"}]
        }

        r = requests.post(url, json=payload)
        j = r.json()
        print("✅ 响应内容：")
        print(">>>", j["choices"][0]["message"]["content"])

    except Exception as e:
        print("❌ requests 测试失败：", e)


# ================================
# 4) 错误处理测试
# ================================
def test_error_case():
    print("\n✅ 测试：错误情况（模型名错误）")
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

        _ = client.chat.completions.create(
            model="WrongModelName",
            messages=[{"role": "user", "content": "测试错误"}],
        )
    except Exception as e:
        print("✅ 预期错误出现：", e)


# ================================
# 主函数
# ================================
if __name__ == "__main__":
    print("====================================")
    print("🚀 vLLM OpenAI API 自动化测试程序开始")
    print("====================================")

    time.sleep(1)

    test_basic_chat()
    time.sleep(0.5)

    test_streaming_chat()
    time.sleep(0.5)

    test_raw_request()
    time.sleep(0.5)

    test_error_case()

    print("\n====================================")
    print("✅ 全部测试完成")
    print("====================================")
