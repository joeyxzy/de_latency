#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-pressure stress test to FORCE preemption in vLLM.
PRE-GENERATES all prompts at startup using real tokenizer.
Avoids runtime tokenization bottleneck.
"""

import time
import asyncio
import random
import os
import numpy as np
from openai import AsyncOpenAI

# ================================
# 🔧 CONFIGURATION
# ================================
BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"
MODEL_PATH = os.getenv("DE_LATENCY_MODEL", "Qwen/Qwen1.5-4B-Chat")

MAX_MODEL_LEN = 4096
MIN_OUTPUT_LEN = 200
MAX_OUTPUT_LEN = 500
SAFETY_MARGIN = 16

TOTAL_REQUESTS = 120
CONCURRENCY = 80

# ================================
# 🧠 Pre-generate Prompts (at startup)
# ================================
def pre_generate_prompts(total_requests, min_output, max_output, max_model_len, safety_margin):
    print("Loading tokenizer and pre-generating prompts...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    sentences = [
        "The rapid advancement of artificial intelligence is reshaping industries worldwide. ",
        "Large language models demonstrate remarkable capabilities in understanding human language. ",
        "Efficient memory management is crucial for high-throughput inference serving systems. ",
        "Transformer architectures rely on self-attention mechanisms to process sequences. ",
        "Tokenization converts raw text into numerical representations for neural networks. ",
        "Context length limitations require careful scheduling in production deployments. ",
        "Throughput and latency are key performance indicators for LLM serving engines. ",
        "Concurrency control helps balance resource utilization across multiple requests. ",
        "Preemption allows the system to handle more requests than fit in GPU memory. ",
        "Real-world workloads often mix short queries with long document processing. ",
        "KV cache efficiency directly impacts the number of concurrent requests supported. ",
        "Prefix caching can reduce redundant computation but may hide true system pressure. ",
        "GPU memory bandwidth often becomes the bottleneck in large-batch inference. ",
        "Asynchronous request handling improves overall system responsiveness. "
    ]

    prompts_and_outputs = []
    for i in range(total_requests):
        output_len = random.randint(min_output, max_output)
        prompt_token_budget = max_model_len - output_len - safety_margin
        if prompt_token_budget <= 50:
            prompt_token_budget = 100

        # Build prompt efficiently
        prompt = ""
        current_tokens = 0
        while current_tokens < prompt_token_budget:
            sent = random.choice(sentences)
            temp_prompt = prompt + sent
            tokens = tokenizer.encode(temp_prompt, add_special_tokens=False)
            if len(tokens) > prompt_token_budget:
                break
            prompt = temp_prompt
            current_tokens = len(tokens)

        # Final truncate if needed
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > prompt_token_budget:
            tokens = tokens[:prompt_token_budget]
            prompt = tokenizer.decode(tokens, skip_special_tokens=True)

        prompts_and_outputs.append((prompt, output_len))
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{total_requests} prompts...")

    print("✅ All prompts pre-generated.")
    return prompts_and_outputs

# ================================
# 🧪 Stress Tester
# ================================
class PreemptionStressTester:
    def __init__(self, prompts_and_outputs):
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.prompts_and_outputs = prompts_and_outputs
        self.latencies = []
        self.ttfts = []
        self.tpots = []
        self.completed = 0
        self.errors = 0

    async def send_request(self, req_id: int):
        if req_id >= len(self.prompts_and_outputs):
            return
        prompt, output_len = self.prompts_and_outputs[req_id]

        req_start = time.time()
        first_token_time = None
        token_count = 0

        try:
            stream = await self.client.chat.completions.create(
                model=MODEL_PATH,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_len,
                temperature=0.0,
                stream=True,
            )

            async for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.time()
                if chunk.choices[0].delta.content is not None:
                    token_count += 1

            total_time = time.time() - req_start
            ttft = (first_token_time - req_start) if first_token_time else total_time
            decode_time = total_time - ttft
            tpot = decode_time / (token_count - 1) if token_count > 1 else 0.0

            self.latencies.append(total_time)
            self.ttfts.append(ttft)
            self.tpots.append(tpot)
            self.completed += 1

            if self.completed % 10 == 0:
                recent = self.latencies[-min(10, len(self.latencies)):]
                avg_lat = np.mean(recent) if recent else 0
                print(f"✅ {self.completed}/{TOTAL_REQUESTS} | Avg Lat (last 10): {avg_lat:.2f}s")

        except Exception as e:
            self.errors += 1
            error_str = str(e).replace('\n', ' ')[:120]
            print(f"❌ Req {req_id} failed: {error_str}")

    async def worker(self, queue: asyncio.Queue):
        while True:
            req_id = await queue.get()
            await self.send_request(req_id)
            queue.task_done()

    async def run(self):
        print("🔥 Starting PREEMPTION-FORCING stress test")
        print(f"🎯 Target: {BASE_URL}")
        print(f"🧠 Model: {MODEL_PATH}")
        print(f"📤 Output: {MIN_OUTPUT_LEN}–{MAX_OUTPUT_LEN} tokens")
        print(f"👥 Concurrency: {CONCURRENCY} | Total: {TOTAL_REQUESTS}")
        print("⚠️  Expect queuing, high latency, and PREEMPTION!\n")

        queue = asyncio.Queue()
        for i in range(len(self.prompts_and_outputs)):
            queue.put_nowait(i)

        start_time = time.time()
        workers = [asyncio.create_task(self.worker(queue)) for _ in range(CONCURRENCY)]

        await queue.join()
        for w in workers:
            w.cancel()

        total_time = time.time() - start_time
        self.print_stats(total_time)

    def print_stats(self, total_time: float):
        print("\n" + "="*50)
        print("📊 PREEMPTION TEST RESULTS")
        print("="*50)
        print(f"Completed: {self.completed} / {len(self.prompts_and_outputs)}")
        print(f"Errors:    {self.errors}")
        print(f"Total time: {total_time:.2f} s")
        print(f"RPS:       {self.completed / total_time:.2f}")
        if self.latencies:
            print(f"E2E P99:   {np.percentile(self.latencies, 99):.2f} s")
            print(f"TTFT P99:  {np.percentile(self.ttfts, 99):.2f} s")
        valid_tpots = [t for t in self.tpots if t > 0]
        if valid_tpots:
            print(f"TPOT P99:  {np.percentile(valid_tpots, 99):.4f} s/token")
        print("="*50)

# ================================
# 🚀 Main
# ================================
if __name__ == "__main__":
    # Pre-generate all prompts (takes ~10-30 seconds, but only once)
    prompts_and_outputs = pre_generate_prompts(
        TOTAL_REQUESTS, MIN_OUTPUT_LEN, MAX_OUTPUT_LEN, MAX_MODEL_LEN, SAFETY_MARGIN
    )
    
    tester = PreemptionStressTester(prompts_and_outputs)
    asyncio.run(tester.run())
