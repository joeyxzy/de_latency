#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import asyncio
import random
import numpy as np
from openai import AsyncOpenAI

# ================================
# ⚙️ 压测配置 (根据你的显存大小调整)
# ================================
BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"
#MODEL = "Qwen/Qwen1.5-4B-Chat" # 确保和你的启动参数一致
MODEL = "/home/joeyxzy/models/Qwen1.5-4B-Chat"
# 压测参数
TOTAL_REQUESTS = 200      # 总共发送多少个请求 (轰炸量)
CONCURRENCY = 50          # 同时维持多少个并发请求 (太高会 OOM，太低没压力)
MAX_TOKENS_RANGE = (100, 500) # 输出长度范围 (让 Decode 阶段更长)

# 长文本 Prompt 模板 (增加 Prefill 压力)
LONG_PROMPT_TEMPLATE = """
人工智能的历史源远流长。在古代的神话传说中，技艺高超的工匠可以制作人造人，并为其赋予智能或意识。[1]现代意义上的AI始于古典哲学家试图将人类的思维过程描述为对符号的机械操作。20世纪40年代，基于抽象数学推理的可编程数字电脑的发明使一批科学家开始严肃地探讨构造一个电子大脑的可能性。

1956年，人工智能的研究领域确立于在达特茅斯学院举行的会议。此次会议的参加者在接下来的数十年间成为AI研究领域的领军人物。[2]他们中有许多人曾预言，与人类具有同等智能水平的机器将在不超过一代人的时间中出现。同时，上千万美元被投入到AI研究中，以期实现这一目标。

然而，研究人员发现自己大大低估了这一工程的难度，人工智能史上共出现过几次低潮（也被称作AI之冬）。由于詹姆斯·莱特希尔爵士的批评和国会方面的压力，美国和英国政府于1973年停止向没有明确目标的人工智能研究项目拨款。七年之后受到日本政府研究规划的刺激，美国政府和企业再次在AI领域投入数十亿研究经费，但这些投资者在80年代末重新撤回了投资。AI研究领域诸如此类的高潮和低谷不断交替出现；至今仍有人对AI的前景作出异常乐观的预测。[3]

尽管在政府官僚和风投资本家那里经历了大起大落，AI领域仍在取得进展。某些在20世纪70年代被认为不可能解决的问题今天已经获得了圆满解决并已成功应用在商业产品上。与第一代AI研究人员的乐观估计不同，具有与人类同等智能水平的机器至今仍未出现。图灵在1950年发表的一篇催生现代智能机器研究的著名论文中称，“我们只能看到眼前的一小段距离……但是，我们可以看到仍有许多工作要做”。[4]

在21世纪的第一个十年，机器学习得益于新方法的出现、性能强大的计算机硬件的应用庞大数据集的收集，被广泛应用解决学术和工业上的问题，这重新引发了人们对AI的投资和兴趣。

计算历史
硬件
1960年代之前1960年代至今
软件
软件Unix自由和开源软件
计算机科学
人工智能编译器构造计算机科学操作系统编程语言杰出先驱者软件工程
现代概念
通用CPU图形用户界面互联网个人电脑笔记型电脑电子游戏万维网
按国家
保加利亚波兰罗马尼亚苏联集团国家苏联南斯拉夫
计算年表
1950年之前1950–19791980–19891990–19992000–20092010–2019更多年表……
计算机科学词汇
 分类
查论编
先驱
奥特曼写道[1]：“某种形式上的人工智能是一个遍布于西方知识分子历史的观点，是一个急需被实现的梦想，”先民对人工智能的追求表现在诸多神话，传说，故事，预言以及制作机器人偶（自动机）的实践之中。[5]

神话，幻想和预言中的AI
希腊神话中已经出现了机械人和人造人，如赫淮斯托斯的黄金机器人和皮格马利翁的伽拉忒亚。[6]中世纪出现了使用巫术或炼金术将意识赋予无生命物质的传说，如贾比尔的Takwin，帕拉塞尔苏斯的何蒙库鲁兹和Judah Loew的魔像。[7]19世纪的幻想小说中出现了人造人和会思考的机器之类题材，例如玛丽·雪莱的《弗兰肯斯坦》和卡雷尔·恰佩克的《罗素姆的万能机器人》。[8]Samuel Butler的《机器中的达尔文（Darwin among the Machines）》一文（1863）探讨了机器通过自然选择进化出智能的可能性。[9]至今人工智能仍然是科幻小说的重要元素。

自动人偶
主条目：自动机

加扎利的可编程自动人偶（1206年）
许多文明中都有创造自动人偶的杰出工匠，例如偃师（中国西周）[10]，希罗（希腊）[11]，加扎利[12]和Wolfgang von Kempelen[13] 等等。已知最古老的“机器人”是古埃及和古希腊的圣像，忠实的信徒认为工匠为这些神像赋予了思想，使它们具有智慧和激情。赫耳墨斯·特里斯墨吉斯忒斯（赫耳墨斯·特里斯墨吉斯忒斯）写道“当发现神的本性时，人就能够重现他”[14][15]。

形式推理
人工智能的基本假设是人类的思考过程可以机械化。对于机械化推理（即所谓“形式推理（formal reasoning）”）的研究已有很长历史。中国，印度和希腊哲学家均已在公元前的第一个千年里提出了形式推理的结构化方法。他们的想法为后世的哲学家所继承和发展，其中著名的有亚里士多德（对三段论逻辑进行了形式分析），欧几里得（其著作《几何原本》是形式推理的典范），花剌子密（代数学的先驱，“algorithm”一词由他的名字演变而来）以及一些欧洲经院哲学家，如奥卡姆的威廉和邓斯·司各脱。[16]

马略卡哲学家拉蒙·柳利（1232-1315）开发了一些“逻辑机”，试图通过逻辑方法获取知识。[17] 柳利的机器能够将基本的，无可否认的真理通过机械手段用简单的逻辑操作进行组合，以求生成所有可能的知识。[18]Llull的工作对莱布尼兹产生了很大影响，后者进一步发展了他的思想。[19]


莱布尼兹猜测人类的思想可以简化为机械计算
在17世纪中，莱布尼兹，托马斯·霍布斯和笛卡儿尝试将理性的思考系统化为代数学或几何学那样的体系。[20]霍布斯在其著作《利维坦》中有一句名言：“推理就是计算（reason is nothing but reckoning）。” [21]莱布尼兹设想了一种用于推理的普适语言（他的通用表意文字），能将推理规约为计算，从而使“哲学家之间，就像会计师之间一样，不再需要争辩。他们只需拿出铅笔放在石板上，然后向对方说（如果想要的话，可以请一位朋友作为证人）：‘我们开始算吧。’”[22] 这些哲学家已经开始明确提出形式符号系统的假设，而这一假设将成为AI研究的指导思想。

在20世纪，数理逻辑研究上的突破使得人工智能好像呼之欲出。这方面的基础著作包括布尔的《思维的定律》与弗雷格的《概念文字》。基于弗雷格的系统，罗素和怀特海在他们于1913年出版的巨著《数学原理》中对数学的基础给出了形式化描述。这一成就激励了希尔伯特，后者向20世纪20年代和30年代的数学家提出了一个基础性的难题：“能否将所有的数学推理形式化?” [16]这个问题的最终回答由哥德尔不完备定理，图灵机和Alonzo Church的λ演算给出。[16][23]他们的答案令人震惊：首先，他们证明了数理逻辑的局限性；其次（这一点对AI更重要），他们的工作隐含了任何形式的数学推理都能在这些限制之下机械化的可能性。


"""

# 短 Prompt (增加调度频率)
SHORT_PROMPTS = [
    "1+1=?",
    "写一首关于春天的诗。",
    "用Python写个冒泡排序。",
    "把'Hello World'翻译成法语。",
    "解释一下什么是Transformer架构。",
]

# ================================

class StressTester:
    def __init__(self):
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.latencies = []
        self.ttfts = []
        self.tpots = []
        self.request_id_map = {}
        self.per_request_stats = []  # ← 新增：存储每条请求的完整统计
        self.errors = 0
        self.completed = 0
        self.start_time = 0

    async def send_request(self, req_id):
        is_long = random.random() < 0.2
        prompt = LONG_PROMPT_TEMPLATE if is_long else random.choice(SHORT_PROMPTS)
        max_tokens = random.randint(*MAX_TOKENS_RANGE)
        
        req_start = time.time()
        first_token_time = None
        token_count = 0
        vllm_request_id = None

        try:
            stream = await self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True,
            )

            async for chunk in stream:
                if vllm_request_id is None:
                    vllm_request_id = chunk.id

                delta_content = chunk.choices[0].delta.content
                if delta_content is not None:
                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += 1

            total_time = time.time() - req_start

            # 计算 TTFT / TPOT
            if token_count == 0:
                ttft = total_time
                tpot = 0.0
            else:
                ttft = (first_token_time - req_start) if first_token_time else total_time
                decode_time = total_time - ttft
                tpot = decode_time / (token_count - 1) if token_count > 1 else 0.0

            # 保存全局统计
            self.latencies.append(total_time)
            self.ttfts.append(ttft)
            self.tpots.append(tpot)
            self.request_id_map[req_id] = vllm_request_id

            # ✅ 保存 per-request 详细信息
            self.per_request_stats.append({
                "req_id": req_id,
                "vllm_id": vllm_request_id,
                "ttft": ttft,
                "tpot": tpot,
                "total_output_tokens": token_count,
                "e2e_latency": total_time,
            })

            self.completed += 1
            if self.completed % 10 == 0:
                avg_lat = np.mean(self.latencies[-10:])
                print(f"✅ Progress: {self.completed}/{TOTAL_REQUESTS} | Last 10 Avg Latency: {avg_lat:.2f}s")

        except Exception as e:
            print(f"❌ [Req {req_id}] Failed: {e}")
            self.errors += 1

    async def worker(self, queue):
        while True:
            req_id = await queue.get()
            await self.send_request(req_id)
            queue.task_done()

    async def run(self):
        print(f"🔥 Starting Stress Test: {TOTAL_REQUESTS} reqs, {CONCURRENCY} concurrency")
        print(f"🎯 Target: {BASE_URL} | Model: {MODEL}")
        
        queue = asyncio.Queue()
        for i in range(TOTAL_REQUESTS):
            queue.put_nowait(i)

        self.start_time = time.time()

        # 启动 Worker
        workers = []
        for _ in range(CONCURRENCY):
            task = asyncio.create_task(self.worker(queue))
            workers.append(task)

        # 等待队列清空
        await queue.join()

        # 取消 Worker
        for task in workers:
            task.cancel()

        total_time = time.time() - self.start_time
        self.print_stats(total_time)

    def print_stats(self, total_time):
        print("\n========================================")
        print("📊 Stress Test Results")
        print("========================================")
        print(f"Total Requests  : {TOTAL_REQUESTS}")
        print(f"Concurrency     : {CONCURRENCY}")
        print(f"Total Time      : {total_time:.2f} s")
        print(f"Throughput (RPS): {self.completed / total_time:.2f} req/s")
        print(f"Success Rate    : {100 * self.completed / TOTAL_REQUESTS:.1f}%")
        print("----------------------------------------")
        if self.latencies:
            print(f"Latency E2E (Avg): {np.mean(self.latencies):.2f} s")
            print(f"Latency E2E (P95): {np.percentile(self.latencies, 95):.2f} s")
        if self.ttfts:
            print(f"TTFT (Avg)       : {np.mean(self.ttfts):.3f} s")
            print(f"TTFT (P95)       : {np.percentile(self.ttfts, 95):.3f} s")
        if self.tpots:
            valid_tpots = [t for t in self.tpots if t > 0]
            if valid_tpots:
                print(f"TPOT (Avg)       : {np.mean(valid_tpots):.3f} s/token")
                print(f"TPOT (P95)       : {np.percentile(valid_tpots, 95):.3f} s/token")
        print("========================================")

        # ✅ 新增：逐条打印每个请求的详细指标
        print("\n📋 Per-Request Detailed Metrics (Success Only):")
        print(f"{'ReqID':<6} {'vLLM ID':<36} {'TTFT(s)':<8} {'TPOT(s/tok)':<12} {'Tokens':<8}")
        print("-" * 80)
        for stat in sorted(self.per_request_stats, key=lambda x: x["req_id"]):
            print(
                f"{stat['req_id']:<6} "
                f"{stat['vllm_id']:<36} "
                f"{stat['ttft']:<8.3f} "
                f"{stat['tpot']:<12.4f} "
                f"{stat['total_output_tokens']:<8}"
            )

if __name__ == "__main__":
    tester = StressTester()
    asyncio.run(tester.run())