import json, time, urllib.request, sys
sys.stdout.reconfigure(encoding='utf-8')
prompt = '你是一名知识科普作者。请把下面这段知识文本改写成教科书讲义风格：事实必须准确，条理清晰，语言通俗易懂，直接输出改写后的正文。原文：光合作用是植物利用光能把二氧化碳和水转化为有机物并释放氧气的过程，是地球上生命能量来源的基础。'
body = json.dumps({'model': 'qwen3-1.7b', 'prompt': prompt, 'stream': False,
                   'options': {'num_predict': 128, 'temperature': 0.7, 'top_p': 0.9}}).encode('utf-8')
req = urllib.request.Request('http://localhost:11434/api/generate', data=body,
                             headers={'Content-Type': 'application/json'})
t0 = time.time()
with urllib.request.urlopen(req, timeout=120) as r:
    resp = json.loads(r.read().decode('utf-8'))
dt = time.time() - t0
out = resp.get('response', '')
print('elapsed:', round(dt, 1), 's')
print('prompt_eval:', resp.get('prompt_eval_count'), '| eval:', resp.get('eval_count'))
print('rate:', round(resp.get('eval_count', 0) / max(0.01, resp.get('eval_duration', 1e9) * 1e-9), 1), 'tok/s')
print('output:', out[:200].replace(chr(10), ' / '))
