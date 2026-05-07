# eval_simple.py - Quick evaluation without model loading issues
import os, sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.getcwd(), 'BACKEND'))
os.environ['CHROMA_PATH'] = os.path.join(os.getcwd(), 'chroma_db')
os.environ['POLICIES_PATH'] = os.path.join(os.getcwd(), 'DATA', 'policies')
os.environ['OPENROUTER_API_KEY'] = os.environ.get('OPENROUTER_API_KEY', '')

print("Loading system...", flush=True)

from app.ingestion import load_vectorstore, load_embeddings
from app.rag_system import KolroseRAG
from app.evaluation import EVALUATION_QUESTIONS, GroundednessEvaluator

vectorstore = load_vectorstore()
embeddings = load_embeddings()
rag = KolroseRAG(vectorstore)

questions = EVALUATION_QUESTIONS[:10]
results = []

print(f"Testing {len(questions)} questions...\n", flush=True)

for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...", flush=True)
    
    start = time.time()
    result = rag.query(q['question'], enable_guardrails=False)
    elapsed = (time.time() - start) * 1000
    
    answer = result.answer.lower()
    keywords = q.get('expected_keywords', [])
    found = [kw for kw in keywords if kw.lower() in answer]
    score = len(found) / len(keywords) if keywords else 1.0
    
    results.append({
        'id': q['id'],
        'question': q['question'],
        'keyword_score': round(score, 3),
        'latency_ms': round(elapsed, 0),
        'citations': result.citations,
    })
    
    print(f"  Score: {score:.0%} | Latency: {elapsed:.0f}ms | Citations: {len(result.citations)}", flush=True)

os.makedirs('evaluation_results', exist_ok=True)
avg_score = sum(r['keyword_score'] for r in results) / len(results)
avg_latency = sum(r['latency_ms'] for r in results) / len(results)

summary = {
    'timestamp': datetime.now().isoformat(),
    'questions': len(results),
    'avg_keyword_score': round(avg_score, 3),
    'avg_latency_ms': round(avg_latency, 0),
    'results': results,
}

with open('evaluation_results/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nDone! Avg Score: {avg_score:.0%} | Avg Latency: {avg_latency:.0f}ms")
print(f"Results: evaluation_results/summary.json")