# run_evaluation.py
"""
Quick evaluation runner for Kolrose Limited RAG System.
Generates all evaluation reports and saves to evaluation_results/
"""

import os
import sys
import json
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.join(os.getcwd(), 'BACKEND'))

# Set environment variables
os.environ['CHROMA_PATH'] = os.path.join(os.getcwd(), 'chroma_db')
os.environ['POLICIES_PATH'] = os.path.join(os.getcwd(), 'DATA', 'policies')
os.environ['OPENROUTER_API_KEY'] = os.environ.get('OPENROUTER_API_KEY', '')

print("=" * 70)
print("📊 KOLROSE LIMITED - EVALUATION RUNNER")
print("=" * 70)

# Import evaluation module
from app.evaluation import (
    EVALUATION_QUESTIONS,
    GroundednessEvaluator,
    CitationAccuracyEvaluator,
    GoldAnswerMatcher,
    LatencyBenchmark,
    AblationStudy,
)
from app.ingestion import load_vectorstore, load_embeddings
from app.rag_system import KolroseRAG

# Initialize system
print("\n📂 Loading vector store...")
vectorstore = load_vectorstore()
if vectorstore is None:
    print("❌ Vector store not found. Run ingestion first.")
    exit(1)

print(f"   ✅ Loaded {vectorstore._collection.count()} vectors")

print("🔧 Initializing RAG system...")
rag = KolroseRAG(vectorstore)

embeddings = load_embeddings()
print("   ✅ RAG system ready")

# Create output directory
output_dir = os.path.join(os.getcwd(), 'evaluation_results')
os.makedirs(output_dir, exist_ok=True)

# Subset for quick run
quick = '--quick' in sys.argv
questions = EVALUATION_QUESTIONS[:10] if quick else EVALUATION_QUESTIONS
latency_questions = [q['question'] for q in questions[:15]]

print(f"\n📋 Evaluating {len(questions)} questions...")

# =========================================================================
# 1. Groundedness
# =========================================================================
print("\n" + "=" * 70)
print("📋 PHASE 1/4: GROUNDEDNESS")
print("=" * 70)

groundedness_eval = GroundednessEvaluator(vectorstore, embeddings)
groundedness_report = groundedness_eval.evaluate_all(rag, questions, verbose=True)

with open(os.path.join(output_dir, 'groundedness.json'), 'w') as f:
    json.dump(groundedness_report, f, indent=2, default=str)

print(f"   💾 Saved: groundedness.json")
print(f"   📊 Score: {groundedness_report['aggregate']['avg_groundedness']:.1%}")

# =========================================================================
# 2. Citation Accuracy
# =========================================================================
print("\n" + "=" * 70)
print("📋 PHASE 2/4: CITATION ACCURACY")
print("=" * 70)

citation_eval = CitationAccuracyEvaluator(vectorstore, embeddings)
citation_report = citation_eval.evaluate_all(rag, questions, verbose=True)

with open(os.path.join(output_dir, 'citation_accuracy.json'), 'w') as f:
    json.dump(citation_report, f, indent=2, default=str)

print(f"   💾 Saved: citation_accuracy.json")
if citation_report['aggregate'].get('questions_with_citations', 0) > 0:
    print(f"   📊 Score: {citation_report['aggregate']['avg_citation_accuracy']:.1%}")

# =========================================================================
# 3. Gold Answer Matching
# =========================================================================
print("\n" + "=" * 70)
print("📋 PHASE 3/4: GOLD ANSWER MATCHING")
print("=" * 70)

gold_eval = GoldAnswerMatcher(embeddings)
gold_report = gold_eval.evaluate_all(rag, questions, verbose=True)

with open(os.path.join(output_dir, 'gold_matching.json'), 'w') as f:
    json.dump(gold_report, f, indent=2, default=str)

print(f"   💾 Saved: gold_matching.json")
if gold_report['aggregate'].get('partial_match_pct'):
    print(f"   📊 Partial Match: {gold_report['aggregate']['partial_match_pct']}%")

# =========================================================================
# 4. Latency Benchmark
# =========================================================================
print("\n" + "=" * 70)
print("📋 PHASE 4/4: LATENCY BENCHMARK")
print("=" * 70)

bench = LatencyBenchmark()
latency_report = bench.benchmark(rag, latency_questions, iterations=2, verbose=True)

with open(os.path.join(output_dir, 'latency.json'), 'w') as f:
    json.dump(latency_report, f, indent=2, default=str)

print(f"   💾 Saved: latency.json")
print(f"   📊 P50: {latency_report['aggregate'].get('p50_ms', 0)}ms")
print(f"   📊 P95: {latency_report['aggregate'].get('p95_ms', 0)}ms")

# =========================================================================
# Generate Summary
# =========================================================================
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_questions': len(questions),
    'groundedness': {
        'avg_score': groundedness_report['aggregate']['avg_groundedness'],
        'fully_grounded_pct': groundedness_report['aggregate']['fully_grounded_pct'],
    },
    'citation_accuracy': {
        'avg_score': citation_report['aggregate'].get('avg_citation_accuracy', 0),
        'fully_accurate_pct': citation_report['aggregate'].get('fully_accurate_pct', 0),
    },
    'gold_matching': {
        'exact_match_pct': gold_report['aggregate'].get('exact_match_pct', 0),
        'partial_match_pct': gold_report['aggregate'].get('partial_match_pct', 0),
        'avg_semantic_similarity': gold_report['aggregate'].get('avg_semantic_similarity', 0),
    },
    'latency': {
        'p50_ms': latency_report['aggregate'].get('p50_ms', 0),
        'p95_ms': latency_report['aggregate'].get('p95_ms', 0),
        'p99_ms': latency_report['aggregate'].get('p99_ms', 0),
        'mean_ms': latency_report['aggregate'].get('mean_ms', 0),
    },
}

with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE!")
print("=" * 70)
print(f"\n📊 RESULTS:")
print(f"   Groundedness: {summary['groundedness']['avg_score']:.1%}")
print(f"   Citation Accuracy: {summary['citation_accuracy']['avg_score']:.1%}")
print(f"   P50 Latency: {summary['latency']['p50_ms']}ms")
print(f"   P95 Latency: {summary['latency']['p95_ms']}ms")
print(f"\n📁 Reports saved to: {output_dir}/")

# run_evaluation.py (FIXED)
"""
Quick evaluation runner for Kolrose Limited RAG System.
Generates all evaluation reports and saves to evaluation_results/
"""

import os
import sys
import json
import time
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.join(os.getcwd(), 'BACKEND'))

os.environ['CHROMA_PATH'] = os.path.join(os.getcwd(), 'chroma_db')
os.environ['POLICIES_PATH'] = os.path.join(os.getcwd(), 'DATA', 'policies')

print("=" * 60)
print("KOLROSE LIMITED - EVALUATION RUNNER")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Path: {os.getcwd()}")
print()

# Step 1: Load vector store
print("[1/3] Loading vector store...")
try:
    from app.ingestion import load_vectorstore
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("ERROR: Vector store not found!")
        print("Run: python -m app.ingestion")
        sys.exit(1)
    count = vectorstore._collection.count()
    print(f"  OK: {count} vectors loaded")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Step 2: Load embeddings
print("[2/3] Loading embedding model...")
try:
    from app.ingestion import load_embeddings
    embeddings = load_embeddings()
    print("  OK: Embedding model ready")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# Step 3: Run evaluation
print("[3/3] Running evaluation...")
try:
    from app.rag_system import KolroseRAG
    from app.evaluation import GroundednessEvaluator, EVALUATION_QUESTIONS
    
    rag = KolroseRAG(vectorstore)
    questions = EVALUATION_QUESTIONS[:10]  # First 10 for quick run
    
    print(f"  Evaluating {len(questions)} questions...")
    print()
    
    evaluator = GroundednessEvaluator(vectorstore, embeddings)
    report = evaluator.evaluate_all(rag, questions, verbose=True)
    
    # Save results
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/groundedness.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    with open('evaluation_results/summary.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'questions': len(questions),
            'avg_groundedness': report['aggregate']['avg_groundedness'],
            'fully_grounded_pct': report['aggregate']['fully_grounded_pct'],
            'avg_latency_ms': report['aggregate']['avg_latency_ms'],
            'p95_latency_ms': report['aggregate']['p95_latency_ms'],
        }, f, indent=2)
    
    print()
    print("=" * 60)
    print("EVALUATION COMPLETE!")
    print(f"  Groundedness: {report['aggregate']['avg_groundedness']:.1%}")
    print(f"  Fully Grounded: {report['aggregate']['fully_grounded_pct']}%")
    print(f"  Avg Latency: {report['aggregate']['avg_latency_ms']:.0f}ms")
    print(f"  Results: evaluation_results/")
    print("=" * 60)
    
except Exception as e:
    print(f"ERROR in evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)