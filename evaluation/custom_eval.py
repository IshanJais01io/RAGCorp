import os
import sys
import json
import re
from groq import Groq

RAGCORP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAGCORP_PATH not in sys.path:
    sys.path.insert(0, RAGCORP_PATH)

from serving.generation import retrieve_context, generate_answer

EVAL_SYSTEM_PROMPT = """You are an expert RAG Benchmarking Judge. Evaluate the given (Query, Context, Answer) triplet.
Provide 3 numerical scores between 0.0 and 1.0:
1. "faithfulness": Is the answer strictly derived from context without hallucination? (1.0 = Fully grounded, 0.0 = Pure hallucination)
2. "answer_relevancy": Does the answer directly and completely address the query? (1.0 = Perfectly relevant, 0.0 = Irrelevant)
3. "context_precision": Are the retrieved context chunks highly relevant to the query? (1.0 = All chunks relevant, 0.0 = No relevant context)

OUTPUT FORMAT: Return strictly valid JSON with no markdown wrapping:
{"faithfulness": 0.95, "answer_relevancy": 1.0, "context_precision": 0.90, "reasoning": "Brief evaluation explanation."}
"""

BENCHMARK_TEST_SET = [
    "About STRING AND PERCUSSION INSTRUMENTS",
    "Why do most people work for money according to Rich Dad Poor Dad?",
    "Why teach financial literacy according to Rich Dad Poor Dad?",
    "What is the main concept of feedback loop from hell in The Subtle Art?",
    "How do simple mechanical tools or levers work in The Way Things Work Now?"
]

def run_evaluation():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not set.")
        return

    client = Groq(api_key=api_key)
    results = []
    
    print("==================================================")
    print("🚀 Starting RAGCorp Benchmark Evaluation Suite")
    print("==================================================\n")

    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_precision = 0.0

    for idx, query in enumerate(BENCHMARK_TEST_SET, 1):
        print(f"[{idx}/{len(BENCHMARK_TEST_SET)}] Evaluating Query: '{query}'")
        
        chunks = retrieve_context(query, top_k=12)
        answer, _ = generate_answer(query, chunks)
        
        context_str = "\n\n".join([c.get("text", "") for c in chunks]) if chunks else "No context retrieved."

        try:
            eval_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_str[:3000]}\n\nGenerated Answer:\n{answer[:2000]}"}
                ],
                temperature=0.0
            )
            
            raw_eval = eval_response.choices[0].message.content.strip()
            clean_eval = re.sub(r'```json\s*|```', '', raw_eval).strip()
            eval_data = json.loads(clean_eval)
            
            faith = float(eval_data.get("faithfulness", 0.0))
            rel = float(eval_data.get("answer_relevancy", 0.0))
            prec = float(eval_data.get("context_precision", 0.0))

            total_faithfulness += faith
            total_relevancy += rel
            total_precision += prec

            results.append({
                "query": query,
                "scores": {"faithfulness": faith, "answer_relevancy": rel, "context_precision": prec},
                "reasoning": eval_data.get("reasoning", "")
            })
            
            print(f"   - Faithfulness: {faith:.2f} | Relevancy: {rel:.2f} | Context Precision: {prec:.2f}")
            print(f"   - Reasoning: {eval_data.get('reasoning')}\n")
            
        except Exception as e:
            print(f"   ⚠️ Evaluation parsing error on query '{query}': {e}\n")

    num_samples = len(BENCHMARK_TEST_SET)
    avg_faithfulness = total_faithfulness / num_samples
    avg_relevancy = total_relevancy / num_samples
    avg_precision = total_precision / num_samples

    overall_score = (avg_faithfulness + avg_relevancy + avg_precision) / 3.0

    print("==================================================")
    print("📊 OVERALL BENCHMARK RESULTS")
    print("==================================================")
    print(f"• Average Faithfulness:        {avg_faithfulness * 100:.1f}%")
    print(f"• Average Answer Relevancy:    {avg_relevancy * 100:.1f}%")
    print(f"• Average Context Precision:   {avg_precision * 100:.1f}%")
    print(f"• Overall System Score:        {overall_score * 100:.1f}%")
    print("==================================================\n")

    summary_path = "/content/RAGCorp/evaluation/benchmark_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "overall_score": overall_score,
            "avg_faithfulness": avg_faithfulness,
            "avg_answer_relevancy": avg_relevancy,
            "avg_context_precision": avg_precision,
            "detailed_results": results
        }, f, indent=2)
    
    print(f"✅ Saved updated benchmark report to {summary_path}")

if __name__ == "__main__":
    run_evaluation()
