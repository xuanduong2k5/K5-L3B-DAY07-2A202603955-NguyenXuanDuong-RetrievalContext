"""Compare TF-IDF, Day 6 dense, and RRF hybrid retrieval on one benchmark split."""
import argparse
import json
import statistics
import time
from pathlib import Path

from baseline import hit, pack_context, validate
from convert_day6 import convert_question
from day6_retrieval import load_day6_retriever


def evaluate(retriever, questions, *, budget=800, top_k=3):
    traces = []
    for question in questions:
        started = time.perf_counter()
        results = retriever.search(
            question['question'],
            filters=question.get('filter'),
            top_k=top_k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**row, 'hit': hit(row, question['gold'])} for row in results]
        context, context_ids = pack_context(results, budget)
        traces.append({
            'query_id': question['id'],
            'split': question['split'],
            'question': question['question'],
            'method': retriever.method,
            'day6_strategy': retriever.strategy,
            'candidate_k': retriever.candidate_k,
            'top_k': top_k,
            'budget_chars': budget,
            'results': marked,
            'context': context,
            'context_ids': context_ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(row['hit'] for row in marked),
            'context_hit': any(row['hit'] and row['id'] in context_ids for row in marked),
        })
    if not traces:
        raise ValueError('No answerable questions found for the selected split')
    return traces


def main():
    parser = argparse.ArgumentParser(description='Benchmark three RAG retrieval methods')
    parser.add_argument('--day6-starter', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--method', required=True, choices=('dense', 'tfidf', 'hybrid'))
    parser.add_argument('--strategy', default='C')
    parser.add_argument('--candidate-k', type=int, default=12)
    parser.add_argument('--budget', type=int, default=800)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.candidate_k < 1 or args.budget < 1:
        parser.error('--candidate-k and --budget must be positive')

    raw_questions = json.loads(args.questions.read_text(encoding='utf-8'))
    if not isinstance(raw_questions, list):
        raise ValueError('questions input must be a JSON array')
    selected = [
        convert_question(question)
        for question in raw_questions
        if question.get('split') == args.split and question.get('expected_answer')
    ]
    retriever = load_day6_retriever(
        args.day6_starter,
        strategy=args.strategy,
        method=args.method,
        candidate_k=args.candidate_k,
    )
    validate(retriever.chunks, selected)
    traces = evaluate(retriever, selected, budget=args.budget)
    summary = {
        'method': args.method,
        'day6_strategy': args.strategy,
        'split': args.split,
        'n': len(traces),
        'hit1': sum(row['hit1'] for row in traces),
        'hit3': sum(row['hit3'] for row in traces),
        'context_hit': sum(row['context_hit'] for row in traces),
        'avg_context_chars': statistics.mean(row['context_chars'] for row in traces),
        'median_ms': statistics.median(row['elapsed_ms'] for row in traces),
        'candidate_k': args.candidate_k,
        'top_k': 3,
        'budget_chars': args.budget,
        'service_cost_usd': 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'traces.json').write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
