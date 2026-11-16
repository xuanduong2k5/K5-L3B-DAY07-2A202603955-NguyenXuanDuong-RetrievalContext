"""Run the real tool-calling agent and save traces for human review."""
import argparse
import json
from pathlib import Path

import psycopg

from agentic import run_agent
from convert_day6 import convert_question
from day6_retrieval import load_day6_retriever
from llm_controller import ModelRequestError, OpenAICompatibleController


def load_questions(day6_questions_path, abstentions_path):
    day6_questions = json.loads(day6_questions_path.read_text(encoding='utf-8'))
    abstentions = json.loads(abstentions_path.read_text(encoding='utf-8'))
    if not isinstance(day6_questions, list) or not isinstance(abstentions, list):
        raise ValueError('both benchmark inputs must be JSON arrays')
    cases = []
    for raw in day6_questions:
        case = convert_question(raw)
        case['answerable'] = True
        cases.append(case)
    for case in abstentions:
        if case.get('answerable') is not False or case.get('split') not in ('dev', 'test'):
            raise ValueError(f"abstention case {case.get('id')!r} must be unanswerable and split")
        cases.append(case)
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('benchmark question IDs must be unique')
    return cases


def main():
    parser = argparse.ArgumentParser(description='Run the Day 7 agent with a real model API')
    parser.add_argument('--day6-starter', required=True, type=Path)
    parser.add_argument('--day6-questions', required=True, type=Path)
    parser.add_argument('--abstentions', type=Path, default=Path('data/abstention_questions.json'))
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--method', required=True, choices=('dense', 'tfidf', 'hybrid'))
    parser.add_argument('--strategy', default='C')
    parser.add_argument('--candidate-k', type=int, default=12)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=800)
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if min(args.candidate_k, args.top_k, args.budget, args.max_searches) < 1:
        parser.error('candidate-k, top-k, budget, and max-searches must be positive')

    cases = [case for case in load_questions(args.day6_questions, args.abstentions)
             if case.get('split') == args.split]
    answerable = sum(case['answerable'] for case in cases)
    unanswerable = len(cases) - answerable
    minimums = {'dev': (15, 2), 'test': (5, 2)}
    if answerable < minimums[args.split][0] or unanswerable < minimums[args.split][1]:
        raise ValueError(
            f"{args.split} needs at least {minimums[args.split][0]} answerable and "
            f"{minimums[args.split][1]} unanswerable cases"
        )

    retriever = load_day6_retriever(
        args.day6_starter,
        strategy=args.strategy,
        method=args.method,
        candidate_k=args.candidate_k,
    )
    controller = OpenAICompatibleController.from_environment()
    args.output.mkdir(parents=True, exist_ok=True)
    traces = []
    trace_path = args.output / 'traces.json'
    for case in cases:
        try:
            trace = run_agent(
                case['question'],
                controller,
                retriever,
                filters=case.get('filter'),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
        except (ModelRequestError, psycopg.Error) as error:
            trace = {
                'status': 'error',
                'answer': '',
                'citations': [],
                'events': [{'event': 'error', 'type': type(error).__name__, 'message': str(error)}],
            }
        trace['query_id'] = case['id']
        traces.append(trace)
        trace_path.write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )

    (args.output / 'cases.json').write_text(
        json.dumps(
            [{'id': case['id'], 'split': case['split'], 'answerable': case['answerable']}
             for case in cases],
            ensure_ascii=False,
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )
    run_info = {
        'split': args.split,
        'method': args.method,
        'day6_strategy': args.strategy,
        'model': controller.model,
        'model_base_url': controller.base_url,
        'embedding_model': retriever.embedder.model_name if retriever.embedder else None,
        'embedding_dimensions': retriever.embedder.dimensions if retriever.embedder else None,
        'top_k': args.top_k,
        'candidate_k': args.candidate_k,
        'budget_chars': args.budget,
        'max_searches': args.max_searches,
    }
    (args.output / 'run.json').write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(
        {
            'split': args.split,
            'cases': len(cases),
            'errors': sum(trace['status'] == 'error' for trace in traces),
            'output': str(args.output.resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == '__main__':
    main()
