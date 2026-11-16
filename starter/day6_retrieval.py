"""Adapters for the Day 6 PostgreSQL/pgvector corpus and three retrieval methods."""
import json
import os
import sys
from pathlib import Path

from baseline import TfidfIndex


class Day6Retriever:
    def __init__(self, store, embedder, chunks, *, strategy, method, candidate_k=12):
        if method not in ('dense', 'tfidf', 'hybrid'):
            raise ValueError('method must be dense, tfidf, or hybrid')
        if not strategy:
            raise ValueError('Day 6 strategy must not be empty')
        if candidate_k < 1:
            raise ValueError('candidate_k must be positive')
        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        self.strategy = strategy
        self.method = method
        self.candidate_k = candidate_k
        self.lexical = TfidfIndex(chunks)

    @staticmethod
    def _normalize(rows, retrieval_method):
        normalized = []
        for row in rows:
            normalized.append({
                'id': row.get('id', row.get('chunk_id')),
                'text': row['text'],
                'source_id': row.get('source_id', row.get('document_id')),
                'source': row['source'],
                'page': row['page'],
                'metadata': row.get('metadata') or {},
                'score': float(row['score']),
                'strategy': retrieval_method,
            })
        return normalized

    def _dense(self, query, filters, top_k):
        vector = self.embedder.embed_query(query)
        rows = self.store.search(self.strategy, vector, top_k=top_k, filters=filters)
        return self._normalize(rows, 'dense')

    def search(self, query, *, filters=None, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string')
        if top_k < 1:
            raise ValueError('top_k must be positive')
        if filters and set(filters) - {'section'}:
            raise ValueError('Day 6 supports only the section metadata filter')

        if self.method == 'dense':
            return self._dense(query, filters, top_k)

        if self.method == 'tfidf':
            return self.lexical.search(query, top_k=top_k, filters=filters)

        pool_size = min(max(top_k, self.candidate_k), len(self.chunks))
        dense_rows = self._dense(query, filters, pool_size)
        lexical_rows = self.lexical.search(query, top_k=pool_size, filters=filters)
        by_id = {}
        reciprocal_rank_scores = {}
        for rows in (dense_rows, lexical_rows):
            for rank, row in enumerate(rows, start=1):
                chunk_id = row['id']
                by_id[chunk_id] = row
                reciprocal_rank_scores[chunk_id] = (
                    reciprocal_rank_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
                )
        ranked_ids = sorted(
            reciprocal_rank_scores,
            key=lambda chunk_id: (-reciprocal_rank_scores[chunk_id], chunk_id),
        )
        return [
            {**by_id[chunk_id], 'score': reciprocal_rank_scores[chunk_id], 'strategy': 'hybrid_rrf'}
            for chunk_id in ranked_ids[:top_k]
        ]


def load_day6_retriever(
    starter_dir,
    *,
    strategy='C',
    method='dense',
    candidate_k=12,
):
    """Load the selected Day 6 strategy without changing its database or index."""
    starter = Path(starter_dir).expanduser().resolve()
    if not (starter / 'src' / 'store.py').is_file():
        raise FileNotFoundError(f'Day 6 starter source not found under {starter}')
    if not (starter / '.env').is_file():
        raise FileNotFoundError(f'Day 6 environment file not found under {starter}')

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError('Install python-dotenv from starter/requirements.txt') from error
    load_dotenv(starter / '.env')

    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise RuntimeError('DATABASE_URL is missing from the environment and Day 6 .env')

    model_name = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    summary_path = starter / 'results' / 'ingestion_summary.json'
    if method != 'tfidf' and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        indexed_model = summary.get('embedding_model')
        if indexed_model and indexed_model != model_name:
            raise ValueError(
                f'Configured embedding model {model_name!r} does not match the '
                f'Day 6 index model {indexed_model!r}'
            )

    starter_path = str(starter)
    if starter_path not in sys.path:
        sys.path.insert(0, starter_path)
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError('Install psycopg from starter/requirements.txt') from error

    embedder = None
    store = None
    if method != 'tfidf':
        try:
            from src.embeddings import MyEmbedder
            from src.store import PgVectorStore
        except ImportError as error:
            raise RuntimeError('Could not import the Day 6 embedding and store adapters') from error
        embedder = MyEmbedder(model_name)
        store = PgVectorStore(database_url, embedder.dimensions)
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, chunk_text, source, document_id, page, metadata
                FROM day6_chunks
                WHERE strategy = %s
                ORDER BY chunk_id
                """,
                (strategy,),
            )
            rows = cursor.fetchall()
    chunks = [
        {
            'id': row[0],
            'text': row[1],
            'source': row[2],
            'source_id': row[3],
            'page': row[4],
            'metadata': row[5] or {},
        }
        for row in rows
    ]
    if not chunks:
        raise RuntimeError(
            f'No chunks found for Day 6 strategy {strategy!r}; run the Day 6 ingestion first'
        )
    return Day6Retriever(
        store,
        embedder,
        chunks,
        strategy=strategy,
        method=method,
        candidate_k=candidate_k,
    )
