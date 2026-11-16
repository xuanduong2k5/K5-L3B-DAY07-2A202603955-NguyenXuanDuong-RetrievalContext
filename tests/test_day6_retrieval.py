"""Unit checks for the Day 6 adapter and hybrid ranking."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from day6_retrieval import Day6Retriever


def chunk(chunk_id, text, section='Introduction'):
    return {
        'id': chunk_id,
        'text': text,
        'source_id': 'paper',
        'source': 'paper.pdf',
        'page': 1,
        'metadata': {'section': section},
    }


class Embedder:
    def embed_query(self, query):
        return [1.0, 0.0]


class Store:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, strategy, vector, *, top_k, filters):
        self.calls.append((strategy, vector, top_k, filters))
        rows = self.rows
        if filters:
            rows = [row for row in rows if row['metadata'].get('section') == filters['section']]
        return rows[:top_k]


class Day6RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk('dense-first', 'a result about algebraic curves'),
            chunk('lexical-first', 'equidistribution of curves in hypersurfaces'),
            chunk('other-section', 'equidistribution of algebraic curves', 'Preliminaries'),
        ]
        self.dense_rows = [
            {
                'chunk_id': 'dense-first',
                'text': self.chunks[0]['text'],
                'document_id': 'paper',
                'source': 'paper.pdf',
                'page': 1,
                'metadata': {'section': 'Introduction'},
                'score': 0.9,
            },
            {
                'chunk_id': 'lexical-first',
                'text': self.chunks[1]['text'],
                'document_id': 'paper',
                'source': 'paper.pdf',
                'page': 1,
                'metadata': {'section': 'Introduction'},
                'score': 0.8,
            },
        ]

    def make_retriever(self, method):
        return Day6Retriever(
            Store(self.dense_rows),
            Embedder(),
            self.chunks,
            strategy='C',
            method=method,
            candidate_k=3,
        )

    def test_dense_uses_day6_vector_search_and_normalizes_chunk_ids(self):
        retriever = self.make_retriever('dense')
        results = retriever.search('equidistribution', top_k=1)
        self.assertEqual(results[0]['id'], 'dense-first')
        self.assertEqual(results[0]['source_id'], 'paper')
        self.assertEqual(retriever.store.calls[0][0], 'C')

    def test_tfidf_honors_day6_section_filter(self):
        retriever = self.make_retriever('tfidf')
        results = retriever.search(
            'equidistribution curves',
            filters={'section': 'Introduction'},
            top_k=3,
        )
        self.assertTrue(results)
        self.assertTrue(all(row['metadata']['section'] == 'Introduction' for row in results))

    def test_hybrid_fuses_dense_and_lexical_ranks_without_duplicate_ids(self):
        retriever = self.make_retriever('hybrid')
        results = retriever.search('equidistribution curves', top_k=3)
        ids = [row['id'] for row in results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, ['dense-first', 'lexical-first', 'other-section'])
        self.assertGreater(results[0]['score'], results[1]['score'])
        self.assertTrue(all(row['strategy'] == 'hybrid_rrf' for row in results))

    def test_invalid_metadata_filter_is_rejected(self):
        retriever = self.make_retriever('hybrid')
        with self.assertRaises(ValueError):
            retriever.search('query', filters={'version': '2025'})


if __name__ == '__main__':
    unittest.main()
