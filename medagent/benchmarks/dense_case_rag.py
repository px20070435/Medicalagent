"""Local symptom embeddings + cosine Top-3; paper mechanism, substitute encoder.

Queries are encoded offline from chief complaint only, the field used by the
released MedChain main.py. No current-case labels enter the query cache. An
unseen query is an explicit error, never silently replaced by BM25 or a label.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from medagent.benchmarks.case_rag import CaseRetriever, sha


def query_key(value):
    return hashlib.sha256(value.encode()).hexdigest()


class DenseCaseRetriever(CaseRetriever):
    def __init__(self, corpus_dir, index_dir):
        super().__init__(corpus_dir)
        index_dir = Path(index_dir)
        self.dense_manifest = json.loads((index_dir / 'manifest.json').read_text())
        manifest = self.dense_manifest
        if sha(self.directory / 'manifest.json') != manifest['corpus_manifest_sha256']:
            raise ValueError('Dense index was built for a different frozen corpus')
        for name, digest in manifest['files_sha256'].items():
            if sha(index_dir / name) != digest:
                raise ValueError('Dense index hash mismatch: ' + name)
        self.vectors = np.load(index_dir / 'case_embeddings.npy', allow_pickle=False)
        self.query_vectors = np.load(index_dir / 'query_embeddings.npy', allow_pickle=False)
        keys = json.loads((index_dir / 'query_keys.json').read_text())
        ids = json.loads((index_dir / 'case_ids.json').read_text())
        if ids != [doc['case_id'] for doc in self.docs]:
            raise ValueError('Dense case vectors are not aligned with workbook rows')
        if len(keys) != len(set(keys)) or len(keys) != len(self.query_vectors):
            raise ValueError('Invalid query cache mapping')
        if (self.vectors.ndim != 2 or self.query_vectors.ndim != 2
                or self.vectors.shape[1] != self.query_vectors.shape[1]):
            raise ValueError('Dense vector dimensions do not match')
        for matrix in (self.vectors, self.query_vectors):
            if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
                raise ValueError('Dense vectors must be finite and unit-normalized')
        self.query_lookup = {key: i for i, key in enumerate(keys)}

    def search(self, query, department=None, k=3):
        if k <= 0:
            raise ValueError('k must be positive')
        key = query_key(query)
        if not query.strip():
            ranked, scores = [], None
        else:
            if key not in self.query_lookup:
                raise ValueError('Query is not encoded in the local cache; rebuild the index for this input file')
            scores = self.vectors @ self.query_vectors[self.query_lookup[key]]
            # No arbitrary cosine threshold: the paper specifies rank-based Top-3.
            candidates = [i for i, doc in enumerate(self.docs)
                if department is None or doc['department'] == department]
            ranked = sorted(candidates, key=lambda i: (-float(scores[i]), self.docs[i]['case_id']))[:k]
        return {'query': query, 'query_sha256': key, 'department_filter': department,
            'requested_k': k, 'returned_k': len(ranked),
            'backend': 'medchain-local-dense-cosine-v1',
            'embedding_model': self.dense_manifest['embedding_model'],
            'embedding_revision': self.dense_manifest['embedding_revision'],
            'query_policy': self.dense_manifest['query_policy'],
            'hits': [{'case_id': self.docs[i]['case_id'], 'score': round(float(scores[i]), 8),
                      **self.locations[self.docs[i]['case_id']]} for i in ranked]}
