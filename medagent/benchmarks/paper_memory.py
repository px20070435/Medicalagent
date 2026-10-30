"""Dense verified-first retrieval with deterministic prior-wave pseudo memories."""
import json
from pathlib import Path
from urllib.request import Request, ProxyHandler, build_opener

import numpy as np

from medagent.benchmarks.dense_case_rag import DenseCaseRetriever, query_key


def local_embed(value):
    opener = build_opener(ProxyHandler({}))
    request = Request('http://127.0.0.1:23830/embed',
        data=json.dumps({'texts': [value]}).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
    return result


class PaperMemory(DenseCaseRetriever):
    def __init__(self, corpus_dir, index_dir):
        super().__init__(corpus_dir, index_dir)
        self.pseudo = {}
        self.raw_rows = {d['case_id']: d['row'] for d in self.docs}
        self.last_vector = None

    def load_wave(self, directory, ids):
        for cid in ids:
            if cid not in self.pseudo:
                item = json.loads((Path(directory) / (cid + '.json')).read_text())
                if item['case_id'] != cid or item['provenance'] != 'model_predictions_and_observed_facts_only':
                    raise ValueError('Invalid pseudo-case provenance')
                vector = np.asarray(item['embedding'], dtype=np.float32)
                if vector.shape != (self.vectors.shape[1],) or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                    raise ValueError('Invalid pseudo-case embedding')
                self.pseudo[cid] = item

    def search(self, query, department=None, k=3, run=None):
        key = query_key(query)
        if key in self.query_lookup:
            vector = self.query_vectors[self.query_lookup[key]]
        else:
            def encode():
                result = local_embed(query)
                if (result['model'] != self.dense_manifest['embedding_model']
                        or result['revision'] != self.dense_manifest['embedding_revision']):
                    raise ValueError('Live encoder differs from frozen index')
                return dict(result, input=query)
            result = run('embedding/' + key, encode) if run else {'status': 'ok', 'result': encode()}
            if result['status'] != 'ok':
                raise RuntimeError('Local symptom encoding failed')
            vector = np.asarray(result['result']['vectors'][0], dtype=np.float32)
        if not query.strip() or vector.shape != (self.vectors.shape[1],):
            raise ValueError('Invalid symptom query or vector dimensions')
        self.last_vector = vector
        scores = self.vectors @ vector
        indices = [i for i, d in enumerate(self.docs) if department is None or d['department'] == department]
        selected = sorted(indices, key=lambda i: (-float(scores[i]), self.docs[i]['case_id']))[:k]
        hits = [{'case_id': self.docs[i]['case_id'], 'score': float(scores[i]), 'source': 'verified',
                 **self.locations[self.docs[i]['case_id']]} for i in selected]
        if len(hits) < k:
            candidates = [item for item in self.pseudo.values()
                if department is None or item['department'] == department]
            candidates.sort(key=lambda d: (-float(np.asarray(d['embedding']) @ vector), d['case_id']))
            hits += [{'case_id': d['case_id'], 'score': float(np.asarray(d['embedding']) @ vector),
                      'source': 'pseudo', 'row': d['row']} for d in candidates[:k - len(hits)]]
        return {'query': query, 'backend': 'local_bge_m3_verified_first_cosine',
            'embedding_revision': self.dense_manifest['embedding_revision'],
            'department_filter': department, 'returned_k': len(hits), 'hits': hits,
            'available_prior_pseudo_cases': len(self.pseudo)}

    def row_for(self, hit):
        return hit['row'] if hit['source'] == 'pseudo' else self.raw_rows[hit['case_id']]
