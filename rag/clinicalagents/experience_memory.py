"""Static local guidance and CDC evidence memory, replacing unavailable AGRAG.

Historical cases supply candidate test categories, never patient findings.
All material is immutable; test cases and rollout outputs are never added.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
import os
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def load_index(directory):
    folder = Path(directory)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, checksum in manifest['files_sha256'].items():
        if digest(folder / name) != checksum:
            raise ValueError('Experience index hash mismatch: ' + name)
    guides = json.loads((folder / 'guidelines.json').read_text())
    cases = json.loads((folder / 'cases.json').read_text())
    gv = np.load(folder / 'guideline_vectors.npy', allow_pickle=False)
    cv = np.load(folder / 'case_vectors.npy', allow_pickle=False)
    if len(guides) != len(gv) or len(cases) != len(cv):
        raise ValueError('Misaligned memory vectors')
    for matrix in (gv, cv):
        if matrix.ndim != 2 or matrix.shape[1] != 1024 or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ValueError('Invalid memory vectors')
        matrix.flags.writeable = False
    return manifest, guides, cases, gv, cv


def query_from_state(state):
    # Explicit whitelist: no diagnosis_result, treatment_plan, unobserved exams,
    # cached retrieval content, case ID, reference answer or state snapshot.
    fields = ('chief_complaint', 'present_illness', 'past_history', 'personal_info',
              'physical_exams', 'auxiliary_exams', 'imaging_results', 'hypothesis_illness')
    return json.dumps({k: state[k] for k in fields if state.get(k)}, ensure_ascii=False, sort_keys=True)


class ExperienceMemory:
    def __init__(self, guide_path, cdc_path, held_out_case_ids=None,
                 allow_dataset_fallback=False, guide_retriever=None):
        if allow_dataset_fallback or guide_retriever is not None:
            raise ValueError('This version permits only the pinned static local index')
        if Path(guide_path).parent != Path(cdc_path).parent:
            raise ValueError('Guide/case index provenance must match')
        self.manifest, self.guides, self.cases, self.gv, self.cv = load_index(str(Path(guide_path).parent))
        heldout = set(held_out_case_ids or [])
        if heldout & {c['case_id'] for c in self.cases}:
            raise ValueError('Heldout cases present in historical memory')
        self.available = bool(self.guides and self.cases)
        self.frozen = False
        self.cache = {}

    def freeze(self):
        self.frozen = True
        return self

    def _embed(self, text):
        port = int(os.environ.get('CA_EMBED_PORT', '23831'))
        request = Request(f'http://127.0.0.1:{port}/embed',
            data=json.dumps({'texts': [text]}).encode(), headers={'Content-Type': 'application/json'})
        with build_opener(ProxyHandler({})).open(request, timeout=240) as response:
            result = json.load(response)
        expected = self.manifest['embedding']
        if (result['model'], result['revision']) != (expected['model'], expected['revision']):
            raise ValueError('Query and document encoder revisions differ')
        vector = np.asarray(result['vectors'][0], dtype=np.float32)
        if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise ValueError('Invalid query vector')
        return vector

    def retrieve(self, state):
        if not self.frozen:
            raise ValueError('Experience memory must be frozen')
        query = query_from_state(state)
        if query in self.cache:
            return deepcopy(self.cache[query])
        vector = self._embed(query)
        gs, cs = self.gv @ vector, self.cv @ vector
        # Untuned pilot cutoffs, not clinically calibrated relevance thresholds.
        # Keep at most three distinct sections from a ten-chunk candidate pool.
        selected, seen = [], set()
        for i in np.argsort(-gs, kind='stable')[:10]:
            item = self.guides[int(i)]
            key = (item['document_id'], item['section'])
            if float(gs[i]) < 0.35 or key in seen:
                continue
            seen.add(key)
            selected.append(dict(item, score=float(gs[i])))
            if len(selected) == 3:
                break
        candidates = [int(i) for i in np.argsort(-cs, kind='stable')[:3] if float(cs[i]) >= 0.35]
        hits = [{'case_id': self.cases[i]['case_id'], 'score': float(cs[i])} for i in candidates]
        observed = set(state.get('physical_exams', {})) | set(state.get('auxiliary_exams', {})) | set(state.get('imaging_results', {}))
        importance, support = {}, {}
        for i in candidates:
            row = self.cases[i]['evidence']
            for field in ('体格检查', '辅助检查'):
                values = row.get(field) or {}
                if not isinstance(values, dict):
                    continue
                for exam in values:
                    if exam in observed:
                        continue
                    importance[exam] = importance.get(exam, 0) + max(0, float(cs[i])) / max(1, len(candidates))
                    support.setdefault(exam, []).append(self.cases[i]['case_id'])
        # Category-level adaptation of Eq. (12), not entity-level AGRAG.
        missing = [dict(item=k, importance=v, historical_case_ids=support[k],
                    status='candidate_to_verify_not_patient_fact')
                   for k, v in sorted(importance.items(), key=lambda x: (-x[1], x[0])) if v > 0.25]
        knowledge = [f"[External guideline, not a patient finding] [{g['chunk_id']}] {g['title']} | {g['section']} | {g['url']}\n{g['text']}" for g in selected]
        value = dict(knowledge=knowledge, retrieved_guidelines=selected, retrieved_cdc_cases=hits,
            potential_missing_evidence=missing, evidence_importance=importance,
            retrieval_backend='local_bge_m3_static_cdc_guidance_and_historical_test_categories',
            query=query, query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            guideline_cutoff=0.35, historical_cutoff=0.35, missing_importance_cutoff=0.25,
            thresholds_calibrated=False, original_agrag=False)
        self.cache[query] = deepcopy(value)
        # Append only the retrieval trace, never add test records to the memory.
        log = os.environ.get('CA_RETRIEVAL_LOG')
        if log:
            with open(log, 'a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        return value
