"""Frozen MedChain case corpus, held-out deduplication and CPU BM25 retrieval.

No current-case labels or titles enter search. This is a local adaptation, not
the original MedCase-RAG embedding model or its unreleased MDDB artifacts.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from medagent.benchmarks.medchain import opaque_id

VERSION = 'medchain-case-bm25-v1'
NEAR_THRESHOLD = 0.85


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return '；'.join(str(k) + '：' + text(v) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return '；'.join(text(v) for v in value)
    return str(value)


def normalize(value):
    value = unicodedata.normalize('NFKC', text(value)).lower()
    return ''.join(re.findall(r'[\u4e00-\u9fff a-z0-9]', value)).replace(' ', '')


def visible(case):
    intro = case.get('【病案介绍】') or {}
    return {k: intro.get(k) for k in ('主诉', '现病史', '既往史', '查体')}


def clinical_text(case):
    return text(visible(case))


def fingerprint(case):
    # Field-aware normalization avoids dictionary order / formatting differences.
    values = visible(case)
    return hashlib.sha256(dump({k: normalize(v) for k, v in values.items()}).encode()).hexdigest()


def source_id(raw_id):
    match = re.match(r'^(\d+)(?:_|$)', raw_id)
    return match.group(1) if match else raw_id


def shingles(case):
    value = normalize(clinical_text(case))
    return {value[i:i+5] for i in range(max(1, len(value)-4))} if value else set()


def prefix(items, threshold):
    # Shared total ordering; Jaccard prefix filtering guarantees candidates for
    # pairs with similarity >= threshold (then verified using the full sets).
    return sorted(items)[:len(items) - math.ceil(threshold * len(items)) + 1]


class NearIndex:
    def __init__(self, threshold=NEAR_THRESHOLD):
        self.threshold = threshold
        self.sets = {}
        self.postings = defaultdict(set)

    def add(self, key, items):
        self.sets[key] = items
        for item in prefix(items, self.threshold) if items else ():
            self.postings[item].add(key)

    def match(self, items):
        candidates = set()
        for item in prefix(items, self.threshold) if items else ():
            candidates.update(self.postings.get(item, ()))
        for key in sorted(candidates):
            other = self.sets[key]
            if min(len(items), len(other)) / max(len(items), len(other)) < self.threshold:
                continue
            similarity = len(items & other) / len(items | other)
            if similarity >= self.threshold:
                return key, similarity
        return None


def tokens(value):
    # Reproducible Chinese bigrams + Latin/numeric terms, no model downloads.
    result = []
    for item in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+',
                           unicodedata.normalize('NFKC', value).lower()):
        if re.fullmatch(r'[\u4e00-\u9fff]+', item):
            result.extend(item[i:i+2] for i in range(max(1, len(item)-1)))
        else:
            result.append(item)
    return result


def build_corpus(source, heldout, output, threshold=NEAR_THRESHOLD):
    source, heldout, output = Path(source), Path(heldout), Path(output)
    if not 0 < threshold <= 1:
        raise ValueError('Require 0 < near-duplicate threshold <= 1')
    if output.exists():
        raise FileExistsError('Frozen corpus directory already exists: ' + str(output))
    full = json.loads(source.read_text())
    test = json.loads(heldout.read_text())
    held_ids = {source_id(k) for k in test}
    held_fps = {fingerprint(v): opaque_id(k) for k, v in test.items()}
    held_near = NearIndex(threshold)
    for key, case in test.items():
        held_near.add(opaque_id(key), shingles(case))
    kept_near = NearIndex(threshold)
    seen_fps, seen_sources = {}, set()
    counts, exclusions, docs = Counter(), [], []
    for key, case in full.items():
        cid, sid, fp = opaque_id(key), source_id(key), fingerprint(case)
        reason, match, similarity = None, None, None
        items = None
        if key in test:
            reason = 'heldout_id'
        elif sid in held_ids:
            reason = 'heldout_source_id'
        elif fp in held_fps:
            reason, match = 'heldout_exact_clinical_text', held_fps[fp]
        else:
            items = shingles(case)
            near = held_near.match(items)
            if near:
                reason, (match, similarity) = 'heldout_near_clinical_text', near
        if not reason:
            if not tokens(clinical_text(case)):
                reason = 'empty_clinical_text'
            elif sid in seen_sources:
                reason = 'corpus_same_source_id'
            elif fp in seen_fps:
                reason, match = 'corpus_exact_clinical_text', seen_fps[fp]
            else:
                near = kept_near.match(items)
                if near:
                    reason, (match, similarity) = 'corpus_near_clinical_text', near
        if reason:
            counts[reason] += 1
            exclusions.append({'case_id': cid, 'reason': reason, 'match_id': match,
                               'jaccard': similarity})
            continue
        intro = case.get('【病案介绍】') or {}
        exams = intro.get('查体') or {}
        exams = exams if isinstance(exams, dict) else {}
        departments = (case.get('tags') or {}).get('科室') or []
        diagnosis = (case.get('【诊治过程】') or {}).get('诊断结果')
        diagnosis = diagnosis or (case.get('tags') or {}).get('病种') or []
        docs.append({'case_id': cid, 'clinical_fingerprint': fp,
            'search_text': clinical_text(case),
            'department': departments[0] if departments else '',
            'row': {'主诉': text(intro.get('主诉')), '症状': text(intro.get('现病史')),
                    '既往史': text(intro.get('既往史')), '体格检查': exams.get('体格检查') or {},
                    '辅助检查': exams.get('辅助检查') or {}, '诊断结果': text(diagnosis),
                    '治疗项目': text(case.get('【治疗项目】')),
                    '一级科室': departments[0] if departments else '',
                    '二级科室': ','.join(departments[1:])}})
        seen_fps[fp] = cid
        seen_sources.add(sid)
        kept_near.add(cid, items)
    if len(docs) < 3:
        raise ValueError('Fewer than three clean corpus cases')
    output.mkdir(parents=True)
    (output / 'corpus.json').write_text(dump(docs) + '\n')
    (output / 'exclusions.json').write_text(dump(exclusions) + '\n')
    manifest = {'version': VERSION, 'source_sha256': sha(source), 'heldout_sha256': sha(heldout),
        'source_count': len(full), 'heldout_count': len(test), 'corpus_count': len(docs),
        'exclusion_counts': dict(counts), 'near_duplicate_threshold': threshold,
        'near_duplicate_metric': 'Jaccard of normalized clinical-text character 5-gram sets',
        'dedup_scope': 'heldout and internal corpus; source numeric IDs and clinical fields only',
        'search_fields': ['主诉', '现病史', '既往史', '查体'],
        'no_diagnosis_or_treatment_or_title_in_index': True,
        'not_verified_official_training_split': True,
        'tokenizer': 'Chinese bigrams plus Latin/numeric terms; NFKC lowercase',
        'bm25': {'k1': 1.5, 'b': 0.75, 'idf': 'log(1+(N-df+0.5)/(df+0.5))'},
        'corpus_sha256': sha(output / 'corpus.json'),
        'exclusions_sha256': sha(output / 'exclusions.json')}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def display_row(row):
    """Bound historical examples only; preserve exam keys and valid dict syntax."""
    limits = {'主诉': 400, '症状': 1000, '既往史': 600, '诊断结果': 400, '治疗项目': 300,
              '一级科室': 80, '二级科室': 200}
    result, truncated = {}, []
    for key, limit in limits.items():
        value = text(row.get(key))
        if len(value) > limit:
            value = value[:limit] + '〔历史字段截短〕'
            truncated.append(key)
        result[key] = value
    for key in ('体格检查', '辅助检查'):
        value = row.get(key) or {}
        if not isinstance(value, dict):
            raise ValueError('Historical examination must be a dictionary')
        per_value = max(0, (1200 - sum(len(str(k)) + 8 for k in value)) // max(1, len(value)))
        bounded = {}
        for name, raw in value.items():
            item = text(raw)
            if len(item) > per_value:
                item = item[:per_value] + '〔历史字段截短〕'
                if key not in truncated:
                    truncated.append(key)
            bounded[name] = item
        result[key] = repr(bounded)
    return result, truncated


class CaseRetriever:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        for name, field in [('corpus.json', 'corpus_sha256'), ('exclusions.json', 'exclusions_sha256')]:
            if sha(self.directory / name) != self.manifest[field]:
                raise ValueError('Frozen corpus hash mismatch: ' + name)
        self.docs = json.loads((self.directory / 'corpus.json').read_text())
        self.postings = defaultdict(list)
        self.lengths = []
        self.rows = defaultdict(list)
        self.locations = {}
        for idx, doc in enumerate(self.docs):
            terms = Counter(tokens(doc['search_text']))
            self.lengths.append(sum(terms.values()))
            for term, count in terms.items():
                self.postings[term].append((idx, count))
            department = doc['department'] or '未分类'
            row, truncated = display_row(doc['row'])
            self.locations[doc['case_id']] = {
                'sheet': department, 'row_index': len(self.rows[department]),
                'truncated_history_fields': truncated}
            self.rows[department].append(row)
        self.avg_length = sum(self.lengths) / max(1, len(self.docs))

    def search(self, query, department=None, k=3):
        if k <= 0:
            raise ValueError('k must be positive')
        scores = defaultdict(float)
        for term in set(tokens(query)):
            postings = self.postings.get(term, ())
            idf = math.log(1 + (len(self.docs) - len(postings) + .5) / (len(postings) + .5))
            for idx, tf in postings:
                if department is not None and self.docs[idx]['department'] != department:
                    continue
                denom = tf + 1.5 * (1 - .75 + .75 * self.lengths[idx] / self.avg_length)
                scores[idx] += idf * tf * 2.5 / denom
        ranked = sorted(scores, key=lambda idx: (-scores[idx], self.docs[idx]['case_id']))[:k]
        return {'query': query, 'department_filter': department, 'requested_k': k,
                'returned_k': len(ranked), 'backend': VERSION,
                'hits': [{'case_id': self.docs[i]['case_id'], 'score': round(scores[i], 8),
                          **self.locations[self.docs[i]['case_id']]} for i in ranked]}

    def export_workbook(self, path):
        import pandas as pd
        path = Path(path)
        if path.exists():
            raise FileExistsError('Refuse to overwrite corpus workbook')
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for department, rows in sorted(self.rows.items()):
                pd.DataFrame(rows).to_excel(writer, sheet_name=department, index=False)

    def triage_examples(self, retrieval):
        first, second = [], []
        for rank, hit in enumerate(retrieval['hits'], 1):
            row = self.rows[hit['sheet']][hit['row_index']]
            common = f'参考病例{rank}\n主诉：{row["主诉"]}\n症状：{row["症状"]}\n'
            first.append(common + '一级科室：' + row['一级科室'])
            second.append(common + '二级科室：' + row['二级科室'])
        return ['\n\n'.join(first), '\n\n'.join(second)]
