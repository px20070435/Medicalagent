"""Fetch a small, versioned CDC guidance corpus and build local BGE-M3 indexes.

No current test diagnosis or treatment is used to select or index guidance.
The CDC guidance library is an initial coverage-limited resource, not AGRAG.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ('cdc_sti_2021', 'Sexually Transmitted Infections Treatment Guidelines, 2021',
     'https://www.cdc.gov/mmwr/volumes/70/rr/rr7004a1.htm', '2021', 'formal guideline'),
    ('cdc_opioid_2022', 'CDC Clinical Practice Guideline for Prescribing Opioids for Pain, 2022',
     'https://www.cdc.gov/mmwr/volumes/71/rr/rr7103a1.htm', '2022', 'formal guideline'),
    ('cdc_adult_outpatient', 'Outpatient Clinical Care for Adults: Antibiotic Prescribing',
     'https://www.cdc.gov/antibiotic-use/hcp/clinical-care/adult-outpatient.html', None, 'clinical guidance'),
    ('cdc_pediatric_outpatient', 'Outpatient Clinical Care for Pediatric Populations: Antibiotic Prescribing',
     'https://www.cdc.gov/antibiotic-use/hcp/clinical-care/pediatric-outpatient.html', None, 'clinical guidance'),
    ('cdc_tb_diagnosis', 'Clinical and Laboratory Diagnosis for Tuberculosis',
     'https://www.cdc.gov/tb/hcp/testing-diagnosis/clinical-and-laboratory-diagnosis.html', None, 'clinical guidance'),
    ('cdc_tb_treatment', 'Treatment for Drug-Susceptible Tuberculosis Disease',
     'https://www.cdc.gov/tb/hcp/treatment/tuberculosis-disease.html', None, 'clinical guidance'),
    ('cdc_cdiff', 'Clinical Guidance for C. diff Infection Prevention in Acute Care Facilities',
     'https://www.cdc.gov/c-diff/hcp/clinical-guidance/index.html', None, 'clinical guidance'),
    ('cdc_lyme', 'Clinical Care of Lyme Disease',
     'https://www.cdc.gov/lyme/hcp/clinical-care/index.html', None, 'clinical guidance'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class ClinicalHTML(HTMLParser):
    """Extract body blocks under main, excluding navigation/scripts/references."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.blocks = []
        self.text = []
        self.section = ''
        self.main = False
        self.stopped = False

    def flush(self, heading=False):
        value = re.sub(r'\s+', ' ', ''.join(self.text)).strip()
        self.text = []
        if value:
            if heading:
                self.section = value
                if value.lower() in ('references', 'acknowledgments', 'acknowledgements'):
                    self.stopped = True
            if not self.stopped:
                self.blocks.append((self.section, value))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('br', 'hr', 'img', 'meta', 'input', 'link', 'source', 'wbr'):
            if tag == 'br':
                self.text.append('\n')
            return
        if tag == 'main':
            self.main = True
        if tag in ('td', 'th'):
            self.text.append(' | ')
        if self.main and tag in ('p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'tr'):
            self.flush()
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.main and tag in ('p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'tr'):
            self.flush(heading=tag.startswith('h') and tag != 'hr')
        if tag == 'main':
            self.flush()
            self.main = False
        if tag in self.stack:
            self.stack = self.stack[:len(self.stack) - self.stack[::-1].index(tag) - 1]

    def handle_data(self, data):
        if self.main and not self.stopped and not any(t in self.stack for t in ('script', 'style', 'nav', 'footer')):
            self.text.append(data)


def chunks_for(doc, blocks):
    """Pack complete paragraphs within a heading; avoid separating conditions.

    Long paragraphs/tables remain intact (hard limit checked by tokenizer).
    """
    result, pending, section = [], [], ''
    def emit():
        if not pending:
            return
        body = '\n'.join(pending)
        if len(body) >= 120:
            result.append(dict(document_id=doc['document_id'], title=doc['title'],
                section=section, url=doc['url'], year=doc['year'], issuer='CDC',
                document_kind=doc['document_kind'], text=body,
                chunk_id=f"{doc['document_id']}_{len(result):04d}"))
        pending.clear()
    for heading, text in blocks:
        if heading.lower().strip() in ('on this page', 'key points', 'conflicts of interest',
                'disclaimer', 'acknowledgments', 'acknowledgements', 'references',
                'author affiliations', 'suggested citation'):
            continue
        if len(text) < 25 or text.lower() in ('on this page', 'at a glance'):
            continue
        if pending and (heading != section or sum(map(len, pending)) + len(text) > 1800):
            emit()
        section = heading
        pending.append(text)
    emit()
    return result


def fetch(directory):
    import requests
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / 'raw'
    raw.mkdir()
    docs, chunks, failures = [], [], []
    for docid, title, url, year, kind in SOURCES:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            response.encoding = 'utf-8'
            if '<main' not in response.text.lower():
                raise ValueError('No clinical main element')
            filename = raw / f'{docid}.html'
            filename.write_bytes(response.content)
            parser = ClinicalHTML()
            parser.feed(response.text)
            parser.flush()
            meta = dict(document_id=docid, title=title, url=url, resolved_url=response.url,
                year=year, document_kind=kind, issuer='CDC', raw_sha256=sha(filename),
                fetched_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                rights='US government CDC text; external/third-party media excluded; retain attribution',
                revision_policy='URL + raw file SHA256 + retrieval timestamp; no inferred publication year')
            new = chunks_for(meta, parser.blocks)
            if not new:
                raise ValueError('Empty clinical extraction')
            docs.append(dict(meta, chunks=len(new)))
            chunks.extend(new)
            print(json.dumps({'document': docid, 'chunks': len(new)}), flush=True)
        except Exception as error:
            failures.append({'document_id': docid, 'url': url, 'error': str(error)})
    if len(docs) < 4:
        raise RuntimeError('Insufficient actual guidance sources; inspect raw downloads')
    save(directory / 'guidelines.json', chunks)
    save(directory / 'sources.json', docs)
    save(directory / 'manifest.json', dict(sources=len(docs), chunks=len(chunks),
        failures=failures, corpus_kind='initial English CDC formal guidelines and clinical guidance',
        full_specialty_coverage=False, original_clinicalagents_corpus=False,
        guideline_sha256=sha(directory / 'guidelines.json'),
        attribution='Centers for Disease Control and Prevention; titles and source URLs retained per chunk',
        selection='Fixed source list, independent of test reference labels',
        chunk_policy='Section-aware full paragraphs, soft target 1800 characters; no synthesized recommendations'))


def build(directory, guidance):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    model_dir = ROOT / 'data/models/bge-m3'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().cuda()
    if directory.exists():
        raise ValueError('Use a new index directory')
    directory.mkdir(parents=True)
    source = ROOT / 'data/benchmarks/medchain-rag-bm25-v1-20261028'
    manifest = json.loads((source / 'manifest.json').read_text())
    if sha(source / 'corpus.json') != manifest['corpus_sha256']:
        raise ValueError('Historical corpus hash mismatch')
    if sha(ROOT / 'datasets/filtered_data_test_set.json') != manifest['heldout_sha256']:
        raise ValueError('Different heldout test file')
    docs = json.loads((source / 'corpus.json').read_text())
    guidelines = json.loads((guidance / 'guidelines.json').read_text())
    # Clinical evidence only is embedded. Historical labels remain on disk,
    # are not included in retrieval inputs or missing-evidence suggestions.
    labels = {'诊断结果', '治疗项目', '一级科室', '二级科室'}
    cases = []
    for doc in docs:
        evidence = {k: v for k, v in doc['row'].items() if k not in labels and v}
        cases.append({'case_id': doc['case_id'], 'evidence': evidence})
    texts = [json.dumps(c['evidence'], ensure_ascii=False) for c in cases]
    guide_texts = [g['title'] + '\n' + g['section'] + '\n' + g['text'] for g in guidelines]
    stats = {'max_tokens': 0, 'truncated_inputs': 0, 'batch_size': 4}
    def encode(values, name):
        result = np.empty((len(values), 1024), dtype=np.float32)
        order = sorted(range(len(values)), key=lambda i: len(values[i]))
        for start in range(0, len(order), 4):
            ids = order[start:start + 4]
            encoded = tokenizer([values[i] for i in ids], padding=True, truncation=False, return_tensors='pt')
            length = encoded['input_ids'].shape[1]
            stats['max_tokens'] = max(length, stats['max_tokens'])
            if length > 8192:
                raise ValueError(f'{name} input exceeds BGE-M3 limit; no silent truncation')
            with torch.inference_mode():
                vector = torch.nn.functional.normalize(model(**{k: v.cuda() for k, v in encoded.items()}).last_hidden_state[:, 0], p=2, dim=1)
            result[ids] = vector.float().cpu().numpy()
            if start % 200 == 0:
                print(json.dumps({'group': name, 'encoded': start + len(ids), 'total': len(values)}), flush=True)
        return result
    np.save(directory / 'guideline_vectors.npy', encode(guide_texts, 'guidelines'), allow_pickle=False)
    np.save(directory / 'case_vectors.npy', encode(texts, 'cases'), allow_pickle=False)
    save(directory / 'guidelines.json', guidelines)
    save(directory / 'cases.json', cases)
    save(directory / 'manifest.json', dict(embedding=json.loads((model_dir / 'download_manifest.json').read_text()),
        dimensions=1024, pooling='CLS + L2 normalization', encoding_stats=stats,
        historical_cases=len(cases), guideline_chunks=len(guidelines),
        historical_manifest_sha256=sha(source / 'manifest.json'),
        guidance_manifest_sha256=sha(guidance / 'manifest.json'),
        heldout_sha256=sha(ROOT / 'datasets/filtered_data_test_set.json'),
        indexed_case_fields='clinical evidence only, no historical diagnosis/treatment labels',
        pseudo_case_writeback=False, static_memory=True,
        files_sha256={p.name: sha(p) for p in directory.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fetch', 'index'])
    parser.add_argument('--guidance', type=Path, required=True)
    parser.add_argument('--index', type=Path)
    args = parser.parse_args()
    if args.action == 'fetch':
        fetch(args.guidance)
    else:
        build(args.index, args.guidance)
