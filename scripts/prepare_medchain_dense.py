"""Build local-only BGE-M3 vectors; run using the existing internvl2 environment.

Loads an already downloaded, pinned model with local_files_only=True. This does
not contact an inference API, download a model, or modify any Python environment.
The original benchmark and historical corpus are separate inputs.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medagent.benchmarks.case_rag import sha
from medagent.benchmarks.dense_case_rag import query_key
from medagent.benchmarks.medchain import opaque_id


def chief(raw):
    return ','.join((raw.get('【病案介绍】') or {}).get('主诉') or [])


def record12(raw):
    intro = raw.get('【病案介绍】') or {}
    exams = intro.get('查体') or {}
    exams = exams if isinstance(exams, dict) else {}
    complaint = chief(raw)
    age = re.search(r'(\d+)岁', complaint)
    sex = re.search(r'(男|女)', complaint)
    tags = raw.get('tags') or {}
    departments = tags.get('科室') or []
    return {'年龄': age.group(1) + '岁' if age else None, '性别': sex.group(1) if sex else None,
        '主诉': complaint, '症状': intro.get('现病史'), '既往史': intro.get('既往史'),
        '体格检查': exams.get('体格检查'), '辅助检查': exams.get('辅助检查'),
        '影像报告': intro.get('影像报告'), '诊断结果': tags.get('病种'),
        '治疗项目': raw.get('【治疗项目】'), '一级科室': departments[0] if departments else None,
        '二级科室': departments[1:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'data/models/bge-m3')
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets/merged_cases.json')
    parser.add_argument('--queries', type=Path, default=ROOT / 'datasets/filtered_data_test_set.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
        help='cuda requires externally selecting a permitted, available GPU by UUID')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--symptom-summaries', type=Path, help='Optional frozen map of corpus IDs to <=70-character symptoms')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; use a new frozen directory')
    if args.batch_size <= 0:
        parser.error('batch size must be positive')
    corpus_manifest = json.loads((args.corpus_dir / 'manifest.json').read_text())
    if sha(args.source) != corpus_manifest['source_sha256'] or sha(args.queries) != corpus_manifest['heldout_sha256']:
        raise ValueError('Source/query input differs from frozen corpus provenance')
    if sha(args.corpus_dir / 'corpus.json') != corpus_manifest['corpus_sha256']:
        raise ValueError('Frozen corpus changed')
    model_info = json.loads((args.model_dir / 'download_manifest.json').read_text())
    if model_info['model'] != 'BAAI/bge-m3':
        raise ValueError('This encoder implementation is pinned to BAAI/bge-m3')
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    full = json.loads(args.source.read_text())
    raw_by_id = {opaque_id(k): v for k, v in full.items()}
    docs = json.loads((args.corpus_dir / 'corpus.json').read_text())
    ids = [doc['case_id'] for doc in docs]
    # Crucially, historical answer fields are stored but never encoded.
    texts = [chief(raw_by_id[cid]) for cid in ids]
    if args.symptom_summaries:
        summaries = json.loads(args.symptom_summaries.read_text())
        if set(summaries) != set(ids) or not all(isinstance(v, str) and 0 < len(v) <= 70 for v in summaries.values()):
            raise ValueError('Symptom summaries must cover the exact corpus with 1..70 characters each')
        texts = [summaries[cid] for cid in ids]
    if any(not value.strip() for value in texts):
        raise ValueError('Historical case has no chief complaint; define an explicit missing-query policy')
    raw_queries = json.loads(args.queries.read_text())
    query_texts = list(OrderedDict.fromkeys(chief(raw) for raw in raw_queries.values() if chief(raw).strip()))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model.eval().to(args.device)
    max_tokens = min(int(tokenizer.model_max_length), int(model.config.max_position_embeddings) - 2)
    stats = {'batches': 0, 'max_input_tokens': 0, 'truncated_inputs': 0}
    usage = {'corpus': len(texts), 'unique_queries': len(query_texts)}

    def encode(values):
        result = np.empty((len(values), model.config.hidden_size), dtype=np.float32)
        # Length-sorted batches reduce padding; restore original case order.
        order = sorted(range(len(values)), key=lambda i: len(values[i]))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            encoded = tokenizer([values[i] for i in indices], padding=True,
                truncation=False, return_tensors='pt')
            length = encoded['input_ids'].shape[1]
            stats['max_input_tokens'] = max(stats['max_input_tokens'], length)
            if length > max_tokens:
                raise ValueError(f'Input has {length} tokens, above {max_tokens}; no silent truncation')
            with torch.inference_mode():
                outputs = model(**{k: v.to(args.device) for k, v in encoded.items()})
                vectors = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
            result[indices] = vectors.cpu().float().numpy()
            stats['batches'] += 1
            if stats['batches'] % 50 == 0:
                print(json.dumps({'batches': stats['batches'], 'encoded_in_group': start + len(indices),
                    'group_size': len(values)}), flush=True)
        return result

    vectors = encode(texts)
    queries = encode(query_texts)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / 'case_embeddings.npy', vectors, allow_pickle=False)
    np.save(args.output_dir / 'query_embeddings.npy', queries, allow_pickle=False)
    values = {'case_ids.json': ids, 'query_keys.json': [query_key(q) for q in query_texts],
        'indexed_symptoms.json': [{'case_id': cid, 'text': value} for cid, value in zip(ids, texts)],
        'records12.json': [{'case_id': cid, 'fields': record12(raw_by_id[cid])} for cid in ids]}
    for name, value in values.items():
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False) + '\n')
    model_files = [p for p in args.model_dir.iterdir() if p.is_file() and p.name != 'download_manifest.json']
    manifest = {'version': 'medchain-local-dense-cosine-v1', 'embedding_model': model_info['model'],
        'embedding_revision': model_info['revision'],
        'embedding_files_sha256': {p.name: sha(p) for p in sorted(model_files)},
        'corpus_manifest_sha256': sha(args.corpus_dir / 'manifest.json'),
        'query_input_sha256': sha(args.queries), 'query_fields': ['主诉'],
        'query_policy': 'chief_complaint_as_released_main_py; no 70-character LLM refinement',
        'pooling': 'CLS + L2 normalization', 'query_instruction': None,
        'vector_dimensions': model.config.hidden_size, 'dtype': 'float32',
        'corpus_count': len(ids), 'query_cache': usage, 'encoding_stats': stats,
        'files_sha256': {p.name: sha(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
        'build_code_sha256': sha(Path(__file__)),
        'pseudo_data_writeback': False, 'original_ada002_reproduction': False,
        'runtime': {'torch': torch.__version__, 'device': args.device}}
    if args.symptom_summaries:
        manifest.update(query_policy='<=70-character symptoms; short chiefs preserved, long chiefs refined by InternVL2',
            symptom_summaries_sha256=sha(args.symptom_summaries))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
