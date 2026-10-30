"""Loopback-only local BGE-M3 encoder for live symptom queries; CPU, no downloads."""
import argparse
import json
from pathlib import Path
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=23830)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModel
    from fastapi import FastAPI, HTTPException
    import uvicorn
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False).eval()
    info = json.loads((args.model_dir / 'download_manifest.json').read_text())
    app = FastAPI()
    lock = threading.Lock()

    @app.get('/metadata')
    def metadata():
        return dict(info, pooling='CLS + L2 normalization', dimensions=model.config.hidden_size)

    @app.post('/embed')
    def embed(body: dict):
        values = body.get('texts')
        if (not isinstance(values, list) or not 0 < len(values) <= 16
                or not all(isinstance(v, str) and v.strip() for v in values)):
            raise HTTPException(400, 'Require 1..16 nonempty texts')
        with lock:
            inputs = tokenizer(values, padding=True, truncation=False, return_tensors='pt')
            if inputs['input_ids'].shape[1] > 8192:
                raise HTTPException(400, 'Input too long; refusing to truncate')
            with torch.inference_mode():
                vectors = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0], p=2, dim=1)
        return {'model': info['model'], 'revision': info['revision'], 'vectors': vectors.tolist()}

    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
