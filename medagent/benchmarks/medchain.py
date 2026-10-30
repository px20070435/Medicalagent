from copy import deepcopy
import hashlib
import json
from pathlib import Path

from medagent.core.contracts import CaseInput, Example


def opaque_id(raw_id):
    # Source keys often contain the diagnosis! Never use them as model inputs.
    return 'mc_' + hashlib.sha256(raw_id.encode()).hexdigest()[:20]


def legacy_sample(case):
    """Reconstruct ONLY whitelisted prediction input, never labels/reports."""
    payload = case.payload
    intro = {key: deepcopy(payload.get(key, [])) for key in ('主诉', '现病史', '既往史')}
    intro['查体'] = {'辅助检查': deepcopy(payload.get('辅助检查', '无'))}
    intro['图像'] = [{'文件名': image['path'], '分类': [image['category']]} for image in payload['images']]
    return case.case_id, {'【病案介绍】': intro}


class MedChain:
    schema = 'clinical_case.legacy_aux.v1'

    def load(self, dataset, image_root):
        cases = json.loads(Path(dataset).read_text(encoding='utf-8'))
        result = []
        for raw_id, raw in cases.items():
            intro = raw.get('【病案介绍】', {})
            exams = intro.get('查体') or {}
            images = [{'path': str((Path(image_root) / item['文件名']).resolve(strict=True)),
                       'category': item['分类'][0]} for item in (intro.get('图像') or [])]
            visible = {key: deepcopy(intro.get(key, ['None'])) for key in ('主诉', '现病史', '既往史')}
            visible.update(images=images, 辅助检查=deepcopy(exams.get('辅助检查', '无')) if isinstance(exams, dict) else '无')
            # This preserves legacy oracle auxiliary findings. Interactive methods
            # must use a gated schema, not silently inherit this information.
            case = CaseInput(opaque_id(raw_id), self.schema, visible)
            reference = {key: deepcopy(raw.get(key)) for key in ('tags', '【治疗项目】', '【诊治过程】')}
            # The frozen scorer expects mappings. Missing diagnosis metadata must
            # reach its existing tags/病种 fallback, not become an explicit None.
            for key in ('tags', '【诊治过程】'):
                if reference[key] is None:
                    reference[key] = {}
                elif not isinstance(reference[key], dict):
                    raise ValueError(f'{case.case_id}: reference {key} must be an object or null')
            reference['【病案介绍】'] = {key: deepcopy(intro.get(key)) for key in ('查体', '影像报告', '图像')}
            if not images:
                stratum = 'no_image'
            elif not intro.get('影像报告'):
                stratum = 'image_no_reference'
            elif len(images) == 1:
                stratum = 'single_image_reference'
            else:
                stratum = 'multi_image_reference'
            result.append(Example(case, reference, stratum))
        return result

    def evaluate(self, example, records, run, judge):
        # Shared scoring adapter; baseline implementation is never the judge.
        from medagent.evaluation.medchain import evaluate
        evaluate(example, records, run, judge)
