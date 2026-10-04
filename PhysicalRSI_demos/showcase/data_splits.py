"""Keep evaluation layouts separate from all demonstration layouts in a cycle."""
import json


def check_disjoint(training_manifests, evaluation_manifest):
    evaluation = json.loads(evaluation_manifest.read_text())['layouts']
    fields = ('seed', 'qpos', 'mouse_pose')
    def values(rows, field):
        return {tuple(row[field]) if isinstance(row[field], list) else row[field]
                for row in rows}
    for manifest in training_manifests:
        training = json.loads(manifest.read_text())['layouts']
        for field in fields:
            if values(training, field) & values(evaluation, field):
                raise ValueError(f'Training/evaluation overlap in {field}: {manifest}')
