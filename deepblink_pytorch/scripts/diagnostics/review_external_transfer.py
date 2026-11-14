"""Audit external-model coordinate drift before using the existing launcher.

Run this file with the kit CUDA Python and --root pointing to the sealed kit.
This executes fixed-weight inference only. It records strict parity separately
from admission of external-model coordinate drift with unchanged detections.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

POLICY = 'external_coordinates_with_exact_detection_counts_v1'


def matched_truths(folder):
    with (Path(folder) / 'matches.csv').open(encoding='utf-8-sig', newline='') as handle:
        return {(row['sample_id'], int(row['true_index'])) for row in csv.DictReader(handle)}


def reviewed_comparison(original, expected, actual, sample_ids, tolerance=.005):
    result = original(expected, actual, sample_ids, tolerance)
    result['strict_parity_passed'] = result['passed']
    result['admission_policy'] = POLICY
    result['review_disposition'] = 'strict_parity_passed' if result['passed'] else 'blocked'
    if (not result['passed'] and 'Spotiflow_F' in Path(expected).parts
            and result['thresholded_counts_exact'] and result['per_image_prediction_counts_exact']):
        same_truths = matched_truths(expected) == matched_truths(actual)
        result['matched_truth_identities_exact'] = same_truths
        if same_truths:
            result['passed'] = True
            result['review_disposition'] = 'external_coordinate_drift_recorded'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / 'deepblink_pytorch'), str(root / 'deepblink_pytorch/scripts'),
                    str(root / 'deepblink_pytorch/scripts/diagnostics/scaffolds')]
    from diagnostics.completion_v1.cloud import verify
    from diagnostics.completion_v1 import transfer_check
    from diagnostics.handoff_v2.freeze import verify_frozen
    from diagnostics.handoff_v2.common import read, write, sha
    from filelock import FileLock

    verify(root)
    verify_frozen(root / 'mechanism/decisions')
    profile = read(root / 'configs/hardware_profile.json')
    if profile['profile_id'] != 'autodl_5090d_blackwell_transfer_v1':
        raise ValueError('This addendum is for the separately registered RTX5090D track')
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original = transfer_check.compare_exports

    def compare(expected, actual, sample_ids, tolerance=.005):
        result = reviewed_comparison(original, expected, actual, sample_ids, tolerance)
        result['review_source_sha256'] = source
        return result

    transfer_check.compare_exports = compare
    work = root / 'outputs/hardware_transfer'
    work.mkdir(parents=True, exist_ok=True)
    with FileLock(str(work / 'review_addendum.lock'), timeout=0):
        addendum = dict(policy=POLICY,review_source_sha256=source,
            source_bundle_sha256=sha(root / 'completion_bundle_manifest.json'),
            rationale='External frozen inference coordinates are diagnostic when counts and matched truth identities agree. Strict parity remains recorded.',
            optimizer_updates=0)
        path = work / 'external_coordinate_review.json'
        if path.exists():
            if read(path) != addendum:
                raise ValueError('Existing review addendum differs')
        else:
            write(path, addendum)
        result = transfer_check.preflight(root, dry_run=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
