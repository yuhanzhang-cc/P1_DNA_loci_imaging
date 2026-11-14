from pathlib import Path
import csv
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_external_transfer import reviewed_comparison


@pytest.mark.parametrize('method,counts,truth_equal,expected',[
    ('Spotiflow_F', True, True, True),
    ('Spotiflow_F', False, True, False),
    ('Spotiflow_F', True, False, False),
    ('H0', True, True, False),
])
def test_only_external_coordinate_drift_is_admitted(tmp_path, method, counts, truth_equal, expected):
    old = tmp_path / method / 'old'
    new = tmp_path / method / 'new'
    for folder, truth in ((old, 0), (new, 0 if truth_equal else 1)):
        folder.mkdir(parents=True)
        with (folder / 'matches.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['sample_id', 'true_index'])
            writer.writeheader()
            writer.writerow(dict(sample_id='SYN', true_index=truth))

    def original(*args):
        return dict(passed=False, thresholded_counts_exact=counts,
            per_image_prediction_counts_exact=True, max_coordinate_difference_px=.06530020841126022,
            coordinate_tolerance_px=.005)

    result = reviewed_comparison(original, old, new, ['SYN'])
    assert result['passed'] is expected
    assert result['strict_parity_passed'] is False
    assert result['coordinate_tolerance_px'] == .005
    assert result['max_coordinate_difference_px'] == .06530020841126022
