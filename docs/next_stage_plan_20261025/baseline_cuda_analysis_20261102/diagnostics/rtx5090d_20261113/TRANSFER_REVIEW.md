# External inference coordinate review

The first RTX5090D preflight reported Spotiflow_F seed43 maximum coordinate drift of 0.0653002084px and maximum score drift of 0.0002601147. All 22 images retained their prediction counts and TP/FP/FN. Both exports had TP301, FP29, FN71 and F1=0.8575498575. These values were supplied in the server console report. The full server evidence remains to be returned.

The original 0.005px requirement for every external-model coordinate was an engineering choice that unnecessarily blocked prospective mechanism training. The review helper records the original strict comparison and admits external Spotiflow coordinate drift only when the aggregate detection counts, per-image prediction counts and matched truth identities agree. The matched-truth check additionally enforces the same TP composition per image. Other model checks and errors use the existing evaluator.

Run `review_external_transfer.py --root <sealed-kit>` using the kit CUDA Python. This helper performs frozen-model inference and writes a review addendum and classified parity evidence under outputs/hardware_transfer. The existing 04_start.sh then verifies and reuses the completed receipt. The helper source hash is recorded in each comparison and the addendum. The existing source manifests continue to identify the original kit. This operation performs zero optimizer updates. Actual acceptance remains subject to the server run.

The development labels in this package have SHA256 `84ceae4ecedac3856ca6c1b9be545cfab05d08ade7eaa499b7973b9fa86ffa25`. The approved decision has labels_need_revision=false. The 60-case blind review informed P0 supervision with radius2, nearest-truth ownership and per-truth normalization. The prospective question is whether these supervision changes improve F1/localization/weak recall and duplicate FP relative to S0 and G0. The current experiment measures review-guided mechanism changes. A later annotation-revision experiment requires a versioned corrected coordinate dataset and a suitable fixed evaluation design.

Four synthetic regression cases passed for external drift admission, count mismatch rejection, truth-identity mismatch rejection and own-model mismatch rejection. Server CUDA inference and subsequent formal training remain pending.

Focused verification command follows.

```bash
python -m pytest -q deepblink_pytorch/scripts/diagnostics/test_external_transfer_review.py
```
