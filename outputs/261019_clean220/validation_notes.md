# Validation notes

- New preprocessing, masked losses, training CLI, checkpoint inference, evaluation,
  context architecture, padding compatibility and coordinate output tests:
  **29 passed** in the isolated XPU environment (`validation_tests.log`).
- The initial full legacy suite had 37 passes and 5 failures caused by missing
  historical fixtures: TF baseline CSV files and excluded `sample_1`/`sample_4`
  raw files. Those failures are not evidence of a clean220 pipeline regression.
  No removed/confidential raw files were restored to satisfy tests.
- Real-data audit: 220 paired TIFF/CSV samples; exclusions are sample_230 and
  sample_286. 5076 finite in-bounds coordinates, no exact duplicate images or
  coordinates. Original shapes and unit declarations are retained in manifests.
- Original old113 images and converted coordinates were checked against the
  historical NPZ and matched before preparing the new archive.
- XPU smoke checks exercised forward/backward and confirmed finite gradients
  in float32 and bfloat16. Training uses bfloat16 autocast, float32 losses and
  channels-last memory layout; final inference/evaluation uses float32.
- The CPU-only shared conda environment was preserved. The optional Intel GPU
  build is isolated under the ignored `.venv/clean220_xpu` directory.
- Architecture weights and loss semantics were preserved for the baseline;
  changes apply to units, fixed-canvas preparation and ignoring artificial cells.
  Historical checkpoints without input_size retain their old inference geometry.
