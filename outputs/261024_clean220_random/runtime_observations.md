# Runtime observations

- During seed42, system physical memory usage was about 98%, with less than
  1 GB available. A brief slow epoch was followed by recovery to 22-24 seconds.
  No model losses became non-finite and no training settings were changed.
- During seed43 epoch36, both the host wait and training wall time showed an
  approximately 9200-second system-level pause. The process resumed, retained
  its checkpoint, and continued normally; this was not counted as extra epochs.
- A temporary Windows SetThreadExecutionState system-awake request was attached
  to the live training process. Display sleep and explicit user power actions
  are not overridden, and no permanent power settings are changed. The helper
  releases its request when the training process exits (12-hour maximum).
- Training remains 3 seeds x 100 epochs with the original frozen split and
  hyperparameters. Epoch wall times include any host pauses; they are not pure
  GPU throughput measurements.
