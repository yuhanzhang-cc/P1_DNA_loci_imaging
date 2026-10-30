"""Representative CPU update/validation measurement; never choose by accuracy."""
import json
from pathlib import Path
import statistics
import time

import numpy as np
from spotiflow.model import Spotiflow
from .engine import default_training, train_native
from .common import OUT, WEIGHTS, dump, status


def measure(data):
    path=OUT/"00_engineering/fullsize_cpu_pilot"
    if (path/"runtime.json").exists():
        return json.loads((path/"runtime.json").read_text())
    path.mkdir(parents=True,exist_ok=False)
    train,valid=data["train"],data["valid"]
    selected=[int(i) for size in (256,512) for i in np.flatnonzero(train["original_shape"][:,0]==size)[:16]]
    if len(selected)!=32:raise ValueError("Pilot needs 16 training images per geometry.")
    status("runtime_pilot",activity="8 native batch=4 CPU updates, mixed original geometries, full native validation")
    model=Spotiflow.from_folder(WEIGHTS,which="best",inference_mode=False)
    tick=time.perf_counter()
    callback=train_native(model,[train["images"][i] for i in selected],[train["y"][i] for i in selected],
                          valid["images"],list(valid["y"]),path,config=default_training(epochs=1),period=5)
    fit_seconds=time.perf_counter()-tick
    update_seconds=statistics.median(callback.batch_wall_seconds[2:])
    native_val=callback.history[0]["native_validation_wall_seconds"]
    z1=json.loads((OUT/"01_pretrained/Z1_validation_calibrated/selection.json").read_text())
    snapshot_eval=z1["total_seconds"]
    seed_seconds=100*(44*update_seconds+native_val+.5)+20*(snapshot_eval+.5)+30
    estimate=3*seed_seconds
    record=dict(batch_size=4,actual_crop_size=256,original_training_geometries=[256,512],
        discarded_engineering_updates=8,measured_train_images=32,native_validation_images=22,
        median_update_seconds_after_two_warmups=update_seconds,
        individual_update_wall_seconds=callback.batch_wall_seconds,
        native_validation_seconds=native_val,fullimage_snapshot_evaluation_seconds=snapshot_eval,
        native_fit_including_startup_seconds=fit_seconds,expected_updates_per_formal_epoch=44,
        estimated_seconds_per_seed=seed_seconds,estimated_three_seed_hours=estimate/3600,
        estimated_three_seed_hours_with_25pct_margin=1.25*estimate/3600,
        cpu_window_gate_hours=5.5,run_three_seeds_on_cpu=(1.25*estimate/3600)<=5.5,
        selection_used_accuracy=False,pilot_weights_discarded=True,test_arrays_loaded=False)
    dump(path/"runtime.json",record)
    return record
