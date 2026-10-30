"""Native Spotiflow objectives with epoch-deterministic recovery and snapshots."""
from contextlib import contextmanager
from pathlib import Path
import random
import time
from types import MethodType

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from spotiflow.model.config import SpotiflowTrainingConfig
from spotiflow.model.trainer import SpotiflowModelCheckpoint, SpotiflowTrainingWrapper, collate_spots
from deepblink_torch.experiment_log import sha256, write_json
from deepblink_torch.spotiflow_bridge import evaluate_checkpoints, save_candidates, spotiflow_candidates, write_csv


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        set_rng(state)


def sample_seed(seed, epoch, index, validation=False):
    return (seed*1000003 + epoch*1009 + index*17 + int(validation)*100000007) % (2**31-1)


class DeterministicSamples(Dataset):
    """Keep native transforms but key augmentation to seed, epoch and image."""
    def __init__(self, native, seed, epoch, validation=False):
        self.native, self.seed, self.epoch, self.validation = native, seed, epoch, validation
        self._sigma = native._sigma

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        value = sample_seed(self.seed, self.epoch(), index, self.validation)
        with preserve_rng():
            random.seed(value)
            np.random.seed(value)
            torch.manual_seed(value)
            return self.native[index]


class EpochSampler(Sampler):
    def __init__(self, data, seed, epoch):
        self.data, self.seed, self.epoch = data, seed, epoch

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 10000*self.epoch())
        return iter(torch.randperm(len(self.data), generator=generator).tolist())


class SnapshotCallback(SpotiflowModelCheckpoint):
    def __init__(self, folder, config, seed, period=5, evaluate=None, event=None, stop_after=None):
        super().__init__(folder, config)
        self.folder, self.config, self.seed = Path(folder), config, seed
        self.period, self.evaluate, self.event, self.stop_after = period, evaluate, event, stop_after
        self.history, self.snapshots, self.batch_seconds = [], [], []
        self.batch_wall_seconds = []
        self.restored_rng, self.loader_states = None, {}
        self.generators = {}

    @property
    def state_key(self):
        return "SpotiflowOvernightSnapshotV1"

    def state_dict(self):
        return dict(history=self.history, snapshots=self.snapshots, rng=rng_state(), seed=self.seed,
                    period=self.period, loader_states={k:g.get_state() for k,g in self.generators.items()},
                    augmentation_rng="keyed_by_seed_epoch_image", epoch_boundary_resume=True)

    def load_state_dict(self, state):
        if state["seed"] != self.seed or state["period"] != self.period:
            raise ValueError("Resume seed/snapshot protocol differs.")
        self.history, self.snapshots = state["history"], state["snapshots"]
        self.restored_rng, self.loader_states = state["rng"], state["loader_states"]

    def on_fit_start(self, trainer, module):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.config.save(self.folder / "train_config.yaml")
        if self.restored_rng is not None:
            set_rng(self.restored_rng)
            for key,state in self.loader_states.items():
                self.generators[key].set_state(state.cpu())

    def on_train_epoch_start(self, trainer, module):
        if module.device.type=="cuda":torch.cuda.synchronize()
        self.tick = time.perf_counter()
        self.last_batch_end = self.tick
        self.train_wall_seconds = None

    def on_validation_epoch_start(self, trainer, module):
        self.train_wall_seconds = time.perf_counter()-self.tick

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if module.device.type=="cuda":torch.cuda.synchronize()
        self.batch_tick = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if module.device.type=="cuda":torch.cuda.synchronize()
        self.batch_seconds.append(time.perf_counter()-self.batch_tick)
        stamp = time.perf_counter()
        self.batch_wall_seconds.append(stamp-self.last_batch_end)
        self.last_batch_end = stamp

    def on_validation_end(self, trainer, module):
        pass  # Current-epoch snapshots must not copy historical native-best.

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch+1
        metrics = {k:float(trainer.callback_metrics[k]) for k in ("train_loss", "val_loss")}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite native training/validation loss.")
        row = dict(epoch=epoch, **metrics, seconds=time.perf_counter()-self.tick,
                   optimizer_updates=trainer.global_step, learning_rate=trainer.optimizers[0].param_groups[0]["lr"])
        row["train_wall_seconds"] = self.train_wall_seconds
        row["native_validation_wall_seconds"] = row["seconds"]-self.train_wall_seconds
        if epoch % self.period == 0:
            snapshot = self.folder / f"epoch_{epoch:03d}"
            if snapshot.exists():
                raise FileExistsError("Epoch snapshot already exists: "+str(snapshot))
            with preserve_rng():
                module.model.save(snapshot, which="best", update_thresholds=True)
                record = dict(epoch=epoch, native_val_loss=metrics["val_loss"], weight_sha256=sha256(snapshot/"best.pt"))
                if self.evaluate:
                    was_training = module.training
                    try:
                        record.update(self.evaluate(module.model, snapshot, epoch))
                    finally:
                        module.train(was_training)
            self.snapshots.append(record)
            write_json(snapshot/"snapshot.json", record)
        row["wall_seconds_with_snapshot"] = time.perf_counter()-self.tick
        self.history.append(row)
        write_csv(self.folder/"training_metrics.csv", self.history, list(self.history[0]))
        write_json(self.folder/"snapshots.json", self.snapshots)
        write_json(self.folder/"progress.json", dict(seed=self.seed, **row, checkpoints=len(self.snapshots), max_epochs=self.config.num_epochs))
        if self.event:
            self.event("epoch_completed", seed=self.seed, **row, checkpoints=len(self.snapshots))
        if self.stop_after is not None and epoch >= self.stop_after:
            trainer.should_stop = True

    def on_train_end(self, trainer, module):
        pass


def default_training(epochs=100, batch_size=4, crop_size=256):
    return SpotiflowTrainingConfig(crop_size=crop_size, smart_crop=True, heatmap_loss_f="bce", flow_loss_f="l1",
        loss_levels=None, num_train_samples=None, pos_weight=10., lr=.0003, optimizer="adamw", batch_size=batch_size,
        lr_reduce_patience=10, num_epochs=epochs, finetuned_from="general", early_stopping_patience=0, optimize_threshold=False)


def train_native(model, train_x, train_y, val_x, val_y, folder, *, seed=42, config=None, period=5,
                 evaluate=None, event=None, resume=None, stop_after=None, augment=True, max_train_batches=None, device="cpu"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = default_training() if config is None else config
    if config.num_train_samples is not None:
        raise ValueError("Protocol requires no-replacement epoch traversal.")
    pl.seed_everything(seed, workers=True)
    callback = SnapshotCallback(folder, config, seed, period, evaluate, event, stop_after)
    holder = {}

    def fit_dataset(self, train_ds, val_ds, train_config, **kwargs):
        if not train_ds._sigma == val_ds._sigma == self.config.sigma:
            raise ValueError("Native target/model sigma differs.")
        checkpoint = ModelCheckpoint(dirpath=folder/"recovery", save_top_k=0, save_last=True,
                                      every_n_epochs=1, save_on_train_epoch_end=True)
        trainer = pl.Trainer(accelerator=device, devices=1, precision="32-true", callbacks=[callback, checkpoint],
            logger=kwargs["logger"], max_epochs=train_config.num_epochs, deterministic=True, benchmark=False,
            num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1,
            default_root_dir=str(folder), limit_train_batches=1. if max_train_batches is None else max_train_batches)
        epoch = lambda:trainer.current_epoch
        train, valid = DeterministicSamples(train_ds, seed, epoch), DeterministicSamples(val_ds, seed, epoch, True)
        callback.generators = {"train":torch.Generator().manual_seed(seed+200000), "valid":torch.Generator().manual_seed(seed+300000)}
        train_dl = DataLoader(train, batch_size=train_config.batch_size, sampler=EpochSampler(train, seed, epoch),
                              num_workers=0, generator=callback.generators["train"], collate_fn=collate_spots)
        valid_dl = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0,
                              generator=callback.generators["valid"], collate_fn=collate_spots)
        wrapper = SpotiflowTrainingWrapper(self, train_config)
        holder["trainer"] = trainer
        write_json(folder/"resolved_pipeline.json", dict(seed=seed, crop_size=train_config.crop_size,
            min_train_shape=min(min(i.shape) for i in train_x), training_images=len(train), validation_images=len(valid),
            train_batches=len(train_dl), num_workers=0, native_model_config=vars(self.config), train_config=vars(train_config),
            augmentation=repr(train_ds.augmenter), validation=repr(val_ds.augmenter), precision="32-true", device=device,
            sampler="independent_epoch_generator_without_replacement", augmentation_rng="seed_epoch_image_keyed", test_arrays_loaded=False))
        trainer.fit(wrapper, train_dl, valid_dl, ckpt_path=str(resume) if resume else None, weights_only=False)

    model.fit_dataset = MethodType(fit_dataset, model)
    model.fit(train_x, train_y, val_x, val_y, augment_train=augment, save_dir=folder, train_config=config,
              device=device, logger="tensorboard", callbacks=[callback], num_workers=0)
    trainer = holder["trainer"]
    write_json(folder/"optimizer_summary.json", dict(optimizer=type(trainer.optimizers[0]).__name__,
        defaults={k:v for k,v in trainer.optimizers[0].defaults.items() if isinstance(v,(str,int,float,bool,tuple,type(None)))},
        scheduler=[type(x.scheduler).__name__ for x in trainer.lr_scheduler_configs],
        completed_epochs=len(callback.history), optimizer_updates=trainer.global_step))
    return callback


def evaluate_snapshot(data, device="cpu"):
    def evaluate(model, snapshot, epoch):
        tick = time.perf_counter()
        candidates = [spotiflow_candidates(model, image, device) for image in data["images"]]
        inference_seconds = time.perf_counter()-tick
        save_candidates(snapshot/"candidates.npz", candidates, data["sample_id"], data["original_shape"])
        rows, selected, per_image = evaluate_checkpoints([(epoch,candidates)], data)
        write_csv(snapshot/"threshold_metrics.csv", rows, list(rows[0]))
        write_json(snapshot/"threshold_metrics.json", rows)
        write_csv(snapshot/"per_image_selected.csv", per_image, list(per_image[0]))
        return dict(best_at_epoch=selected, fullimage_inference_seconds=inference_seconds,
                    fullimage_evaluation_seconds=time.perf_counter()-tick, candidate_sha256=sha256(snapshot/"candidates.npz"))
    return evaluate
