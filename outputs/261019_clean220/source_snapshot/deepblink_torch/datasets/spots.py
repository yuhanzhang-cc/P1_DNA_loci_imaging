"""PyTorch dataset for deepBlink spot localization data.

This module mirrors the behavior of the original TensorFlow deepBlink
``deepblink/datasets/spots.py`` as closely as possible while exposing a
``torch.utils.data.Dataset`` interface.

Behavioral notes
----------------
- Input ``.npz`` files are expected to contain the standard deepBlink arrays:
  ``x_train``, ``y_train``, ``x_valid``, ``y_valid``, ``x_test``, ``y_test``.
  Optional pixel-size metadata arrays are loaded for traceability and ignored
  by model training.
- Images are normalized to zero mean and unit standard deviation, matching the
  original ``normalize_image`` behavior.
- Coordinate labels are converted into deepBlink prediction matrices via
  ``deepblink_torch.data_protocol.get_prediction_matrix``.
- Probability channel smoothing follows the original implementation:
  positive cells are set to ``smooth_factor`` and negative cells to
  ``1 - smooth_factor``.
- No data augmentation is performed inside this dataset. This matches the
  original design, where augmentation happens in the training pipeline rather
  than in ``SpotsDataset`` itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data_protocol import get_prediction_matrix
from ..data_protocol import next_power
from ..preprocessing import normalize_image
from ..preprocessing import prepare_image


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize one image to zero mean and unit standard deviation.

    This matches the original deepBlink implementation:

    ``image = (image - image.mean()) / image.std()``

    Parameters
    ----------
    image:
        Input image array.

    Returns
    -------
    np.ndarray
        Normalized ``float32`` image with the same spatial shape.
    """
    return normalize_image(image)


def _load_npz(fname: str | Path) -> dict[str, np.ndarray]:
    """Load the standard deepBlink dataset archive.

    Parameters
    ----------
    fname:
        Path to the input ``.npz`` file.

    Returns
    -------
    dict[str, np.ndarray]
        All arrays stored in the archive. The standard deepBlink arrays are
        required; optional metadata arrays are preserved for callers that need
        traceability but are ignored by the training target conversion.

    Raises
    ------
    ValueError
        If the archive does not contain all required arrays.
    """
    expected = ["x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test"]
    with np.load(fname, allow_pickle=True) as data:
        if not all(name in data.files for name in expected):
            raise ValueError(f"{expected} must be present. Only found {data.files}.")
        return {name: data[name] for name in data.files}


class SpotsDataset(Dataset):
    """Spot-localization dataset with deepBlink-compatible label encoding.

    Parameters
    ----------
    name:
        Path to the deepBlink ``.npz`` dataset.
    cell_size:
        Number of original-image pixels represented by one prediction cell.
    subset:
        Which split to expose. Must be one of ``'train'``, ``'valid'``,
        or ``'test'``.
    smooth_factor:
        Probability-channel smoothing factor. Positive cells are assigned
        ``smooth_factor`` and negative cells ``1 - smooth_factor``.

    Attributes
    ----------
    x:
        Selected image subset after normalization.
    y:
        Selected label subset after conversion to prediction matrices.

    Notes
    -----
    ``__getitem__`` returns:

    - ``image`` as a ``torch.float32`` tensor of shape ``(1, H, W)``
    - ``target`` as a ``torch.float32`` tensor of shape ``(nrow, ncol, 3)``

    The target shape intentionally remains channel-last to preserve the same
    semantic layout as the original deepBlink implementation.
    """

    def __init__(
        self,
        name: str | Path,
        cell_size: int,
        subset: str = "train",
        smooth_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.cell_size = cell_size
        self.smooth_factor = smooth_factor

        valid_subsets = {"train", "valid", "test"}
        if subset not in valid_subsets:
            raise ValueError(
                f"subset must be one of {sorted(valid_subsets)}, but got {subset!r}."
            )
        self.subset = subset

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None
        self.pixel_size_x_train = None
        self.pixel_size_y_train = None
        self.pixel_size_source_train = None
        self.pixel_size_x_valid = None
        self.pixel_size_y_valid = None
        self.pixel_size_source_valid = None
        self.pixel_size_x_test = None
        self.pixel_size_y_test = None
        self.pixel_size_source_test = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.pixel_size_source = None

        self.load_data()
        self.select_subset()

    @property
    def data_filename(self) -> str:
        """Return the absolute path to the dataset archive."""
        return str(Path(self.name).resolve())

    @property
    def image_size(self) -> int:
        """Return the common image side length after validating all splits.

        This follows the intent of the original implementation: all images must
        have the same shape, must be square, and the side length must be a
        power of two according to deepBlink's ``next_power`` convention.
        """
        datasets = [
            dataset
            for dataset in [self.x_train, self.x_valid, self.x_test]
            if dataset is not None and len(dataset) > 0
        ]
        if not datasets:
            raise ValueError("No image data loaded.")

        base_shape = datasets[0][0].shape
        if not all(base_shape == image.shape for dataset in datasets for image in dataset):
            raise ValueError("All images must have the same shape.")
        if base_shape[0] != base_shape[1]:
            raise ValueError("Images must be square.")
        if base_shape[0] != next_power(base_shape[0]):
            raise ValueError(
                f"Images sidelength must be a power of two. {base_shape[0]} is not."
            )
        return int(base_shape[0])

    def load_data(self) -> None:
        """Load, encode, and normalize the full dataset archive."""
        data = _load_npz(self.data_filename)
        self.x_train = data["x_train"]
        self.y_train = data["y_train"]
        self.x_valid = data["x_valid"]
        self.y_valid = data["y_valid"]
        self.x_test = data["x_test"]
        self.y_test = data["y_test"]
        self.original_shapes = {
            subset: data.get(f"original_shape_{subset}")
            for subset in ("train", "valid", "test")
        }
        self.input_size = int(data["input_size"]) if "input_size" in data else None

        self.pixel_size_x_train = data.get("pixel_size_x_train")
        self.pixel_size_y_train = data.get("pixel_size_y_train")
        self.pixel_size_source_train = data.get("pixel_size_source_train")
        self.pixel_size_x_valid = data.get("pixel_size_x_valid")
        self.pixel_size_y_valid = data.get("pixel_size_y_valid")
        self.pixel_size_source_valid = data.get("pixel_size_source_valid")
        self.pixel_size_x_test = data.get("pixel_size_x_test")
        self.pixel_size_y_test = data.get("pixel_size_y_test")
        self.pixel_size_source_test = data.get("pixel_size_source_test")

        self.prepare_data()
        self.normalize_dataset()

    def prepare_data(self) -> None:
        """Convert coordinate labels into deepBlink prediction matrices."""

        def _convert(dataset: np.ndarray, image_size: int, cell_size: int) -> np.ndarray:
            labels = []
            for coords in dataset:
                matrix = get_prediction_matrix(coords, image_size, cell_size)
                matrix[..., 0] = np.where(
                    matrix[..., 0], self.smooth_factor, 1 - self.smooth_factor
                )
                labels.append(matrix.astype(np.float32, copy=False))
            return np.asarray(labels, dtype=np.float32)

        image_size = self.image_size
        self.y_train = _convert(self.y_train, image_size, self.cell_size)
        self.y_valid = _convert(self.y_valid, image_size, self.cell_size)
        self.y_test = _convert(self.y_test, image_size, self.cell_size)
        for subset in ("train", "valid", "test"):
            shapes = self.original_shapes[subset]
            if shapes is not None:
                targets = getattr(self, f"y_{subset}")
                for target, (height, width) in zip(targets, shapes):
                    # -1 is an ignored occupancy target, never a negative example.
                    target[int(height)//self.cell_size:, :, 0] = -1
                    target[:, int(width)//self.cell_size:, 0] = -1

    def normalize_dataset(self) -> None:
        """Normalize all images to zero mean and unit standard deviation."""

        def _normalize(dataset: np.ndarray) -> np.ndarray:
            return np.asarray(
                [_normalize_image(image) for image in dataset], dtype=np.float32
            )

        for subset in ("train", "valid", "test"):
            images = getattr(self, f"x_{subset}")
            shapes = self.original_shapes[subset]
            if shapes is None:
                normalized = _normalize(images)
            else:
                normalized = np.asarray([
                    prepare_image(image[:int(height), :int(width)], self.input_size)
                    for image, (height, width) in zip(images, shapes)
                ], dtype=np.float32)
            setattr(self, f"x_{subset}", normalized)

    def select_subset(self) -> None:
        """Select the requested split and expose it as ``self.x`` / ``self.y``."""
        mapping: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "train": (self.x_train, self.y_train),
            "valid": (self.x_valid, self.y_valid),
            "test": (self.x_test, self.y_test),
        }
        self.x, self.y = mapping[self.subset]

        pixel_size_mapping = {
            "train": (
                self.pixel_size_x_train,
                self.pixel_size_y_train,
                self.pixel_size_source_train,
            ),
            "valid": (
                self.pixel_size_x_valid,
                self.pixel_size_y_valid,
                self.pixel_size_source_valid,
            ),
            "test": (
                self.pixel_size_x_test,
                self.pixel_size_y_test,
                self.pixel_size_source_test,
            ),
        }
        self.pixel_size_x, self.pixel_size_y, self.pixel_size_source = pixel_size_mapping[
            self.subset
        ]

    def __len__(self) -> int:
        """Return the number of items in the selected subset."""
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one normalized image and its encoded target.

        Parameters
        ----------
        index:
            Item index within the selected subset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(image, target)`` where:
            - ``image`` has shape ``(1, H, W)``
            - ``target`` has shape ``(nrow, ncol, 3)``
        """
        image = self.x[index].astype(np.float32, copy=False)
        target = self.y[index].astype(np.float32, copy=False)

        image = np.expand_dims(image, axis=0)

        return torch.from_numpy(image), torch.from_numpy(target)
