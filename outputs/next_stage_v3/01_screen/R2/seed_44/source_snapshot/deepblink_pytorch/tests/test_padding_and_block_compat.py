import numpy as np

from deepblink_torch.cli.predict import _pad_to_square_power2
from deepblink_torch.networks.unet import UNetDeepBlink


def test_pad_to_square_power2_uses_reflect_padding() -> None:
    image = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    padded, original_shape = _pad_to_square_power2(image)

    assert original_shape == (2, 3)
    np.testing.assert_allclose(
        padded,
        np.asarray(
            [
                [1, 2, 3, 2],
                [4, 5, 6, 5],
                [1, 2, 3, 2],
                [4, 5, 6, 5],
            ],
            dtype=np.float32,
        ),
    )


def test_unet_inception_block_can_be_constructed() -> None:
    model = UNetDeepBlink(block="inception")
    assert model.block_name == "inception"
