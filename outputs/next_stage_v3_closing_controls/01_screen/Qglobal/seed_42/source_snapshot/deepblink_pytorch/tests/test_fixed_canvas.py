"""Regression checks for geometry, units, ignored padding and checkpoint inference."""

import numpy as np
import torch
import tifffile

from deepblink_torch.cli import create, predict
from deepblink_torch.datasets.spots import SpotsDataset
from deepblink_torch.losses import combined_dice_rmse_loss
from deepblink_torch.preprocessing import prepare_image


def test_canvas_preserves_signal_and_does_not_use_padding_for_normalization():
    image = np.arange(256*256, dtype=np.float32).reshape(256, 256)
    prepared = prepare_image(image, 512)
    np.testing.assert_allclose(prepared[:256, :256].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(prepared[:256, :256].std(), 1, atol=1e-6)
    assert np.count_nonzero(prepared[256:, :]) == 0
    assert np.count_nonzero(prepared[:, 256:]) == 0


def test_padding_has_no_loss_or_gradient():
    truth = torch.zeros(1, 4, 4, 3)
    truth[0, 0, 0] = torch.tensor([1., .2, .3])
    truth[:, 2:, :, 0] = -1
    pred = torch.full_like(truth, .4, requires_grad=True)
    loss = combined_dice_rmse_loss(truth, pred)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad[:, 2:]) == 0
    assert torch.allclose(loss, combined_dice_rmse_loss(truth[:, :2], pred[:, :2]))


def test_mixed_size_create_keeps_pixel_and_micron_labels_aligned(tmp_path):
    images, labels = tmp_path/'images', tmp_path/'labels'
    images.mkdir()
    labels.mkdir()
    for index, size in enumerate([256, 512, 256]):
        stem = f'sample_{index}'
        image = np.arange(size*size, dtype=np.uint16).reshape(size, size)
        tifffile.imwrite(images/f'{stem}.tif', image, imagej=True,
                         resolution=(5, 5), metadata={'unit': 'um'})
        unit = 'pixel' if index == 0 else 'micron'
        value = 10 if index == 0 else 2
        (labels/f'{stem}.csv').write_text(
            f'POSITION_X,POSITION_Y\nX,Y\nX,Y\n({unit}),({unit})\n{value},{value}\n'
        )
    manifest = tmp_path/'frozen.csv'
    manifest.write_text('sample_id,subset\nsample_0,train\nsample_1,valid\nsample_2,test\n')
    args = create.parse_args(['--images', str(images), '--labels', str(labels),
                              '--output', str(tmp_path/'mixed.npz'), '--input-size', '512',
                              '--split-manifest', str(manifest)])
    archive = create.run_create(args)
    with np.load(archive, allow_pickle=True) as data:
        assert str(data['split_method']) == 'frozen_manifest'
        assert np.isclose(data['train_ratio'], 1/3)
        for subset in ['train', 'valid', 'test']:
            np.testing.assert_allclose(np.asarray(data[f'y_{subset}'][0], dtype=float), [[10, 10]])
            dataset = SpotsDataset(archive, cell_size=4, subset=subset)
            image, target = dataset[0]
            height, width = data[f'original_shape_{subset}'][0]
            expected = prepare_image(tifffile.imread(images/str(data[f'image_name_{subset}'][0])), 512)
            np.testing.assert_array_equal(image[0], expected)
            if height == 256:
                assert torch.all(target[64:, :, 0] == -1)
                assert torch.all(target[:, 64:, 0] == -1)


def test_checkpoint_canvas_is_used_for_prediction(tmp_path):
    from deepblink_torch import UNetDeepBlink
    checkpoint = tmp_path/'model.pt'
    torch.save({'model_state_dict': UNetDeepBlink().state_dict(), 'input_size': 512}, checkpoint)
    model, _ = predict._load_model(checkpoint, torch.device('cpu'))
    assert model.input_size == 512
