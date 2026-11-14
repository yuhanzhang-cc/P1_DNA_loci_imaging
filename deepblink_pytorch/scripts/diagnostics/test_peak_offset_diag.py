"""Source-cell attribution and coordinate conventions; no optimizer or real data."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from peak_offset_diag import capture_sources, decompose, edge_distance, link_export


def test_dense_roundout_is_recorded_before_filter_and_not_clipped():
    prediction = np.zeros((256,256,3),np.float32)
    prediction[...,0] = -10
    prediction[255,3] = [10, np.nextafter(np.float32(1),np.float32(0)), .2]
    source, audit = capture_sources(prediction, np.asarray([256,256]), True)
    assert audit["rounded_out_candidates"] == 1
    assert not np.any(source["coordinates"] >= 256)


def test_offset_one_does_not_erase_the_originating_peak():
    prediction = np.zeros((8,8,3),np.float32)
    prediction[...,0] = -10
    prediction[2,4] = [10,1.,.25]
    source, _ = capture_sources(prediction,np.asarray([8,8]),True)
    i = np.argmax(source["scores"])
    np.testing.assert_array_equal(source["anchors"][i],[2,4])
    np.testing.assert_array_equal(source["coordinates"][i],[3,4.25])


@pytest.mark.parametrize("cell",[1,4])
def test_decomposition_is_exact_after_cast_and_backend_perturbation(cell):
    truth = np.asarray([12.300005,24.70001],np.float32)
    anchor = np.asarray([12-cell,24])
    offset = np.asarray([.2,.8],np.float32)*cell
    raw = anchor + offset
    serialized = raw.astype(np.float32)
    exported = (serialized + np.asarray([.0001,-.0001],np.float32)).astype(np.float32)
    target_anchor,target_offset,shift,error,cast,backend=decompose(exported,truth,anchor,offset,raw,serialized,cell)
    np.testing.assert_allclose(shift+error+cast+backend,exported.astype(float)-truth.astype(float),atol=1e-10,rtol=0)
    assert backend[0]>0 and backend[1]<0


def test_grid_source_anchor_and_offset_are_in_pixel_units():
    grid=np.zeros((2,2,3),np.float32)
    grid[1,1]=[.9,.5,.25]
    source,_=capture_sources(grid,np.asarray([8,8]),False)
    i=np.argmax(source["scores"])
    np.testing.assert_array_equal(source["anchors"][i],[4,4])
    np.testing.assert_array_equal(source["offsets"][i],[2,1])
    np.testing.assert_array_equal(source["coordinates"][i],[6,5])


def test_parity_does_not_guess_ambiguous_duplicate_sources():
    source=dict(coordinates=np.asarray([[1,1],[1,1]],np.float32),scores=np.asarray([.9,.8],np.float32))
    mapping,kind=link_export(source,np.asarray([[1,1]],np.float32),np.asarray([.9]))
    assert not mapping and kind=="unique_coordinate_match"
    mapping,kind=link_export(source,source["coordinates"],source["scores"])
    assert mapping=={0:0,1:1} and kind=="ordered"


def test_original_fov_edges_do_not_use_padded_canvas():
    assert edge_distance([255.,128.],[256,256])==1
    assert edge_distance([0.,128.],[256,256])==0
    assert edge_distance([128.,128.],[256,256])==128
