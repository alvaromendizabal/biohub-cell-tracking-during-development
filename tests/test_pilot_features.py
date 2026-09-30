import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from biohub_tracking.pilot_features import normalize_volume,physical_log,patch_descriptors,phase_shift,pair_features

ROOT=Path(__file__).resolve().parents[1]
def cfg():return json.loads((ROOT/'configs/pilot.json').read_text())

def test_normalization_flat_is_explicit():
    image,stats=normalize_volume(np.ones((5,5,5)))
    assert stats['flat'] and np.count_nonzero(image)==0

def test_nonfinite_image_rejected():
    with pytest.raises(ValueError):normalize_volume(np.full((5,5,5),np.nan))

def test_physical_log_center_response_positive():
    axis=np.arange(-12,13);z,y,x=np.meshgrid(axis,axis,axis,indexing='ij')
    v=np.exp(-(z*z+y*y+x*x)/(2*3.**2)).astype(np.float32)
    response=physical_log(v,[1,1,1],3.)
    assert response[12,12,12]>0
    assert np.argmax(response)==np.ravel_multi_index((12,12,12),v.shape)

def test_bad_spacing_rejected():
    with pytest.raises(ValueError):physical_log(np.zeros((5,5,5)),[0,1,1],2.)

def test_patch_contrast_exceeds_flat_background():
    raw=np.ones((21,21,21))*10;raw[9:12,9:12,9:12]=50
    f=patch_descriptors(raw,[10,10,10],[1,1,1])
    assert f['shell_median_raw']==10 and f['core_minus_shell_raw']>0
    assert f['weighted_variance_large_um2']>=f['weighted_variance_small_um2']

def test_incomplete_patch_rejected():
    with pytest.raises(ValueError):patch_descriptors(np.ones((21,21,21)),[0,0,0],[1,1,1])

def test_phase_shift_sign_is_source_to_target():
    rng=np.random.default_rng(19);a=rng.normal(size=(12,13,14));b=np.roll(a,(2,-3,1),axis=(0,1,2))
    delta,_=phase_shift(a,b)
    np.testing.assert_allclose(delta,[2,-3,1])

def test_duplicate_roi_not_a_motion_claim():
    a=np.ones((8,8,8));d,report=phase_shift(a,a.copy())
    np.testing.assert_array_equal(d,[0,0,0]);assert report['identical_roi']

def node(t,z):return pd.DataFrame([{'node_id':t,'t':t,'z_um':z,'y_um':0.,'x_um':0.,
    'core_mean_raw':10.,'robust_contrast':2.,'best_sigma_um':3.,'candidate_neighbors_within_10um':0}])

def test_pair_displacement_and_drift_units():
    table=pair_features(node(0,0.),node(1,1.625),[1.625,0,0],cfg())
    assert len(table)==1 and table.iloc[0]['distance_um']==1.625
    assert table.iloc[0]['drift_residual_um']==0 and bool(table.iloc[0]['mutual_nearest'])

def test_no_gt_arguments_in_feature_api():
    import inspect
    assert 'gt_graph' not in inspect.signature(pair_features).parameters
    assert 'labels' not in inspect.signature(pair_features).parameters

def test_nonadjacent_pairs_rejected():
    with pytest.raises(ValueError):pair_features(node(0,0.),node(3,1.),[0,0,0],cfg())
