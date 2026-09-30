import numpy as np
import pytest
from feature_fixtures import constructed
from biohub_tracking.feature_math import normalize,patch,moments,sample_scale_features
from biohub_tracking.feature_audit import audit,planned_ablations

FAMILIES=['radial','scale','moments','texture','structure','context','quality','scale_shape']

@pytest.fixture(scope='module')
def scene():return constructed()

@pytest.mark.parametrize('family',FAMILIES)
def test_round1_each_family_has_sixteen_declared_columns(scene,family):
    cfg,*_=scene;table=scene[7]
    expected={f'r1_{family}__{x}' for x in cfg['round1']['families'][family]}
    actual={c for c in table if c.startswith(f'r1_{family}__')}
    assert actual==expected and len(actual)==16

@pytest.mark.parametrize('family',FAMILIES)
def test_round1_each_family_has_no_infinite_values(scene,family):
    v=scene[7].filter(regex='^r1_'+family+'__').to_numpy(float)
    assert not np.isinf(v).any() and np.isfinite(v).any()


def test_round1_keeps_candidate_ids_and_frames(scene):
    assert set(scene[7].node_id)==set(scene[5].node_id)
    assert scene[7].node_id.is_unique
    assert set(scene[7].t)=={0,1,2,3}


def test_radial_signal_decreases_for_synthetic_bright_centers(scene):
    t=scene[7]
    assert np.all(t.r1_radial__ring0_mean>t.r1_radial__ring3_mean)


def test_positive_integrated_mass_and_moment_eigenvalues(scene):
    t=scene[7]
    assert np.all(t.r1_moments__r5_mass>0)
    assert np.all(t.r1_moments__r5_eig_min>=0)
    assert np.all(t.r1_moments__r5_eig_max>=t.r1_moments__r5_eig_min)


def test_quality_has_no_saturation_in_synthetic_data(scene):
    assert (scene[7].r1_quality__saturation_fraction==0).all()


def test_audit_does_not_remove_constant_features(scene):
    q,r=audit(scene[7],'r1_')
    assert len(q)==128 and q.constant_in_pilot.any()
    assert q.decision.eq('unlabeled_quality_diagnostic_only').all()


def test_round1_addition_and_removal_ablations_are_complete():
    p=planned_ablations(FAMILIES)
    assert len(p)==18
    assert sum(x['experiment'].startswith('add_') for x in p)==8
    assert sum(x['experiment'].startswith('remove_') for x in p)==8


def test_local_patch_translation_does_not_change_moments(scene):
    raw=scene[1][0];normal,_=normalize(raw);center=np.array([10,12,12]);spacing=scene[4]
    a=patch(raw,normal,center,spacing)
    moved=np.roll(raw,1,axis=2);nm,_=normalize(moved);b=patch(moved,nm,center+[0,0,1],spacing)
    np.testing.assert_allclose(moments(a)[2],moments(b)[2],atol=1e-10)


def test_scale_derivative_at_gaussian_center_is_positive():
    grid=np.indices((25,25,25),dtype=float);v=np.exp(-sum((grid[a]-12)**2 for a in range(3))/8).astype(np.float32)
    s=sample_scale_features(v,[1,1,1],np.array([[12,12,12]]))[0]
    assert s['s2p5_log']>0 and s['s2p5_gradient']<1e-5


def test_candidate_context_counts_are_monotone_in_radius(scene):
    t=scene[7]
    values=t[[f'r1_context__r{r}_count' for r in [8,12,16,20]]].to_numpy(float)
    assert np.all(np.diff(values,axis=1)>=0)


def test_glcm_energy_and_homogeneity_ranges(scene):
    for name in ['z','y','x','xy']:
        for metric in ['homogeneity','energy']:
            v=scene[7][f'r1_texture__{name}_{metric}']
            assert v.between(0.,1.+1e-12).all()


def test_structure_eigenvalues_ordered(scene):
    for radius in [3,5]:
        values=scene[7][[f'r1_structure__r{radius}_tensor_{s}' for s in ['min','mid','max']]].to_numpy(float)
        assert (values>=0).all() and (np.diff(values,axis=1)>=-1e-12).all()


def test_scale_shape_reports_only_a_declared_scale(scene):
    assert set(scene[7].r1_scale_shape__best_scale).issubset({1.5,2.5,3.5,5.0})
