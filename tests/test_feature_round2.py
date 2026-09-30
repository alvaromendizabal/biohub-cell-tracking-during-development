import numpy as np
import pytest
from feature_fixtures import constructed
from biohub_tracking.feature_round2 import pair_geometry,correspondence,history,sibling_hypotheses,competition
from biohub_tracking.feature_audit import audit,planned_ablations

FAMILIES=['geometry','appearance','correspondence','competition','history','neighborhood','conservation','division']

@pytest.fixture(scope='module')
def scene():return constructed()

@pytest.mark.parametrize('family',FAMILIES)
def test_round2_each_family_has_sixteen_declared_columns(scene,family):
    expected={f'r2_{family}__{x}' for x in scene[0]['round2']['families'][family]}
    actual={c for c in scene[8] if c.startswith(f'r2_{family}__')}
    assert actual==expected and len(actual)==16

@pytest.mark.parametrize('family',FAMILIES)
def test_round2_each_family_has_no_infinite_values(scene,family):
    v=scene[8].filter(regex='^r2_'+family+'__').to_numpy(float)
    assert not np.isinf(v).any() and np.isfinite(v).any()


def test_round2_keeps_exact_pair_set(scene):
    assert set(zip(scene[8].source_id,scene[8].target_id))==set(zip(scene[6].source_id,scene[6].target_id))
    assert not scene[8].duplicated(['source_id','target_id']).any()


def test_source_history_missing_in_first_frame(scene):
    t=scene[8].loc[scene[8].t_source==0]
    assert t.r2_history__available.eq(0).all()
    assert t.r2_history__prediction_residual.isna().all()


def test_history_does_not_access_a_frame_after_target(scene):
    contexts=scene[10];frames={0:scene[11][0],1:scene[11][1]}
    h=history(contexts[0],contexts[10],contexts,frames)
    assert h['available']==0


def test_pair_displacement_reverses_and_norm_stays_same(scene):
    a,b=scene[10][0],scene[10][10];x=pair_geometry(a,b);y=pair_geometry(b,a)
    assert np.isclose(x['distance'],y['distance'])
    np.testing.assert_allclose([x['dz'],x['dy'],x['dx']],-np.array([y['dz'],y['dy'],y['dx']]))


def test_identical_patch_self_similarity_is_one(scene):
    s=scene[10][0];c=correspondence(s,s)
    assert np.isclose(c['r3_ncc'],1.) and np.isclose(c['r5_ncc'],1.) and c['r5_rmse']==0


def test_division_hypotheses_have_distinct_canonical_daughters(scene):
    d=scene[9]
    assert len(d)>0 and (d.daughter_a<d.daughter_b).all()
    assert not d.duplicated(['source_id','daughter_a','daughter_b']).any()


def test_round2_audit_includes_missing_history_instead_of_fake_zeros(scene):
    q,r=audit(scene[8],'r2_')
    assert len(q)==128
    assert q.loc[q.feature=='r2_history__prediction_residual','missing_fraction'].iloc[0]>0


def test_round2_addition_and_removal_ablations_are_complete():
    p=planned_ablations(FAMILIES)
    assert len(p)==18
    assert sum(x['experiment'].startswith('add_') for x in p)==8
    assert sum(x['experiment'].startswith('remove_') for x in p)==8


def test_appearance_relative_change_is_bounded(scene):
    columns=[c for c in scene[8] if c.startswith('r2_appearance__') and c.endswith('relative_difference')]
    values=scene[8][columns].to_numpy(float);finite=values[np.isfinite(values)]
    assert len(columns)==8 and np.all(abs(finite)<=1.+1e-12)


def test_competitor_weights_sum_to_one_but_are_not_labels(scene):
    table=scene[8]
    for temperature in [2,5]:
        assert np.allclose(table.groupby('source_id')[f'r2_competition__out_prob_{temperature}'].sum(),1.)
        assert np.allclose(table.groupby('target_id')[f'r2_competition__in_prob_{temperature}'].sum(),1.)


def test_local_motion_residuals_are_nonnegative(scene):
    for k in [3,5]:
        values=scene[8][f'r2_neighborhood__k{k}_motion_residual'].dropna()
        assert (values>=0).all()


def test_conservation_orientation_is_sign_invariant_cosine_range(scene):
    v=scene[8].r2_conservation__orientation_alignment.dropna()
    assert v.between(0.,1.+1e-12).all()


def test_history_is_unchanged_when_future_frame_is_removed(scene):
    full=history(scene[10][10],scene[10][20],scene[10],scene[11])
    limited=history(scene[10][10],scene[10][20],scene[10],{k:v for k,v in scene[11].items() if k<=2})
    assert full.keys()==limited.keys()
    np.testing.assert_allclose(list(full.values()),list(limited.values()),equal_nan=True)
