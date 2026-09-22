"""Offline fixtures only. These are deliberately not competition-performance tests."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from biohub_tracking.pilot_contract import checked_relative,validate_config,image_geometry,chunk_plan,centered_roi

ROOT=Path(__file__).resolve().parents[1]

def cfg():return json.loads((ROOT/'configs/pilot.json').read_text())
def array():return {'zarr_format':3,'node_type':'array','shape':[100,64,256,256],'data_type':'uint16',
    'chunk_grid':{'name':'regular','configuration':{'chunk_shape':[1,64,256,256]}},
    'chunk_key_encoding':{'name':'default','configuration':{'separator':'/'}},
    'codecs':[{'name':'bytes','configuration':{'endian':'little'}}],'fill_value':0}
def group():return {'zarr_format':3,'node_type':'group','attributes':{'multiscales':[{
    'axes':[{'name':'t','type':'time','unit':'second'},*({'name':n,'type':'space','unit':'micrometer'} for n in ['z','y','x'])],
    'datasets':[{'path':'0','coordinateTransformations':[{'type':'scale','scale':[1,1.625,.40625,.40625]}]}]}]}}

@pytest.mark.parametrize('path',['../x','/root','a/../b','a\\b','a\x00b','http://x'])
def test_reject_unsafe_paths(path):
    with pytest.raises(ValueError):checked_relative(path)

def test_four_frame_plan_is_exact():
    plan=chunk_plan(array(),cfg())
    assert plan['chunk_keys']==[f'0/c/{t}/0/0/0' for t in range(4)]
    assert plan['selected_decoded_bytes']==4*64*256*256*2
    assert plan['incomplete_store'] is True

def test_temporal_chunks_deduplicate():
    a=array();a['chunk_grid']['configuration']['chunk_shape'][0]=2
    assert chunk_plan(a,cfg())['chunk_keys']==['0/c/0/0/0/0','0/c/1/0/0/0']

def test_excess_chunks_fail_before_downloading():
    a=array();a['chunk_grid']['configuration']['chunk_shape']=[1,8,32,32]
    with pytest.raises(ValueError,match='chunks exceed'):chunk_plan(a,cfg())

def test_sharded_codec_stops():
    a=array();a['codecs']=[{'name':'sharding_indexed','configuration':{}}]
    with pytest.raises(ValueError,match='codecs'):chunk_plan(a,cfg())

def test_metadata_physical_units():
    x=image_geometry(group(),array(),cfg())
    assert x['spacing_source']=='metadata'
    np.testing.assert_allclose(x['spacing_um'],[1.625,.40625,.40625])

def test_missing_scale_is_not_fabricated_as_measured():
    g=group();g['attributes']={}
    assert 'fallback' in image_geometry(g,array(),cfg())['spacing_source']

def test_axes_cannot_be_silently_reordered():
    g=group();g['attributes']['multiscales'][0]['axes'].reverse()
    with pytest.raises(ValueError):image_geometry(g,array(),cfg())

def test_spacing_mismatch_fails():
    g=group();g['attributes']['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][1]=1.
    with pytest.raises(ValueError,match='spacing'):image_geometry(g,array(),cfg())

def test_center_roi_keeps_source_offsets():
    slices,offsets=centered_roi([64,256,256],[32,128,128])
    assert offsets==[16,64,64]
    assert [(s.start,s.stop) for s in slices]==[(16,48),(64,192),(64,192)]

def test_do_not_allow_test_split():
    c=cfg();c['split']='test'
    with pytest.raises(ValueError):validate_config(c)

def test_no_expanded_candidate_search():
    c=cfg();c['max_candidates_per_frame']=1000
    with pytest.raises(ValueError):validate_config(c)
