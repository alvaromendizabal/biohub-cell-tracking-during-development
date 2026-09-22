"""Pure metadata/geometry contracts for a deliberately small, incomplete Zarr store."""
from __future__ import annotations
from itertools import product
import math
from pathlib import PurePosixPath
import numpy as np


def checked_relative(value: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in ('\\', '\x00', ':')):
        raise ValueError('Unsafe relative path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or p.as_posix() in ('.', ''):
        raise ValueError('Unsafe relative path')
    return p.as_posix()


def validate_config(cfg: dict) -> None:
    if cfg.get('competition') != 'biohub-cell-tracking-during-development' or cfg.get('split') != 'train':
        raise ValueError('This pilot is restricted to the named competition training split.')
    if cfg.get('sample_id') != '44b6_0113de3b' or cfg.get('frames') != [0, 1, 2, 3]:
        raise ValueError('Do not expand this pilot or select a new sample before reviewing its results.')
    if cfg.get('roi_shape_zyx') != [32,128,128] or cfg.get('sigma_um') != [2.0,3.0,4.0]:
        raise ValueError('The first pilot uses the documented, fixed ROI and three scales.')
    limits = {'max_chunk_files':16,'max_download_file_bytes':64*1024**2,
              'max_selected_decoded_bytes':128*1024**2,'max_single_decoded_chunk_bytes':64*1024**2,
              'max_download_cache_bytes':256*1024**2, 'max_rss_bytes':8*1024**3,
              'max_candidates_per_frame':64,'max_peak_proposals':4096,'pair_top_k':5,
              'per_request_timeout_seconds':45}
    for key, cap in limits.items():
        value=cfg.get(key)
        if not isinstance(value,int) or isinstance(value,bool) or not 0 < value <= cap:
            raise ValueError(f'Invalid or expanded safety limit: {key}')
    if cfg.get('training_fits_allowed') != 0:
        raise ValueError('No training is authorized by this pilot configuration.')
    if not 0.95 <= cfg['candidate_quantile'] < 1:
        raise ValueError('Invalid pilot candidate quantile.')
    for key in ('nms_distance_um','pair_radius_um'):
        if not 0 < cfg[key] <= 20:
            raise ValueError(f'Invalid {key}')


def canonical_tzyx(names, *, field="axes"):
    """Accept this dataset's ASCII upper/lowercase aliases without sorting/transposing.

    This is a Biohub reader compatibility rule, NOT a general claim that arbitrary
    OME axis names are case-insensitive. Original names remain in the receipt.
    """
    if not isinstance(names, (list, tuple)) or len(names) != 4:
        raise ValueError(f"STOP: {field} must declare exactly four TZYX axes.")
    allowed = {"T": "t", "t": "t", "Z": "z", "z": "z",
               "Y": "y", "y": "y", "X": "x", "x": "x"}
    if any(not isinstance(name, str) or name not in allowed for name in names):
        raise ValueError(f"STOP: unsupported axis name in {field}; inspect metadata.")
    normalized = [allowed[name] for name in names]
    if normalized != ["t", "z", "y", "x"]:
        raise ValueError(f"STOP: unexpected axis order in {field}; no transpose is permitted.")
    return normalized


def image_geometry(root_meta: dict, array_meta: dict, cfg: dict) -> dict:
    """Read declared axes/scale; an absent scale is explicitly sourced to organizer docs."""
    if root_meta.get('zarr_format') != 3 or root_meta.get('node_type') != 'group':
        raise ValueError('STOP: this first adapter expects Zarr-v3 group metadata; return metadata, do not guess.')
    attrs = root_meta.get('attributes', {})
    multiscales = attrs.get('multiscales', attrs.get('ome',{}).get('multiscales',[]))
    spacing=np.asarray(cfg['reference_spacing_um'],dtype=float)
    axes_source='organizer_documented_TZYX_fallback'
    scale_source='organizer_documented_spacing_fallback_metadata_missing'
    time_step=None
    time_unit=None
    declared_axis_names=None
    declared_dimension_names=array_meta.get('dimension_names')
    if multiscales:
        if len(multiscales)!=1:
            raise ValueError('STOP: multiple image series require explicit inspection.')
        ms=multiscales[0]
        axes=ms.get('axes')
        if axes is not None:
            if not isinstance(axes, list):
                raise ValueError('STOP: axes metadata must be a list.')
            names=[a.get('name') if isinstance(a,dict) else a for a in axes]
            canonical_tzyx(names, field='multiscales.axes')
            declared_axis_names=list(names)
            for axis, expected_type in zip(axes, ('time', 'space', 'space', 'space')):
                if isinstance(axis, dict) and axis.get('type') not in (None, expected_type):
                    raise ValueError('STOP: declared axis type contradicts the TZYX mapping.')
            axes_source='metadata'
            for a in axes[1:]:
                unit=a.get('unit') if isinstance(a,dict) else None
                if unit not in (None,'micrometer','micrometre','um','µm','μm'):
                    raise ValueError('STOP: spatial units are not micrometers.')
            time_unit=axes[0].get('unit') if isinstance(axes[0],dict) else None
        entries=[d for d in ms.get('datasets',[]) if d.get('path')=='0']
        if len(entries)!=1:
            raise ValueError('STOP: expected an explicitly listed resolution-0 array.')
        transforms=entries[0].get('coordinateTransformations',[])+ms.get('coordinateTransformations',[])
        scales=[]
        for transform in transforms:
            if transform.get('type')=='scale':
                values=np.asarray(transform['scale'],dtype=float)
                if values.shape!=(4,) or not np.all(np.isfinite(values)) or np.any(values<=0):
                    raise ValueError('Invalid coordinate scale.')
                scales.append(values)
            elif transform.get('type')!='translation':
                raise ValueError('STOP: unsupported coordinate transform.')
        if scales:
            combined=np.prod(np.stack(scales),axis=0)
            spacing=combined[1:]; time_step=float(combined[0]); scale_source='metadata'
    names=array_meta.get('dimension_names')
    if names is not None:
        canonical_tzyx(names, field='array.dimension_names')
    if not np.allclose(spacing,cfg['reference_spacing_um'],rtol=1e-6,atol=1e-8):
        raise ValueError('STOP: measured spacing differs from the native-scale contract; inspect before processing.')
    return {'axis_order':['t','z','y','x'],'axes_source':axes_source,
            'declared_axis_names':declared_axis_names,
            'declared_dimension_names':declared_dimension_names,
            'axis_mapping':'ASCII T/t, Z/z, Y/y, X/x aliases only; array order unchanged',
            'spacing_um':spacing.tolist(),'spacing_source':scale_source,
            'declared_time_step':time_step,'declared_time_unit':time_unit,
            'motion_units':'micrometers per frame; acquisition regularity is NOT assumed',
            'coordinates':'crop-local coordinates, not a global embryo coordinate system'}


def chunk_plan(array_meta: dict, cfg: dict) -> dict:
    validate_config(cfg)
    if array_meta.get('zarr_format')!=3 or array_meta.get('node_type')!='array':
        raise ValueError('STOP: unsupported array metadata version/type.')
    shape=array_meta.get('shape'); grid=array_meta.get('chunk_grid',{})
    chunks=grid.get('configuration',{}).get('chunk_shape')
    if grid.get('name')!='regular' or not isinstance(shape,list) or not isinstance(chunks,list):
        raise ValueError('STOP: expected a regular four-dimensional chunk grid.')
    if len(shape)!=4 or len(chunks)!=4 or any(not isinstance(n,int) or isinstance(n,bool) or n<=0 for n in shape+chunks):
        raise ValueError('Invalid shape or chunk dimensions.')
    if shape[0]<=max(cfg['frames']) or any(n<r for n,r in zip(shape[1:],cfg['roi_shape_zyx'])):
        raise ValueError('STOP: selected frames/ROI do not fit this array.')
    dtype_name=array_meta.get('data_type')
    if dtype_name not in ('uint8','uint16','uint32','int8','int16','int32','float32','float64'):
        raise ValueError('STOP: unsupported scalar image dtype.')
    dtype=np.dtype(dtype_name)
    selected_bytes=len(cfg['frames'])*math.prod(shape[1:])*dtype.itemsize
    decoded_chunk_bytes=math.prod(chunks)*dtype.itemsize
    if selected_bytes>cfg['max_selected_decoded_bytes'] or decoded_chunk_bytes>cfg['max_single_decoded_chunk_bytes']:
        raise ValueError('STOP: decoded sample/chunk exceeds the memory contract.')
    if array_meta.get('storage_transformers'):
        raise ValueError('STOP: storage transformers need a separate adapter.')
    codecs=array_meta.get('codecs',[])
    if not codecs or any(not isinstance(c,dict) or c.get('name') not in ('bytes','blosc','gzip','zstd','crc32c','transpose') for c in codecs):
        raise ValueError('STOP: inspect unsupported codecs (including sharding) before downloading data.')
    key=array_meta.get('chunk_key_encoding',{})
    name=key.get('name'); sep=key.get('configuration',{}).get('separator','/' if name=='default' else '.')
    if name not in ('default','v2') or sep not in ('/','.'):
        raise ValueError('STOP: unsupported chunk key encoding.')
    t_chunks=sorted(set(t//chunks[0] for t in cfg['frames']))
    ranges=[range(math.ceil(n/c)) for n,c in zip(shape[1:],chunks[1:])]
    count=len(t_chunks)*math.prod(len(r) for r in ranges)
    if count>cfg['max_chunk_files']:
        raise ValueError(f'STOP: {count} chunks exceed the {cfg["max_chunk_files"]}-chunk pilot limit.')
    keys=[]
    for index in product(t_chunks,*ranges):
        encoded=sep.join(map(str,index))
        if name=='default': encoded='c'+sep+encoded
        keys.append(checked_relative('0/'+encoded))
    return {'shape_tzyx':shape,'chunk_shape_tzyx':chunks,'dtype':dtype_name,
            'frames':cfg['frames'],'array_path':'0','chunk_keys':keys,'chunk_file_count':count,
            'selected_decoded_bytes':selected_bytes,'single_decoded_chunk_bytes':decoded_chunk_bytes,
            'incomplete_store':True,'full_dataset_downloaded':False,
            'guard':'Every required chunk must exist and match its receipt BEFORE reading any selected frame.'}


def centered_roi(shape_zyx, roi_shape):
    if len(shape_zyx)!=3 or len(roi_shape)!=3 or any(n<r or r<=0 for n,r in zip(shape_zyx,roi_shape)):
        raise ValueError('Invalid center ROI.')
    offsets=[(int(n)-int(r))//2 for n,r in zip(shape_zyx,roi_shape)]
    return tuple(slice(o,o+int(r)) for o,r in zip(offsets,roi_shape)),offsets
