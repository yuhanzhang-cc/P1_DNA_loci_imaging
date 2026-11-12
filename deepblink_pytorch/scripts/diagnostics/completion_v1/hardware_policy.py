"""Explicit alternate hardware track; never relax initialization or recipe."""
from pathlib import Path
import re
from diagnostics.handoff_v2.common import read,write,sha,canonical

PROFILE_ID='autodl_vgpu32_ada_transfer_v1'
GPU_PATTERN=r'(?:NVIDIA\s+)?(?:GeForce\s+RTX\s+4080\s*SUPER(?:\s+32GB)?|vGPU(?:[-\s]?32(?:GB|G))?)'


def profile(root):
    p=Path(root)/'configs/hardware_profile.json'
    if not p.exists():return None
    spec=read(p);validate(spec);return spec


def validate(spec):
    if (spec.get('profile_id')!=PROFILE_ID or spec.get('torch')!='2.11.0+cu128' or spec.get('cuda_runtime')!='12.8'
        or spec.get('compute_capability')!=[8,9] or spec.get('require_historical_initializations') is not True
        or spec.get('same_hardware_as_historical_baselines') is not False
        or spec.get('experiment_track')!='vgpu32_development_v1'):raise ValueError('Unknown or weakened hardware profile')
    return spec


def snapshot(torch):
    p=torch.cuda.get_device_properties(0)
    return dict(gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),compute_capability=[int(p.major),int(p.minor)],memory_bytes=int(p.total_memory),
        multiprocessors=int(p.multi_processor_count),gpu_uuid=str(getattr(p,'uuid','unavailable')))


def judge(spec,actual,initializations):
    validate(spec)
    checks=dict(name=bool(re.fullmatch(GPU_PATTERN,actual.get('gpu',''),flags=re.I)),
        torch=actual.get('torch')==spec['torch'],cuda_runtime=actual.get('cuda_runtime')==spec['cuda_runtime'],
        compute_capability=actual.get('compute_capability')==spec['compute_capability'],
        memory=30_000_000_000<=actual.get('memory_bytes',0)<=35_000_000_000,
        all_three_seeds=[r.get('seed') for r in initializations]==[42,43,44],
        paired_initializations=all(r.get('paired_exact') is True for r in initializations),
        historical_initializations=all(r.get('historical_exact') is True for r in initializations))
    return dict(passed=all(checks.values()),checks=checks,profile_id=spec['profile_id'],
        experiment_track=spec['experiment_track'],same_hardware_as_historical_baselines=False,
        strict_within_track_contrast='P0-S0',historical_contrasts='cross-hardware development references; not strict hardware-isolated ablations')


def binding(root,torch,device):
    spec=profile(root)
    if spec is None:return {}
    return dict(hardware_profile_sha256=canonical(spec),hardware_identity=snapshot(torch) if device=='cuda' else dict(device='cpu_engineering_only'))


def lock_backend(path,identity):
    """A resumed vGPU run must retain the initially accepted hardware identity."""
    path=Path(path)
    if path.exists():
        if read(path)!=identity:raise ValueError('vGPU hardware identity changed; do not mix devices in one resumed run')
    else:write(path,identity)
    return True


def context(root):
    """Keep mixed-hardware interpretation attached to every new report."""
    root=Path(root);spec=profile(root)
    if spec is None:return None
    receipt=root/'outputs/hardware_transfer/complete.json'
    return dict(profile=spec,profile_sha256=sha(root/'configs/hardware_profile.json'),
        preflight_receipt_sha256=sha(receipt) if receipt.exists() else None,
        strict_within_track_contrast='P0-S0',historical_comparisons='cross-hardware development references',
        training_equivalence_proven=False)
