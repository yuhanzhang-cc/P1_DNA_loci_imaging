"""Fail closed until a scientifically reviewable core ablation is defined."""
from .common import canonical_hash


def validate(plan):
    cells=plan["cells"]
    if len(cells)!=4 or {c["id"] for c in cells}!={"base","capacity","full","minus"}:raise ValueError("Expected four core ablation roles")
    if any(c["enabled"] for c in cells):
        if not all(c["enabled"] for c in cells):raise ValueError("A partial ablation matrix cannot be launched")
        required=("core_rationale","related_work_delta","capacity_matching_rationale","removal_legality","implementation_hashes","locked_recipe")
        if any(not plan.get(k) for k in required):raise ValueError("Core rationale, implementation and legal controls must be locked first")
        if any(not c["model_key"] or c["seeds"]!=[42,43,44] for c in cells):raise ValueError("Unspecified model or seed budget")
        return dict(executable=True,tasks=12,plan_hash=canonical_hash(plan))
    if any(c["model_key"] is not None for c in cells):raise ValueError("Disabled placeholders should not imply a selected core")
    return dict(executable=False,tasks=0,reason="Core method not selected; previous controls remain historical evidence",plan_hash=canonical_hash(plan))
