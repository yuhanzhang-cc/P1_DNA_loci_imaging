"""Shared scaffold contracts. No training adapters or optimizer imports."""
from pathlib import Path
import copy
import csv
import hashlib
import json

CODE_ROOT=Path(__file__).resolve().parents[4]
REL="docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103"
PARAMETERS=671719


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8"))


def csv_new(path,data,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data)


def table(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def base(root):return Path(root).resolve()/REL


def todos(value,prefix=""):
    if isinstance(value,dict):
        return [p for k,v in value.items() for p in todos(v,prefix+"."+k if prefix else k)]
    if isinstance(value,list):return [p for i,v in enumerate(value) for p in todos(v,prefix+f"[{i}]" )]
    return [prefix] if value is None or isinstance(value,str) and value.startswith("TODO") else []


def architecture_signature(config):
    return json.dumps(config["architecture"],sort_keys=True)


def validate_config(config):
    if config["status"]!="not_started" or config["training_enabled"] or config["adapter_implemented"]:
        raise ValueError("Scaffold must remain disabled/not_started/unimplemented")
    architecture=config["architecture"]
    if architecture!={"base":"H0_frozen_structure","input_channels":1,"head_channels":3,"added_trainable_parameters":0}:
        raise ValueError("No implemented capacity-matching rule for changed architecture")
    if config["expected_parameters"]!=PARAMETERS:raise ValueError("Parameter budget differs")
    if config["seeds"]!=[42,43,44]:raise ValueError("Registered training seeds differ")
    components=config["components"]
    if set(components)!={"signed_offsets","neighborhood_supervision"} or any(not isinstance(v,bool) for v in components.values()):
        raise ValueError("Unknown or non-boolean scaffold component")
    if components["neighborhood_supervision"] and not components["signed_offsets"]:
        raise ValueError("Illegal bounded-offset neighborhood: targets can be unrepresentable")
    recipe=config["recipe"]
    if recipe!={"epochs":100,"updates":2200,"batch_size":8,"canvas":512,"optimizer":"Adam","lr":.001,"period":5,"threshold_count":31,"precision":"float32"}:
        raise ValueError("Frozen recipe contract changed")
    return dict(schema_valid=True,launch_ready=False,status="not_started",dry_run=True,optimizer_updates=0,
                architecture_instantiated=False,adapter_implemented=False,expected_parameters=PARAMETERS,
                unresolved=todos(config),reason="scaffold only; no executable S0/P0 adapter is implemented")


def matched_control(config):
    validate_config(config)
    control=copy.deepcopy(config)
    control.update(id="CAP_"+config["id"],alias_of="S0",counts_as_new_trial=False,
                   role="parameter_matched_control",capacity_signature=architecture_signature(config))
    control["components"]["neighborhood_supervision"]=False
    control["supervision"].update(radius_px=0,ownership="not_applicable_single_anchor",normalization="anchor_mean")
    validate_config(control)
    assert architecture_signature(control)==architecture_signature(config)
    return control
