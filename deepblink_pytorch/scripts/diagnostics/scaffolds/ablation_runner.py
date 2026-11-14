"""Validate every matrix cell, deduplicate aliases and print dry-run commands."""
from pathlib import Path
import argparse,copy,json,subprocess,sys
from scaffold_common import read,validate_config


def plan(path):
    matrix=read(path);folder=Path(path).parent
    if matrix["automatic_training_allowed"]:raise ValueError("Automatic training must remain off")
    output=[];seen=set();commands=[]
    for cell in matrix["cells"]:
        if cell.get("reuse_existing"):
            output.append(dict(id=cell["id"],status="reuse_completed_H0",new_runs=0));continue
        if cell.get("expected_invalid"):
            invalid=copy.deepcopy(read(folder/"P0.json"));invalid["components"]["signed_offsets"]=False
            try:validate_config(invalid)
            except ValueError as error:output.append(dict(id=cell["id"],status="excluded_invalid",reason=str(error),new_runs=0));continue
            raise ValueError("Illegal matrix control unexpectedly accepted")
        config=read(folder/cell["config"]);validate_config(config)
        if any(config["components"][k]!=cell[k] for k in matrix["components"]):raise ValueError("Matrix/config component mismatch")
        method=config.get("alias_of",config["id"])
        if method in seen:
            output.append(dict(id=cell["id"],status="alias_deduplicated",alias_of=method,new_runs=0));continue
        seen.add(method)
        for seed in config["seeds"]:
            commands.append([sys.executable,str(Path(__file__).with_name("train_mechanism.py")),"--config",str(folder/cell["config"]),"--seed",str(seed),"--dry-run"])
        output.append(dict(id=cell["id"],status="schema_valid_scaffold_only",method=method,new_runs=len(config["seeds"])))
    if len(commands)!=matrix["prospective_new_runs"]:raise ValueError("Unique run budget differs")
    return dict(passed=True,dry_run=True,optimizer_updates=0,cells=output,would_check_commands=commands,commands_executed=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--matrix",type=Path,required=True);p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args();result=plan(args.matrix)
    print(json.dumps(result,indent=2))
    for command in result["would_check_commands"]:print(subprocess.list2cmdline(command))


if __name__=="__main__":main()
