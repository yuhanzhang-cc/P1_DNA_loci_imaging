"""Check scaffold registration without incrementing the completed-study counter."""
from pathlib import Path
import argparse,json
from scaffold_common import CODE_ROOT,table,read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args()
    registry=table(CODE_ROOT/"docs/next_stage_plan_20261025/experiment_registry.csv")
    selected=[r for r in registry if r["study_id"] in ("S0_SCAFFOLD","P0_SCAFFOLD","CAP_P0_SCAFFOLD")]
    assert len(selected)==3 and all(r["status"]=="not_started" and not r["metric_value"] for r in selected)
    assert "scaffold only, not run" in (CODE_ROOT/"README.md").read_text(encoding="utf-8")
    count=read(args.data_root/"outputs/next_stage_v4_recipe_20261101/study_progress.json")["formal_research_tasks_completed"]
    assert count==41
    print(json.dumps(dict(passed=True,scaffold_entries=3,all_status="not_started",completed_training_count=count,new_training=0)))


if __name__=="__main__":main()
