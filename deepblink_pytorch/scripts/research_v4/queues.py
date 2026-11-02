"""Explicit GPU queues; no automatic expansion from recipes into baselines."""
from .common import SEEDS


def queues():
    result={"recipe":[]}
    for seed in SEEDS:
        for augment in (False,True):
            result["recipe"].append(dict(task_id="G0_aug_cuda" if augment else "G0_ref_cuda",seed=seed,key="G0",augment=augment,pipeline="own",epochs=100,updates=2200,
                                        role="fixed_structure_joint_augmentation_control"))
    for queue,key in (("baseline_r0","R0"),("baseline_h0","H0")):
        result[queue]=[dict(task_id=key+"_cuda_reference",seed=seed,key=key,augment=False,pipeline="own",epochs=100,updates=2200,
                           role="direct_reference" if key=="R0" else "dense_heatmap_offset_representation_reference") for seed in SEEDS]
    result["spotiflow_scratch"]=[dict(task_id="Spotiflow_scratch_control",seed=seed,key="Spotiflow",augment="native",pipeline="spotiflow_native",epochs=100,updates=4400,
                                     role="pretraining_control_not_fully_tuned_scratch_baseline") for seed in SEEDS]
    return result


def validate(queue,items):
    registered=queues()
    if queue not in registered or items!=registered[queue]:raise ValueError("Unregistered or expanded GPU task queue")
