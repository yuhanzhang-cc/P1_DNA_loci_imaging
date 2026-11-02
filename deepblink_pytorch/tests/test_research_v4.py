"""Meaningful invariants for geometry, independent RNG, matching and classical thresholds."""
from pathlib import Path
import sys
import json
import unittest
import tempfile
from copy import deepcopy
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from research_v4.augmentation import transform,prepare_sample
from research_v4.classical import peaks,image_candidates,response_policies,configurations
from research_v4.common import load_spec
from research_v4.evaluation import sweep,export_predictions,keep_scores
from deepblink_torch.preprocessing import prepare_image
from research_v4.common import write_csv,sha256,read_json,dump
from research_v4.external import load_manifest,evaluate,pixel_hash
from research_v4.ablation import validate


class ResearchV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=load_spec(Path(__file__).resolve().parents[2]/"docs/next_stage_plan_20261025/overnight_protocol_v4_proposed_20261101.yaml")

    def test_rectangular_pixel_coordinate_d4(self):
        image=np.zeros((8,12),np.float32);image[2,7]=1
        for k in range(4):
            for flip in (False,True):
                x,y,_=transform(image,[[2,7]],k,flip)
                np.testing.assert_array_equal(np.argwhere(x==1),y)
                u,c,_=transform(x,y,0,flip);u,c,_=transform(u,c,(-k)%4)
                np.testing.assert_array_equal(u,image);np.testing.assert_array_equal(c,[[2,7]])

    def test_logged_roundoff_does_not_hide_real_bounds_error(self):
        image=np.zeros((8,8),np.float32);truth=np.asarray([[7.00001,2]],np.float32)
        _,coords,logs=transform(image,truth,2)
        self.assertEqual(len(logs),1);self.assertEqual(coords[0,0],0);self.assertGreater(truth[0,0],7)
        with self.assertRaises(ValueError):transform(image,[[7.2,2]],2)
        with self.assertRaises(ValueError):transform(image,[[-.01,2]],0)

    def test_affine_is_effective_after_zscore_and_padding_is_zero(self):
        x=np.arange(256*256,dtype=np.float32).reshape(256,256)
        base,_,_,_=prepare_sample(x,[[10.5,20.5]])
        changed,target,_,_=prepare_sample(x,[[10.5,20.5]],intensity=(1.2,.1))
        np.testing.assert_array_equal(base[0],prepare_image(x,512))
        self.assertAlmostEqual(float(changed[0,:256,:256].mean()),.1,places=5)
        self.assertAlmostEqual(float(changed[0,:256,:256].std()),1.2,places=5)
        self.assertFalse(changed[:,256:,:].any());self.assertFalse(changed[:,:,256:].any())
        self.assertEqual(int((target[...,0]>=0).sum()),64*64)

    def test_keyed_augmentation_does_not_consume_model_rng(self):
        image=np.arange(128*128,dtype=np.float32).reshape(128,128)
        state=torch.get_rng_state().clone();nstate=np.random.get_state()[1].copy()
        a=prepare_sample(image,[[25.2,44.3]],augment=True,seed=42,epoch=4,index=2,canvas=128)
        b=prepare_sample(image,[[25.2,44.3]],augment=True,seed=42,epoch=4,index=2,canvas=128)
        self.assertTrue(torch.equal(state,torch.get_rng_state()));np.testing.assert_array_equal(nstate,np.random.get_state()[1])
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_classical_budget_and_serializable_boundaries(self):
        for name in ("LoG","DoG"):self.assertEqual(len(configurations(self.spec,name)),24)
        policies=response_policies(np.asarray([.2,.4,.8,2.,4.]),self.spec)
        self.assertLessEqual(len(policies),31)
        json.dumps(policies,allow_nan=False)
        self.assertTrue(keep_scores(np.asarray([.2,4.]),policies[0]).all())
        self.assertFalse(keep_scores(np.asarray([.2,4.]),policies[-1]).any())

    def test_nms_ties_and_flat_images(self):
        response=np.zeros((8,8),np.float32);response[2,2]=response[2,4]=1
        self.assertEqual(len(peaks(response,1)[0]),2)
        np.testing.assert_array_equal(peaks(response,2)[0],[[2,2]])
        for method in ("LoG","DoG"):
            out=image_candidates(np.zeros((32,32),np.float32),configurations(self.spec,method)[:1],method)
            self.assertEqual(len(out[0][0]),0)

    def test_bright_gaussian_peak_coordinate_order(self):
        y,x=np.mgrid[:48,:64];image=np.exp(-((y-17)**2+(x-39)**2)/2.).astype(np.float32)
        for method in ("LoG","DoG"):
            coords,scores=image_candidates(image,configurations(self.spec,method)[:1],method)[0]
            self.assertLessEqual(float(np.linalg.norm(coords[np.argmax(scores)]-[17,39])),1.)

    def test_threshold_matching_and_empty_predictions(self):
        data=dict(y=[np.asarray([[0,0],[0,3]],np.float32)])
        candidates=[(np.asarray([[0,1],[0,0]],np.float32),np.asarray([2.,4.],np.float32))]
        result=sweep(candidates,data,[dict(kind="all",value=None),dict(kind="finite",value=3.),dict(kind="none",value=None)])
        self.assertEqual([r["tp"] for r in result],[2,1,0]);self.assertEqual(result[0]["f1"],1.)
        self.assertEqual(result[-1]["fp"],0);self.assertEqual(result[-1]["fn"],2)

    def test_ablation_does_not_enable_unspecified_core(self):
        plan=dict(cells=[dict(id=i,enabled=False,model_key=None,seeds=[42,43,44]) for i in ("base","capacity","full","minus")])
        self.assertEqual(validate(plan)["tasks"],0)
        for cell in plan["cells"]:cell["enabled"]=True
        with self.assertRaises(ValueError):validate(plan)

    def external_fixture(self,folder):
        import tifffile
        image=np.arange(32*32,dtype=np.float32).reshape(32,32)
        tifffile.imwrite(folder/"new.tif",image)
        write_csv(folder/"new.csv",[dict(y=10.,x=15.)])
        row=dict(sample_id="engineering_new",image_file="new.tif",label_file="new.csv",height=32,width=32,coordinate_unit="pixel",um_per_pixel="",
                 biological_group_id="synthetic_group",acquisition_id="synthetic_acquisition",source_container="synthetic_source",
                 collection_date="2026-11-02",dataset_role="synthetic_engineering",image_sha256=sha256(folder/"new.tif"),label_sha256=sha256(folder/"new.csv"),annotation_version="synthetic_v1")
        write_csv(folder/"manifest.csv",[row])
        return row,image,dict(pixel_hashes=[],sample_ids=[],source_containers=[])

    def test_external_rejects_missing_groups_duplicates_bad_units_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);row,image,exclusions=self.external_fixture(folder)
            data,_=load_manifest(folder/"manifest.csv",exclusions,engineering=True)
            self.assertEqual(len(data["x"]),1)
            for key,value in (("biological_group_id","unknown"),("coordinate_unit","micron"),("image_sha256","changed"),("dataset_role","external_new")):
                bad=dict(row,**{key:value});write_csv(folder/"manifest.csv",[bad])
                with self.assertRaises(ValueError):load_manifest(folder/"manifest.csv",exclusions,engineering=True)
            write_csv(folder/"manifest.csv",[row])
            with self.assertRaises(ValueError):load_manifest(folder/"manifest.csv",dict(exclusions,pixel_hashes=[pixel_hash(image)]),engineering=True)
            write_csv(folder/"new.csv",[],["x","y"]);row["label_sha256"]=sha256(folder/"new.csv");write_csv(folder/"manifest.csv",[row])
            with self.assertRaises(ValueError):load_manifest(folder/"manifest.csv",exclusions,engineering=True)

    def test_external_matching_export_and_first_readout_preservation(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);_,_,exclusions=self.external_fixture(folder)
            lock=dict(method="synthetic_adapter",seed=42,threshold=.5)
            candidates=[(np.asarray([[10.,15.],[20.,20.]],np.float32),np.asarray([.9,.8],np.float32))]
            with patch("research_v4.external.locked_candidates",return_value=candidates):
                result=evaluate(folder/"manifest.csv",[lock],folder,exclusions,folder/"result",engineering=True)
                self.assertEqual((result[0]["tp"],result[0]["fp"],result[0]["fn"]),(1,1,0))
                self.assertTrue(read_json(folder/"result/synthetic_adapter/seed_42/metrics.json")["engineering_only"])
                with self.assertRaises(ValueError):evaluate(folder/"manifest.csv",[lock],folder,exclusions,folder/"result",engineering=True)

    def test_analysis_paired_exact_floor_and_image_group_contract(self):
        from research_v4.analysis import summarize
        seed_rows=[dict(method=m,seed=s,f1=f,precision=f,recall=f,tp=t,fp=1,fn=1) for m,f,t in (("a",.8,4),("b",.6,3)) for s in (42,43,44)]
        image_rows=[dict(method=r["method"],seed=r["seed"],sample_id=i,tp=r["tp"],fp=1,fn=1) for r in seed_rows for i in ("i1","i2")]
        with tempfile.TemporaryDirectory() as tmp:
            result=summarize(seed_rows,image_rows,Path(tmp),[("a","b")])
            self.assertAlmostEqual(result["comparisons"][0]["p_exact"],.25)
            with self.assertRaises(ValueError):summarize(seed_rows,image_rows[:-1],Path(tmp)/"bad",[("a","b")])

    def test_formal_completion_rejects_smoke_foreign_registration_and_changed_weights(self):
        from cloud_recipe_train import check_complete
        from research_v4.queues import queues
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);items=queues()["recipe"];config=dict(queue="recipe",tasks=items,manifest_sha256="synthetic_test_lock")
            dump(output/"recipe/protocol.json",config)
            for item in items:
                folder=output/"recipe"/item["task_id"]/f"seed_{item['seed']}";folder.mkdir(parents=True)
                # Only a file-integrity fixture; not a loadable model or training result.
                (folder/"best_model.pt").write_bytes(b"synthetic primary")
                (folder/"native_selected_model.pt").write_bytes(b"synthetic secondary")
                dump(folder/"task_protocol.json",dict(item=item,queue=config))
                dump(folder/"status.json",dict(status="complete",epoch=100,updates=2200))
                dump(folder/"registered_config.json",dict(engineering_only=False,device="cuda",source_lock="synthetic_test_lock"))
                primary=dict(tp=8,fp=1,fn=1,f1=8/9,checkpoint_sha256=sha256(folder/"best_model.pt"))
                dump(folder/"selection.json",dict(checkpoint_count=20,epoch_threshold_pairs=620,primary=primary,
                     native_secondary=dict(checkpoint_sha256=sha256(folder/"native_selected_model.pt"))))
                dump(folder/"primary/metrics.json",{k:primary[k] for k in ("tp","fp","fn","f1")})
            check_complete(output,"recipe")
            dump(folder/"status.json",dict(status="complete",epoch=2,updates=44))
            with self.assertRaises(ValueError):check_complete(output,"recipe")
            dump(folder/"status.json",dict(status="complete",epoch=100,updates=2200))
            dump(folder/"registered_config.json",dict(engineering_only=True,device="cuda",source_lock="synthetic_test_lock"))
            with self.assertRaises(ValueError):check_complete(output,"recipe")
            dump(folder/"registered_config.json",dict(engineering_only=False,device="cuda",source_lock="synthetic_test_lock"))
            (folder/"best_model.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):check_complete(output,"recipe")


if __name__=="__main__":unittest.main()
