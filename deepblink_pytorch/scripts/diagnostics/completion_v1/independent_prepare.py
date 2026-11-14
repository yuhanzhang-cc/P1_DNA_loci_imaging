"""Hash supplied independent files after an explicit provenance decision."""
from pathlib import Path
from diagnostics.handoff_v2.common import table,sha,write,csv_write


def prepare(metadata,evidence,reviewer,reviewed_at,notes,output,confirmed=False):
    metadata=Path(metadata).resolve();evidence=Path(evidence).resolve();output=Path(output).resolve();root=metadata.parent
    if not confirmed or not all(isinstance(v,str) and v.strip() and not v.startswith('TODO') for v in (reviewer,reviewed_at,notes)):raise ValueError('Explicit real provenance review is required; cannot infer independence')
    if output.parent!=root or not evidence.is_relative_to(root):raise ValueError('Manifest and provenance evidence must stay in the dataset folder')
    if output.exists():raise FileExistsError('Existing independent lock protected')
    rows=table(metadata)
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate metadata')
    required=('sample_id','image_file','label_file','height','width','biological_group_id','acquisition_id','source_container','collection_date','annotation_version')
    for r in rows:
        if any(not r.get(k,'').strip() or r[k].strip().casefold() in ('todo','unknown','na','none') or r[k].startswith('TODO') for k in required):raise ValueError('Incomplete actual acquisition metadata')
        for key,dest in (('image_file','image_sha256'),('label_file','label_sha256')):
            p=(root/r[key]).resolve()
            if not p.is_relative_to(root):raise ValueError('Dataset path escapes its root')
            r[dest]=sha(p)
        if r.get('coordinate_unit')!='pixel' or r.get('dataset_role')!='external_new':raise ValueError('Declare native pixel coordinates and new external role')
    locked=root/(output.stem+'_images.csv');fields=list(dict.fromkeys(k for r in rows for k in r))
    csv_write(locked,[{k:r.get(k,'') for k in fields} for r in rows],fields)
    spec=dict(dataset_role='independent_test',synthetic_fixture=False,independence_reviewed=True,reviewer=reviewer,reviewed_at=reviewed_at,independence_notes=notes,
        images_csv=locked.name,images_csv_sha256=sha(locked),independence_evidence_file=evidence.relative_to(root).as_posix(),independence_evidence_sha256=sha(evidence))
    write(output,spec)
    return dict(manifest=str(output),images=len(rows),test_evaluated=False,images_decoded=0,provenance_is_human_supplied=True)
