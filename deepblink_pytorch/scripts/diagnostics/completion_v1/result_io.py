"""Fresh-directory result ingestion for the composite delivery and scratch tar."""
from pathlib import Path,PurePosixPath
import zipfile,tarfile,gzip,stat
from diagnostics.handoff_v2.common import sha,read,fresh,inside


def digest_check(archive,checksum):
    expected=Path(checksum).read_text(encoding='utf-8-sig').split()[0]
    if len(expected)!=64 or sha(archive)!=expected:raise ValueError('Downloaded SHA256 differs')


def safe_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in name or ':' in name or any(s in ('.git','.env') for s in p.parts):raise ValueError('Unsafe archive member')
    return p


def verify_files(root,manifest_name):
    m=read(Path(root)/manifest_name);actual={p.relative_to(root).as_posix() for p in Path(root).rglob('*') if p.is_file()}
    if actual!=set(m['files'])|{manifest_name}:raise ValueError('Undeclared/missing result members')
    for name,digest in m['files'].items():
        if sha(inside(root,name))!=digest:raise ValueError('Extracted file hash differs')
    return m


def delivery(archive,checksum,source_manifest,output,max_bytes=8*1024**3):
    return zip_result(archive,checksum,source_manifest,output,'delivery_manifest.json','completion_delivery_v1',max_bytes)


def zip_result(archive,checksum,source_manifest,output,manifest_name,schema,max_bytes=8*1024**3):
    digest_check(archive,checksum)
    with zipfile.ZipFile(archive) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        if len(files)>20000 or sum(i.file_size for i in files)>max_bytes:raise ValueError('Archive exceeds budget')
        names=[];prefix=set()
        for i in files:
            p=safe_member(i.filename)
            if len(p.parts)<2 or stat.S_ISLNK(i.external_attr>>16):raise ValueError('Unsafe ZIP layout/link')
            prefix.add(p.parts[0]);names.append(PurePosixPath(*p.parts[1:]).as_posix())
        if len(prefix)!=1 or len(set(names))!=len(names):raise ValueError('Duplicate or inconsistent ZIP roots')
        if z.testzip():raise ValueError('ZIP CRC failed')
        output=fresh(output)
        for i,name in zip(files,names):
            q=inside(output,name);q.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as src,q.open('xb') as dst:
                while block:=src.read(1024*1024):dst.write(block)
    m=verify_files(output,manifest_name)
    if m['schema']!=schema or m['completion_bundle_sha256']!=sha(source_manifest):raise ValueError('Delivery belongs to a different uploaded composite')
    return dict(output=str(output),verified=True,files=len(m['files']),source_binding_verified=True)


def all_results(archive,checksum,source_manifest,output):
    zip_result(archive,checksum,source_manifest,output,'all_results_manifest.json','completion_results_v1')
    output=Path(output);source=Path(source_manifest).parent;manifest=read(output/'all_results_manifest.json')
    if manifest.get('engineering_only') is not False:raise ValueError('Engineering collection cannot complete a formal return')
    required={'mechanism_results.tar.gz','delivery.zip','frozen_evaluation.zip'}
    if not required<=set(manifest['components']) or set(manifest['components'])-required-{'scratch_results.tar.gz'}:raise ValueError('Unknown/missing result components')
    result=dict(delivery=delivery(output/'delivery.zip',output/'delivery.zip.sha256',source_manifest,output/'verified_delivery'))
    from diagnostics.handoff_v2.archive import ingest
    result['mechanism']=ingest(output/'mechanism_results.tar.gz',output/'mechanism_results.tar.gz.sha256',output/'verified_mechanism',source_manifest=source/'mechanism/bundle_manifest.json')
    if 'scratch_results.tar.gz' in manifest['components']:
        result['scratch']=scratch(output/'scratch_results.tar.gz',output/'scratch_results.tar.gz.sha256',source/'p1_v4_training/bundle_manifest.json',output/'verified_scratch')
    write_path=output/'LOCAL_VERIFICATION.json'
    from diagnostics.handoff_v2.common import write
    write(write_path,result);return dict(output=str(output),all_downloaded_results_verified=True,components=result)


def scratch(archive,checksum,source_manifest,output,max_bytes=8*1024**3):
    digest_check(archive,checksum)
    with gzip.open(archive,'rb') as f:
        total=0
        while block:=f.read(8*1024*1024):
            total+=len(block)
            if total>max_bytes:raise ValueError('Archive exceeds budget')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[]
        for m in members:
            safe_member(m.name)
            if not m.isfile():raise ValueError('Only regular tar files accepted')
            names.append(m.name)
        if len(names)>20000 or len(set(names))!=len(names):raise ValueError('Duplicate/oversized tar')
        output=fresh(output)
        for m in members:
            q=inside(output,m.name);q.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,q.open('xb') as dst:
                while block:=src.read(1024*1024):dst.write(block)
    from .scratch import check_return
    result=check_return(output)
    if read(output/'scratch_return_manifest.json')['source_bundle_sha256']!=sha(source_manifest):raise ValueError('Scratch return belongs to another uploaded child')
    return dict(**result,output=str(output),source_binding_verified=True)
