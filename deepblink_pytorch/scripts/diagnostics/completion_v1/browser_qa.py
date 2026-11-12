"""Real headless-browser DOM checks on a SYNTHETIC copy; never real review."""
from pathlib import Path
import argparse,json,subprocess,re,html,base64,uuid
from diagnostics.handoff_v2.common import write,fresh,read

SCRIPT='''
<script>
setTimeout(()=>{try{
const ensure=(x,m)=>{if(!x)throw Error(m);};
ensure(packet.synthetic_fixture===true,'not synthetic');
ensure(document.getElementById('exportCsv').disabled,'blank export allowed');
ensure(document.getElementById('raw').naturalWidth===64,'image not loaded');
document.getElementById('reviewer').value='SYNTHETIC_BROWSER_QA';
document.getElementById('reviewer').dispatchEvent(new Event('input',{bubbles:true}));
document.querySelector('input[name="presence"][value="uncertain"]').checked=true;
document.querySelector('input[name="presence"][value="uncertain"]').dispatchEvent(new Event('change',{bubbles:true}));
document.getElementById('countUncertain').checked=true;
document.getElementById('countUncertain').dispatchEvent(new Event('change',{bubbles:true}));
document.querySelector('input[name="confidence"][value="low"]').checked=true;
document.querySelector('input[name="confidence"][value="low"]').dispatchEvent(new Event('change',{bubbles:true}));
ensure(document.getElementById('progress').value===1,'real DOM first answer failed');
const box=document.getElementById('rawView').getBoundingClientRect();
const click=new MouseEvent('click',{clientX:box.left+box.width/2,clientY:box.top+box.height/2,bubbles:true});
document.getElementById('rawView').dispatchEvent(click);
const expectedX=Math.min(63.999,Math.max(0,(click.clientX-box.left)*64/box.width-.5));
ensure(Math.abs(Number(rowFor(packet.cases[0]).suggested_center_x_patch_px)-expectedX)<.001,'native coordinate wrong');
document.getElementById('next').click();ensure(index===1,'navigation failed');
document.getElementById('prev').click();ensure(index===0,'back failed');
const backup=draft();validateDraft(backup);
for(const c of packet.cases)answers[c.anonymous_id]={...rowFor(c),candidate_is_real_spot:'uncertain',number_of_spots:'uncertain',review_confidence:'low'};
render();ensure(document.getElementById('progress').value===60,'complete count failed');
ensure(!document.getElementById('exportCsv').disabled,'complete export disabled');
const csv=csvText();ensure(csv.includes('"True"'),'synthetic flag missing');
localStorage.removeItem(storageKey);
const result={passed:true,synthetic_fixture:true,real_human_answers_entered:false,images_loaded:true,blank_export_blocked:true,
real_DOM_answer_navigation_and_center:true,complete_60_export:true,csv_base64:btoa(unescape(encodeURIComponent(csv)))};
const node=document.createElement('pre');node.id='browser-qa-result';node.textContent=JSON.stringify(result);document.body.append(node);
}catch(e){const node=document.createElement('pre');node.id='browser-qa-result';node.textContent=JSON.stringify({passed:false,error:String(e),synthetic_fixture:true});document.body.append(node);}},500);
</script>
'''


def run(public,output,chrome):
    from diagnostics.handoff_v2.reviewer import build,validate_answers
    from diagnostics.handoff_v2.common import table
    output=fresh(output);reviewer=output/'synthetic_reviewer';build(public,reviewer,synthetic=True)
    page=reviewer/'index.html';page.write_text(page.read_text(encoding='utf-8')+SCRIPT,encoding='utf-8')
    command=[str(chrome),'--headless=new','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check',
        '--user-data-dir='+str(output/'isolated_profile'),'--virtual-time-budget=2000','--dump-dom','--window-size=1200,1400',
        '--screenshot='+str(output/'synthetic_ui.png'),page.resolve().as_uri()]
    result=subprocess.run(command,capture_output=True,timeout=60)
    (output/'browser_stderr.txt').write_bytes(result.stderr)
    text=result.stdout.decode('utf-8',errors='replace');match=re.search(r'<pre id="browser-qa-result">(.*?)</pre>',text,re.S)
    if not match:raise RuntimeError('Browser did not produce a QA result; inspect stderr')
    check=json.loads(html.unescape(match.group(1)))
    if not check['passed']:raise AssertionError(check)
    csv_path=output/'SYNTHETIC_BROWSER_QA.csv';csv_path.write_bytes(base64.b64decode(check.pop('csv_base64')))
    rows=table(csv_path);validated=validate_answers(rows,[r['anonymous_id'] for r in rows],synthetic=True)
    if validated['human_review_completed']:raise AssertionError('Synthetic browser QA cannot complete human review')
    check.update(python_csv_roundtrip=True,browser_exit_code=result.returncode,review_packet_id=read(reviewer/'packet_checks.json')['packet_id'])
    write(output/'browser_qa.json',check);return check


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--chrome',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.public,a.output,a.chrome),indent=2))
