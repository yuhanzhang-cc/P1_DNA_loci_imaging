"use strict";
// Pure JS DOM-double integration test. This is NOT a browser visual test and
// never opens the real review packet or submits real human answers.
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict"),path=require("node:path");
class Element{
 constructor(tag="div"){this.tag=tag;this.children=[];this.handlers={};this.value="";this.checked=false;this.disabled=false;this.textContent="";}
 append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}
 addEventListener(name,fn){this.handlers[name]=fn;}setAttribute(name,value){this[name]=value;}
 getBoundingClientRect(){return {left:0,top:0,width:384,height:384};}
 click(){if(this.tag==="a")downloads.push({name:this.download,blob:blobs.get(this.href)});else if(this.onclick)this.onclick({});}
}
const ids=["packet","reviewer","completed","progress","exportCsv","notice","caseTitle","raw","marked","caseSelect","prev","next","position","presence","confidence","countUncertain","count","background","notes","crosshair","centerInfo","nextMissing","rawView","clearCenter","saveDraft","restore"];
const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
const fields=["anonymous_id","raw_file","marked_file","candidate_is_real_spot","number_of_spots","background_description","suggested_center_y_patch_px","suggested_center_x_patch_px","review_confidence","reviewer","notes","adjudicated_detection_label","adjudication_phase","synthetic_fixture","review_packet_id"];
const packet={schema:"anonymous_review_ui_v2",packet_id:"SYNTHETIC_ONLY",synthetic_fixture:true,fields,cases:Array.from({length:60},(_,i)=>({anonymous_id:`SIM${String(i).padStart(4,"0")}`,raw_file:"synthetic_raw.png",marked_file:"synthetic_marked.png",raw:"data:image/png;base64,SYNTHETIC",marked:"data:image/png;base64,SYNTHETIC"}))};
elements.packet.textContent=JSON.stringify(packet);
const storage=new Map(),blobs=new Map(),downloads=[];
function descendants(e){return [e,...e.children.flatMap(c=>typeof c==="object"?descendants(c):[])];}
const context={console,Blob,Date,setTimeout:()=>0,URL:{createObjectURL:b=>{const k="blob:"+blobs.size;blobs.set(k,b);return k;},revokeObjectURL:()=>{}},
 localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
 document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),createTextNode:s=>s,
 querySelector:s=>{const name=s.match(/name="([^"]+)"/)[1];return descendants(elements[name]).find(e=>e.name===name&&e.checked)||null;}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,"reviewer.js"),"utf8"),context);
function select(group,value){const all=descendants(elements[group]).filter(e=>e.name===group);for(const e of all)e.checked=e.value===value;all.find(e=>e.value===value).handlers.change();}
async function main(){
 assert.equal(elements.progress.value,0);assert.equal(elements.exportCsv.disabled,true);
 elements.exportCsv.onclick();assert.equal(downloads.length,0);
 elements.reviewer.value="SYNTHETIC_QA";elements.reviewer.handlers.input();
 select("presence","yes");elements.count.value="1";elements.count.handlers.input();select("confidence","high");
 assert.equal(elements.progress.value,1);assert.equal(elements.exportCsv.disabled,true);
 elements.rawView.onclick({clientX:195,clientY:195});assert.match(elements.centerInfo.textContent,/y=32\.00, x=32\.00/);
 assert.equal(elements.crosshair.children[0].d,"M 30.5 32.5 h 4 M 32.5 30.5 v 4");
 elements.notes.value='comma, quote " and\nnewline';elements.notes.handlers.input();
 elements.saveDraft.onclick();assert.match(downloads[0].name,/SYNTHETIC_QA/);
 const d=JSON.parse(await downloads[0].blob.text());assert.equal(d.synthetic_fixture,true);assert.equal(d.answers.SIM0000.suggested_center_y_patch_px,"32.000");
 context.P1Review.validateDraft(d);
 assert.throws(()=>context.P1Review.validateDraft({...d,packet_id:"OTHER"}));
 const bad=structuredClone(d);bad.answers.SIM0000.sample_id="UNBLINDING";assert.throws(()=>context.P1Review.validateDraft(bad));
 elements.next.onclick();assert.match(elements.caseTitle.textContent,/SIM0001/);assert.equal(elements.progress.value,1);
 elements.prev.onclick();assert.equal(elements.notes.value,'comma, quote " and\nnewline');
 elements.restore.files=[{text:async()=>JSON.stringify(d)}];await elements.restore.onchange();assert.equal(elements.progress.value,1);assert.equal(elements.notes.value,'comma, quote " and\nnewline');
 for(let i=1;i<60;i++){
  elements.caseSelect.value=String(i);elements.caseSelect.onchange();select("presence",i%3?"yes":"uncertain");
  if(i%3){elements.count.value="1";elements.count.handlers.input();}else{elements.countUncertain.checked=true;elements.countUncertain.handlers.change();}
  select("confidence","medium");
 }
 assert.equal(elements.progress.value,60);assert.equal(elements.exportCsv.disabled,false);
 elements.exportCsv.onclick();const item=downloads.at(-1);assert.match(item.name,/SYNTHETIC_QA/);const csv=await item.blob.text();
 assert.match(csv,/"synthetic_fixture"/);assert.match(csv,/"True"/);assert.match(csv,/quote "" and\nnewline/);
 if(process.argv[2])fs.writeFileSync(process.argv[2],csv);
 console.log(JSON.stringify({passed:true,synthetic_fixture:true,answers_never_real:true,checks:["blank_by_default","incomplete_export_blocked","one_case_progress","native_pixel_click_mapping","draft_backup_and_import","packet_binding","unblinding_rejected","navigation_restores","60_case_csv_export","quotes_and_newlines"],browser_visual_tested:false}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
