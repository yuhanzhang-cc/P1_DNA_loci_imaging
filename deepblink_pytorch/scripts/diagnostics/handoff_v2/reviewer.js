"use strict";
const packet=JSON.parse(document.getElementById("packet").textContent);
const $=id=>document.getElementById(id);
const storageKey="p1-blind-v2-"+packet.packet_id+(packet.synthetic_fixture?"-QA":"");
let index=0, reviewer="", answers={};
const blank=()=>Object.fromEntries(packet.fields.map(k=>[k,""]));
function rowFor(c){return {...blank(),...(answers[c.anonymous_id]||{}),anonymous_id:c.anonymous_id,raw_file:c.raw_file,marked_file:c.marked_file,reviewer,synthetic_fixture:packet.synthetic_fixture?"True":"False",review_packet_id:packet.packet_id};}
function valid(r){return reviewer.trim() && ["yes","no","uncertain"].includes(r.candidate_is_real_spot) && (r.number_of_spots==="uncertain" || /^\d+$/.test(r.number_of_spots)) && ["low","medium","high"].includes(r.review_confidence);}
function notify(s){$("notice").textContent=s;}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(draft()));}catch(e){notify("浏览器未能保存本地草稿；请下载 JSON 备份。");}}
function draft(){return {schema:"anonymous_review_answers_v2",packet_id:packet.packet_id,synthetic_fixture:packet.synthetic_fixture,reviewer,index,answers,saved_at:new Date().toISOString()};}
function validateDraft(d){
 if(d.schema!=="anonymous_review_answers_v2" || d.packet_id!==packet.packet_id || d.synthetic_fixture!==packet.synthetic_fixture || typeof d.reviewer!=="string" || !d.answers || Array.isArray(d.answers))throw Error("草稿不属于本审阅包");
 const ids=new Set(packet.cases.map(c=>c.anonymous_id));
 for(const [id,r] of Object.entries(d.answers)){
  if(!ids.has(id)||typeof r!=="object"||r===null)throw Error("未知匿名 ID");
  if(Object.keys(r).some(k=>!packet.fields.includes(k)))throw Error("草稿包含不允许的字段");
  if(Object.values(r).some(v=>typeof v!=="string"))throw Error("草稿字段必须为文本");
  if(r.anonymous_id && r.anonymous_id!==id)throw Error("草稿匿名 ID 不一致");
  if(r.adjudicated_detection_label || r.adjudication_phase)throw Error("盲审草稿不能包含解盲判定");
 }
 return d;
}
function useDraft(d){validateDraft(d);answers=d.answers;reviewer=d.reviewer;index=Number.isInteger(d.index)&&d.index>=0&&d.index<60?d.index:0;$("reviewer").value=reviewer;render();}
function update(){
 const c=packet.cases[index],r=rowFor(c);
 r.candidate_is_real_spot=document.querySelector('input[name="presence"]:checked')?.value||"";
 r.review_confidence=document.querySelector('input[name="confidence"]:checked')?.value||"";
 r.number_of_spots=$("countUncertain").checked?"uncertain":$("count").value;
 r.background_description=$("background").value;r.notes=$("notes").value;
 answers[c.anonymous_id]=r;persist();progress();
}
function progress(){const n=packet.cases.filter(c=>valid(rowFor(c))).length;$("completed").textContent=`已完成 ${n}/60`;$("progress").value=n;$("exportCsv").disabled=n!==60;}
function radios(where,name,values,selected){$(where).replaceChildren();for(const [value,label] of values){const l=document.createElement("label");l.className="option";const i=document.createElement("input");i.type="radio";i.name=name;i.value=value;i.checked=value===selected;i.addEventListener("change",update);l.append(i,document.createTextNode(label));$(where).append(l);}}
function cross(r){$("crosshair").replaceChildren();if(r.suggested_center_y_patch_px==="" || r.suggested_center_x_patch_px===""){$("centerInfo").textContent="建议中心：未填写（可选）";return;}const y=Number(r.suggested_center_y_patch_px),x=Number(r.suggested_center_x_patch_px);if(!Number.isFinite(x)||!Number.isFinite(y)||x<0||y<0||x>=64||y>=64)return;const p=document.createElementNS("http://www.w3.org/2000/svg","path");const vx=x+.5,vy=y+.5;/* SVG origin is a pixel edge; annotation coordinates use pixel centers. */p.setAttribute("d",`M ${vx-2} ${vy} h 4 M ${vx} ${vy-2} v 4`);p.setAttribute("stroke","#ffb300");p.setAttribute("stroke-width",".4");$("crosshair").append(p);$("centerInfo").textContent=`可选建议中心已记录（原生像素 y=${y.toFixed(2)}, x=${x.toFixed(2)}）；不自动改标签。`;}
function render(){
 const c=packet.cases[index],r=rowFor(c);$("caseTitle").textContent=c.anonymous_id+(packet.synthetic_fixture?" · SYNTHETIC QA":"");$("raw").src=c.raw;$("marked").src=c.marked;
 $("caseSelect").value=String(index);$("position").textContent=`${index+1}/60`;$("prev").disabled=index===0;$("next").disabled=index===59;
 radios("presence","presence",[["yes","有可信 spot"],["no","没有可信 spot"],["uncertain","不确定"]],r.candidate_is_real_spot);
 radios("confidence","confidence",[["low","低"],["medium","中"],["high","高"]],r.review_confidence);
 $("countUncertain").checked=r.number_of_spots==="uncertain";$("count").value=r.number_of_spots==="uncertain"?"":r.number_of_spots;$("count").disabled=$("countUncertain").checked;$("background").value=r.background_description;$("notes").value=r.notes;cross(r);progress();
}
function move(i){update();index=i;render();persist();}
function download(name,text,type){const blob=new Blob([text],{type});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csvQuote(v){return '"'+String(v??"").replaceAll('"','""')+'"';}
function csvText(){return packet.fields.map(csvQuote).join(",")+"\r\n"+packet.cases.map(c=>{const r=rowFor(c);return packet.fields.map(k=>csvQuote(r[k])).join(",");}).join("\r\n")+"\r\n";}
for(let i=0;i<60;i++){const option=document.createElement("option");option.value=String(i);option.textContent=packet.cases[i].anonymous_id;$("caseSelect").append(option);}
$("reviewer").addEventListener("input",()=>{reviewer=$("reviewer").value;persist();progress();});
for(const id of ["count","background","notes"])$(id).addEventListener("input",update);
$("countUncertain").addEventListener("change",()=>{$("count").disabled=$("countUncertain").checked;update();});
$("prev").onclick=()=>move(Math.max(0,index-1));$("next").onclick=()=>move(Math.min(59,index+1));$("caseSelect").onchange=()=>move(Number($("caseSelect").value));
$("nextMissing").onclick=()=>{update();for(let n=1;n<=60;n++){const i=(index+n)%60;if(!valid(rowFor(packet.cases[i]))){move(i);return;}}notify("全部必填项已完成，可以导出 CSV。");};
$("rawView").onclick=e=>{update();const box=$("rawView").getBoundingClientRect();const x=Math.min(63.999,Math.max(0,(e.clientX-box.left)*64/box.width-.5));const y=Math.min(63.999,Math.max(0,(e.clientY-box.top)*64/box.height-.5));const r=rowFor(packet.cases[index]);r.suggested_center_y_patch_px=y.toFixed(3);r.suggested_center_x_patch_px=x.toFixed(3);answers[r.anonymous_id]=r;cross(r);persist();};
$("clearCenter").onclick=()=>{const r=rowFor(packet.cases[index]);r.suggested_center_y_patch_px="";r.suggested_center_x_patch_px="";answers[r.anonymous_id]=r;cross(r);persist();};
$("saveDraft").onclick=()=>{update();download((packet.synthetic_fixture?"SYNTHETIC_QA_":"")+"review_draft.json",JSON.stringify(draft(),null,2),"application/json");notify("已请求下载草稿；请确认下载文件存在。");};
$("restore").onchange=async()=>{try{const file=$("restore").files[0];if(!file)return;useDraft(JSON.parse(await file.text()));persist();notify("草稿已恢复。");}catch(e){notify("恢复失败："+e.message);}};
$("exportCsv").onclick=()=>{update();if(!packet.cases.every(c=>valid(rowFor(c)))){notify("尚有未完成项。");return;}download((packet.synthetic_fixture?"SYNTHETIC_QA_":"")+"blind_audit_responses.csv","\ufeff"+csvText(),"text/csv;charset=utf-8");notify("已请求下载完整评分；请保留原件，不编辑旧标注。");};
try{const saved=localStorage.getItem(storageKey);if(saved)useDraft(JSON.parse(saved));else render();}catch(e){render();notify("未恢复本地草稿，可从 JSON 文件导入。");}
if(packet.synthetic_fixture)notify("SYNTHETIC QA：这里的点击仅测试界面，不能作为真实人工审查。");
// Explicit pure helpers support Node tests without inserting any reviewer answers.
globalThis.P1Review={validateDraft,csvQuote,packetId:packet.packet_id};
