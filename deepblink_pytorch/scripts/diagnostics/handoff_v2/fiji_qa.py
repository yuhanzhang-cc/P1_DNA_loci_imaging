"""Headless core macro QA using an existing Fiji/JDK; no app-window control."""
from pathlib import Path
import json,subprocess
from .common import fresh,write,table,sha

JAVA=r'''import java.nio.file.*;
import java.nio.charset.StandardCharsets;
public class ImageJMacroQA {
 public static void main(String[] args) throws Exception {
  System.setProperty("java.awt.headless", "true");
  String macro=new String(Files.readAllBytes(Paths.get(args[0])),StandardCharsets.UTF_8);
  new ij.macro.Interpreter().run(macro);
  if(!Files.exists(Paths.get(args[1])))throw new IllegalStateException("Expected synthetic CSV not created");
  System.out.println("IMAGEJ_CORE_HEADLESS_MACRO_QA_PASS");
 }
}
'''
MACRO=r'''// SYNTHETIC ONLY: no original images, no manual answers.
requires("1.52i");
setBatchMode(true);
newImage("SYNTHETIC_QA", "32-bit black",64,64,1);
setPixel(4,2,17);
makeOval(0,0,8,8);
Overlay.addSelection("cyan",1);
setColor("cyan");
Overlay.drawString("SYNTHETIC",8,8);
makeSelection("point small green cross",newArray(4.5,10.125),newArray(2.25,20.875));
if(selectionType()!=10) exit("QA FAIL: wrong selection type");
getSelectionCoordinates(xs,ys);
if(xs.length!=2 || xs[0]!=4.5 || ys[0]!=2.25 || xs[1]!=10.125 || ys[1]!=20.875) exit("QA FAIL: subpixel coordinates changed");
if(getPixel(4,2)!=17) exit("QA FAIL: source pixels changed");
out=__OUTPUT__;
if(File.exists(out)) exit("QA output already exists");
text="sample_id,point_index,y,x\n";
for(i=0;i<xs.length;i++) text=text+"SYNTHETIC_QA,"+i+","+d2s(ys[i],6)+","+d2s(xs[i],6)+"\n";
File.saveString(text,out);
print("SYNTHETIC_MACRO_QA_PASS");
close();
'''


def run(fiji_root,output):
    fiji=Path(fiji_root);jars=list((fiji/"jars").glob("ij-*.jar"));javac=list((fiji/"java").rglob("bin/javac.exe"))
    if len(jars)!=1 or not javac:raise ValueError("An existing Fiji ImageJ core jar and JDK are required; nothing is installed automatically")
    javac=javac[0];java=javac.with_name("java.exe");output=fresh(output)
    source=output/"ImageJMacroQA.java";source.write_bytes(JAVA.encode())
    csv=output/"SYNTHETIC_points.csv";macro=output/"synthetic.ijm"
    macro.write_bytes(MACRO.replace("__OUTPUT__",json.dumps(csv.as_posix(),ensure_ascii=False)).encode())
    commands=[[str(javac),"-cp",str(jars[0]),"-d",str(output),str(source)],
        [str(java),"-Djava.awt.headless=true","-cp",str(jars[0])+";"+str(output),"ImageJMacroQA",str(macro),str(csv)]]
    logs=[]
    for cmd in commands:
        p=subprocess.run(cmd,capture_output=True,timeout=35)
        logs.append(dict(command=cmd,returncode=p.returncode,stdout=p.stdout.decode("utf-8",errors="replace"),stderr=p.stderr.decode("utf-8",errors="replace")))
        if p.returncode!=0:write(output/"checks.json",dict(passed=False,commands=logs,synthetic_fixture=True));raise RuntimeError("ImageJ core QA failed")
    rows=table(csv)
    passed=len(rows)==2 and [(float(r["y"]),float(r["x"])) for r in rows]==[(2.25,4.5),(20.875,10.125)]
    result=dict(passed=passed,synthetic_fixture=True,imagej_core_sha256=sha(jars[0]),commands=logs,
        subpixel_coordinates_preserved=passed,overlay_leaves_pixels_unchanged=passed,csv_export_verified=passed,
        browser_visual_tested=False,fiji_interactive_dragging_tested=False,original_images_modified=False)
    write(output/"checks.json",result)
    if not passed:raise AssertionError("Synthetic point CSV differs")
    return result
