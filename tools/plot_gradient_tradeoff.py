"""Compact displacement/P1 trade-off from completed summary JSON only.

Default points are the lambda pair, train-calibrated weight and BC-role pair.
Unfinished runs are skipped. No fields, models, checkpoints or CUDA are read.
For later studies, repeat --run LABEL SUMMARY GROUP (at most five points).
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if os.name=="nt" and (ROOT/".local_deps").is_dir():sys.path.insert(0,str(ROOT/".local_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,ScalarFormatter
from matplotlib.text import Text
from matplotlib.transforms import Bbox

DEFAULTS = [
    (r"$\lambda=0$","displacement_gradient_zero_10k","Gradient weight pair"),
    (r"$\lambda=1$","displacement_gradient_actual_10k","Gradient weight pair"),
    ("Train-calibrated lambda","displacement_gradient_calibrated_10k","Calibrated weight"),
    ("BC roles: zero","displacement_bc_role_zero_10k","BC-role pair"),
    ("BC roles: actual","displacement_bc_role_actual_10k","BC-role pair"),
]
MARKERS = ("o","D","s","^","v")
COLORS = ("#3073ac","#ce5832","#7b559f","#227e72","#ba8731")


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT/value


def read_points(specs):
    points = []; skipped = []
    for label,source,group in specs:
        source = path(source)
        if not source.is_file():
            skipped.append({"label":label,"summary":str(source),"reason":"summary not saved"});continue
        content = source.read_bytes(); summary = json.loads(content)
        if summary.get("status")!="completed":
            skipped.append({"label":label,"summary":str(source),"reason":f"status={summary.get('status','unknown')}"});continue
        best = summary["best_metrics"]
        validation = best.get("validation")
        if not validation or "strain" not in validation:
            skipped.append({"label":label,"summary":str(source),"reason":"no paired validation metrics; train-only diagnostics are not validation points"});continue
        x = float(validation["case_mean_relative_l2"])*100
        y = float(validation["strain"]["case_mean_relative_l2"])*100
        best_step = int(summary["best_step"])
        if not math.isfinite(x) or x<0 or not math.isfinite(y) or y<=0 or best_step<0 or ("step" in best and int(best["step"])!=best_step):
            raise ValueError(f"Invalid same-checkpoint physical/P1 summary: {source}")
        points.append({"label":label,"group":group,"summary":str(source),"summary_sha256":hashlib.sha256(content).hexdigest(),
                       "best_step":best_step,"gradient_weight":summary.get("gradient_weight"),
                       "validation_displacement_case_mean_relative_L2_percent":x,
                       "validation_P1_tensor_case_mean_relative_L2_percent":y})
    if not points:raise ValueError("No completed summary has both validation displacement and P1 metrics")
    if len(points)>5:raise ValueError("Use at most five points for this compact plot")
    return points,skipped


def annotate_points(ax,points,colors):
    """Place only a few short labels, avoiding prior label boxes and plot edge."""
    offsets = [(7,8),(7,-16),(-7,8),(-7,-16),(7,22),(-7,22),(7,-30),(-7,-30)]
    renderer = ax.figure.canvas.get_renderer(); occupied=[]
    marker_boxes=[]
    for point in points:
        px,py=ax.transData.transform((point["validation_displacement_case_mean_relative_L2_percent"],point["validation_P1_tensor_case_mean_relative_L2_percent"]))
        marker_boxes.append(Bbox.from_extents(px-9,py-9,px+9,py+9))
    for point,color in zip(points,colors):
        x=point["validation_displacement_case_mean_relative_L2_percent"]
        y=point["validation_P1_tensor_case_mean_relative_L2_percent"]
        options=[]
        for dx,dy in offsets:
            annotation=ax.annotate(point["label"],(x,y),xytext=(dx,dy),textcoords="offset points",
                                   ha="left" if dx>0 else "right",va="bottom" if dy>0 else "top",fontsize=8.6,
                                   color=color,bbox={"facecolor":"white","edgecolor":"none","alpha":.88,"pad":1.4},
                                   arrowprops={"arrowstyle":"-","color":color,"linewidth":.55,"alpha":.5,"shrinkA":2,"shrinkB":5})
            annotation.update_positions(renderer)
            box=Text.get_window_extent(annotation,renderer)
            padded=Bbox.from_extents(box.x0-3,box.y0-3,box.x1+3,box.y1+3)
            inside=ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1)
            collisions=sum(padded.overlaps(previous) for previous in occupied+marker_boxes)
            options.append((collisions+10*(not inside),annotation,padded))
        best=min(options,key=lambda option:option[0])
        for option in options:
            if option is not best:option[1].remove()
        occupied.append(best[2])


def make_plot(points,output):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(6.8,4.4),dpi=200)
    fig.subplots_adjust(left=.13,right=.97,bottom=.29,top=.86)
    groups=list(dict.fromkeys(point["group"] for point in points))
    group_markers={group:MARKERS[index] for index,group in enumerate(groups)}
    colors=COLORS[:len(points)]
    xs=[point["validation_displacement_case_mean_relative_L2_percent"] for point in points]
    ys=[point["validation_P1_tensor_case_mean_relative_L2_percent"] for point in points]
    xspread=max(xs)-min(xs);xpad=max(2.,xspread*.13)
    ax.set_xlim(max(0,min(xs)-xpad),max(xs)+xpad)
    ax.set_yscale("log");ax.set_ylim(min(min(ys),100)*.52,max(max(ys),100)*1.7)
    ticks=[value for value in (1,3,10,30,100,300,1000,3000,10000) if ax.get_ylim()[0]<=value<=ax.get_ylim()[1]]
    ax.yaxis.set_major_locator(FixedLocator(ticks));formatter=ScalarFormatter();formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter);ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis="both",which="major",color="#e4e7eb",linewidth=.7,zorder=0)
    ax.axhline(100,color="#7a7a7a",linewidth=1,linestyle=(0,(4,3)),zorder=1)
    ax.text(.014,100,"Zero-strain predictor: 100%",transform=ax.get_yaxis_transform(),va="bottom",fontsize=7.6,
            color="#666666",bbox={"facecolor":"white","edgecolor":"none","alpha":.92,"pad":1.4})
    for point,color in zip(points,colors):
        ax.scatter(point["validation_displacement_case_mean_relative_L2_percent"],
                   point["validation_P1_tensor_case_mean_relative_L2_percent"],
                   s=47,marker=group_markers[point["group"]],color=color,edgecolor="white",linewidth=.65,zorder=3)
    ax.set_xlabel("Validation displacement vector relative L2 (%)",labelpad=7)
    ax.set_ylabel("Validation P1 tensor relative L2 (%)",labelpad=7)
    ax.set_title("Displacement accuracy and P1 gradient accuracy",fontsize=11.8,pad=12)
    handles=[Line2D([],[],linestyle="none",marker=group_markers[group],color="#59636d",markersize=5,label=group) for group in groups]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.54,.13),ncol=min(3,len(groups)),frameon=False,fontsize=7.8,
               handletextpad=.5,columnspacing=1.4)
    fig.text(.13,.065,"Development validation; each point uses its displacement-selected checkpoint.",fontsize=7.7,color="#5f646b")
    fig.text(.13,.03,"P1 proxy from known-load u, not solver tensor labels. Final test is not evaluated.",fontsize=7.7,color="#5f646b")
    fig.canvas.draw();annotate_points(ax,points,colors)
    fig.savefig(output,dpi=200,facecolor="white");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root",default="runs/geobc_exploration")
    parser.add_argument("--output",default="results/geobc_exploration/gradient_tradeoff.png")
    parser.add_argument("--run",action="append",nargs=3,metavar=("LABEL","SUMMARY","GROUP"),help="Override default roster; repeat at most five times")
    args=parser.parse_args()
    specs=args.run or [(label,path(args.runs_root)/run/"summary.json",group) for label,run,group in DEFAULTS]
    if len(specs)>5:parser.error("At most five run specifications")
    points,skipped=read_points(specs);output=path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    make_plot(points,output)
    record={"status":"completed","created_at_utc":datetime.now(timezone.utc).isoformat(),"figure":str(output),"points":points,"skipped":skipped,
            "plot_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "x":"Known-load development-validation physical displacement vector case-mean relative L2 percent",
            "y":"Same displacement-selected best checkpoint; all-node incident-volume-averaged corner-P1 symmetric tensor Frobenius case-mean relative L2 percent",
            "y_scale":"log","reference_line_percent":100,"reference_line":"zero predicted strain gives100% relativeL2 for nonzero reference",
            "source_data":"completed run summary JSON only","arrays_or_checkpoints_read":False,"GPU_used":False,
            "limitations":"P1 proxy recovered from known-load displacement, not solver GT tensor; developmentvalidation reused; marker groups distinguish studies and matched controls. BC effects compare only new zero vs actual roles, not the historical xyz3 calibrated-weight point. No final-test inference or generalization conclusion"}
    metadata=output.with_suffix(".json");temp=metadata.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");temp.replace(metadata)
    print(json.dumps({"status":"completed","points":len(points),"skipped":len(skipped),"figure":str(output),"metrics":str(metadata)}),flush=True)


if __name__=="__main__":main()
