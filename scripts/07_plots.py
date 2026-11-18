"""Step 7 - report figures, all generated from the saved experiment outputs.

Reads reports/*.csv and data/processed/*.npy, writes reports/figures/*.png.
Run:  python scripts/07_plots.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from pathsnap import config
from pathsnap.preprocess.geo import make_projector, project
from pathsnap.preprocess.io import list_trips, load_trip

R, P = config.REPORTS_DIR, config.PROCESSED_DIR
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150)
    plt.close()


def model_comparison():
    d = pd.read_csv(R / "model_comparison.csv")
    names = d["model"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, d.cv_macro_f1_mean, yerr=d.cv_macro_f1_std, capsize=4, color="#1f6feb")
    ax.set_ylim(0.7, 1.0); ax.set_ylabel("CV macro-F1 (5-fold GroupKFold, 80 train trips)")
    ax.set_title("Cross-validation macro-F1 (error bars = std over folds)")
    plt.xticks(rotation=30, ha="right"); save("cv_macro_f1")

    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(d)); w = 0.38
    ax.bar(x - w / 2, d.test_top1_all, w, label="Top-1 accuracy (obs-level)", color="#1f6feb")
    ax.bar(x + w / 2, d.test_macro_f1, w, label="Macro-F1 (candidate-level)", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.6, 1.0); ax.legend(); ax.set_title("Student-test performance (10 trips)")
    save("student_test_top1_f1")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, d.test_log_loss, color="#2ca02c"); ax.set_ylabel("Log loss (student-test)")
    ax.set_title("Log loss - lower is better"); plt.xticks(rotation=30, ha="right"); save("log_loss")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(d.train_fit_top1, d.test_top1_all)
    for _, r in d.iterrows():
        ax.annotate(r["model"], (r.train_fit_top1, r.test_top1_all), fontsize=8)
    lim = [min(d.train_fit_top1.min(), d.test_top1_all.min()) - .02, 1.0]
    ax.plot(lim, lim, "k--", lw=.8); ax.set_xlabel("Top-1 on training trips (refit)")
    ax.set_ylabel("Top-1 on student-test"); ax.set_title("Over/under-fitting check (dashed: y = x)")
    save("overfitting_check")


def confusion_matrices():
    te = pd.read_parquet(P / "student_test_candidates.parquet")
    d = pd.read_csv(R / "model_comparison.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, name in zip(axes.ravel(), d["model"]):
        p = np.load(P / f"student_{name}.npy")
        cm = confusion_matrix(te["label"], (p >= .5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(name)
    save("confusion_matrices_student_test")


def mlp_curve():
    f = R / "mlp_loss_curve.csv"
    if f.exists():
        d = pd.read_csv(f)
        plt.figure(figsize=(6, 4)); plt.plot(d.epoch, d.train_loss, marker="o")
        plt.xlabel("epoch"); plt.ylabel("training loss"); plt.title("MLP training loss (only model with epoch history)")
        save("mlp_training_loss")


def bayes():
    f = R / "bayes_trials.csv"
    if f.exists():
        d = pd.read_csv(f)
        plt.figure(figsize=(6, 4)); plt.plot(d.number, d.value, "o-", label="trial")
        plt.plot(d.number, d.value.cummax(), "r-", label="best so far")
        plt.xlabel("trial"); plt.ylabel("CV macro-F1"); plt.legend(); plt.title("Bayesian optimisation (Optuna TPE)")
        save("bayes_trials")
    f = R / "bayes_untuned_vs_tuned.csv"
    if f.exists():
        d = pd.read_csv(f)
        cols = ["cv_macro_f1", "test_macro_f1", "test_top1_all"]
        d.set_index("model")[cols].plot.bar(figsize=(7, 4), ylim=(0.8, 1.0), rot=0)
        plt.title("Untuned vs Bayesian-tuned HistGradientBoosting"); save("untuned_vs_tuned")


def hmm():
    f = R / "hmm_comparison.csv"
    if f.exists():
        d = pd.read_csv(f)
        piv = d.pivot(index="partition", columns="method", values=["top1_all", "route_continuity"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ["top1_all", "route_continuity"]):
            piv[metric].plot.bar(ax=ax, rot=15, ylim=(0.6, 1.0)); ax.set_title(metric)
        save("hmm_vs_classifier")


def eda():
    for part in ["train", "student_test"]:
        inv = pd.read_csv(R / f"inventory_{part}.csv", dtype={"trip_id": str})
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].hist(inv.n_obs, bins=20); axes[0].set_title(f"{part}: GPS observations per trip")
        axes[1].hist(inv.pct_dt_1s, bins=20); axes[1].set_title("% of steps exactly 1 s apart")
        axes[2].hist(np.log10(inv.n_arcs), bins=20); axes[2].set_title("log10(road arcs per trip)")
        save(f"eda_{part}_distributions")
    cs = pd.read_csv(R / "candidate_stats_train.csv", dtype={"trip_id": str}).sort_values("miss_rate", ascending=False)
    plt.figure(figsize=(9, 3.6)); plt.bar(range(len(cs)), cs.miss_rate)
    plt.xlabel("training trips, sorted"); plt.ylabel("candidate miss rate")
    plt.title("Share of observations whose reference arc is not within 50 m (train)"); save("eda_candidate_miss_rate")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, tid in zip(axes.ravel(), list_trips("train")[:6]):
        t = load_trip(tid, "train")
        tr = make_projector(t.track[:, 0], t.track[:, 1])
        xy = project(tr, t.track[:, 0], t.track[:, 1])
        nxy = project(tr, t.nodes[:, 0], t.nodes[:, 1])
        box = (np.abs(nxy[:, 0] - xy[:, 0].mean()) < np.ptp(xy[:, 0]) + 300) & (np.abs(nxy[:, 1] - xy[:, 1].mean()) < np.ptp(xy[:, 1]) + 300)
        arcs = t.arcs[box[t.arcs[:, 0]] & box[t.arcs[:, 1]]][::4]
        for a, b in arcs:
            ax.plot([nxy[a, 0], nxy[b, 0]], [nxy[a, 1], nxy[b, 1]], color="#bbb", lw=.4)
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.5, color="#1f6feb")
        ax.set_aspect("equal"); ax.set_title(f"train trip {tid}", fontsize=9)
    fig.suptitle("GPS trajectories (blue) over the road network (grey), local metric plane (m)")
    save("eda_trajectories")


if __name__ == "__main__":
    eda(); model_comparison(); confusion_matrices(); mlp_curve(); bayes(); hmm()
    print("figures written to", FIG)
