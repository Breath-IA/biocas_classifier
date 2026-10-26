"""
report.py  (gate 2 · e03)
─────────────────────────
Parte A: ¿las features físicas de plan.md separan W de FC y sobreviven al cambio de año? (+ clasificadores simples)
Parte B: resultados de la CNN (train_cnn.py) con y sin features físicas, contra las referencias de e02 (RF/LR con features).

Salida: results/{g4_*.png, ph_*.csv, report_auto.md}
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from analysis_utils import YCOL, YEARS, auc, groups, md  # noqa: E402

warnings.filterwarnings("ignore")
G3, G4 = paths.G2_E02 / "results", paths.G2_E03 / "results"   # referencias (e02) y resultados (e03)
TR = ["2022", "2023", "2024"]
CCOL = {"W": "#54A24B", "FC": "#E45756"}


def main():
    rep = ["# Gate 2 · e03 — CNN + features físicas de plan.md (W contra FC) — informe automático\n"]
    base = pd.read_csv(paths.SEGMENTS_CSV)
    base["source"] = base["source"].astype(str)
    base = base[base["class"].isin(["W", "FC"])]
    ph = pd.read_csv(paths.PHYSICS_CSV)
    pcols = [c for c in ph.columns if c.startswith("ph_")]
    df = base.merge(ph, on=["stem", "event_idx"], how="inner").reset_index(drop=True)
    isFC = lambda s: (s["class"] == "FC").values  # noqa: E731
    G = groups([c for c in base.columns if c not in {
        "stem", "source", "split", "patient_id", "age", "gender", "location", "event_idx", "class", "start_ms", "end_ms",
        "rec_duration_s", "n_events_rec", "start_frac", "poor_quality", "n_frames"} and not c.startswith("sp_")])
    robust = G["shape"] + G["tonal"] + G["temporal"]

    # ── A1. univariada de las features físicas ────────────────────────────
    rep += ["## A. Features físicas de plan.md\n", f"{len(df)} eventos W/FC; {len(pcols)} features. Valores faltantes:\n",
            md(df[pcols].isna().mean().round(3).sort_values(ascending=False).head(8).to_frame("fracción NaN")), "\n"]
    auc_y = pd.DataFrame({y: pd.Series({f: auc(df[df.source == y][f].values, isFC(df[df.source == y])) for f in pcols + ["env_kurtosis", "crest_factor", "spec_flux", "n_spec_peaks"]})
                          for y in YEARS})
    dom = {}
    for c in ("W", "FC"):
        s = df[df["class"] == c]
        a_, b_ = s[s.source.isin(TR)], s[s.source == "2025"]
        s = pd.concat([a_.sample(min(2500, len(a_)), random_state=42), b_.sample(min(2500, len(b_)), random_state=42)])
        yy = (s.source == "2025").values
        dom[c] = pd.Series({f: auc(s[f].values, yy) for f in auc_y.index})
    dom = (pd.DataFrame(dom) - .5).abs().mean(1) * 2
    M = auc_y - .5
    sg = np.sign(M.mean(1))
    ok = (M.mul(sg, axis=0) > 0).all(1) & (M.notna().sum(1) >= 3)
    margen = M.mul(sg, axis=0).min(1).where(ok, 0.0) * sg
    tab = auc_y.copy()
    tab.columns = [f"y{c}" for c in tab.columns]
    tab["margen"], tab["shift"], tab["score"] = margen, dom, margen.abs() * (1 - dom)
    tab = tab.sort_values("score", ascending=False)
    tab.to_csv(G4 / "ph_univariate.csv")
    rep += ["### AUC por año, W contra FC (>0,5 = mayor en FC); las cuatro últimas filas de referencia son features de gate 3\n",
            md(tab.loc[[i for i in tab.index if i.startswith("ph_")]]), "\n", "Referencias (gate 3):\n",
            md(tab.loc[["env_kurtosis", "crest_factor", "spec_flux", "n_spec_peaks"]]), "\n"]

    top = [i for i in tab.index if i.startswith("ph_")]
    fig, a = plt.subplots(figsize=(6.5, 9))
    im = a.imshow(auc_y.loc[top].values, cmap="RdBu_r", vmin=.15, vmax=.85, aspect="auto")
    a.set_xticks(range(4))
    a.set_xticklabels(YEARS)
    a.set_yticks(range(len(top)))
    a.set_yticklabels(top, fontsize=8)
    for i, f in enumerate(top):
        for j, y in enumerate(YEARS):
            v = auc_y.loc[f, y]
            a.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, shrink=.5, label="AUC (rojo: mayor en FC)")
    a.set_title("Features físicas — AUC W vs FC por año")
    plt.tight_layout()
    plt.savefig(G4 / "g4_ph_auc_by_year.png", dpi=105)
    plt.close()

    fl = top[:12]
    fig, axs = plt.subplots(3, 4, figsize=(20, 11))
    for a, f in zip(axs.ravel(), fl):
        for i, c in enumerate(["W", "FC"]):
            for j, y in enumerate(YEARS):
                v = df[(df["class"] == c) & (df.source == y)][f].dropna().values
                if len(v) < 8:
                    continue
                bp = a.boxplot(v, positions=[i * 5 + j], widths=.8, showfliers=False, patch_artist=True)
                bp["boxes"][0].set_facecolor(YCOL[y])
        a.set_xticks([1.5, 6.5])
        a.set_xticklabels(["W", "FC"])
        a.set_title(f, fontsize=10)
    fig.suptitle("Features físicas — mejores por score (azul 2022, verde 2023, naranja 2024, rojo 2025)")
    plt.tight_layout()
    plt.savefig(G4 / "g4_ph_dist_top12.png", dpi=95)
    plt.close()

    # ── A2. clasificadores simples con features (LOYO) ────────────────────
    def feats_eval(cols, kind, H):
        tr, te = df[df.source != H], df[df.source == H]
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(tr[cols].replace([np.inf, -np.inf], np.nan))
        Xte = imp.transform(te[cols].replace([np.inf, -np.inf], np.nan))
        if kind == "LR":
            m = make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0),
                              LogisticRegression(C=.3, class_weight="balanced", max_iter=3000))
        else:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=4, random_state=42)
        m.fit(Xtr, isFC(tr))
        p = m.predict_proba(Xte)[:, 1]
        return roc_auc_score(isFC(te), p), balanced_accuracy_score(isFC(te), p >= .5), p

    sets = {"físicas (ph_*)": pcols, "gate 3 `robust`": robust, "robust + físicas": robust + pcols}
    rows, ens = [], {}
    for sn, cols in sets.items():
        for kind in ("LR", "RF"):
            for H in YEARS:
                a_, b_, p = feats_eval(cols, kind, H)
                rows.append({"features": sn, "modelo": kind, "fold": H, "auc": a_, "bal_acc": b_, "n_feat": len(cols)})
                ens[(sn, kind, H)] = p
    fr = pd.DataFrame(rows)
    fr.to_csv(G4 / "feature_models.csv", index=False)
    rep.append("### Clasificadores simples (paso 2 del protocolo: ¿las features solas separan?)\n")
    for m in ("auc", "bal_acc"):
        p = fr.pivot_table(index=["features", "modelo"], columns="fold", values=m)
        p["media"] = p.mean(1)
        rep += [f"**{m}**\n", md(p.sort_values("media", ascending=False)), "\n"]

    # ── B. CNN ────────────────────────────────────────────────────────────
    rp = G4 / "cnn_results.csv"
    if not rp.exists():
        (G4 / "report_auto.md").write_text("\n".join(rep), encoding="utf-8")
        print("sin resultados de CNN todavía; informe parcial escrito")
        return
    res = pd.read_csv(rp)
    res["fold"] = res["fold"].astype(str)
    rep.append("## B. CNN (4 bloques conv, log-mel 64×94) con y sin features físicas\n")
    nseeds = res.groupby(["fold", "variant"]).size().min()
    rep.append(f"Semillas por celda (mínimo): {nseeds}. Early stopping con 15 % de pacientes de los años de entrenamiento.\n")
    agg = res.groupby(["fold", "variant"]).agg(auc=("auc", "mean"), auc_sd=("auc", "std"), bal_acc=("bal_acc", "mean"),
                                                 macro_f1=("macro_f1", "mean"), rec_W=("rec_W", "mean"), rec_FC=("rec_FC", "mean"),
                                                 epochs=("epochs", "mean"), val_auc=("best_val_auc", "mean"), secs=("secs", "mean")).reset_index()
    agg.to_csv(G4 / "cnn_agg.csv", index=False)
    for m in ("auc", "bal_acc", "macro_f1"):
        p = agg.pivot_table(index="variant", columns="fold", values=m)
        p["media"] = p.mean(1)
        rep += [f"### {m} (media de semillas)\n", md(p), "\n"]
    rep += ["### Detalle por fold\n", md(agg.set_index(["fold", "variant"])), "\n"]

    pr = pd.read_csv(G4 / "cnn_predictions.csv")
    pr["fold"] = pr["fold"].astype(str)
    ensm = pr.groupby(["fold", "variant", "stem", "event_idx"]).agg(p=("p", "mean"), y=("y", "first")).reset_index()
    ea = ensm.groupby(["fold", "variant"]).apply(lambda g: pd.Series({"auc_ens": roc_auc_score(g.y, g.p),
                                                                     "bal_acc_ens": balanced_accuracy_score(g.y, g.p >= .5)})).reset_index()
    rep += ["### Conjunto de semillas (promedio de probabilidades)\n", md(ea.pivot_table(index="variant", columns="fold", values="auc_ens")), "\n"]

    # diferencia pareada fusión − baseline
    w = res.pivot_table(index=["fold", "seed"], columns="variant", values="auc")
    if {"cnn", "cnn+phys"} <= set(w.columns):
        d = (w["cnn+phys"] - w["cnn"]).groupby("fold").agg(["mean", "std", "count"])
        d.loc["media"] = [d["mean"].mean(), np.nan, d["count"].sum()]
        rep += ["### ΔAUC pareada (cnn+phys − cnn), por fold y semilla\n", md(d), "\n"]

    # referencias de gate 3
    g3 = pd.read_csv(G3 / "classification_agg.csv")
    g3["fold"] = g3["fold"].astype(str)
    refs = {"LR robust (gate 3)": g3[(g3.model == "LR") & (g3.train == "B0") & (g3.features == "robust")],
            "RF robust (gate 3)": g3[(g3.model == "RF") & (g3.train == "B0") & (g3.features == "robust")],
            "RF all (gate 3)": g3[(g3.model == "RF") & (g3.train == "B0") & (g3.features == "all")]}
    comp = {k: v.set_index("fold")["auc"] for k, v in refs.items()}
    for (sn, kind), g in fr.groupby(["features", "modelo"]):
        if sn != "gate 3 `robust`":
            comp[f"{kind} {sn}"] = g.set_index("fold")["auc"]
    for v, g in agg.groupby("variant"):
        comp[v.upper()] = g.set_index("fold")["auc"]
    ct = pd.DataFrame(comp).T[YEARS]
    ct["media"] = ct.mean(1)
    rep += ["### Comparación general (ROC AUC por año retenido)\n", md(ct.sort_values("media", ascending=False)), "\n"]
    ct.to_csv(G4 / "comparison_auc.csv")

    fig, a = plt.subplots(figsize=(12, 5))
    ct2 = ct.sort_values("media")
    ct2[YEARS].plot.barh(ax=a, width=.85, color=[YCOL[y] for y in YEARS])
    a.set_xlim(.6, None)
    a.set_xlabel("ROC AUC")
    a.set_title("W vs FC — AUC por año retenido: CNN y baselines con features")
    a.grid(alpha=.3, axis="x")
    plt.tight_layout()
    plt.savefig(G4 / "g4_comparison.png", dpi=105)
    plt.close()

    hist = json.loads((G4 / "cnn_history.json").read_text())
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for a, H in zip(axs, YEARS):
        for var, col in (("cnn", "#4C78A8"), ("cnn+phys", "#E45756")):
            for k, h in hist.items():
                f, v, s = k.split("|")
                if f == H and v == var:
                    a.plot([e["epoch"] for e in h], [e["val_auc"] for e in h], color=col, alpha=.7, label=var if s == "0" else None)
        a.set_title(f"AUC de validación (pacientes de entrenamiento) — fold {H}", fontsize=9)
        a.set_xlabel("época")
        a.grid(alpha=.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(G4 / "g4_training_curves.png", dpi=100)
    plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for i, var in enumerate(("cnn", "cnn+phys")):
        for a, H in zip(axs[i], YEARS):
            g = ensm[(ensm.fold == H) & (ensm.variant == var)]
            if g.empty:
                a.axis("off")
                continue
            cm = confusion_matrix(g.y, (g.p >= .5).astype(int), labels=[0, 1]).astype(float)
            cmn = cm / cm.sum(1, keepdims=True)
            a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            for r in range(2):
                for c in range(2):
                    a.text(c, r, f"{cmn[r, c]:.2f}\n({int(cm[r, c])})", ha="center", va="center", fontsize=9,
                           color="white" if cmn[r, c] > .5 else "black")
            a.set_xticks([0, 1])
            a.set_yticks([0, 1])
            a.set_xticklabels(["W", "FC"])
            a.set_yticklabels(["W", "FC"])
            a.set_title(f"{var} · {H} (conjunto de semillas)", fontsize=9)
    plt.tight_layout()
    plt.savefig(G4 / "g4_confusion.png", dpi=100)
    plt.close()

    ip = G4 / "ph_importance.csv"
    if ip.exists():
        imp = pd.read_csv(ip)
        imp["fold"] = imp["fold"].astype(str)
        pv = imp.pivot_table(index="feature", columns="fold", values="drop_auc")
        pv["media"] = pv.mean(1)
        pv = pv.sort_values("media", ascending=False)
        rep += ["### Importancia de las features físicas en la fusión (permutación sobre el año retenido, caída de AUC)\n",
                md(pv.head(15), 4), "\n"]
        top = pv.head(15)[::-1]
        fig, a = plt.subplots(figsize=(8, 6))
        a.barh(top.index, top["media"], xerr=pv.loc[top.index, [c for c in pv.columns if c != "media"]].std(1))
        a.set_xlabel("caída de AUC (permutación)")
        a.set_title("Fusión CNN + física: importancia de cada feature física")
        plt.tight_layout()
        plt.savefig(G4 / "g4_ph_importance.png", dpi=105)
        plt.close()

    (G4 / "report_auto.md").write_text("\n".join(rep), encoding="utf-8")
    print("OK ->", G4 / "report_auto.md")


if __name__ == "__main__":
    main()
