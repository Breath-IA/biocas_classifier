"""
analyze.py  (gate 2 · e02)
──────────────────────────
Igual que e01 pero SIN la clase N: clasificación binaria W (sibilancia) contra FC (crepitante fino).

Entradas : experiments/common/features/segments_spectral.csv, ../e01_flat_N_W_FC_augmentation/cache/aug_<variante>.csv
Salidas  : results/{g3_*.png, *.csv, report_auto.md}

Validación: dejar un año fuera (2022–2025); solo se aumenta el entrenamiento. Cada configuración se ajusta con
3 semillas para medir el ruido (media ± desviación entre semillas).
"""
from __future__ import annotations

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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from analysis_utils import META, YCOL, YEARS, auc, groups, md  # noqa: E402

warnings.filterwarnings("ignore")
G3 = paths.G2_E02 / "results"
AUG_DIR = paths.G2_E01 / "cache"     # copias aumentadas de W y FC (e01/augment.py)
CLS = ["W", "FC"]
CCOL = {"W": "#54A24B", "FC": "#E45756"}
VARIANTS = ["full", "eq", "noise", "jitter"]
SEEDS = [42, 7, 123]
TR_YEARS = ["2022", "2023", "2024"]


def main():
    G3.mkdir(parents=True, exist_ok=True)
    rep = ["# Gate 2 · e02 — W contra FC — informe automático\n"]
    df = pd.read_csv(paths.SEGMENTS_CSV)
    df["source"] = df["source"].astype(str)
    df = df[df["class"].isin(CLS)].reset_index(drop=True)
    aug = {}
    for v in VARIANTS:
        a = pd.read_csv(AUG_DIR / f"aug_{v}.csv")
        a["source"] = a["source"].astype(str)
        aug[v] = a[a["class"].isin(CLS)].reset_index(drop=True)
    feats = [c for c in df.columns if c not in META and not c.startswith("sp_") and pd.api.types.is_numeric_dtype(df[c])]
    sp = [f"sp_{k:02d}" for k in range(32)]
    G = groups(feats)
    gmap = {f: g for g, fs in G.items() for f in fs}
    gmap.update({f: "sp32" for f in sp})
    allcols = feats + sp
    sets = {
        "all": feats,
        "robust": G["shape"] + G["tonal"] + G["temporal"],
        "robust+rel": G["shape"] + G["tonal"] + G["temporal"] + G["rel"],
        "robust+sp32": G["shape"] + G["tonal"] + G["temporal"] + sp,
    }
    isFC = lambda s: (s["class"] == "FC").values  # noqa: E731

    # ── 1. Datos ──────────────────────────────────────────────────────────
    cnt = df.groupby(["class", "source"]).size().unstack(fill_value=0).reindex(CLS)
    cnt.loc["% FC"] = (100 * cnt.loc["FC"] / cnt.sum()).round(0)
    dur = df.groupby(["class", "source"])["duration_s"].median().unstack().reindex(CLS)
    comp = pd.DataFrame({"originales 2022–24": df[df.source.isin(TR_YEARS)].groupby("class").size()})
    for v, a in aug.items():
        comp[f"+aug {v}"] = a[a.source.isin(TR_YEARS)].groupby("class").size()
    comp["A total"] = comp["originales 2022–24"] + comp["+aug full"]
    comp["% aumentadas"] = (100 * comp["+aug full"] / comp["A total"]).round(0)
    rep += ["## 1. Datos\n", "Eventos originales (y % de FC por año):\n", md(cnt, 0), "\n",
            "Duración mediana (s):\n", md(dur, 2), "\n",
            "Entrenamiento 2022–24 (B0 = originales, sin submuestreo: W y FC ya están casi equilibradas):\n",
            md(comp.reindex(CLS), 0), "\n"]

    # ── 2. Features (originales) ──────────────────────────────────────────
    auc_y = pd.DataFrame({y: pd.Series({f: auc(df[df.source == y][f].values, isFC(df[df.source == y])) for f in allcols})
                          for y in YEARS})                      # >0.5 → mayor en FC
    dom = {}
    for c in CLS:
        s = df[df["class"] == c]
        a_, b_ = s[s.source.isin(TR_YEARS)], s[s.source == "2025"]
        s = pd.concat([a_.sample(min(2500, len(a_)), random_state=42), b_.sample(min(2500, len(b_)), random_state=42)])
        yy = (s.source == "2025").values
        dom[c] = pd.Series({f: auc(s[f].values, yy) for f in allcols})
    dom = (pd.DataFrame(dom) - .5).abs().mean(1) * 2
    M = auc_y - .5
    sg = np.sign(M.mean(1))
    ok = (M.mul(sg, axis=0) > 0).all(1) & (M.notna().sum(1) >= 3)
    margen = M.mul(sg, axis=0).min(1).where(ok, 0.0) * sg
    tab = auc_y.copy()
    tab.columns = [f"y{c}" for c in tab.columns]
    tab["margen"], tab["shift"], tab["score"], tab["grupo"] = margen, dom, margen.abs() * (1 - dom), pd.Series(gmap)
    tab = tab.sort_values("score", ascending=False)
    tab.to_csv(G3 / "features_W_vs_FC.csv")
    rep += ["## 2. Features: W contra FC por año\n",
            "AUC por año (**>0.5 = mayor en FC**). `margen` = peor |AUC−0.5| con el mismo signo en los 4 años; `shift` = "
            "desplazamiento train(2022–24) → 2025 dentro de la misma clase; `score = margen·(1−shift)`.\n",
            md(tab.head(25)), "\n"]
    gs = tab.groupby("grupo").agg(n=("score", "size"), mejor=("score", "max"), media_top5=("score", lambda x: x.nlargest(5).mean()),
                                   shift_medio=("shift", "mean"))
    rep += ["Resumen por grupo de features:\n", md(gs.sort_values("mejor", ascending=False)), "\n"]
    new = ["tonal_run_s", "tonal_frac", "harm_ratio_db", "itv_cv"]
    rep += ["Features nuevas:\n", md(tab.loc[new, ["y2022", "y2023", "y2024", "y2025", "margen", "shift"]]), "\n"]

    # control de duración: AUC por estrato de duración
    top8 = [f for f in tab.index if gmap.get(f) not in ("sp32",)][:8]
    strata = [("<1 s", 0, 1), ("1–2 s", 1, 2), (">=2 s", 2, 99)]
    cols = {}
    for nm, lo, hi in strata:
        for y in YEARS:
            s = df[(df.source == y) & (df.duration_s >= lo) & (df.duration_s < hi)]
            cols[(nm, y)] = pd.Series({f: auc(s[f].values, isFC(s)) for f in top8})
    st = pd.DataFrame(cols)
    st.columns = [f"{a}|{b}" for a, b in st.columns]
    ncnt = {f"{nm}|{y}": f"W{int(((df.source == y) & (df.duration_s >= lo) & (df.duration_s < hi) & (df['class'] == 'W')).sum())}/"
                          f"FC{int(((df.source == y) & (df.duration_s >= lo) & (df.duration_s < hi) & (df['class'] == 'FC')).sum())}"
            for nm, lo, hi in strata for y in YEARS}
    rep += ["### ¿Las features dependen de la duración anotada? AUC por estrato de duración\n",
            "(NaN = menos de 15 eventos en alguna clase). Tamaños: " + ", ".join(f"{k}: {v}" for k, v in ncnt.items()) + "\n",
            md(st, 2), "\n"]
    dauc = pd.Series({y: auc(df[df.source == y]["duration_s"].values, isFC(df[df.source == y])) for y in YEARS})
    rep += [f"AUC de `duration_s` por sí sola (W→FC): " + ", ".join(f"{y}: {v:.2f}" for y, v in dauc.items()) + "\n"]

    # redundancia
    top_ = [f for f in tab.index[:25]]
    corr = df[top_].rank().corr()
    sel, red = [], {}
    for f in top_:
        hit = [s for s in sel if abs(corr.loc[f, s]) > .85]
        if hit:
            red.setdefault(hit[0], []).append(f)
        else:
            sel.append(f)
    rep += ["### Redundancia (|ρ|>0.85 entre las 25 mejores)\n",
            "\n".join(f"- **{k}** ≈ {', '.join(v)}" for k, v in red.items()) or "(ninguna)", "\n",
            "Representantes: " + ", ".join(sel) + "\n"]

    # figuras de features
    fx = np.arange(32) * 62.5 + 31.25
    med = {(c, y): df[(df["class"] == c) & (df.source == y)][sp].median() for c in CLS for y in YEARS}
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8), sharex=True)
    for a, c in zip(ax[:2], CLS):
        for y in YEARS:
            a.plot(fx, med[(c, y)], color=YCOL[y], label=y)
        a.set_title(f"Espectro mediano de {c} por año")
    for y in YEARS:
        ax[2].plot(fx, med[("FC", y)] - med[("W", y)], color=YCOL[y], label=y)
    ax[2].axhline(0, color="k", lw=.6)
    ax[2].set_title("FC − W (dB relativo) por año")
    for a in ax:
        a.set_xlabel("Hz")
        a.grid(alpha=.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(G3 / "g3_spectra_by_year.png", dpi=105)
    plt.close()

    keep = list(tab.index[:30])
    fig, a = plt.subplots(figsize=(6, 10))
    im = a.imshow(auc_y.loc[keep].values, cmap="RdBu_r", vmin=.15, vmax=.85, aspect="auto")
    a.set_xticks(range(4))
    a.set_xticklabels(YEARS)
    a.set_yticks(range(len(keep)))
    a.set_yticklabels([f"{f} [{gmap.get(f, '')}]" for f in keep], fontsize=8)
    for i, f in enumerate(keep):
        for j, y in enumerate(YEARS):
            v = auc_y.loc[f, y]
            a.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, shrink=.5, label="AUC (rojo: mayor en FC)")
    a.set_title("W vs FC — AUC univariada por año")
    plt.tight_layout()
    plt.savefig(G3 / "g3_auc_by_year.png", dpi=105)
    plt.close()

    fl = [f for f in tab.index if gmap.get(f) not in ("sp32", "duration", "level", "mfcc")][:12]
    fig, axs = plt.subplots(3, 4, figsize=(20, 11))
    for a, f in zip(axs.ravel(), fl):
        for i, c in enumerate(CLS):
            for j, y in enumerate(YEARS):
                v = df[(df["class"] == c) & (df.source == y)][f].dropna().values
                if len(v) < 8:
                    continue
                bp = a.boxplot(v, positions=[i * 5 + j], widths=.8, showfliers=False, patch_artist=True)
                bp["boxes"][0].set_facecolor(YCOL[y])
                bp["boxes"][0].set_alpha(.85)
        a.set_xticks([1.5, 6.5])
        a.set_xticklabels(CLS)
        a.set_title(f"{f} [{gmap.get(f, '')}]", fontsize=10)
    fig.suptitle("W vs FC — mejores features por año (azul 2022, verde 2023, naranja 2024, rojo 2025)")
    plt.tight_layout()
    plt.savefig(G3 / "g3_dist_top12.png", dpi=95)
    plt.close()

    f1, f2 = fl[:2]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8))
    for a, y in zip(axs, YEARS):
        s = df[df.source == y]
        for c in CLS:
            q = s[s["class"] == c]
            a.scatter(q[f1], q[f2], s=8, alpha=.5, color=CCOL[c], label=f"{c} (n={len(q)})")
        a.set(xlim=tuple(np.nanpercentile(df[f1], [1, 99])), ylim=tuple(np.nanpercentile(df[f2], [1, 99])),
              xlabel=f1, ylabel=f2, title=y)
        a.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(G3 / "g3_scatter_best2.png", dpi=100)
    plt.close()

    # ── 3. Aumentación vs original ────────────────────────────────────────
    trk, t25 = df[df.source.isin(TR_YEARS)], df[df.source == "2025"]
    rob = sets["robust"]

    def mean_shift(rows, fl_):
        out = []
        for c in CLS:
            a_, b_ = rows[rows["class"] == c], t25[t25["class"] == c]
            s = pd.concat([a_.sample(min(len(a_), 2500), random_state=42), b_.sample(min(len(b_), 2500), random_state=42)])
            yy = np.r_[np.zeros(min(len(a_), 2500)), np.ones(min(len(b_), 2500))].astype(bool)
            out.append([abs(auc(s[f].values, yy) - .5) * 2 for f in fl_])
        return np.nanmean(out)

    key = ["spec_slope_db_khz", "spec_centroid", "band_600_800", "band_1250_1500", "rel_tilt_db", "spec_rolloff85"]
    rows = [{"entrenamiento": "originales 2022–24", "shift medio (robust)": mean_shift(trk, rob), "shift (forma espectral)": mean_shift(trk, key)}]
    for v, a in aug.items():
        comb = pd.concat([trk, a[a.source.isin(TR_YEARS)]])
        rows.append({"entrenamiento": f"orig + aug {v}", "shift medio (robust)": mean_shift(comb, rob), "shift (forma espectral)": mean_shift(comb, key)})
    rep += ["## 3. Aumentación y desplazamiento de dominio\n",
            "|2·AUC−1| entre entrenamiento y 2025 dentro de cada clase (menor = más parecido a 2025):\n",
            md(pd.DataFrame(rows).set_index("entrenamiento")), "\n"]

    # ── 4. Clasificación binaria ──────────────────────────────────────────
    def trainset(H, mode):
        o = df[df.source != H]
        if mode == "B0":
            return o
        a = aug[mode.split("_", 1)[1]]
        return pd.concat([o, a[a.source != H]], ignore_index=True)

    modes = ["B0"] + [f"A_{v}" for v in VARIANTS]

    def stable_cols(tr, k=15):
        ys = [y for y in YEARS if (tr.source == y).any()]
        Mm = pd.DataFrame({y: pd.Series({f: auc(tr[tr.source == y][f].values, isFC(tr[tr.source == y])) for f in feats}) for y in ys}) - .5
        sc = Mm.mul(np.sign(Mm.mean(1)), axis=0).min(1)
        return list(sc[sc > 0].sort_values(ascending=False).head(k).index)

    def make(kind, seed):
        if kind == "RF":
            return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                          n_jobs=4, random_state=seed)
        return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed),
                             LogisticRegression(C=.3, class_weight="balanced", max_iter=3000))

    def fit_eval(trd, ted, cols, kind, seed):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(trd[cols].replace([np.inf, -np.inf], np.nan))
        Xte = imp.transform(ted[cols].replace([np.inf, -np.inf], np.nan))
        ytr, yte = isFC(trd).astype(int), isFC(ted).astype(int)
        m = make(kind, seed).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= .5).astype(int)
        return {"auc": roc_auc_score(yte, p), "bal_acc": balanced_accuracy_score(yte, pred),
                "macro_f1": f1_score(yte, pred, average="macro"),
                "rec_W": float(((pred == 0) & (yte == 0)).sum() / max((yte == 0).sum(), 1)),
                "rec_FC": float(((pred == 1) & (yte == 1)).sum() / max((yte == 1).sum(), 1))}, pred, m, Xte

    rows, confs = [], {}
    for H in YEARS:
        ted = df[df.source == H]
        sets_ext = dict(sets)
        sets_ext["stable"] = stable_cols(df[df.source != H])
        for mode in modes:
            trd = trainset(H, mode)
            for sname, cols in sets_ext.items():
                for kind in (["RF", "LR"] if sname in ("robust", "stable") else ["RF"]):
                    for seed in (SEEDS if kind == "RF" else SEEDS[:1]):
                        r, pred, _, _ = fit_eval(trd, ted, cols, kind, seed)
                        rows.append({"fold": H, "train": mode, "features": sname, "model": kind, "seed": seed,
                                     "n_feat": len(cols), "n_train": len(trd), **r})
                        if seed == 42:
                            confs[f"{H}|{mode}|{sname}|{kind}"] = confusion_matrix(isFC(ted).astype(int), pred, labels=[0, 1]).tolist()
        print(f"  fold {H} listo", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(G3 / "classification_results.csv", index=False)

    agg = res.groupby(["fold", "train", "features", "model"]).agg(
        auc=("auc", "mean"), auc_sd=("auc", "std"), bal_acc=("bal_acc", "mean"), macro_f1=("macro_f1", "mean"),
        rec_W=("rec_W", "mean"), rec_FC=("rec_FC", "mean")).reset_index()
    agg.to_csv(G3 / "classification_agg.csv", index=False)
    rep.append("## 4. Clasificación W contra FC (RandomForest 200 árboles, 3 semillas; LR = regresión logística sobre cuantiles)\n")
    rep.append("`B0` = originales de los años de entrenamiento; `A_x` = B0 + copias aumentadas variante x. `stable` = 15 mejores "
               "features por consistencia entre los años de entrenamiento. Umbral 0,5 con pesos de clase balanceados.\n")
    for metric in ("auc", "bal_acc", "macro_f1"):
        p = agg.pivot_table(index=["model", "train", "features"], columns="fold", values=metric)
        p["media"] = p.mean(1)
        rep += [f"### {metric} (media = promedio de los 4 años; promedio de 3 semillas en RF)\n",
                md(p.sort_values("media", ascending=False)), "\n"]
    sd = res[res.model == "RF"].groupby(["fold", "train", "features"])["auc"].std().groupby(["train", "features"]).mean()
    rep += [f"Ruido por semilla (desviación típica del AUC entre semillas, media sobre folds): "
            f"mediana {sd.median():.4f}, máximo {sd.max():.4f}.\n"]
    d25 = agg[agg.fold == "2025"].set_index(["model", "train", "features"])[["auc", "auc_sd", "bal_acc", "rec_W", "rec_FC"]]
    rep += ["### Detalle en 2025 (split oficial)\n", md(d25.sort_values("auc", ascending=False)), "\n"]

    rf = agg[agg.model == "RF"]
    base = rf[rf.train == "B0"].set_index(["fold", "features"])["auc"]
    dl = []
    for m in modes[1:]:
        t = rf[rf.train == m].set_index(["fold", "features"])["auc"] - base
        dl.append(t.groupby("fold").mean().rename(m))
    dl = pd.concat(dl, axis=1)
    dl.loc["media"] = dl.mean()
    rep += ["### Efecto de la aumentación: ΔAUC contra B0 (media sobre conjuntos de features)\n", md(dl), "\n"]

    fig, axs = plt.subplots(1, 2, figsize=(17, 5.2))
    for a, (nm, sub) in zip(axs, (("media 4 años", rf), ("2025", rf[rf.fold == "2025"]))):
        p = sub.pivot_table(index="features", columns="train", values="auc")[modes]
        p.plot.bar(ax=a, width=.85)
        a.set_ylim(.5, None)
        a.set_ylabel("ROC AUC")
        a.set_title(f"W vs FC — {nm}")
        a.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(G3 / "g3_models.png", dpi=105)
    plt.close()

    pf = rf.pivot_table(index="fold", columns=["train"], values="auc", aggfunc="mean")
    fig, a = plt.subplots(figsize=(8, 4.5))
    pf[modes].plot.bar(ax=a, width=.85)
    a.set_ylim(.5, None)
    a.set_ylabel("ROC AUC (media de conjuntos de features)")
    a.set_title("AUC por año retenido")
    a.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(G3 / "g3_auc_by_fold.png", dpi=105)
    plt.close()

    best = rf.groupby(["train", "features"])["auc"].mean().sort_values(ascending=False).index[0]
    bmode, bfeat = best
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for a, H in zip(axs, YEARS):
        cm = np.array(confs[f"{H}|{bmode}|{bfeat}|RF"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=9,
                       color="white" if cmn[i, j] > .5 else "black")
        a.set_xticks([0, 1])
        a.set_yticks([0, 1])
        a.set_xticklabels(CLS)
        a.set_yticklabels(CLS)
        a.set_xlabel("predicho")
        a.set_title(f"{H} · {bmode} · {bfeat}", fontsize=9)
    plt.tight_layout()
    plt.savefig(G3 / "g3_confusion.png", dpi=105)
    plt.close()

    # ── 5. Importancia fuera de dominio ───────────────────────────────────
    rep.append(f"## 5. Importancia fuera de dominio (config ganadora por AUC medio: `{bmode}` + `{bfeat}`, RF)\n")
    imps, gini = [], None
    for H in YEARS:
        ted = df[df.source == H]
        sets_b = dict(sets)
        sets_b["stable"] = stable_cols(df[df.source != H])
        cols = sets_b[bfeat]
        r, pred, m, Xte = fit_eval(trainset(H, bmode), ted, cols, "RF", 42)
        pi = permutation_importance(m, Xte, isFC(ted).astype(int), scoring="roc_auc", n_repeats=5, n_jobs=4, random_state=42)
        imps.append(pd.Series(pi.importances_mean, index=cols))
        if H == "2025":
            gini = pd.Series(m.feature_importances_, index=cols)
    I = pd.concat(imps, axis=1)
    I.columns = YEARS
    I["media"], I["gini_2025"], I["grupo"] = I.mean(1), gini, pd.Series(gmap)
    I = I.sort_values("media", ascending=False)
    I.to_csv(G3 / "permutation_importance.csv")
    rep += ["Permutación (caída de AUC en el año retenido al barajar la feature):\n", md(I.head(20)), "\n"]
    pos = I[I["media"] > 0].groupby("grupo")["media"].sum().sort_values(ascending=False)
    rep += ["Importancia por grupo (solo contribuciones positivas):\n", md(pos.to_frame("suma")), "\n"]
    top = I.head(20)[::-1]
    fig, a = plt.subplots(figsize=(9, 8))
    colmap = {"shape": "#4C78A8", "tonal": "#54A24B", "temporal": "#E45756", "rel": "#F58518", "mfcc": "#9D755D",
              "level": "#9D755D", "duration": "#9D755D", "sp32": "#B279A2"}
    a.barh(top.index, top["media"], xerr=I.loc[top.index, YEARS].std(1), color=[colmap.get(g, "gray") for g in top["grupo"]])
    a.set_xlabel("caída de AUC (permutación, año retenido)")
    a.set_title(f"Top 20 — {bmode} + {bfeat}\n(azul forma · verde tonal · rojo temporal · naranja relativas)")
    plt.tight_layout()
    plt.savefig(G3 / "g3_importance.png", dpi=105)
    plt.close()

    (G3 / "report_auto.md").write_text("\n".join(rep), encoding="utf-8")
    print("OK ->", G3 / "report_auto.md", "| mejor:", best)


if __name__ == "__main__":
    main()
