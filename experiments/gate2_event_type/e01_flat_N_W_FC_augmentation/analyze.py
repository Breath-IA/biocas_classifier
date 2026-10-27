"""
analyze.py  (gate 2 · e01)
──────────────────────────
Clasificación de 3 clases N / W / FC directamente (sin pasar por el gate 1), con y sin aumentación.

Entradas : experiments/common/features/segments_spectral.csv, results/plan.csv, cache/aug_<variante>.csv
Salidas  : results/{g2_*.png, *.csv, report_auto.md}

Validación: dejar un año fuera (2022, 2023, 2024, 2025). Solo se aumenta el conjunto de ENTRENAMIENTO;
el año retenido se evalúa siempre con eventos originales y con su prevalencia real.
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths  # noqa: E402
from analysis_utils import META, YCOL, YEARS, auc, groups, md  # noqa: E402

G2 = paths.G2_E01 / "results"          # informes, figuras y CSV pequeños
CACHE = paths.G2_E01 / "cache"         # copias aumentadas (augment.py), ignoradas por git
RNG = 42
CLS = ["N", "W", "FC"]
CCOL = {"N": "#4C78A8", "W": "#54A24B", "FC": "#E45756"}
PAIRS = [("N", "W"), ("N", "FC"), ("W", "FC")]
VARIANTS = ["full", "eq", "noise", "jitter"]


def main():
    rep = ["# Gate 2 · e01 — N / W / FC plano (sin gate 1) — informe automático\n"]
    df = pd.read_csv(paths.SEGMENTS_CSV)
    df["source"] = df["source"].astype(str)
    df = df[df["class"].isin(CLS)].reset_index(drop=True)
    plan = pd.read_csv(G2 / "plan.csv")[["stem", "event_idx", "keep_orig", "n_aug"]]
    df = df.merge(plan, on=["stem", "event_idx"], how="left")
    df["keep_orig"] = df["keep_orig"].fillna(True).astype(bool)
    aug = {}
    for v in VARIANTS:
        p = CACHE / f"aug_{v}.csv"
        if p.exists():
            a = pd.read_csv(p)
            a["source"] = a["source"].astype(str)
            aug[v] = a
    feats = [c for c in df.columns if c not in META and not c.startswith("sp_") and pd.api.types.is_numeric_dtype(df[c])]
    sp = [f"sp_{k:02d}" for k in range(32)]
    G = groups(feats)
    sets = {
        "all": feats,
        "robust": G["shape"] + G["tonal"] + G["temporal"],
        "robust+rel": G["shape"] + G["tonal"] + G["temporal"] + G["rel"],
        "robust+sp32": G["shape"] + G["tonal"] + G["temporal"] + sp,
    }
    allcols = feats + sp

    # ── 1. Datos ──────────────────────────────────────────────────────────
    cnt = df.groupby(["class", "source"]).size().unstack(fill_value=0).reindex(CLS)
    dur = df.groupby(["class", "source"])["duration_s"].median().unstack().reindex(CLS)
    rep += ["## 1. Datos\n", "Eventos originales por clase y año:\n", md(cnt, 0), "\n",
            "Duración mediana del evento (s) — la anotación cambia entre años:\n", md(dur, 2), "\n"]
    tr_years = ["2022", "2023", "2024"]
    comp = pd.DataFrame({
        "B0 originales (todo)": df[df.source.isin(tr_years)].groupby("class").size(),
        "B1 originales (plan)": df[df.source.isin(tr_years) & df.keep_orig].groupby("class").size(),
    })
    for v, a in aug.items():
        comp[f"+aug {v}"] = a[a.source.isin(tr_years)].groupby("class").size()
    comp["A_full total"] = comp["B1 originales (plan)"] + comp.get("+aug full", 0)
    comp["% aumentadas (full)"] = (100 * comp.get("+aug full", 0) / comp["A_full total"]).round(0)
    rep += ["Composición del entrenamiento (2022–24):\n", md(comp.reindex(CLS), 0), "\n"]

    # ── 2. Reporte de features (solo originales) ──────────────────────────
    d = df
    auc_y = {}
    for a, b in PAIRS:
        rows = {}
        for y in YEARS:
            s = d[(d.source == y) & d["class"].isin([a, b])]
            yy = (s["class"] == b).values
            rows[y] = pd.Series({f: auc(s[f].values, yy) for f in allcols})
        auc_y[f"{a}_vs_{b}"] = pd.DataFrame(rows)       # AUC > .5 → mayor en la 2ª clase
    # desplazamiento de dominio: train (2022–24) vs 2025, misma clase
    dom = {}
    for c in CLS:
        s = d[d["class"] == c]
        s = pd.concat([s[s.source.isin(tr_years)].sample(min(2500, (s.source.isin(tr_years)).sum()), random_state=RNG),
                       s[s.source == "2025"].sample(min(2500, (s.source == "2025").sum()), random_state=RNG)])
        yy = (s.source == "2025").values
        dom[c] = pd.Series({f: auc(s[f].values, yy) for f in allcols})
    dom = (pd.DataFrame(dom) - 0.5).abs().mean(1) * 2
    gmap = {f: g for g, fs in G.items() for f in fs}
    gmap.update({f: "sp32" for f in sp})
    stab = {}
    for k, t in auc_y.items():
        M = t - 0.5
        sg = np.sign(M.mean(1))
        ok = (M.mul(sg, axis=0) > 0).all(1) & (M.notna().sum(1) >= 3)
        stab[k] = M.mul(sg, axis=0).min(1).where(ok, 0.0) * sg
    stab = pd.DataFrame(stab)
    score = stab.abs().mul(1 - dom, axis=0)
    pd.concat({k: t for k, t in auc_y.items()}, axis=1).to_csv(G2 / "auc_by_year.csv")
    dom.to_csv(G2 / "domain_shift.csv")

    rep.append("## 2. Features: separabilidad por par y año\n")
    rep.append("AUC por año; **>0.5 = mayor en la segunda clase del par**. `margen` = peor |AUC−0.5| entre los 4 años "
               "con el mismo signo (0 si cambia de signo). `shift` = desplazamiento train→2025 dentro de las mismas clases "
               "(0 = ninguno, 1 = separables). `score = margen·(1−shift)`.\n")
    tops = {}
    for k, t in auc_y.items():
        tt = t.copy()
        tt.columns = [f"y{c}" for c in tt.columns]
        tt["margen"], tt["shift"], tt["score"], tt["grupo"] = stab[k], dom, score[k], pd.Series(gmap)
        tt = tt.sort_values("score", ascending=False)
        tops[k] = tt
        rep += [f"### {k.replace('_vs_', ' vs ')}\n", md(tt.head(12)), "\n"]
        tt.to_csv(G2 / f"features_{k}.csv")

    # features nuevas
    new = ["tonal_run_s", "tonal_frac", "harm_ratio_db", "itv_cv"]
    nt = pd.concat({k: auc_y[k].loc[new].mean(1) for k in auc_y}, axis=1).round(3)
    nt["shift"] = dom.loc[new].round(3)
    rep += ["### Features nuevas (AUC media sobre los 4 años)\n", md(nt), "\n"]

    # redundancia
    top_union = list(dict.fromkeys(sum([list(tops[k].index[:15]) for k in tops], [])))
    sub = df[df.source.isin(YEARS)][top_union].rank()
    corr = sub.corr()
    sel, red = [], {}
    for f in top_union:
        hit = [s for s in sel if abs(corr.loc[f, s]) > 0.85]
        if hit:
            red.setdefault(hit[0], []).append(f)
        else:
            sel.append(f)
    rep += ["### Redundancia (|ρ Spearman| > 0.85 entre las mejores)\n",
            "Representantes → features que aportan casi lo mismo:\n",
            "\n".join(f"- **{k}** ≈ {', '.join(v)}" for k, v in red.items()) or "(ninguna)", "\n"]

    # figuras de features
    fx = np.arange(32) * 62.5 + 31.25
    fig, ax = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    med = {(c, y): df[(df["class"] == c) & (df.source == y)][sp].median() for c in CLS for y in YEARS}
    for a, c in zip(ax[0], CLS):
        for y in YEARS:
            a.plot(fx, med[(c, y)], color=YCOL[y], label=y)
        a.set_title(f"Espectro mediano de {c} por año")
        a.legend()
    for a, (c1, c2) in zip(ax[1], PAIRS):
        for y in YEARS:
            a.plot(fx, med[(c2, y)] - med[(c1, y)], color=YCOL[y], label=y)
        a.axhline(0, color="k", lw=.6)
        a.set_title(f"{c2} − {c1} (dB relativo) por año")
        a.legend()
    for a in ax.ravel():
        a.set_xlabel("Hz")
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(G2 / "g2_spectra_by_year.png", dpi=105)
    plt.close()

    # heatmap de AUC por año
    keep = list(dict.fromkeys(sum([list(tops[k].index[:12]) for k in tops], [])))[:34]
    mat = pd.concat({k: auc_y[k].loc[keep] for k in auc_y}, axis=1)
    fig, a = plt.subplots(figsize=(11, 12))
    im = a.imshow(mat.values, cmap="RdBu_r", vmin=0.15, vmax=0.85, aspect="auto")
    a.set_xticks(range(mat.shape[1]))
    a.set_xticklabels([f"{k.replace('_vs_', '→')}\n{y}" for k, y in mat.columns], fontsize=7)
    a.set_yticks(range(len(keep)))
    a.set_yticklabels([f"{f} [{gmap.get(f, '')}]" for f in keep], fontsize=8)
    for x in (3.5, 7.5):
        a.axvline(x, color="k", lw=1.2)
    fig.colorbar(im, shrink=.5, label="AUC (rojo: mayor en la 2ª clase)")
    a.set_title("AUC univariada por par y año — ¿se mantiene el signo?")
    plt.tight_layout()
    plt.savefig(G2 / "g2_auc_by_year.png", dpi=105)
    plt.close()

    # distribuciones por clase×año
    for k, tt in tops.items():
        fl = list(tt.index[:6])
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        for a, f in zip(axs.ravel(), fl):
            for i, c in enumerate(CLS):
                for j, y in enumerate(YEARS):
                    v = df[(df["class"] == c) & (df.source == y)][f].dropna().values
                    if len(v) < 8:
                        continue
                    bp = a.boxplot(v, positions=[i * 5 + j], widths=.8, showfliers=False, patch_artist=True)
                    bp["boxes"][0].set_facecolor(YCOL[y])
                    bp["boxes"][0].set_alpha(.8)
            a.set_xticks([i * 5 + 1.5 for i in range(3)])
            a.set_xticklabels(CLS)
            a.set_title(f"{f}  [{gmap.get(f, '')}]", fontsize=10)
        fig.suptitle(f"{k.replace('_vs_', ' vs ')} — mejores features (colores = año: "
                     + ", ".join(f"{y}" for y in YEARS) + ")", fontsize=12)
        plt.tight_layout()
        plt.savefig(G2 / f"g2_dist_{k}.png", dpi=100)
        plt.close()

    # dispersión de las 2 mejores features por par (train vs 2025)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for j, (k, tt) in enumerate(tops.items()):
        f1, f2 = [f for f in tt.index if gmap.get(f) not in ("duration", "level", "mfcc", "sp32")][:2]
        a, b = k.split("_vs_")
        for i, (nm, mask) in enumerate((("train 2022–24", df.source.isin(tr_years)), ("test 2025", df.source == "2025"))):
            s = df[mask & df["class"].isin([a, b])]
            s = pd.concat([g.sample(min(len(g), 700), random_state=RNG) for _, g in s.groupby("class")])
            for c in (a, b):
                q = s[s["class"] == c]
                axs[i, j].scatter(q[f1], q[f2], s=7, alpha=.45, color=CCOL[c], label=c)
            lo1, hi1 = np.nanpercentile(df[f1], [1, 99])
            lo2, hi2 = np.nanpercentile(df[f2], [1, 99])
            axs[i, j].set(xlim=(lo1, hi1), ylim=(lo2, hi2), xlabel=f1, ylabel=f2, title=f"{a} vs {b} — {nm}")
            axs[i, j].legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(G2 / "g2_scatter_best2.png", dpi=100)
    plt.close()

    # ── 3. Calidad de la aumentación ──────────────────────────────────────
    rep.append("## 3. Aumentación: ¿qué hizo con las features?\n")
    if aug:
        trk = df[df.source.isin(tr_years)]
        t25 = df[df.source == "2025"]
        rob = sets["robust"]

        def mean_shift(train_rows, fl):
            out = []
            for c in CLS:
                a_, b_ = train_rows[train_rows["class"] == c], t25[t25["class"] == c]
                a_, b_ = a_.sample(min(len(a_), 2500), random_state=RNG), b_.sample(min(len(b_), 2500), random_state=RNG)
                s = pd.concat([a_, b_])
                yy = (np.r_[np.zeros(len(a_)), np.ones(len(b_))]).astype(bool)
                out.append([abs(auc(s[f].values, yy) - .5) * 2 for f in fl])
            return np.nanmean(out)

        key = ["spec_slope_db_khz", "spec_centroid", "band_600_800", "band_1250_1500", "rel_tilt_db", "spec_rolloff85"]
        rows = [{"entrenamiento": "originales 2022–24", "shift medio (robust)": mean_shift(trk, rob),
                 "shift (features de forma espectral)": mean_shift(trk, key)}]
        for v, a in aug.items():
            comb = pd.concat([trk[trk.keep_orig], a[a.source.isin(tr_years)]])
            rows.append({"entrenamiento": f"plan + aug {v}", "shift medio (robust)": mean_shift(comb, rob),
                         "shift (features de forma espectral)": mean_shift(comb, key)})
        sh = pd.DataFrame(rows).set_index("entrenamiento")
        rep += ["Desplazamiento de dominio (|2·AUC−1| train vs 2025, misma clase; menor = más parecido a 2025):\n", md(sh), "\n"]
        fl = ["spec_centroid", "spec_slope_db_khz", "rel_tilt_db", "snr_vs_floor_db", "env_kurtosis", "duration_s"]
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        for a, f in zip(axs.ravel(), fl):
            for i, c in enumerate(CLS):
                for j, (nm, q) in enumerate((("orig", trk[trk["class"] == c]),
                                             ("aug", aug["full"][(aug["full"]["class"] == c) & aug["full"].source.isin(tr_years)]),
                                             ("2025", t25[t25["class"] == c]))):
                    v = q[f].dropna().values
                    if len(v) < 8:
                        continue
                    bp = a.boxplot(v, positions=[i * 4 + j], widths=.8, showfliers=False, patch_artist=True)
                    bp["boxes"][0].set_facecolor(["#9ecae1", "#fdae6b", "#e45756"][j])
            a.set_xticks([i * 4 + 1 for i in range(3)])
            a.set_xticklabels(CLS)
            a.set_title(f)
        fig.suptitle("Originales 2022–24 (azul) · aumentadas full (naranja) · 2025 (rojo)")
        plt.tight_layout()
        plt.savefig(G2 / "g2_aug_vs_orig.png", dpi=100)
        plt.close()

    # ── 4. Clasificación 3 clases, dejando un año fuera ───────────────────
    def trainset(H, mode):
        o = df[df.source != H]
        if mode == "B0":
            return o
        o = o[o.keep_orig]
        if mode == "B1":
            return o
        a = aug[mode.split("_", 1)[1]]
        return pd.concat([o, a[a.source != H]], ignore_index=True)

    modes = ["B0", "B1"] + [f"A_{v}" for v in VARIANTS if v in aug]

    def stable_cols(tr, k=10):
        ys = [y for y in YEARS if (tr.source == y).any()]
        cols = []
        for a, b in PAIRS:
            M = pd.DataFrame({y: pd.Series({f: auc(tr[(tr.source == y) & tr["class"].isin([a, b])][f].values,
                                                    (tr[(tr.source == y) & tr["class"].isin([a, b])]["class"] == b).values)
                                            for f in feats}) for y in ys}) - .5
            sg = np.sign(M.mean(1))
            sc = M.mul(sg, axis=0).min(1)
            cols += list(sc[sc > 0].sort_values(ascending=False).head(k).index)
        return list(dict.fromkeys(cols))

    def fit_predict(trd, ted, cols):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(trd[cols].replace([np.inf, -np.inf], np.nan))
        Xte = imp.transform(ted[cols].replace([np.inf, -np.inf], np.nan))
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                    n_jobs=4, random_state=RNG).fit(Xtr, trd["class"].values)
        return rf, Xte, rf.predict(Xte), rf.predict_proba(Xte)

    def metrics(y, pred, proba, classes):
        r = {"bal_acc": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", labels=CLS)}
        for c in CLS:
            r[f"rec_{c}"] = float(((pred == c) & (y == c)).sum() / max((y == c).sum(), 1))
        ci = {c: i for i, c in enumerate(classes)}
        for a, b in PAIRS:
            m = np.isin(y, [a, b])
            sc = proba[m, ci[b]] / (proba[m, ci[a]] + proba[m, ci[b]] + 1e-9)
            r[f"auc_{a}_{b}"] = roc_auc_score(y[m] == b, sc) if len(set(y[m])) == 2 else np.nan
        return r

    rows, confs = [], {}
    sets_ext = dict(sets)
    for H in YEARS:
        ted = df[df.source == H]
        sets_ext["stable"] = stable_cols(df[df.source != H])
        for mode in modes:
            trd = trainset(H, mode)
            for sname, cols in sets_ext.items():
                rf, Xte, pred, proba = fit_predict(trd, ted, cols)
                m = metrics(ted["class"].values, pred, proba, list(rf.classes_))
                rows.append({"fold": H, "train": mode, "features": sname, "n_feat": len(cols), "n_train": len(trd), **m})
                confs[f"{H}|{mode}|{sname}"] = confusion_matrix(ted["class"].values, pred, labels=CLS).tolist()
        print(f"  fold {H} listo", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(G2 / "classification_results.csv", index=False)
    (G2 / "confusions.json").write_text(json.dumps(confs))

    rep.append("## 4. Clasificación N / W / FC (RandomForest 200 árboles, class_weight balanceado)\n")
    rep.append("`B0` = todos los originales de los años de entrenamiento. `B1` = originales del plan (N submuestreada, "
               "control sin aumentar). `A_x` = B1 + copias aumentadas variante x. `stable` = 10 mejores features por par "
               "elegidas con los AUC por año **de los años de entrenamiento** (sin ver el año retenido).\n")
    for metric in ("bal_acc", "macro_f1"):
        p = res.pivot_table(index=["train", "features"], columns="fold", values=metric)
        p["media"] = p.mean(1)
        rep += [f"### {metric} (media = promedio de los 4 años)\n", md(p.sort_values("media", ascending=False)), "\n"]
    rec25 = res[res.fold == "2025"].set_index(["train", "features"])[["bal_acc", "rec_N", "rec_W", "rec_FC",
                                                                       "auc_N_W", "auc_N_FC", "auc_W_FC"]]
    rep += ["### Detalle en 2025 (split oficial): recall por clase y AUC por pares\n",
            md(rec25.sort_values("bal_acc", ascending=False)), "\n"]
    allf = res.groupby(["train", "features"])[["bal_acc", "macro_f1", "auc_N_W", "auc_N_FC", "auc_W_FC"]].mean()
    rep += ["### Promedio 4 años: AUC por pares (¿qué par es difícil?)\n", md(allf.sort_values("bal_acc", ascending=False)), "\n"]

    # figura de modelos
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.5))
    for a, (nm, sub) in zip(axs, (("media 4 años", res), ("2025 (split oficial)", res[res.fold == "2025"]))):
        p = sub.pivot_table(index="features", columns="train", values="bal_acc")[modes]
        p.plot.bar(ax=a, width=.85)
        a.set_ylim(0.3, None)
        a.set_ylabel("balanced accuracy")
        a.axhline(1 / 3, color="k", lw=.6)
        a.set_title(f"N/W/FC — {nm}")
        a.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(G2 / "g2_models.png", dpi=105)
    plt.close()

    # confusiones 2025 de la mejor config por modo
    best_feat = allf.reset_index().groupby("features")["bal_acc"].mean().idxmax()
    fig, axs = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 4))
    for a, mode in zip(np.atleast_1d(axs), modes):
        cm = np.array(confs[f"2025|{mode}|{best_feat}"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                       color="white" if cmn[i, j] > .5 else "black")
        a.set_xticks(range(3))
        a.set_yticks(range(3))
        a.set_xticklabels(CLS)
        a.set_yticklabels(CLS)
        a.set_xlabel("predicho")
        a.set_title(f"{mode} · {best_feat}")
    plt.tight_layout()
    plt.savefig(G2 / "g2_confusion_2025.png", dpi=105)
    plt.close()

    # ── 5. Importancia de features fuera de dominio ───────────────────────
    best = allf.sort_values("bal_acc", ascending=False).index[0]
    bmode, bfeat = best
    rep.append(f"## 5. Importancia fuera de dominio (config ganadora: `{bmode}` + `{bfeat}`)\n")
    imps, imp_gini = [], None
    for H in YEARS:
        ted = df[df.source == H]
        sets_ext["stable"] = stable_cols(df[df.source != H])
        cols = sets_ext[bfeat]
        rf, Xte, pred, proba = fit_predict(trainset(H, bmode), ted, cols)
        pi = permutation_importance(rf, Xte, ted["class"].values, scoring="balanced_accuracy", n_repeats=3,
                                    n_jobs=4, random_state=RNG)
        imps.append(pd.Series(pi.importances_mean, index=cols))
        if H == "2025":
            imp_gini = pd.Series(rf.feature_importances_, index=cols)
    I = pd.concat(imps, axis=1)
    I.columns = YEARS
    I["media"] = I.mean(1)
    I["gini_2025"] = imp_gini
    I["grupo"] = pd.Series(gmap)
    I = I.sort_values("media", ascending=False)
    I.to_csv(G2 / "permutation_importance.csv")
    rep += ["Permutación (caída de balanced accuracy en el año retenido al barajar la feature; media sobre 4 años):\n",
            md(I.head(25)), "\n"]
    top = I.head(20)[::-1]
    fig, a = plt.subplots(figsize=(9, 8))
    a.barh(top.index, top["media"], xerr=I.loc[top.index, YEARS].std(1), color=[
        {"shape": "#4C78A8", "tonal": "#54A24B", "temporal": "#E45756", "rel": "#F58518", "mfcc": "#9D755D",
         "level": "#9D755D", "duration": "#9D755D", "sp32": "#B279A2"}.get(g, "gray") for g in top["grupo"]])
    a.set_xlabel("caída de balanced accuracy (permutación, año retenido)")
    a.set_title(f"Top 20 — {bmode} + {bfeat}\n(azul forma · verde tonal · rojo temporal · naranja relativas)")
    plt.tight_layout()
    plt.savefig(G2 / "g2_importance.png", dpi=105)
    plt.close()
    gi = I.groupby("grupo")["media"].sum().sort_values(ascending=False)
    rep += ["Importancia total por grupo de features:\n", md(gi.to_frame("suma")), "\n"]

    (G2 / "report_auto.md").write_text("\n".join(rep), encoding="utf-8")
    print("OK ->", G2 / "report_auto.md", "| mejor:", best)


if __name__ == "__main__":
    main()
