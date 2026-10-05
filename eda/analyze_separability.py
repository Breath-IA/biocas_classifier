"""
analyze_separability.py
───────────────────────
Analiza qué features espectrales/temporales/tonales separan las clases y cuáles
sobreviven al cambio de dominio entre años (2022–24 → 2025).

Entrada : eda/out/segments_spectral.csv   (extract_spectral_features.py)
Salida  : eda/out/fig_*.png, eda/out/*.csv, eda/out/report.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "out"
RNG = 42
CLASSES = ["N", "W", "FC", "CC", "WC", "R", "S"]
COLORS = dict(zip(CLASSES, ["#4C78A8", "#54A24B", "#E45756", "#B279A2", "#F58518", "#9D755D", "#72B7B2"]))
YEARS = ["2022", "2023", "2024", "2025"]

META = {"stem", "source", "split", "patient_id", "age", "gender", "location", "event_idx", "class",
        "start_ms", "end_ms", "rec_duration_s", "n_events_rec", "start_frac", "poor_quality", "n_frames"}

PAIRS = {  # nombre: (positivas, negativas)
    "abn_vs_N":  (["W", "FC", "CC", "WC", "R", "S"], ["N"]),
    "W_vs_N":    (["W"], ["N"]),
    "FC_vs_N":   (["FC"], ["N"]),
    "CC_vs_N":   (["CC"], ["N"]),
    "WC_vs_N":   (["WC"], ["N"]),
    "R_vs_N":    (["R"], ["N"]),
    "S_vs_N":    (["S"], ["N"]),
    "W_vs_crk":  (["W"], ["FC", "CC"]),
    "CC_vs_FC":  (["CC"], ["FC"]),
    "WC_vs_W":   (["WC"], ["W"]),
    "WC_vs_FC":  (["WC"], ["FC"]),
    "R_vs_W":    (["R"], ["W"]),
}


def feature_groups(cols):
    g = {
        "level":    [c for c in cols if c in ("rms_db", "snr_vs_floor_db", "level_vs_rec_db")
                     or c.startswith("rel_")],
        "shape":    [c for c in cols if c.startswith(("spec_", "band_")) or c in ("dom_freq", "lf_hf_db")],
        "tonal":    [c for c in cols if c in ("n_spec_peaks", "max_peak_prom_db", "dom_freq_std", "dom_persist",
                                              "frame_peakiness_db", "if_mean", "if_std", "acf_peak", "f0_acf")],
        "temporal": [c for c in cols if c in ("duration_s", "crest_factor", "kurtosis", "skewness", "zcr",
                                              "n_transients", "transient_rate", "env_kurtosis", "env_cv",
                                              "env_peak_to_median", "mf_peak_ratio", "mf_f0", "mf_dur_ms",
                                              "mf_rate", "energy_cv", "temporal_flatness")],
        "mfcc":     [c for c in cols if c.startswith("mfcc_")],
    }
    return g


def auc(x, y):
    """AUC univariada (x alto → positivo). NaN si <15 muestras por lado."""
    x = np.asarray(x, float)
    m = np.isfinite(x)
    x, y = x[m], np.asarray(y)[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 15 or n0 < 15:
        return np.nan
    r = rankdata(x)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pair_auc(d, feats, pos, neg):
    s = d[d["class"].isin(pos + neg)]
    y = s["class"].isin(pos).values
    return pd.Series({f: auc(s[f].values, y) for f in feats})


def sample(d, n=3000):
    return d.sample(min(len(d), n), random_state=RNG)


def main():
    df = pd.read_csv(OUT / "segments_spectral.csv")
    df["source"] = df["source"].astype(str)
    n0 = len(df)
    df = df[~df["poor_quality"]].reset_index(drop=True)
    feats = [c for c in df.columns if c not in META and not c.startswith("sp_")
             and pd.api.types.is_numeric_dtype(df[c])]
    groups = feature_groups(feats)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    rep = ["# Separabilidad espectral — informe automático\n"]
    rep.append(f"{n0:,} eventos extraídos; {n0 - len(df):,} descartados por 'poor quality' "
               f"(solo anotado en parte del dataset) → {len(df):,}. {len(feats)} features.\n")

    # ── 0. Conteos y duración ──────────────────────────────────────────────
    cnt = df.groupby(["class", "source"]).size().unstack(fill_value=0).reindex(CLASSES)
    cnt["total"] = cnt.sum(1)
    dur = df.groupby("class")["duration_s"].describe()[["count", "25%", "50%", "75%", "max"]].reindex(CLASSES).round(2)
    rep += ["## 1. Eventos por clase y año\n", cnt.to_markdown(), "\n",
            "## 2. Duración por clase (s)\n", dur.to_markdown(), "\n"]
    ov = set(tr.patient_id) & set(te.patient_id)
    rep.append(f"Pacientes en train y test a la vez: {len(ov)}\n")

    # ── 1. Espectros medios por clase ──────────────────────────────────────
    spc = [f"sp_{k:02d}" for k in range(32)]
    fx = np.arange(32) * 62.5 + 31.25
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for c in CLASSES:
        for a, d in ((ax[0, 0], tr), (ax[0, 1], te)):
            s = d[d["class"] == c]
            if len(s) >= 15:
                a.plot(fx, s[spc].median(), color=COLORS[c], label=f"{c} (n={len(s)})")
    ax[0, 0].set_title("Espectro mediano por clase — TRAIN (2022–24)")
    ax[0, 1].set_title("Espectro mediano por clase — TEST (2025)")
    for c in ("N", "W", "FC"):
        ax[1, 0].plot(fx, tr[tr["class"] == c][spc].median(), color=COLORS[c], label=f"{c} train")
        ax[1, 0].plot(fx, te[te["class"] == c][spc].median(), "--", color=COLORS[c], label=f"{c} test")
    ax[1, 0].set_title("Mismas clases en train (—) vs test (--): desplazamiento de dominio")
    nmed_tr, nmed_te = tr[tr["class"] == "N"][spc].median(), te[te["class"] == "N"][spc].median()
    for c in ("W", "FC", "CC", "WC"):
        s1, s2 = tr[tr["class"] == c], te[te["class"] == c]
        if len(s1) >= 15:
            ax[1, 1].plot(fx, s1[spc].median() - nmed_tr, color=COLORS[c], label=f"{c} − N train")
        if len(s2) >= 15:
            ax[1, 1].plot(fx, s2[spc].median() - nmed_te, "--", color=COLORS[c])
    ax[1, 1].axhline(0, color="k", lw=0.6)
    ax[1, 1].set_title("Clase − Normal (mediana, dB relativo): train (—) vs test (--)")
    for a in ax.ravel():
        a.set_xlabel("Hz")
        a.set_ylabel("dB (rel. a potencia total)")
        a.legend(fontsize=7, ncol=2)
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_class_spectra.png", dpi=110)
    plt.close()

    # ── 2. AUC por feature × par × split / año ─────────────────────────────
    auc_tr = pd.DataFrame({k: pair_auc(tr, feats, *v) for k, v in PAIRS.items()})
    auc_te = pd.DataFrame({k: pair_auc(te, feats, *v) for k, v in PAIRS.items()})
    auc_year = {y: pd.DataFrame({k: pair_auc(df[df.source == y], feats, *v) for k, v in PAIRS.items()})
                for y in YEARS}
    auc_tr.to_csv(OUT / "auc_train.csv")
    auc_te.to_csv(OUT / "auc_test.csv")

    # estabilidad: dirección común en los 4 años y mínimo |AUC-0.5| entre años válidos
    stab = {}
    for k in PAIRS:
        M = pd.DataFrame({y: auc_year[y][k] for y in YEARS}) - 0.5
        d = np.sign(M.mean(1))
        agree = (M.mul(d, axis=0) > 0).all(1) | (M.notna().sum(1) == 0)
        stab[k] = (M.mul(d, axis=0).min(1)).where(agree & (M.notna().sum(1) >= 3), 0.0) * d
    stab = pd.DataFrame(stab)
    stab.to_csv(OUT / "auc_year_stability.csv")

    # ── 3. Desplazamiento de dominio por feature (misma clase, train vs test)
    dom = {}
    for c in ("N", "W", "FC"):
        a, b = sample(tr[tr["class"] == c]), sample(te[te["class"] == c])
        dd = pd.concat([a, b])
        y = np.r_[np.zeros(len(a), bool), np.ones(len(b), bool)]
        dom[c] = pd.Series({f: auc(dd[f].values, y) for f in feats})
    dom = pd.DataFrame(dom)
    dom_score = (dom - 0.5).abs().mean(1) * 2          # 0 = sin shift, 1 = separa perfecto
    dom.to_csv(OUT / "domain_shift_auc.csv")

    # tabla de candidatos: separabilidad estable y bajo shift
    def top_table(pair, k=12):
        t = pd.DataFrame({
            "auc_train": auc_tr[pair], "auc_test": auc_te[pair],
            "min_year_margin": stab[pair], "domain_shift": dom_score,
        })
        t["score"] = t["min_year_margin"].abs() * (1 - dom_score)
        return t.sort_values("score", ascending=False).head(k).round(3)

    rep.append("## 3. Features candidatas por par de clases\n")
    rep.append("`auc_*`: AUC univariada (>0.5 = más alto en la 1ª clase del par). "
               "`min_year_margin`: peor |AUC−0.5| entre los 4 años con el mismo signo (0 si cambia de signo). "
               "`domain_shift`: 0 = train/test indistinguibles dentro de las mismas clases, 1 = separables. "
               "`score = margen × (1 − shift)`.\n")
    for pair in ("abn_vs_N", "W_vs_N", "FC_vs_N", "W_vs_crk", "CC_vs_FC", "WC_vs_FC", "WC_vs_W", "R_vs_N", "S_vs_N"):
        rep += [f"### {pair}\n", top_table(pair).to_markdown(), "\n"]

    # ── 4. Heatmap de AUC ──────────────────────────────────────────────────
    order = (auc_tr - 0.5).abs().max(1).sort_values(ascending=False).head(45).index
    fig, ax = plt.subplots(1, 2, figsize=(15, 13), sharey=True)
    for a, t, nm in ((ax[0], auc_tr, "TRAIN"), (ax[1], auc_te, "TEST (2025)")):
        im = a.imshow(t.loc[order].values, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
        a.set_xticks(range(len(PAIRS)))
        a.set_xticklabels(PAIRS, rotation=60, ha="right", fontsize=8)
        a.set_yticks(range(len(order)))
        a.set_yticklabels(order, fontsize=8)
        a.set_title(f"AUC univariada — {nm}")
    fig.colorbar(im, ax=ax, shrink=0.5, label="AUC (rojo: mayor en 1ª clase del par)")
    plt.savefig(OUT / "fig_auc_heatmap.png", dpi=110, bbox_inches="tight")
    plt.close()

    # ── 5. Distribuciones de las mejores features ──────────────────────────
    def boxes(flist, fname, title):
        n = len(flist)
        fig, axs = plt.subplots((n + 3) // 4, 4, figsize=(20, 4.2 * ((n + 3) // 4)))
        for a, f in zip(np.ravel(axs), flist):
            for i, c in enumerate(CLASSES):
                for j, (d, col) in enumerate(((tr, "#4C78A8"), (te, "#F58518"))):
                    v = d.loc[d["class"] == c, f].dropna().values
                    if len(v) < 5:
                        continue
                    bp = a.boxplot(v, positions=[i * 3 + j], widths=0.8, showfliers=False, patch_artist=True)
                    for b in bp["boxes"]:
                        b.set_facecolor(col)
                        b.set_alpha(0.75)
            a.set_xticks([i * 3 + 0.5 for i in range(len(CLASSES))])
            a.set_xticklabels(CLASSES)
            a.set_title(f, fontsize=11)
        for a in np.ravel(axs)[n:]:
            a.axis("off")
        fig.suptitle(title + "  (azul = train 2022–24, naranja = test 2025)", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUT / fname, dpi=100)
        plt.close()

    def best(pair, k):
        return list(top_table(pair, k).index)

    boxes(best("abn_vs_N", 8), "fig_dist_gate1.png", "Gate 1 · mejores features Normal vs Anormal")
    g2 = list(dict.fromkeys(best("W_vs_crk", 6) + best("CC_vs_FC", 4) + best("WC_vs_FC", 2)))[:12]
    boxes(g2, "fig_dist_gate2.png", "Gate 2 · mejores features tipo de sonido")

    # ── 6. Discriminación vs desplazamiento de dominio ─────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for a, pair in zip(ax, ("abn_vs_N", "W_vs_crk")):
        x, y = dom_score, (auc_tr[pair] - 0.5).abs() * 2
        a.scatter(x, y, s=18, c=[("#E45756" if f.startswith("mfcc") else "#4C78A8") for f in feats])
        for f in y.sort_values(ascending=False).head(10).index:
            a.annotate(f, (x[f], y[f]), fontsize=8)
        a.set_xlabel("desplazamiento de dominio (train vs test, misma clase)")
        a.set_ylabel(f"separabilidad en train  |2·AUC−1|  [{pair}]")
        a.set_title(f"{pair}  (rojo = MFCC)")
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_sep_vs_domain.png", dpi=110)
    plt.close()

    # ── 7. Clasificación: dejar un año fuera ───────────────────────────────
    sets = {"all": feats, "shape": groups["shape"], "tonal": groups["tonal"], "temporal": groups["temporal"],
            "mfcc": groups["mfcc"], "level": groups["level"],
            "shape+tonal+temporal": groups["shape"] + groups["tonal"] + groups["temporal"]}

    def year_stable(d, pair, k):
        yrs = [y for y in YEARS if (d.source == y).any()]
        M = pd.DataFrame({y: pair_auc(d[d.source == y], feats, *PAIRS[pair]) for y in yrs}) - 0.5
        sg = np.sign(M.mean(1))
        sc = M.mul(sg, axis=0).min(1)
        return list(sc[sc > 0].sort_values(ascending=False).head(k).index)

    tasks = {
        "gate1 N/abn": (lambda d: d, lambda d: (d["class"] != "N").astype(int).values, "abn_vs_N"),
        "gate2 W/crk": (lambda d: d[d["class"].isin(["W", "FC", "CC"])],
                        lambda d: (d["class"] == "W").astype(int).values, "W_vs_crk"),
        "gate2 6cls":  (lambda d: d[d["class"] != "N"], lambda d: d["class"].values, "W_vs_crk"),
    }

    def fit_eval(trd, ted, cols, ytr, yte, multi):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(trd[cols].replace([np.inf, -np.inf], np.nan))
        Xte = imp.transform(ted[cols].replace([np.inf, -np.inf], np.nan))
        rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced_subsample",
                                    n_jobs=4, random_state=RNG).fit(Xtr, ytr)
        if multi:
            return np.nan, f1_score(yte, rf.predict(Xte), average="macro")
        p = rf.predict_proba(Xte)[:, 1]
        # umbral = prevalencia de train (sin mirar el año de test) → macro-F1 comparable
        thr = np.quantile(rf.predict_proba(Xtr)[:, 1], 1 - ytr.mean())
        return roc_auc_score(yte, p), f1_score(yte, (p >= thr).astype(int), average="macro")

    rows = []
    for tname, (sel, lab, pair) in tasks.items():
        multi = tname.endswith("6cls")
        dd = sel(df).reset_index(drop=True)
        yy = lab(dd)
        for held in YEARS:
            m = (dd.source == held).values
            if m.sum() < 30 or len(np.unique(yy[m])) < 2 or len(np.unique(yy[~m])) < 2:
                continue
            trd, ted = dd[~m], dd[m]
            stable = year_stable(trd, pair, 15)
            for sname, cols in {**sets, "year-stable top15": stable}.items():
                if len(cols) < 2:
                    continue
                a, f1 = fit_eval(trd, ted, cols, yy[~m], yy[m], multi)
                rows.append({"task": tname, "held_out": held, "features": sname, "n_feat": len(cols),
                             "roc_auc": a, "macro_f1": f1})
            print(f"  {tname} · held-out {held} listo", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "loyo_results.csv", index=False)

    rep.append("## 4. Clasificación dejando un año fuera (RandomForest, 250 árboles)\n")
    rep.append("Entrena con los otros 3 años y evalúa en el año retenido (2025 = split oficial). "
               "`year-stable top15` elige las 15 features con AUC consistente entre los años de "
               "**entrenamiento** (sin ver el año retenido). Umbral fijado con la prevalencia de train.\n")
    for tname in tasks:
        r = res[res.task == tname]
        if r.empty:
            continue
        piv = r.pivot_table(index="features", columns="held_out", values="roc_auc" if not tname.endswith("6cls")
                            else "macro_f1")
        piv["media"] = piv.mean(1)
        rep += [f"### {tname} — {'ROC AUC' if not tname.endswith('6cls') else 'macro-F1'}\n",
                piv.sort_values("media", ascending=False).round(3).to_markdown(), "\n"]
        if not tname.endswith("6cls"):
            piv = r.pivot_table(index="features", columns="held_out", values="macro_f1")
            piv["media"] = piv.mean(1)
            rep += [f"{tname} — macro-F1\n", piv.sort_values("media", ascending=False).round(3).to_markdown(), "\n"]

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for a, tname in zip(ax, tasks):
        r = res[res.task == tname]
        if r.empty:
            continue
        val = "roc_auc" if not tname.endswith("6cls") else "macro_f1"
        piv = r.pivot_table(index="features", columns="held_out", values=val)
        piv.loc[piv.mean(1).sort_values().index].plot.barh(ax=a, width=0.8)
        a.set_title(f"{tname} — {val}")
        a.set_xlim(0.1 if val == "macro_f1" else 0.4, None)
        a.axvline(0.5, color="k", lw=0.6)
        a.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT / "fig_loyo.png", dpi=110)
    plt.close()

    # features más estables en los 4 años para el reporte
    rep.append("## 5. Features con separabilidad estable en los 4 años (|margen| > 0.05)\n")
    for pair in ("abn_vs_N", "W_vs_crk", "CC_vs_FC", "WC_vs_FC"):
        s = stab[pair][stab[pair].abs() > 0.05].sort_values(key=abs, ascending=False).head(15)
        rep.append(f"**{pair}**: " + (", ".join(f"{f} ({v:+.2f})" for f, v in s.items()) or "ninguna") + "\n")
    (OUT / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print("OK ->", OUT / "report.md")


if __name__ == "__main__":
    main()
