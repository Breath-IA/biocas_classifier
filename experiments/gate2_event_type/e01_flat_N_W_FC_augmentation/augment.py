"""
gate2_augment.py
────────────────
Aumentación a nivel de GRABACIÓN para las clases N, W y FC (gate 2 sin pasar por gate 1) y extracción
de las mismas features sobre el audio aumentado.

Diseño (por qué así):
  • La aumentación se aplica de forma SIMÉTRICA a las tres clases: si solo W/FC se aumentaran, "suena
    aumentado" se volvería una pista de la etiqueta. Fracción de muestras aumentadas por clase ≈ 45–60 %.
  • Plan (igual para todos los años; ver PLAN abajo):
        N  : 9 % de eventos como original, otro 9 % (distinto) como copia aumentada
        W  : todos originales + 1–2 copias aumentadas por evento (media 1.5)
        FC : todos originales + 1 copia aumentada con prob. 0.82
    → en train 2022-24 ≈ 1500 / 1500 / 1500(N,W-aug...) ≈ 3000 por clase.
  • Transformaciones (se sortean en el mismo orden para todas las variantes → ablación controlada):
        noise  : fondo REAL (otro evento N del mismo año) mezclado a SNR ~ U(8, 25) dB
        eq     : realce/atenuación espectral tipo "otro dispositivo" (shelf ±8 dB, corte 300–900 Hz) + ganancia ±10 dB
        jitter : fronteras del evento desplazadas ±15 % de su duración
    No se usa pitch/time-stretch: desplazan f0 y τ, que son justo lo que define fino vs. sibilancia.
  • El fondo de ruido de una copia sale del MISMO año que la grabación → al dejar un año fuera en la
    validación no se filtra audio del año retenido.

Uso:  python experiments/gate2_event_type/e01_flat_N_W_FC_augmentation/augment.py   (full, eq, noise, jitter)
Salida: results/plan.csv; cache/bg_pool.npz, cache/aug_<variante>.csv
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import extract_spectral_features as ex  # noqa: E402
import paths  # noqa: E402

RES = paths.G2_E01 / "results"
CACHE = paths.G2_E01 / "cache"      # derivados grandes (ignorados por git)
SEED = 42
KEEP_N = 0.09                 # fracción de N que entra como original (y otra igual como copia aumentada)
W_COPIES = 1.5                # media de copias aumentadas por evento W
FC_COPY_P = 0.82              # prob. de una copia aumentada por evento FC
POOL_PER_YEAR = 150
VARIANTS = ["full", "eq", "noise", "jitter"]


def build_plan(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    d = df[df["class"].isin(["N", "W", "FC"])][["stem", "event_idx", "class", "source", "split",
                                                  "start_ms", "end_ms"]].copy().reset_index(drop=True)
    u = rng.random(len(d))
    keep = np.ones(len(d), bool)
    n_aug = np.zeros(len(d), int)
    isN, isW, isFC = (d["class"] == "N").values, (d["class"] == "W").values, (d["class"] == "FC").values
    keep[isN] = u[isN] < KEEP_N
    n_aug[isN] = ((u[isN] >= KEEP_N) & (u[isN] < 2 * KEEP_N)).astype(int)
    n_aug[isW] = 1 + (rng.random(isW.sum()) < (W_COPIES - 1)).astype(int)
    n_aug[isFC] = (rng.random(isFC.sum()) < FC_COPY_P).astype(int)
    d["keep_orig"], d["n_aug"] = keep, n_aug
    return d


def build_pool(df: pd.DataFrame, plan_path: Path) -> Path:
    """Fondos reales: eventos N ≥0.6 s por año, guardados como segmentos de forma de onda."""
    rng = np.random.default_rng(SEED)
    arrays, stems = {}, {}
    for year, g in df[(df["class"] == "N") & (df["duration_s"] >= 0.6)].groupby("source"):
        take = g.sample(min(POOL_PER_YEAR, len(g)), random_state=SEED)
        arrs, ss = [], []
        for _, r in take.iterrows():
            x = ex.load_wav(ex.RAW / r["split"] / "wav" / f"{r['stem']}.wav")
            seg = x[int(r["start_ms"] * ex.SR / 1000): int(r["end_ms"] * ex.SR / 1000)]
            if len(seg) >= 2000:
                arrs.append(seg.astype(np.float32))
                ss.append(r["stem"])
        arrays[str(year)], stems[str(year)] = arrs, ss
    flat = {f"{y}__{i}": a for y, arrs in arrays.items() for i, a in enumerate(arrs)}
    flat.update({f"{y}__stems": np.array(s) for y, s in stems.items()})
    np.savez(plan_path, **flat)
    return plan_path


_POOL = None


def _pool(path):
    global _POOL
    if _POOL is None:
        z = np.load(path)
        _POOL = {}
        for k in z.files:
            y, i = k.split("__")
            if i == "stems":
                _POOL.setdefault(y, {"arr": [], "stems": None})["stems"] = list(z[k])
            else:
                _POOL.setdefault(y, {"arr": [], "stems": None})["arr"].append(z[k])
    return _POOL


def augment_wave(x, s, e, rng, variant, bg_list, bg_stems, stem):
    """Devuelve (x_aug, s_aug, e_aug, descripción). Sorteos en orden fijo para todas las variantes."""
    # sorteos (siempre, para que variantes distintas usen los mismos parámetros)
    snr = rng.uniform(8, 25)
    bg_i = int(rng.integers(len(bg_list))) if bg_list else -1
    bg_off = int(rng.integers(0, 100000))
    use_noise = rng.random() < 0.5
    g_lo, g_hi, fc = rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(300, 900)
    use_eq = rng.random() < 0.8
    gain_db = rng.uniform(-10, 10)
    dj_s, dj_e = rng.uniform(-0.15, 0.15, 2)

    do_noise = {"full": use_noise, "noise": True}.get(variant, False)
    do_eq = {"full": use_eq, "eq": True}.get(variant, False)
    do_gain = variant in ("full", "eq")
    do_jit = variant in ("full", "jitter")

    y = x.copy()
    if do_noise and bg_i >= 0:
        if bg_stems[bg_i] == stem:                     # no mezclar una grabación consigo misma
            bg_i = (bg_i + 1) % len(bg_list)
        bg = np.roll(np.resize(bg_list[bg_i], len(y)), bg_off)
        bg = bg - bg.mean()
        y = y + bg / (np.sqrt(np.mean(bg ** 2)) + 1e-9) * np.sqrt(np.mean(y ** 2)) * 10 ** (-snr / 20)
    if do_eq:
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1 / ex.SR)
        sig = 1 / (1 + np.exp(-(f - fc) / (0.25 * fc)))
        y = np.fft.irfft(Y * 10 ** ((g_lo + (g_hi - g_lo) * sig) / 20), n=len(y))
    if do_gain:
        y = y * 10 ** (gain_db / 20)
    s2, e2 = s, e
    if do_jit:
        dur = e - s
        a, b = int(round(s + dj_s * dur)), int(round(e + dj_e * dur))
        a, b = max(a, 0), min(b, int(len(y) / ex.SR * 1000))
        if b - a >= 20:
            s2, e2 = a, b
    return y.astype(np.float32), s2, e2


def aug_recording(stem, split, meta, copies, variant, pool_path):
    """copies: lista de (event_idx, copy_idx). Devuelve filas de features de las copias aumentadas."""
    import json
    try:
        x = ex.load_wav(ex.RAW / split / "wav" / f"{stem}.wav")
        ann = json.loads((ex.RAW / split / "json" / f"{stem}.json").read_text(encoding="utf-8"))
    except Exception:
        return []
    evs = {k: (c, s, e) for k, c, s, e in ex.parse_events(ann)}
    pool = _pool(pool_path).get(meta["source"], {"arr": [], "stems": []})
    rows = []
    for k, ci in copies:
        if k not in evs:
            continue
        c, s, e = evs[k]
        rng = np.random.default_rng(zlib.crc32(f"{stem}|{k}|{ci}".encode()) ^ SEED)
        y, s2, e2 = augment_wave(x, s, e, rng, variant, pool["arr"], pool["stems"] or [], stem)
        rows += ex.process_waveform(
            y, [(k, c, s2, e2)], meta, poor=False, n_events_rec=len(evs),
            extra={"is_aug": True, "aug_variant": variant, "aug_copy": ci, "orig_start_ms": s, "orig_end_ms": e})
    return rows


def main():
    RES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(paths.SEGMENTS_CSV)
    df["source"] = df["source"].astype(str)
    plan = build_plan(df)
    plan.to_csv(RES / "plan.csv", index=False)
    print(plan.groupby("class")[["keep_orig", "n_aug"]].sum())
    pool_path = CACHE / "bg_pool.npz"
    if not pool_path.exists():
        build_pool(df, pool_path)
        print("pool de fondos creado")

    meta_cols = ["stem", "source", "split", "patient_id", "age", "gender", "location"]
    metas = df.drop_duplicates("stem").set_index("stem")[meta_cols[1:]].to_dict("index")
    jobs = []
    for stem, g in plan[plan.n_aug > 0].groupby("stem"):
        copies = [(int(r.event_idx), ci) for r in g.itertuples() for ci in range(int(r.n_aug))]
        jobs.append((stem, g["split"].iloc[0], {"stem": stem, **metas[stem]}, copies))
    print(f"{len(jobs)} grabaciones con copias, {sum(len(j[3]) for j in jobs)} copias por variante")

    variants = sys.argv[1:] or VARIANTS
    for v in variants:
        res = Parallel(n_jobs=4, verbose=0)(delayed(aug_recording)(*j, v, str(pool_path)) for j in jobs)
        out = pd.DataFrame([r for rs in res for r in rs])
        out.to_csv(CACHE / f"aug_{v}.csv", index=False)
        print(f"[{v}] {len(out)} filas -> aug_{v}.csv", flush=True)
        print(out.groupby(["source", "class"]).size().unstack(0).to_string(), flush=True)


if __name__ == "__main__":
    main()
