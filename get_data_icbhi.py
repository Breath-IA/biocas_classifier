"""
get_data_icbhi.py
─────────────────
Descarga (o toma de una carpeta local) ICBHI 2017 y lo reorganiza igual que
SPRSound, con la partición oficial 60/40:

    <out>/train/{wav,txt}/<stem>.wav|.txt
    <out>/test/{wav,txt}/<stem>.wav|.txt
    <out>/patient_diagnosis.csv        (solo para análisis, no es entrada ni objetivo)

Uso:
    python get_data_icbhi.py                              # descarga + organiza
    python get_data_icbhi.py --src ruta/a/ICBHI_final_database   # ya descargado
    python get_data_icbhi.py --src ICBHI_final_database.zip

Luego: python data_config.py --dataset icbhi   (actualiza workflow/config/pipeline.yaml)

--split-file: ICBHI_challenge_train_test.txt (cols: stem \t train|test). Si no se
encuentra, se descarga; si no se puede, se usa una partición agrupada por paciente 60/40.
"""
from __future__ import annotations

import argparse
import sys

# consolas de Windows (cp1252) no soportan ✓/✗
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE_URL  = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database"
ZIP_URL   = f"{BASE_URL}/ICBHI_final_database.zip"
SPLIT_URL = f"{BASE_URL}/ICBHI_challenge_train_test.txt"


def download(url: str, dst: Path) -> bool:
    print(f"Descargando {url}")
    try:
        urllib.request.urlretrieve(url, dst)
        return True
    except Exception as exc:                     # red caída, sitio movido, etc.
        print(f"  ✗ falló: {exc}")
        dst.unlink(missing_ok=True)
        return False


def find_source(src: Path | None, work: Path) -> Path:
    """Devuelve una carpeta con los .wav/.txt (descomprime o descarga si hace falta)."""
    if src is None:
        zip_path = work / "ICBHI_final_database.zip"
        if not zip_path.exists() and not download(ZIP_URL, zip_path):
            raise SystemExit(
                "No se pudo descargar ICBHI. Bájalo a mano y pasa --src <carpeta|zip>."
            )
        src = zip_path
    if src.is_file() and src.suffix == ".zip":
        extract = work / "ICBHI_raw"
        if not extract.exists():
            print(f"Descomprimiendo {src} → {extract}")
            with zipfile.ZipFile(src) as zf:
                zf.extractall(extract)
        return extract
    if not src.is_dir():
        raise SystemExit(f"--src no existe: {src}")
    return src


def read_split(path: Path) -> dict[str, str]:
    split = {}
    for line in path.read_text().splitlines():
        cols = line.split()
        if len(cols) >= 2:
            split[cols[0]] = cols[1].lower()
    return split


def patient_of(stem: str) -> str:
    return stem.split("_")[0]


def grouped_split(stems: list[str], test_frac: float = 0.4, seed: int = 42) -> dict[str, str]:
    patients = sorted({patient_of(s) for s in stems})
    random.Random(seed).shuffle(patients)
    test_p = set(patients[: round(len(patients) * test_frac)])
    return {s: ("test" if patient_of(s) in test_p else "train") for s in stems}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="carpeta o .zip ya descargado")
    ap.add_argument("--split-file", type=Path, default=None, help="ICBHI_challenge_train_test.txt")
    ap.add_argument("--out", type=Path, default=Path("./data/icbhi/raw_data"))
    ap.add_argument("--work", type=Path, default=Path("./data/icbhi/_download"),
                    help="carpeta temporal para zip/descomprimido")
    ap.add_argument("--symlinks", action="store_true", help="enlazar en vez de copiar")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    root = find_source(args.src, args.work)

    wavs = {p.stem: p for p in root.rglob("*.wav")}
    txts = {p.stem: p for p in root.rglob("*.txt")}
    stems = sorted(s for s in wavs if s in txts)
    print(f"Grabaciones con wav+txt: {len(stems)}  (wav={len(wavs)})")
    if not stems:
        raise SystemExit(f"No se encontraron pares .wav/.txt bajo {root}")

    # ── Partición oficial ────────────────────────────────────────────────
    split_file = args.split_file or next(iter(root.rglob("ICBHI_challenge_train_test.txt")), None)
    if split_file is None:
        split_file = args.work / "ICBHI_challenge_train_test.txt"
        if not split_file.exists() and not download(SPLIT_URL, split_file):
            split_file = None
    if split_file is not None:
        split = read_split(split_file)
        missing = [s for s in stems if s not in split]
        if missing:
            print(f"⚠ {len(missing)} grabaciones sin partición oficial (se omiten): {missing[:3]}…")
            stems = [s for s in stems if s in split]
        print(f"Partición oficial: {split_file}")
    else:
        print("⚠ Sin archivo de partición oficial → partición 60/40 agrupada por paciente")
        split = grouped_split(stems)

    # ── Verificar que no se compartan pacientes entre train y test ───────
    p_train = {patient_of(s) for s in stems if split[s] == "train"}
    p_test  = {patient_of(s) for s in stems if split[s] == "test"}
    shared  = p_train & p_test
    if shared:
        print(f"⚠ {len(shared)} pacientes aparecen en train Y test: {sorted(shared)[:10]}")
        print("  Para una partición sin fuga usa: grouped_split (borra --split-file y reintenta).")
    else:
        print("✓ Ningún paciente compartido entre train y test")

    # ── Copiar / enlazar ─────────────────────────────────────────────────
    counts = {"train": 0, "test": 0}
    for sub in ("train", "test"):
        (args.out / sub / "wav").mkdir(parents=True, exist_ok=True)
        (args.out / sub / "txt").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        sub = split[stem]
        if sub not in counts:
            continue
        for src, folder, ext in ((wavs[stem], "wav", ".wav"), (txts[stem], "txt", ".txt")):
            dst = args.out / sub / folder / f"{stem}{ext}"
            if dst.exists() or dst.is_symlink():
                continue
            if args.symlinks:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        counts[sub] += 1

    diag = next(iter(root.rglob("patient_diagnosis.csv")), None) or \
           next(iter(root.rglob("ICBHI_Challenge_diagnosis.txt")), None)
    if diag is not None:
        shutil.copy2(diag, args.out / diag.name)

    total = sum(counts.values())
    print(f"✓ train={counts['train']} ({counts['train']/max(total,1):.0%})  "
          f"test={counts['test']} ({counts['test']/max(total,1):.0%})  → {args.out.resolve()}")


if __name__ == "__main__":
    main()
