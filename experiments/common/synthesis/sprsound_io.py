"""
sprsound_io.py — write synthetic recordings exactly in the SPRSound format and manage the provenance manifest.

WAV : 8 kHz, mono, PCM-16.
JSON: {"record_annotation": "Normal" | "DAS" | "CAS", "event_annotation": [{"start": "338", "end": "2463", "type": "..."}]}
      (start/end are millisecond strings, like the original files) plus a "synthetic" provenance block that the
      original loaders ignore.
Name: <year>__<patient>_<age>_<gender>_<location>_<recording>.wav with patient = SYN<index>, so the year, age,
      gender and chest location of the host recording are preserved and the file sorts with the real data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import soundfile as sf

from . import FS


def synthetic_name(year: str, idx: int, age, gender, location: str) -> str:
    return f"{year}__SYN{idx:05d}_{age}_{gender}_{location}_{900000 + idx}"


def write_recording(out_dir: Path, name: str, pcm, record_annotation: str, events: List[Dict], provenance: Dict) -> None:
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "wav" / f"{name}.wav"), pcm, FS, subtype="PCM_16")
    body = {"record_annotation": record_annotation,
            "event_annotation": [{"start": str(int(e["start"])), "end": str(int(e["end"])), "type": e["type"]} for e in events],
            "synthetic": provenance}
    (out_dir / "json" / f"{name}.json").write_text(json.dumps(body, indent=2, default=float), encoding="utf-8")


def load_manifest(path: Path, exclude_years: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Manifest of the synthetic recordings. Pass the held-out year in `exclude_years` so a leave-one-year-out
    experiment never trains on synthetic data derived from the evaluation year (synthetic data are train-only).
    """
    m = pd.read_csv(path, dtype={"year": str})
    if exclude_years:
        m = m[~m["year"].isin([str(y) for y in exclude_years])]
    return m.reset_index(drop=True)
