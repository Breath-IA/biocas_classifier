"""
Synthetic auscultation toolkit (plan.md, component 2: synthetic augmentation).

Design rules (plan.md + findings of the previous experiments):

* The synthetic *event* is parametric (damped-sinusoid crackles, self-sustained-oscillator wheezes) but the
  *recording* is always real: events are inserted into real SPRSound normal recordings ("hosts").
* Everything keeps the SPRSound acquisition method: 8 kHz, mono, PCM-16, the dataset's electronic-stethoscope
  frequency response (a causal band-pass applied to the synthetic event only), event-level JSON annotations
  (start/end in ms + type) and record-level tags (Normal / DAS / CAS).
* Event duration, level (SNR vs. the recording floor) and crackle rate are drawn from real events of the same
  class and year, so the annotation style and level distribution stay natural.
* All classes (N, FC, W) go through the same builder so that "synthetic" cannot become a label cue.
* Synthetic data are for training only: every file carries the host year/patient so a leave-one-year-out
  consumer can drop the held-out year.

Modules: channel (SPRSound chain), generators (crackle/wheeze sources), reference (analysis of the MEDZCOOL
clips), insertion (host selection, breath-aware placement, level matching), sprsound_io (WAV/JSON writing).
"""
FS = 8000  # native SPRSound sampling rate
