# Reference clip analysis (MEDZCOOL)

Analysis at the SPRSound native rate (8 kHz) on **unique** breath cycles only. Red marks in the overview figure are the unique cycles used for the statistics.

## FC — Fine Crackles

- native 44100 Hz, stereo L/R correlation 1.000, 99 % of the energy below **904 Hz** (SPRSound's median recording has no energy above ~1 kHz)
- 24 active breath segments, **3 unique** (cluster sizes [8, 8, 8]): the clip is a loop of 3 distinct segment(s), so estimates rest on few independent examples
## CC — Coarse Crackles

- native 44100 Hz, stereo L/R correlation 1.000, 99 % of the energy below **506 Hz** (SPRSound's median recording has no energy above ~1 kHz)
- 34 active breath segments, **2 unique** (cluster sizes [17, 17]): the clip is a loop of 2 distinct segment(s), so estimates rest on few independent examples
## W — Wheezing

- native 44100 Hz, stereo L/R correlation 1.000, 99 % of the energy below **345 Hz** (SPRSound's median recording has no energy above ~1 kHz)
- 33 active breath segments, **2 unique** (cluster sizes [17, 16]): the clip is a loop of 2 distinct segment(s), so estimates rest on few independent examples

### Crackle morphology — FC

- detected 80 transients, 45 isolated crackles fitted (R² ≥ 0.4)
- f0: 611.95 Hz (p10–p90 426.12–660.54, n=45)
- tau: 4.12 ms (p10–p90 2.56–8.69, n=45)
- sensitivity of the medians across 18 detection/fit settings: f0 474–625 Hz, tau 3.4–4.3 ms, 2CD 3.0–3.7 ms
- IDW: 0.50 ms (p10–p90 0.25–0.62, n=44) · 2CD: 3.00 ms (p10–p90 2.62–4.70, n=45)
- rate inside active segments: 27.34 /s (p10–p90 25.44–28.35, n=3)
- inter-crackle interval (<150 ms): 22.12 ms (p10–p90 10.69–52.25, n=76); gamma shape ≈ 1.78
- amplitude / local RMS (log-normal): 2.03 (p10–p90 0.44–3.83, n=80)
- timing in the segment (first/middle/last third): 16%, 48%, 36%

### Crackle morphology — CC

- detected 38 transients, 20 isolated crackles fitted (R² ≥ 0.4)
- f0: 346.65 Hz (p10–p90 235.54–377.00, n=20)
- tau: 5.41 ms (p10–p90 3.23–8.51, n=20)
- sensitivity of the medians across 18 detection/fit settings: f0 195–351 Hz, tau 4.3–7.7 ms, 2CD 4.8–10.9 ms
- IDW: 1.00 ms (p10–p90 0.24–1.50, n=20) · 2CD: 5.06 ms (p10–p90 4.38–8.38, n=20)
- rate inside active segments: n/a
- inter-crackle interval (<150 ms): 20.38 ms (p10–p90 10.38–95.45, n=35); gamma shape ≈ 1.22
- amplitude / local RMS (log-normal): 2.00 (p10–p90 0.85–2.71, n=38)
- timing in the segment (first/middle/last third): 32%, 37%, 32%

### Wheeze

- f0 (dominant persistent spectral track, 120–1200 Hz): 267.58 Hz (p10–p90 262.89–272.27, n=2)
- track duration: 1.26 s (p10–p90 1.19–1.33, n=2)
- f0 jitter (std of residual / f0): 0.01 (p10–p90 0.01–0.01, n=2)
- relative f0 drift over the event: -0.12 (p10–p90 -0.14–-0.11, n=2)
- harmonics vs f0: 2f0 -46.42 dB (p10–p90 -46.76–-46.08, n=2), 3f0 -46.07 dB (p10–p90 -48.62–-43.51, n=2), 4f0 -63.92 dB (p10–p90 -67.15–-60.68, n=2)
- polyphonic (more than one simultaneous strong track) in 50% of unique segments
- track start/end inside the breath segment (fraction): 0.08 (p10–p90 0.08–0.08, n=2) / 0.87 (p10–p90 0.84–0.91, n=2)