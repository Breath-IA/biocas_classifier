# Real SPRSound event statistics (native 8 kHz, after the real acquisition chain)

Up to 300 events per class, stratified by year. Same detectors and fitters as the reference analysis.

## Fine crackles (FC) in SPRSound

- 300 events; 2065 isolated transients fitted with the 150 Hz high-pass
- **f0 is not identifiable at SPRSound's bandwidth**: the fitted f0 follows the high-pass cutoff (100 / 150 / 250 Hz → median f0 174 / 216 / 288 Hz; share of fits within 60 Hz of the cutoff 38% / 46% / 67%). The real crackle energy sits in a very low band, unlike the reference clip (f0 ≈ 610 Hz).
- tau 6.8 ms (p10–p90 4.0–15.0) and 2CD 8.1 ms are longer than in the reference (≈ 4 ms and 3 ms): the chain low-pass stretches the crackles; tau by year {'2022': 6.157535852585693, '2023': 6.573912893360678, '2024': 7.165925548065184, '2025': 7.431295415297042}
- transients per second inside events (150 Hz high-pass, k = 4.5; includes breath-noise bumps): median 11.2 /s (p10–p90 5.7–18.8)
- inter-transient interval median 18.0 ms, gamma shape 1.12
- amplitude / local RMS log-sd 1.00

## Wheezes (W) in SPRSound

- 220 events with a tracked tone
- f0 203 Hz (p10–p90 129–514); by year {'2022': 236.328125, '2023': 207.03125, '2024': 195.3125, '2025': 195.3125}
- tone duration 0.15 s; jitter CV 0.010; relative drift -0.000
- harmonics vs f0: 2f0 -23.9 dB, 3f0 -47.9 dB, 4f0 -65.4 dB
- polyphonic in 36% of the events

## Channel estimate

- relative low-pass vs. the reference device: **375 Hz, order 2** (fit RMSE 12.8 dB). Used only as the starting point of the calibration.

## Per-year noise floor above the stethoscope band

Median PSD at 1.5 kHz per year (dB; −135 dB is the 16-bit quantization floor): 2022: -134, 2023: -134, 2024: -134, 2025: -134