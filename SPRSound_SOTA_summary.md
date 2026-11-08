# SPRSound – State of the Art Summary

**Dataset:** SPRSound (Shanghai Jiao Tong Univ. / Shanghai Children's Medical Center) – first open pediatric respiratory sound database (1 month – 18 years). Original release: 2,683 recordings, 9,089 events, 292 participants, 8 kHz, Yunting II digital stethoscope.
Repo: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound

---

## 1. Tasks defined on the dataset

| Level | Task | Classes |
|---|---|---|
| Event | 1-1 | Normal vs Adventitious |
| Event | 1-2 | Normal, Rhonchi, Wheeze, Stridor, Coarse Crackle, Fine Crackle, Wheeze & Crackle (7) |
| Record | 2-1 | Normal, Adventitious, Poor Quality |
| Record | 2-2 | Normal, CAS, DAS, CAS & DAS, Poor Quality |
| Record (2024–25) | Event detection (SED) | onset/offset + label of each event |
| Record (2024–25) | Compression | compressive sensing of recordings |

## 2. Important note about the metrics

Most papers **do not report Precision / Recall / F1** directly. The official BioCAS challenge metric is:

- **SE (Sensitivity)** = recall of the adventitious classes → *this is the "recall" equivalent*
- **SP (Specificity)** = recall of the Normal class
- **AS** = (SE + SP)/2, **HS** = 2·SE·SP/(SE + SP)  *(HS is F1-like, but between SE and SP, not precision/recall)*
- **Score** = (AS + HS)/2

Event detection tracks (2024–2025) use **event-based F-score** and **Error Rate**.
Results are **not strictly comparable**: train/test splits and test sets (2022 vs 2023 release) differ between papers. `n/r` = not reported.

---

## 3. Models and results

### 3.1 Classification – challenge metric (Score, SE/SP)

| Model (year) | Type | Task 1-1 | Task 1-2 | Task 2-1 | Task 2-2 | SE / SP (recall) |
|---|---|---|---|---|---|---|
| Original baseline – SPRSound paper (2022) | Handcrafted features + ML classifiers | 75.22 | 61.57 | 56.71 | 37.84 | n/r |
| STFT + fine-tuned ResNet18 (Chen et al., 2022) | CNN, transfer learning (ImageNet) | HS 0.89 | HS 0.80 | HS 0.71 | HS 0.36 | 1-1: SE 0.89 / SP 0.90 · 1-2: SE 0.68 / SP 0.94 |
| Inception-Residual + multi-objective loss (Ngo et al., 2023) | CNN on Gammatone + CWT spectrograms | +9.7 % vs base | +15.8 % | **74.5** (Top-1) | **53.9** (Top-1) | n/r |
| Spatio-temporal focusing (Ngo et al., BioCAS 2023) | CNN + multi-spectrogram ensemble | 81.0 | 66.7 | 74.4 | 60.8 | n/r |
| Dual-input CNN (Pessoa et al., BioCAS 2023) | 1D CNN (raw audio) + 2D CNN (STFT) fusion | n/r here | n/r here | n/r here | n/r here | n/r |
| WLANN – Waveform-Logmel (2025) | 1D-CNN + AST (transformer) + Bi-GRU | improves over baselines (abnormal classes) | | | | n/r |
| Bi-SEResNet + DeiT (Wang et al., 2025) | SE-ResNet + Vision Transformer, mel + waveform | **89.26** | **83.63** | – | – | n/r |
| Multi-stage hybrid CNN-Transformer (Shuvo & Hasan, 2025) | CNN + Transformer on scalograms, focal loss | **90.39** | **84.48** | 72.0 | 57.1 | n/r |
| CMBES framework (youth) | DL, score = mean(AS, F1) | 91 ± 3 | 82 ± 3 | 74 ± 2 | 55 ± 3 | n/r |
| Rene (Zhang et al., 2024) | Pre-trained speech model (Whisper-like) + medical records, multimodal | +10.3 % | +16.2 % | +15.3 % | +18.9 % vs baseline | n/r |

### 3.2 Classification – Precision / Recall / F1 reported

| Model (year) | Type | Setup | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Google HeAR embeddings + SVM / RF / MLP (Ehtesham et al., 2025) | Audio foundation model + classic ML | Binary: asthma-indicative vs normal, 2-s segments | high (not detailed in abstract) | high | n/r · **Accuracy > 91 %** |
| PulmoVec (HeAR + LightGBM stacking, 2026) | Foundation model + meta-learning, multi-task | Screening / sound pattern / disease group | n/r | n/r | Patient-level disease: **weighted F1 0.73**, Acc 0.74, macro AUC 0.91 · Event screening AUC 0.96 |
| Neuro-Scattering Mamba-KAN (2026, MDPI) | Lightweight state-space (Mamba) + KAN, no transformer | 4 classes (Normal, Crackle, Wheeze, Both) | Normal 0.89 · Crackle 0.75 · Wheeze 0.39 | Normal 0.94 · Crackle 0.48 · Wheeze 0.47 | Normal 0.92 · Crackle 0.58 · Wheeze 0.42 · **Macro-F1 0.48** · Acc 84.67 % |

### 3.3 Sound event detection (onset/offset)

| Model | Type | Event-based F1 | Error Rate |
|---|---|---|---|
| CNN + VAD (baseline cited by EZhouNet) | CNN + voice activity detection | 6.81 % (class-wise) | 1.26 |
| EZhouNet (2025) | Graph neural network + anchor intervals | improves on CNN+VAD (4 abnormal classes) | lower |

> Event detection is still a weak area: F1 scores are very low compared with event classification.

---

## 4. Families of models used

1. **Classic ML** – handcrafted features (MFCC, spectral) + SVM / RF / ensembles (baseline, FP-TLEM).
2. **CNNs on time-frequency images** – STFT, Mel, Gammatone, wavelet (CWT/scalograms); ResNet, Inception, transfer learning.
3. **Dual-input / multimodal fusion** – raw waveform (1D CNN) + spectrogram (2D CNN / AST).
4. **Hybrid CNN-Transformer** – CNN features + attention (DeiT, AST, Transformer encoders) — current best on Task 1.
5. **Recurrent** – LSTM, Bi-GRU for temporal context.
6. **Foundation / pre-trained audio models** – Google HeAR, speech-recognition models (Rene) + clinical records.
7. **Lightweight / edge models** – Mamba-KAN, compressed Rene for real-time.
8. **Sound event detection** – CNN+VAD, Graph Neural Networks.

## 5. Functional applications addressed

- **Automatic auscultation screening** – normal vs abnormal lung sounds in children.
- **Adventitious sound typing** – wheeze, crackle, rhonchi, stridor (support for diagnosis of asthma, bronchiolitis, pneumonia).
- **Pediatric asthma detection** – HeAR pipeline, noninvasive screening.
- **Disease-group prediction** – pneumonia vs bronchial diseases etc. at patient level (PulmoVec, Rene).
- **Recording quality control** – detecting *Poor Quality* recordings (Task 2).
- **Real-time / wearable edge AI** – Rene real-time system on auscultation devices; lightweight "green AI" models.
- **Telemedicine / remote auscultation** – compression of recordings for digital stethoscopes (2024–25 track).
- **Event localization** – onset/offset detection of abnormal events for physician review.
- **Low-resource settings** – most works motivate use where pediatric specialists are scarce.

## 6. Key takeaways

- **Best event-level results:** hybrid CNN-Transformer (~0.90 binary, ~0.84 7-class Score).
- **Record level is harder:** best Task 2-2 Scores ≈ 0.54–0.61.
- **Rare classes (Stridor, Rhonchi, Coarse Crackle, Wheeze & Crackle) are the bottleneck** – strong class imbalance (e.g. 17 Stridor vs 6,887 Normal events).
- **Precision/Recall/F1 per class are rarely published**; if you need them, you'll likely have to compute them yourself or read full papers.

---

## Sources

- SPRSound dataset paper (IEEE TBioCAS 2022): https://pure.ecnu.edu.cn/en/publications/sprsound-open-source-sjtu-paediatric-respiratory-sound-database/
- SPRSound GitHub: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
- BioCAS 2024/2025 challenges (IEEE DataPort): https://ieee-dataport.org/competitions/respiratory-sound-track-grand-challenge-2024-respiratory-sound-compression-and-event · https://ieee-dataport.org/competitions/respiratory-sound-track-grand-challenge-2025-respiratory-sound-compression-and-event
- STFT + ResNet18: https://arxiv.org/pdf/2208.13943
- Inception-Residual (Ngo et al.): https://arxiv.org/abs/2303.04104v2
- Spatio-temporal focusing: https://arxiv.org/pdf/2306.14929 · https://www.ndcn.ox.ac.uk/publications/1621693
- Dual-input CNN (Pessoa et al.): https://eden.dei.uc.pt/~ruipedro/publications/Conferences/BioCAS_2023_Pessoa.pdf
- WLANN: https://arxiv.org/pdf/2504.17156
- Bi-SEResNet + DeiT: https://research.mpu.edu.mo/en/publications/hybrid-dual-input-model-for-respiratory-sound-classification-with/
- Multi-stage hybrid CNN-Transformer: https://arxiv.org/html/2507.20408v2 · https://arxiv.org/pdf/2507.20408
- CMBES youth framework: https://proceedings.cmbes.ca/index.php/proceedings/article/download/973/962
- Rene: https://www.arxiv.org/abs/2405.07442
- HeAR pediatric asthma: https://www.arxiv.org/abs/2504.20124
- PulmoVec: https://arxiv.org/html/2603.15688
- Mamba-KAN: https://www.mdpi.com/2076-3417/16/16/8342
- EZhouNet: https://arxiv.org/html/2509.01153
