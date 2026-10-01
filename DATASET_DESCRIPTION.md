# VOLTRACER dataset card

**Dataset title:** VOLTRACER Ruapehu DAS Adaptive Cable-Recovery Corpus, version 1.

**Original source:** Leighton Watson, *Distributed Acoustic Sensing at Ruapehu Volcano in 2023*, [Zenodo 20789299](https://zenodo.org/records/20789299), DOI [10.5281/zenodo.20789299](https://doi.org/10.5281/zenodo.20789299). **Source license:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). The redistributed processed recordings retain this attribution and license. There is no copyright-free claim.

The original record describes 50 Hz fiber-optic measurements, 14 channels spaced about 6.38 m apart on a telecommunications cable at Ruapehu, and windows associated with catalogued earthquakes. This derivative uses 13 of its daily archives. Exact source file hashes and selected days are pinned in the organizer's `SOURCE_MANIFEST.json`; transformations are implemented in `src/build_derived.py` and `src/prepare.py`. The challenge does not label eruptions or predict volcanic hazards.

## Prepared file inventory

- `train.csv`, `test.csv`: `case_id,file,anchors_json`, ordered identically. `case_id` is an opaque string identifier; `file` is a relative path; `anchors_json` is an ordered JSON list of three unique channel indices 0–13.
- `arrays/<case_id>.npz`: `traces` signed-int16 array, `(8,14,14900)` in training and `(8,3,14900)` in test. Test-channel order matches `anchors_json`. Each case groups eight independent recorded event windows from a single source day. Original absolute timestamps and event IDs are removed. Sampling is 50 Hz; the 14,900 samples cover 298 seconds. Divide the int16 values by 4096 for dimensionless normalized amplitude; clipping at ±32767 is possible.
- `train_labels.csv`: `case_id,policy_json,truth_file`. The policy is an optimal legal batch policy. The truth file is a relative path in `train_truth/`.
- `train_truth/<case_id>.npz`: `loss` float32 array `(8,14,14)` of per-event, per-two-probe reconstruction loss and `response` uint8 array `(8,14)` of response bins 0–5. Ineligible entries of `loss` are zero placeholders. `loss` is a dimensionless normalized energy-reconstruction error, **not** an earthquake magnitude or eruption probability.
- `sample_submission.csv`: ordered `case_id,policy_json`; legal but non-oracle example policies.
- Organizer-only `answers.csv`: the same two columns, with private policy truth packed inside each JSON cell for deterministic evaluation. Never redistribute this file to participants.

## Generation and split

Cases are selected by valid sample count and grouped in nonoverlapping batches of eight; HMAC-based deterministic sorting obscures original event names. Seven source days become train and six entire days become test. The choice of days was made before utility labels were derived; the six-day test assignment avoids concentrating most evaluation rows in a single source day. Each source waveform is scaled by the 95th percentile of absolute amplitude on its three visible anchors, quantized to signed int16, and cropped to its first 14,900 samples. The primary output metric uses 28 nonoverlapping 500-sample RMS-energy windows over the first 14,000 samples. Response bins come from a late/early energy-ratio cutpoint rule; utility is measured against actual held-out channel energies. See the challenge specification for exact equations and the submitted-policy contract.

## Known limitations

One geographic site, one cable, one year and an earthquake-window sampling frame limit generalization. Channel correlations may be unusually strong or weak for other cable layouts. Day-held-out testing reduces within-day leakage but cannot prove independence of related earthquake sequences across days. Anchor normalization destroys absolute amplitude calibration, while quantization, clipping and cropping can distort subtle signals. The source archives remain public, so high-effort waveform matching could reveal withheld channels even though row IDs and timestamps are removed; rules forbid that attack, but technical secrecy is not claimed. Source events are not confirmed eruptions, and no operational hazard decision should be made from these labels. There may be high class/response-bin imbalance and high correlation between neighboring channels; these should be reported rather than interpreted as volcanic mechanisms.

## Redistribution and attribution

Original measurement attribution: Leighton Watson, *Distributed Acoustic Sensing at Ruapehu Volcano in 2023*, Zenodo 20789299, CC BY 4.0. Changes made for VOLTRACER: day selection, event batching, HMAC identifiers, anchor masking, normalization, quantization, window-energy features, response bins, and interpolation-utility targets. Users must retain attribution, license link and indication of changes when redistributing adapted data.
