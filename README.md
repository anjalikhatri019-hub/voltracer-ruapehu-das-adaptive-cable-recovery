# VOLTRACER: Adaptive Fiber-Cable Recovery at Ruapehu

VOLTRACER is a real-data, general-purpose sensor-acquisition and policy-inference challenge using distributed-acoustic-sensing (DAS) waveforms recorded near Ruapehu volcano. From three initially visible cable channels, a participant chooses one more channel and a response-conditional final channel for each batch of eight seismic windows. The evaluator scores how well those choices recover the unmeasured channels' energy profiles, including a penalty for the two worst-recovered events.

This is **not eruption prediction**. The source contains earthquake-associated windows recorded at a volcanic site. Operational or safety decisions must not be based on this benchmark.

## Which ZIP is which?

- `VOLTRACER_Eris_PRIVATE_INPUT_v1.zip` (550 MB): **organizer/platform upload only**. It contains hidden test targets, the private source map and ID salt. Do not publish it or provide it to solvers.
- `VOLTRACER_Ruapehu_PARTICIPANT_v1_1.zip`: public prepared participant data and documentation; it contains no private answers, salt or source map. This is the file to publish as the source-verifiable dataset release.

The organizer uploads the private ZIP to the challenge creation form, pastes `src/prepare.py` in Prepare and `src/grade.py` in Grade, and uses `PROBLEM_DESCRIPTION.md` as the challenge description. Prepare accepts a ZIP file, opaque renamed upload file, or extracted root containing `manifest.csv` and the source manifest. It emits participant files into its public output and `answers.csv` into its private output. The platform may hold a full submission while grading a public or private answer subset; the grader supports that contract.

## Dataset and license

Original waveforms: Leighton Watson, [Distributed Acoustic Sensing at Ruapehu Volcano in 2023](https://zenodo.org/records/20789299), DOI [10.5281/zenodo.20789299](https://doi.org/10.5281/zenodo.20789299), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source record is the authoritative original-data URL. The public VOLTRACER release URL should be used for the **derived dataset's Source URL** only after it actually exists and its downloadable ZIP is verified. Do not paste an invented release URL.

See [DATASET_DESCRIPTION.md](DATASET_DESCRIPTION.md) for fields, units, modifications, limitations and leakage caveats, and [PROBLEM_DESCRIPTION.md](PROBLEM_DESCRIPTION.md) for the precise objective, formulas, rules and submission contract. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) carries attribution and a changes notice.

## Local validation

From this mission directory, with NumPy installed:

```powershell
python src/prepare.py packages/VOLTRACER_Eris_PRIVATE_INPUT_v1.zip workspace/prepared/public workspace/prepared/private
python src/grade.py workspace/prepared/private/answers.csv workspace/prepared/public/sample_submission.csv
python src/audit20.py
```

The builder additionally needs h5py and the pinned source archives. `reports/BUILD_STATS.json`, `reports/HARDEN_ONCE.json`, and `reports/AUDIT_20.json` record the build and validation outcomes. Never upload `workspace/prepared/private`, `workspace/raw_derived`, `source`, `PRIVATE_ID_SALT.txt` or the organizer ZIP to a public repository.
