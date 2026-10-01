# VOLTRACER: Adaptive Fiber-Cable Recovery at Ruapehu

## Objective

Each case contains eight real seismic windows recorded by a 14-channel distributed-acoustic-sensing cable near Ruapehu volcano. Three channel traces are initially visible. Design **one two-stage sensor-acquisition policy for the entire eight-event batch**: choose a first missing channel, then choose a second missing channel depending on the first channel's measured response. The objective is to reconstruct the 28-window energy profiles of the nine channels left unmeasured, not to predict eruptions or public-safety outcomes.

This is a policy-inference challenge. Test-time first-probe responses and the other eleven traces are withheld; submit the policy before either probe is acquired. The evaluator replays it on the recorded held-out traces. A policy can branch differently across the eight events, but its four integer parameters are fixed for the whole case.

## Data and supervised signal

`train.csv` and `test.csv` have exactly `case_id,file,anchors_json` in that order. `file` is relative to the prepared public directory. IDs are opaque. `anchors_json` lists the three initially visible channel indices, each an integer from 0 to 13. Each `.npz` has one `traces` array: shape `(8,14,14900)` for train and `(8,3,14900)` for test. In test arrays the second axis follows `anchors_json` order. Samples are signed 16-bit, at 50 Hz, representing an event-wise, anchor-normalized, clipped and quantized strain proxy; divide by 4096 to get dimensionless normalized amplitude. This is **not calibrated absolute strain**.

`train_labels.csv` has `case_id,policy_json,truth_file`. The JSON is an optimal legal policy for that training case. `truth_file` points to a public training `.npz` with `loss` `(8,14,14)` and `response` `(8,14)`. `loss[e,i,j]` is the measured event reconstruction loss if distinct non-anchor channels `i,j` are acquired; its diagonal and unavailable combinations are zero placeholders, not legal actions. `response[e,i]` is an integer 0–5 representing the bin observed after probing channel `i`. These training labels permit supervised learning, direct policy optimization, and checking an independent simulator. The public `sample_submission.csv` uses `case_id,policy_json`. The private `answers.csv` has the same ordered columns; its JSON contains evaluator-only truth.

Training and test cases are grouped by source recording day, and no source day crosses the split. All eight events in a case come from the same day and are distinct. Cases are not independent physical volcanoes: this is one cable and one 2023 acquisition period, so generalization to other instruments, sites, and eruption regimes is not established.

## Policy program and replay

For each case submit a JSON object with **exactly** these four integer keys:

`{"first":3,"threshold":3,"second_low":7,"second_high":10}`

`first` is the first additional channel. `second_low` is acquired if its observed response bin is **strictly less** than `threshold`; otherwise `second_high` is acquired. The first and both second-channel choices must be from `0..13` excluding the three anchors. Each second choice must differ from `first`; the two branch choices may equal one another. `threshold` is an integer from 0 to 6. Threshold 0 always chooses the high branch; threshold 6 always chooses the low branch. The same policy is replayed independently across all eight events. No participant-supplied response, waveform, or case-specific action list is accepted.

For the acquired first channel, the evaluator divides its first 14,000 samples into 28 consecutive 500-sample windows. Let `E_w` be each window's RMS of sample/4096. It computes `r = log2((mean(E_14..27)+1e-6)/(mean(E_0..13)+1e-6))`. The response bin is the number of cutpoints in `[-0.8,-0.4,0,0.4,0.8]` that `r` is at least. This yields 0–5. Training `response` is calculated by this rule, including clipping and quantization of the raw signal.

With the three anchors and the two chosen channels known, each remaining channel's 28-window RMS profile is reconstructed by piecewise-linear interpolation in ascending **channel index**, using the nearest known channel on each side. Beyond the outermost known channels, use the nearest known channel's profile without extrapolation. An event's loss is the mean absolute difference between predicted and measured energy across the nine unmeasured channels and 28 windows, divided by `(mean absolute measured energy across all 14 channels and 28 windows + 1e-8)`. The case's risk-adjusted batch loss is **0.75 × mean of all eight event losses + 0.25 × mean of the two largest event losses**. Ties in the largest two are harmless because only their values are used.

## Metric

For each test case, the evaluator enumerates all legal first channels, thresholds, and low/high second-channel choices on the withheld recordings. Let `best` and `worst` be the minimum and maximum **risk-adjusted batch** loss among these policies. A submission's case score is `clip((worst - submitted_loss)/(worst - best), 0, 1)`. If all legal policies tie to numerical tolerance `1e-12`, every legal policy scores 1. The final scalar score is the arithmetic mean of case scores over the evaluator's answer rows. **Higher is better; 1 is optimal.** Every test case has equal weight, irrespective of day or signal amplitude. The 0.75/0.25 mean/tail weights retain average reconstruction as the primary objective while explicitly discouraging a policy that catastrophically misses a minority of the eight events. The tail component is the worst quarter (two of eight) and does not require a subjective loss threshold.

The normalization makes 0 the measured worst legal policy and 1 the measured best legal policy, so no uncalibrated energy scale dominates the ranking. It also means a baseline's absolute score is specific to this action space and set of test days. The score measures sensor-placement utility, not an eruption forecast or seismological event classification.

## Submission and constraints

Submit CSV with exactly `case_id,policy_json` in that order, one row for each test case. `policy_json` must be valid JSON; quote/escape the JSON cell according to CSV rules. Keep policy keys and values exactly as specified. The platform may add a `visibility` column; it is ignored. Duplicate IDs, missing evaluated IDs, wrong columns, or invalid actions receive zero according to the grader contract. A grader invocation may contain a private answer subset against a full submission; extra submission rows outside that subset are ignored.

Allowed: public training waveforms and training truth, public test anchors, general-purpose signal-processing or ML software, and independently developed models. Prohibited: private organizer files; matching test traces against the openly published upstream event windows to recover withheld channels; source event IDs/timestamps or any other target lookup; altering grader or answer files. The original dataset remains public for attribution and independent research, but using it as a test-target lookup defeats this benchmark's intended unseen-channel task.

## Provenance, limitations and risk

The underlying real measurements are from Leighton Watson's [Ruapehu distributed-acoustic-sensing record](https://zenodo.org/records/20789299), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This challenge selects 13 daily archives, groups event windows, normalizes amplitudes using visible anchors, quantizes samples, and derives policy-utility labels. The data are windows around catalogued earthquakes at a volcano-adjacent cable; they are **not labeled eruptions**. A source-level waveform-matching attack remains possible in principle because the source itself is open. The split prevents ordinary row-ID memorization, not access to an external copy of the same waveform. Nearby channels may be highly correlated; the interpolation rule intentionally rewards geometry-aware selection. Train and test also share a physical instrument, so the leaderboard does not establish cross-volcano transfer.
