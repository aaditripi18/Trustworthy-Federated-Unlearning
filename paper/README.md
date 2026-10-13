# Paper artefacts

Scripts, figures and derived results behind the paper
*Attention-Fidelity-Preserving Federated Unlearning with Vision Transformers for Chest X-Rays*.

All numbers are computed from the run outputs in
`ec2_final_run/tfu/experiments/{full_experiment,exp_a1,ga_kaggle}/`.

| File | Purpose |
|---|---|
| `make_overview_fig.py` | Fig. 1, three-panel block diagram |
| `make_paper_figs.py` | Data figures, from `final_results.csv` and `run.log` (the paper uses the training and dose-response figures) |
| `make_qualitative_fig.py` | Fig. 4, attention-rollout maps (needs the model checkpoints) |
| `compute_setting_a_stability.py` | Explanation stability for setting A, computed from the saved checkpoints; writes results only if it reproduces the oracle similarities logged by the original run |
| `setting_a_stability.csv` | Output of the script above (15 rows: 5 models x 3 seeds) |
| `probe_images_needed.txt` | The 243 NIH ChestX-ray14 probe images (first 96 test images per seed) |
| `fetch_probe_images.py` | Downloads only those probe images from a public mirror of the dataset |
| `compute_forgetting_attacks.py` | Loss, Oracle-calibrated and positive-finding membership attacks on the saved checkpoints, with patient-level bootstrap intervals |
| `forgetting_attacks.csv` | Output of the script above (13 models x 3 seeds, including the gradient-ascent models) |
| `ga_results.csv` | Gradient-ascent baseline (ascent only, GA, GA+EPU), from `scripts/run_ga.py` |
| `verify_numbers.py` | Checks the numbers quoted in the paper text against the result files |
| `figures/` | Rendered figures and the cached inputs of Fig. 4 |

Model checkpoints and dataset images are not tracked. `compute_setting_a_stability.py`,
`make_qualitative_fig.py` and `fetch_probe_images.py` expect them under `ec2_final_run/`.

```bash
python paper/verify_numbers.py
python paper/make_paper_figs.py
```
