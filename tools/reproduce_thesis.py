#!/usr/bin/env python3
"""
Runs, in order and with the arguments used for the thesis, every step that produced its results.

Stages (run all of them, or pick some with --stages):
  preprocess    patches of the historical sections and of the three new sections, validity masks
  domain_shift  statistics, figures, closure test and inverse-correction parameters (Chapter 3, Section 4.4)
  train         the 37 training runs of the ablation (Table 5.4), one after the other
  replica       the Baseline replica with the same seed, meant for a second machine (Section 6.1)
  test          test-set evaluation of every run (eval_*.txt in each run folder)
  infer         predictions on the new sections: single pass, TTA, inverse correction of the Baselines
  evaluate      out-of-domain metrics on the annotated crops (gt_ood/, gt_ood_corr/)
  summary       tables of Chapter 6 (output/campaign3/summary/)

Every command is printed before it runs; --dry-run only prints them. A training run whose folder already
contains train_summary.json is skipped, so the train stage can be stopped and restarted.
Training is non-deterministic on GPU (cuDNN), so retrained models match the thesis only up to the
run-to-run variability discussed in Section 6.1.

Usage (from the repository root):
    python tools/reproduce_thesis.py --dry-run
    python tools/reproduce_thesis.py --stages train test infer evaluate summary
    python tools/reproduce_thesis.py --stages train --only gen_060 dd_060      # a subset of the runs
"""

import os
import sys
import argparse
import subprocess

PY = sys.executable
RUNS_DIR = os.path.join('output', 'campaign3', 'MMSFormer')
OUTPUT_ROOT = os.path.join('output', 'campaign3')
ANALYSIS_DIR = os.path.join('output', 'domain_shift_analysis_v2')
DEFAULT_SEED = 3407   # TRAIN.SEED of every config: runs with this seed get no '-seedN' suffix

BASELINE = 'configs/baseline/mortars_v3_baseline-median.yaml'
REPLICA = 'configs/baseline/mortars_v3_baseline-median-micc.yaml'
DD = 'configs/ablation/data_driven'
GEN = 'configs/ablation/generic'

# (config, seeds) of the ablation, as in Table 5.4 of the thesis
TRAINING_RUNS = [
    (BASELINE, [3407, 42, 3408]),
    *[(f'{DD}/{op}/mortars_v3_dd1_{op}-{p}.yaml', [3407]) for op in ('gamma', 'saturation', 'noise', 'blur')
      for p in ('030', '060', '090') if not (op == 'gamma' and p == '090')],
    (f'{DD}/gamma/mortars_v3_dd1_gamma-090.yaml', [3407, 42]),
    *[(f'{DD}/contrast/mortars_v3_dd1_contrast-{p}.yaml', [3407, 42, 3408]) for p in ('030', '060', '090')],
    (f'{DD}/full/mortars_v3_dd-060.yaml', [3407, 42, 3408]),
    (f'{GEN}/mortars_v3_gen-060.yaml', [3407, 42, 3408]),
    (f'{GEN}/mortars_v3_gen-tex-060.yaml', [3407, 42, 3408]),
    (f'{GEN}/mortars_v3_gen-col-060.yaml', [3407, 42, 3408]),
]
BASELINE_KEYS = ['v3_baseline_med', 'v3_baseline_med_seed42', 'v3_baseline_med_seed3408']


def run_folder(cfg_path, seed):
    """Run folder written by tools/train_mm.py: WANDB_NAME [+ -pXXX from P_IMAGE] [+ -seedN]."""
    import yaml
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    name = cfg['WANDB_NAME']
    photometric = cfg['TRAIN']['AUGMENTATION'].get('PHOTOMETRIC')
    if photometric and photometric.get('P_IMAGE') is not None:
        name += f"-p{round(float(photometric['P_IMAGE']) * 100):03d}"
    if seed != DEFAULT_SEED:
        name += f'-seed{seed}'
    return os.path.join(cfg['SAVE_DIR'], name)


def model_key(folder):
    return os.path.basename(folder).lower().replace('mmsf-', '', 1).replace('-', '_')


def all_runs(only=None):
    runs = [(cfg, seed, run_folder(cfg, seed)) for cfg, seeds in TRAINING_RUNS for seed in seeds]
    if only:
        runs = [r for r in runs if any(o in model_key(r[2]) for o in only)]
    return runs


def stage_commands(stage, args):
    if stage == 'preprocess':
        return [
            [PY, 'tools/preprocess.py'],
            [PY, 'tools/inspect_and_mask_new_images.py'],
            [PY, 'tools/preprocess_new_images.py'],
            [PY, 'tools/inspect_and_mask_5x.py'],
            [PY, 'tools/preprocess_5x.py'],
            [PY, 'tools/make_valid_masks.py'],
        ]
    if stage == 'domain_shift':
        return [
            [PY, 'tools/quantify_domain_shift.py', '--output-dir', ANALYSIS_DIR],
            [PY, 'tools/plot_domain_shift.py', '--analysis-dir', ANALYSIS_DIR],
            [PY, 'tools/check_augmentation_params.py', '--analysis-dir', ANALYSIS_DIR,
             '--cfg', f'{DD}/full/mortars_v3_dd-060.yaml'],
            [PY, 'tools/fit_inverse_correction.py', '--analysis-dir', ANALYSIS_DIR],
        ]
    if stage == 'train':
        cmds = []
        for cfg, seed, folder in all_runs(args.only):
            if os.path.isfile(os.path.join(folder, 'train_summary.json')) and not args.dry_run:
                print(f"[skip] {folder} already trained")
                continue
            cmds.append([PY, 'tools/train_mm.py', '--cfg', cfg] + ([] if seed == DEFAULT_SEED else ['--seed', str(seed)]))
        return cmds
    if stage == 'replica':
        return [[PY, 'tools/train_mm.py', '--cfg', REPLICA]]
    if stage == 'test':
        runs = all_runs(args.only) + [(REPLICA, DEFAULT_SEED, run_folder(REPLICA, DEFAULT_SEED))]
        return [[PY, 'tools/val_mm.py', '--cfg', cfg, '--run-dir', folder,
                 '--vis-dir', os.path.join(OUTPUT_ROOT, 'results', model_key(folder))]
                for cfg, _, folder in runs if os.path.isdir(folder) or args.dry_run]
    if stage == 'infer':
        params = os.path.join(ANALYSIS_DIR, 'inverse_correction', 'inverse_correction_params.json')
        return [
            [PY, 'tools/run_all_inferences_softmax.py', '--domains', '1_SCALA', 'UNITO_B', 'ARCHEO_02'],
            [PY, 'tools/run_all_inferences_softmax.py', '--domains', '1_SCALA', 'UNITO_B', '--tta'],
            [PY, 'tools/run_all_inferences_softmax.py', '--models', *BASELINE_KEYS,
             '--domains', '1_SCALA', 'UNITO_B', '--inverse-correction', params],
        ]
    if stage == 'evaluate':
        return [
            [PY, 'tools/eval_ood_gt.py', '--sources', 'single', 'tta',
             '--output-dir', os.path.join(OUTPUT_ROOT, 'gt_ood')],
            [PY, 'tools/eval_ood_gt.py', '--experiments', *BASELINE_KEYS, '--sources', 'single', 'corr',
             '--output-dir', os.path.join(OUTPUT_ROOT, 'gt_ood_corr')],
        ]
    if stage == 'summary':
        return [[PY, 'tools/summarize_results.py',
                 '--corr-csv', os.path.join(OUTPUT_ROOT, 'gt_ood_corr', 'gt_ood_metrics.csv')]]
    raise ValueError(stage)


STAGES = ['preprocess', 'domain_shift', 'train', 'replica', 'test', 'infer', 'evaluate', 'summary']


def main():
    parser = argparse.ArgumentParser(description="Reproduce the experiments of the thesis")
    parser.add_argument('--stages', nargs='+', default=[s for s in STAGES if s != 'replica'], choices=STAGES)
    parser.add_argument('--only', nargs='+', default=None,
                        help="train/test: only the runs whose key contains one of these strings (e.g. gen_060 seed42)")
    parser.add_argument('--dry-run', action='store_true', help="Print the commands without running them")
    args = parser.parse_args()

    for stage in [s for s in STAGES if s in args.stages]:
        print(f"\n{'=' * 70}\nSTAGE {stage}\n{'=' * 70}", flush=True)
        for cmd in stage_commands(stage, args):
            print('$ ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
