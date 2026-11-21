#!/usr/bin/env python3
"""
Summary tables of the ablation (thesis, Chapter 6): one row per run and mean +- standard deviation
(ddof = 1) across seeds for each configuration and group.

Inputs:
  - <runs-dir>/<run>/train_summary.json   best validation mIoU and epochs (tools/train_mm.py)
  - <runs-dir>/<run>/eval_*.txt           test-set mIoU and IoU per class (tools/val_mm.py)
  - gt_ood_metrics.csv                    out-of-domain metrics (tools/eval_ood_gt.py)
  - optionally the gt_ood_metrics.csv of the inverse correction (sources 'single' and 'corr')

Outputs (in --output-dir):
  runs_summary.csv      per run: test mIoU, val mIoU, OOD mIoU per section and mean, aggregate fractions,
                        per-class accuracy, TTA, collapse flag (>= 65% predicted aggregates on a section)
  configs_summary.csv   per configuration and per group: mean and std of the same columns
  correction_summary.csv  (with --corr-csv) OOD mIoU of the Baselines with and without the correction

Only the configurations of the thesis are summarized; the Baseline replica trained on another machine
(run key ending in '_micc') is reported separately and kept out of the means.

Usage (from the repository root):
    python tools/summarize_results.py
    python tools/summarize_results.py --corr-csv output/campaign3/gt_ood_corr/gt_ood_metrics.csv
"""

import os
import re
import csv
import glob
import json
import argparse
import statistics

DEFAULT_RUNS_DIR = os.path.join('output', 'campaign3', 'MMSFormer')
DEFAULT_OOD_CSV = os.path.join('output', 'campaign3', 'gt_ood', 'gt_ood_metrics.csv')
DEFAULT_OUTPUT_DIR = os.path.join('output', 'campaign3', 'summary')
COLLAPSE_THRESHOLD = 65.0   # % of predicted aggregates on at least one section
SECTIONS = ('1_SCALA', 'UNITO_B')

# Configuration (run key without the seed suffix) -> group of the thesis tables
CONFIG_GROUPS = [
    (r'^v3_baseline_med$', 'Baseline'),
    (r'^v3_dd1_(gamma|contrast|saturation|noise|blur)_0[369]0_p080$', 'Single operations'),
    (r'^v3_dd_060_p080$', 'Data-driven (full)'),
    (r'^v3_gen_060_p080$', 'Generic'),
    (r'^v3_gen_tex_060_p080$', 'Generic, high frequency'),
    (r'^v3_gen_col_060_p080$', 'Generic, low frequency'),
]


def run_key(run_dir):
    return os.path.basename(run_dir).lower().replace('mmsf-', '', 1).replace('-', '_')


def split_seed(key):
    """('v3_gen_060_p080', 42) for 'v3_gen_060_p080_seed42'; the seed without suffix is 3407."""
    m = re.match(r'^(.*)_seed(\d+)$', key)
    return (m.group(1), int(m.group(2))) if m else (key, 3407)


def group_of(config):
    for pattern, group in CONFIG_GROUPS:
        if re.match(pattern, config):
            return group
    return None


def read_test_eval(run_dir):
    """(mIoU, IoU binder, IoU aggregates) of the most recent eval_*.txt on the test split, or Nones."""
    for path in sorted(glob.glob(os.path.join(run_dir, 'eval_*.txt')), reverse=True):
        text = open(path, encoding='utf-8').read()
        if 'split: test' not in text:
            continue
        get = lambda name: float(re.search(rf'^{name}\s+([\d.]+)', text, re.M).group(1))
        return get('Mean'), get('Legante'), get('Aggregati')
    return None, None, None


def read_ood(path):
    """{(experiment, source): {section: row}} from a gt_ood_metrics.csv."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out.setdefault((r['experiment'], r['source']), {})[r['domain']] = {k: float(v) for k, v in r.items()
                                                                              if k not in ('domain', 'experiment', 'source')}
    return out


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(m, s, decimals):
    if m is None:
        return '-'
    return f"{m:.{decimals}f}" + (f" +- {s:.{decimals}f}" if s is not None else '')


def main():
    parser = argparse.ArgumentParser(description="Summary tables of the ablation")
    parser.add_argument('--runs-dir', default=DEFAULT_RUNS_DIR)
    parser.add_argument('--ood-csv', default=DEFAULT_OOD_CSV)
    parser.add_argument('--corr-csv', default=None, help="gt_ood_metrics.csv of the inverse correction")
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ood = read_ood(args.ood_csv)

    runs, replicas = [], []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, 'MMSF-*'))):
        key = run_key(run_dir)
        is_replica = key.endswith('_micc')
        config, seed = split_seed(key[:-len('_micc')] if is_replica else key)
        group = group_of(config)
        if group is None:
            continue
        row = {'run': key, 'config': config, 'group': group, 'seed': seed}
        summary_path = os.path.join(run_dir, 'train_summary.json')
        if os.path.isfile(summary_path):
            s = json.load(open(summary_path, encoding='utf-8'))
            row.update(val_miou=s['best_miou'], best_epoch=s['best_epoch'], last_epoch=s['last_epoch'])
        row['test_miou'], row['test_iou_binder'], row['test_iou_aggregates'] = read_test_eval(run_dir)
        for source, prefix in (('single', ''), ('tta', 'tta_')):
            per_section = ood.get((key, source))
            if not per_section or any(sec not in per_section for sec in SECTIONS):
                continue
            for sec in SECTIONS:
                r = per_section[sec]
                row[f'{prefix}miou_{sec}'] = r['miou']
                row[f'{prefix}pred_aggr_{sec}'] = r['pred_aggregate_fraction']
                if not prefix:
                    row[f'acc_binder_{sec}'] = r['acc_binder']
                    row[f'acc_aggregates_{sec}'] = r['acc_aggregates']
                    row[f'iou_binder_{sec}'] = r['iou_binder']
                    row[f'iou_aggregates_{sec}'] = r['iou_aggregates']
            row[f'{prefix}miou_ood'] = statistics.fmean(row[f'{prefix}miou_{sec}'] for sec in SECTIONS)
            row[f'{prefix}collapse'] = any(row[f'{prefix}pred_aggr_{sec}'] >= COLLAPSE_THRESHOLD for sec in SECTIONS)
        (replicas if is_replica else runs).append(row)

    columns = sorted({k for r in runs + replicas for k in r if k not in ('run', 'config', 'group', 'seed')},
                     key=lambda k: (not k.startswith('test'), not k.startswith('val'), k))
    write_csv(os.path.join(args.output_dir, 'runs_summary.csv'),
              [{**{c: r.get(c) for c in ('run', 'config', 'group', 'seed')}, **{c: r.get(c) for c in columns}}
               for r in runs + replicas])

    summary = []
    for level in ('config', 'group'):
        names = list(dict.fromkeys(r[level] for r in runs))
        for name in names:
            members = [r for r in runs if r[level] == name]
            out = {'level': level, 'name': name, 'runs': len(members)}
            for c in columns:
                if any(isinstance(r.get(c), bool) for r in members):
                    out[f'{c}_count'] = sum(bool(r.get(c)) for r in members)
                    continue
                m, s = mean_std([r.get(c) for r in members])
                out[f'{c}_mean'], out[f'{c}_std'] = m, s
            summary.append(out)
    write_csv(os.path.join(args.output_dir, 'configs_summary.csv'), summary)

    print(f"{'Group':26s} {'Runs':>4s}  {'Test':>14s}  {'1_SCALA':>12s}  {'UNITO_B':>12s}  {'Mean OOD':>12s}  {'Collapses':>9s}")
    for out in (o for o in summary if o['level'] == 'group'):
        print(f"{out['name']:26s} {out['runs']:4d}  {fmt(out.get('test_miou_mean'), out.get('test_miou_std'), 2):>14s}  "
              f"{fmt(out.get('miou_1_SCALA_mean'), out.get('miou_1_SCALA_std'), 1):>12s}  "
              f"{fmt(out.get('miou_UNITO_B_mean'), out.get('miou_UNITO_B_std'), 1):>12s}  "
              f"{fmt(out.get('miou_ood_mean'), out.get('miou_ood_std'), 1):>12s}  {out.get('collapse_count', 0):9d}")
    for r in replicas:
        print(f"Replica {r['run']}: test {r.get('test_miou')}, mean OOD {fmt(r.get('miou_ood'), None, 1)}")

    if args.corr_csv:
        corr = read_ood(args.corr_csv)
        rows = []
        for r in (r for r in runs if r['group'] == 'Baseline'):
            per_section = corr.get((r['run'], 'corr'))
            if not per_section:
                continue
            row = {'run': r['run'], 'miou_ood_original': r.get('miou_ood')}
            for sec in SECTIONS:
                row[f'miou_{sec}'] = per_section[sec]['miou']
                row[f'pred_aggr_{sec}'] = per_section[sec]['pred_aggregate_fraction']
            row['miou_ood'] = statistics.fmean(row[f'miou_{sec}'] for sec in SECTIONS)
            rows.append(row)
        write_csv(os.path.join(args.output_dir, 'correction_summary.csv'), rows)
        m, s = mean_std([r['miou_ood'] for r in rows])
        print(f"Inverse correction on the Baselines: mean OOD {fmt(m, s, 1)} ({len(rows)} runs)")
    print(f"\nTables saved to {args.output_dir}")


if __name__ == '__main__':
    main()
