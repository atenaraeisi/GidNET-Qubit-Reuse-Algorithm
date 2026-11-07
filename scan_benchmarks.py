#!/usr/bin/env python3
"""Scan a folder of OpenQASM 2 files (no Qiskit needed) and pick circuits for the scale-up experiment.

Per file: declared / ACTIVE qubits, #gates, #2q, #>=3q, ASAP depth, file size.
Selection (fixed BEFORE looking at any routing result, to avoid cherry-picking):
  active qubits in [--min-active, --max-active], 2q gates in [--min-2q, --max-2q], no >=3q gates unless
  --allow-multi, size <= --max-kb, near-duplicate names (same stem before the last _<number>) capped by --per-family.
Writes scan.csv (everything) and selected.csv (the picks); prints a table.

  python scan_benchmarks.py --dir "<benchmarks>" --out-dir scan_out
"""
import argparse, hashlib, os, re, sys
import pandas as pd

QREG = re.compile(r'^\s*qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;')
ARG = re.compile(r'(\w+)\s*\[\s*(\d+)\s*\]')
SKIP = ('OPENQASM', 'include', 'creg', 'barrier', 'measure', 'reset', 'if', 'gate', '//', '}')


def scan(path):
    regs, off, tot = {}, 0, 0
    last, touched = {}, set()
    n_g = n2 = n3 = 0
    depth = 0
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = QREG.match(s)
            if m:
                regs[m.group(1)] = off
                off += int(m.group(2)); tot = off
                continue
            if s.startswith(SKIP):
                continue
            body = s.split(None, 1)
            if len(body) < 2:
                continue
            qs = [regs[r] + int(i) for r, i in ARG.findall(body[1]) if r in regs]
            if not qs:
                continue
            n_g += 1
            touched.update(qs)
            if len(qs) == 2:
                n2 += 1
            elif len(qs) >= 3:
                n3 += 1
            d = 1 + max(last.get(q, 0) for q in qs)
            for q in qs:
                last[q] = d
            depth = max(depth, d)
    return dict(declared=tot, active=len(touched), gates=n_g, n_2q=n2, n_3q_plus=n3, depth=depth,
                kb=os.path.getsize(path) / 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out-dir', default='scan_out')
    ap.add_argument('--min-active', type=int, default=13)
    ap.add_argument('--max-active', type=int, default=30)
    ap.add_argument('--min-2q', type=int, default=50)
    ap.add_argument('--max-2q', type=int, default=3000)
    ap.add_argument('--max-kb', type=float, default=300)
    ap.add_argument('--allow-multi', action='store_true', help='keep circuits that contain >=3-qubit gates')
    ap.add_argument('--per-family', type=int, default=2, help='max variants per family (e.g. urf1_149, urf1_278)')
    a = ap.parse_args()
    rows = []
    for fn in sorted(os.listdir(a.dir)):
        if not fn.endswith('.qasm'):
            continue
        try:
            r = scan(os.path.join(a.dir, fn))
        except Exception as e:
            r = dict(error=str(e))
        body = re.sub(r'\s+', ' ', open(os.path.join(a.dir, fn), errors='ignore').read())
        rows.append(dict(circuit=fn, family=re.sub(r'[_-]?\d+$', '', fn[:-5]),
                         sha=hashlib.sha1(body.encode()).hexdigest()[:12], **r))
    df = pd.DataFrame(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(os.path.join(a.out_dir, 'scan.csv'), index=False)
    ok = df[(df.active >= a.min_active) & (df.active <= a.max_active) & (df.n_2q >= a.min_2q)
            & (df.n_2q <= a.max_2q) & (df.kb <= a.max_kb)]
    if not a.allow_multi:
        ok = ok[ok.n_3q_plus == 0]
    ok = ok.drop_duplicates('sha')          # identical circuits under different names (e.g. cm42a/pm1)
    ok = ok.sort_values(['family', 'n_2q']).groupby('family').head(a.per_family)
    ok.to_csv(os.path.join(a.out_dir, 'selected.csv'), index=False)
    pd.set_option('display.width', 200)
    print(f'{len(df)} files scanned; {len(ok)} selected')
    print(ok[['circuit', 'declared', 'active', 'n_2q', 'n_3q_plus', 'depth', 'kb']].to_string(index=False))
    print('\nrejected because of >=3q gates only:',
          len(df[(df.active.between(a.min_active, a.max_active)) & (df.n_2q.between(a.min_2q, a.max_2q)) & (df.n_3q_plus > 0)]))


if __name__ == '__main__':
    main()
