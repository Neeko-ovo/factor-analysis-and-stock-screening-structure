# -*- coding: utf-8 -*-
"""
把「量化工作台.ipynb」的代码单元在 mock 数据下真跑一遍。
目的：确认运行端的每个 cell 都能自己跑通（不依赖上一个 cell 的副作用除外——顺序照 notebook 来）。

用法：python exec_workbench_mock.py
"""
import io
import json
import os
import sys
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))
import matplotlib
matplotlib.use('Agg')

NB = os.path.join(HERE, '量化工作台.ipynb')

# ---- 1. mock 数据源（复用 selftest_pct_mock.py 头部）
src = open(os.path.join(HERE, 'selftest_pct_mock.py'), encoding='utf-8').read().splitlines()
cut = next(i for i, l in enumerate(src) if l.startswith("sys.modules['jqdata']"))
M = {'__name__': 'mockjq', '__file__': 'selftest_pct_mock.py'}
exec(compile('\n'.join(src[:cut + 1]), 'mock_head', 'exec'), M)
print('[mock] jqdata 假模块已注入')

import factor_lib as F
import rsi_multiday as R

for name in ('get_price', 'get_index_stocks', 'get_trade_days', 'get_security_info'):
    if name in M:
        setattr(F, name, M[name])
        setattr(R, name, M[name])
        print('  [mock] %s -> F / R' % name)
F.LOCAL_MODE = True

# notebook 里的示例只跑几只，别让 mock 也跑 300 只
F.apply_params(POOL=['000300.XSHG', '510300.XSHG', '512480.XSHG'],
               START='2022-01-01', END='2024-06-28', PERIOD=20, TH=60.0)

# ---- 2. 按 notebook 顺序执行代码单元
nb = json.load(open(NB, encoding='utf-8'))
ns = {'__name__': '__wb__'}
errs = []
n = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    n += 1
    s = ''.join(c['source'])
    s = '\n'.join(l for l in s.splitlines() if not l.strip().startswith('%'))
    try:
        exec(compile(s, 'cell%d' % i, 'exec'), ns)
    except Exception:
        errs.append(i)
        print('\n[FAIL] cell %d 首行: %s' % (i, s.strip().splitlines()[0][:70]))
        traceback.print_exc()
        break

# ---- 3. 关键符号自检
print('\n== 自检 ==')
ic = ns['res'].get('ic') if isinstance(ns.get('res'), dict) else None
checks = {
    'F / R 都在': 'F' in ns and 'R' in ns,
    'res 有 ic': ic is not None and len(ic) > 0,
    'figs 是字典': isinstance(ns.get('figs'), dict),
    't 有 signal': ns.get('t') is not None and ns['t'].get('signal') is not None,
    'setup_overview 出表': hasattr(ns.get('ov'), 'columns'),
    'picks 是 DataFrame': hasattr(ns.get('picks'), 'columns'),
}
for k, v in checks.items():
    print(('  [OK] ' if v else '  [!!] ') + k)

ok = (not errs) and all(checks.values())
print('\n[%s] 代码单元 %d 个，失败 %d 个' % ('PASS' if ok else 'FAIL', n, len(errs)))
sys.exit(0 if ok else 1)
