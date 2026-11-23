# -*- coding: utf-8 -*-
"""模拟「本地没有 get_price」的环境，按顺序跑 notebook 的全部 code cell。"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['MPLCONFIGDIR'] = os.path.join(HERE, '_mplcache')
import matplotlib
matplotlib.use('Agg')

import json

nb = json.load(open(os.path.join(HERE, 'RSI多日图.ipynb'), encoding='utf-8'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print('code cells:', len(cells))

NS = {'__name__': 'rsi_nb'}
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    # 第一个 code cell 是参数区，把默认标的换成 510300 测 ETF
    if i == 0:
        src = src.replace("POOL = ['000300.XSHG']", "POOL = ['510300.XSHG']")
        src = src.replace('BARS = 300', 'BARS = 200')
    try:
        exec(compile(src, '<cell%d>' % i, 'exec'), NS)
    except Exception:
        import traceback
        print('!!! cell %d 失败' % i)
        traceback.print_exc()
        sys.exit(2)
    print('cell %d ok' % i)

print('figs =', len(NS.get('figs', [])))
print('snapshot rows =', len(NS.get('table', [])))
print('全部通过')
