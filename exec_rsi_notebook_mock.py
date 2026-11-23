# -*- coding: utf-8 -*-
"""按顺序执行 RSI多日图.ipynb 的所有代码单元（用假聚宽数据）。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ['MPLCONFIGDIR'] = os.path.join(HERE, '_mplcache')

import matplotlib
matplotlib.use('Agg')

try:  # 测试环境不弹浏览器：plotly 的 show() 直接空转，图本身照常生成
    import plotly.graph_objects as _pgo
    _pgo.Figure.show = lambda self, *a, **k: None
except Exception:
    pass

import selftest_pct_mock as M

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

nb = json.load(open(os.path.join(HERE, 'RSI多日图.ipynb'), encoding='utf-8'))
g = {'__name__': '__main__'}
g.update({
    'get_price': M.get_price,
    'get_index_stocks': M.get_index_stocks,
    'get_trade_days': M.get_trade_days,
    'get_security_info': M.get_security_info,
})

n = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    n += 1
    src = ''.join(c['source'])
    print('\n----- cell %d (%d 行) -----' % (i, len(c['source'])))
    exec(compile(src, '<cell %d>' % i, 'exec'), g)

print('\n单元全部跑通：%d 个 code cell' % n)
print('POOL =', g.get('POOL'), '| FREQ =', g.get('FREQ'), '| RSI_N =', g.get('RSI_N'))
print('figs =', len(g.get('figs', [])))
t = g.get('table')
print('快照表:')
print(t.to_string(index=False) if t is not None else None)
print('calc_rsi →', type(g.get('calc_rsi')), '| to_bars →', type(g.get('to_bars')))
