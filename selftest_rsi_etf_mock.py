# -*- coding: utf-8 -*-
"""ETF 池接入自测（本地 mock，不联网）"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_mplcache')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

import selftest_pct_mock as M
import rsi_multiday as R

R.get_price = M.get_price
R.get_index_stocks = M.get_index_stocks
R._today = lambda: pd.Timestamp('2024-12-31')

fail = []


def chk(name, cond, extra=''):
    print(('  OK  ' if cond else '  FAIL') + '  ' + name + (('  | ' + str(extra)) if extra else ''))
    if not cond:
        fail.append(name)


print('[1] ETF 池名单')
print('  全池 %d 只，分类 %s' % (len(R.ETF_NAME), '/'.join(R.ETF_POOL)))
chk('全池 64 只', len(R.ETF_NAME) == 64, len(R.ETF_NAME))
chk('510300 名字', R.ETF_NAME.get('510300.XSHG') == '沪深300ETF')
chk('代码不重复', len(R.ETF_NAME) == len(set(R.ETF_NAME)))

print('[2] resolve_pool')
for key, n in [('ETF', 64), ('ETF池', 64), ('宽基', 9), ('宽基ETF', 9),
               ('防御', 5), ('行业', 22), ('科技', 9), ('新能源', 6),
               ('医药', 3), ('金融', 3), ('周期', 4),
               ('港股', 4), ('海外', 6), ('商品', 3)]:
    got = R.resolve_pool([key])
    chk('%s -> %d 只' % (key, n), len(got) == n, len(got))
chk('去重（ETF + 宽基）', len(R.resolve_pool(['ETF', '宽基'])) == 64,
    len(R.resolve_pool(['ETF', '宽基'])))
chk('指数池不受影响', R.resolve_pool(['HS300']) != R.resolve_pool(['宽基']))

print('[3] 图标题带 ETF 名字')
rng = np.random.default_rng(7)
n = 120
c = 4 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
df = pd.DataFrame(dict(open=c * (1 + rng.normal(0, .003, n)), high=c * 1.01,
                       low=c * 0.99, close=c, volume=rng.integers(1e6, 9e6, n).astype(float)),
                  index=pd.bdate_range('2024-01-02', periods=n))
fig = R.plot_kline_rsi(df, '510300.XSHG', dict(FREQ='D', BARS=120, RSI_N=14, OB=75, OS=25))
title = fig._suptitle.get_text()
print('  title =', title)
chk('标题含名称', '沪深300ETF' in title)
import matplotlib.pyplot as plt
plt.close(fig)

print('[4] 快照表')
tb = R.rsi_snapshot(POOL=['宽基', 'S000.XSHG'])
print(tb.to_string(index=False))
chk('有名称列', '名称' in tb.columns)
chk('宽基那几只名字非空', tb[tb['代码'].str.startswith(('5103', '5105', '1599', '588'))]['名称'].ne('').all())

print('[5] MAX_CHARTS=0 不限')
orig_load = R.load_bars
R.load_bars = lambda code, cfg: df.copy()
figs = R.run_all(POOL=['宽基'], BARS=100, MAX_CHARTS=0, SHOW_VOLUME=False)
chk('画了 9 张', len(figs) == 9, len(figs))
for f in figs:
    plt.close(f)
R.load_bars = orig_load

print()
if fail:
    print('FAILED:', fail)
    sys.exit(1)
print('全部通过')
