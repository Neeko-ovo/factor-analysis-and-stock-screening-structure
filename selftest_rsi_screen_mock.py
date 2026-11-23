# -*- coding: utf-8 -*-
"""screen_rsi 本地自测（mock 数据源）"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

import matplotlib
matplotlib.use('Agg')

import selftest_pct_mock as M
import rsi_multiday as R

R.get_price = M.get_price
R.get_index_stocks = M.get_index_stocks
R.get_trade_days = M.get_trade_days
R._today = lambda: pd.Timestamp('2024-12-31')


def check(name, cond, extra=''):
    print('  %s %-46s %s' % ('OK  ' if cond else 'FAIL', name, extra))
    if not cond:
        FAILED.append(name)


FAILED = []
CODES = list(M.CODES)
print('[1] 基本筛选（全池 20 只，hit 模式）')
hit = R.screen_rsi(pool=CODES[:20], signal='hit', bars=250, verbose=True)
print(hit.to_string(index=False) if len(hit) else '  （空）')
check('返回 DataFrame', isinstance(hit, pd.DataFrame), str(len(hit)) + ' 行')
check('列名齐全', list(hit.columns) == R._SCREEN_COLS, str(list(hit.columns)))
check('hit 行都有信号或在极值区',
      all((r['信号'] != '') or (r['状态'] in ('超买', '超卖')) for _, r in hit.iterrows()))

print('\n[2] 各模式数量单调性')
cnt = {}
for mode in ['buy', 'sell', 'cross', 'overbought', 'oversold', 'hit', 'all']:
    d = R.screen_rsi(pool=CODES[:20], signal=mode, bars=250, verbose=False)
    cnt[mode] = len(d)
print('   ', cnt)
check('all >= hit', cnt['all'] >= cnt['hit'])
check('hit >= buy + sell 之外的极值也算', cnt['hit'] >= cnt['cross'])
check('cross >= buy / sell', cnt['cross'] >= max(cnt['buy'], cnt['sell']))
check('buy + sell >= cross（同一天可能双信号）', cnt['buy'] + cnt['sell'] >= cnt['cross'])

print('\n[3] 排序：买入信号排最前')
if len(hit):
    has_buy = hit['信号'].str.contains('买入')
    if has_buy.any():
        first_buy = int(np.argmax(has_buy.values))
        last_pri = 0 if '买入' in hit.iloc[0]['信号'] else 9
        check('首行是买入信号', first_buy == 0, hit.iloc[0]['代码'])
    else:
        print('  SKIP 这批数据没有买入信号')

print('\n[4] within 参数：窗口越大命中越多')
c0 = len(R.screen_rsi(pool=CODES[:20], signal='cross', within=0, bars=250, verbose=False))
c5 = len(R.screen_rsi(pool=CODES[:20], signal='cross', within=5, bars=250, verbose=False))
c20 = len(R.screen_rsi(pool=CODES[:20], signal='cross', within=20, bars=250, verbose=False))
print('    within 0/5/20 →', c0, c5, c20)
check('窗口单调不减', c0 <= c5 <= c20)

print('\n[5] 周线筛选')
w = R.screen_rsi(pool=CODES[:10], signal='all', freq='W', bars=200, verbose=False)
check('周线能跑', len(w) > 0, str(len(w)) + ' 行')

print('\n[6] 错误参数')
try:
    R.screen_rsi(pool=CODES[:3], signal='xxx')
    check('未知 signal 报错', False)
except ValueError as e:
    check('未知 signal 报错', True, str(e)[:40])

print('\n[7] 空池 / 限制数量')
e1 = R.screen_rsi(pool=[], signal='hit', verbose=False)
check('空池返回空表', len(e1) == 0 and list(e1.columns) == R._SCREEN_COLS)
l5 = R.screen_rsi(pool=CODES[:20], signal='all', limit=5, bars=250, verbose=False)
check('limit=5 只扫 5 只', len(l5) <= 5, str(len(l5)))

print('\n[8] 信号日期是真实交易日且 <= 最新日期')
if len(hit):
    bad = [r['代码'] for _, r in hit.iterrows()
           if r['信号日期'] and r['信号日期'] > r['日期']]
    check('信号日期不晚于数据日期', len(bad) == 0, str(bad[:3]))

print('\n[9] plot_hits 出图')
R._setup_style(False)
figs = R.plot_hits(pd.DataFrame({'代码': CODES[:2], '名称': ['', '']}),
                   FREQ='D', BARS=200, SHOW_VOLUME=False)
check('画出来 2 张', len(figs) == 2, str(len(figs)))
import matplotlib.pyplot as plt
plt.close('all')

print('\n[10] 构造 RSI 序列，精确验证分区 / 交叉 / 排序 / 窗口')
CASES = {
    'F001.XSHG': [80, 80, 80],    # 超买
    'F002.XSHG': [20, 20, 20],    # 超卖
    'F003.XSHG': [26, 28, 31],    # 今天上穿 30 → 买入
    'F004.XSHG': [74, 72, 68],    # 今天下穿 70 → 卖出
    'F005.XSHG': [50, 50, 50],    # 中性无信号
    'F006.XSHG': [26, 32, 32],    # 昨天上穿、今天维持
}
_CUR = ['']
_orig_load, _orig_rsi = R.load_bars, R.calc_rsi


def _fake_load(code, cfg=None):
    _CUR[0] = code
    n = 40
    idx = pd.bdate_range('2024-01-01', periods=n)
    c = np.full(n, 100.0)
    return pd.DataFrame({'open': c, 'high': c, 'low': c, 'close': c,
                         'volume': np.full(n, 1e6)}, index=idx)


def _fake_rsi(close, n=14):
    tail = CASES[_CUR[0]]
    arr = np.full(len(close), np.nan)
    arr[-len(tail):] = tail
    return pd.Series(arr, index=close.index)


try:
    R.load_bars, R.calc_rsi = _fake_load, _fake_rsi
    h2 = R.screen_rsi(pool=list(CASES), signal='hit', verbose=False)
    got = list(h2['代码'])
    print('    hit →', got)
    check('命中 4 只（超买/超卖/买入/卖出）',
          got == ['F003.XSHG', 'F002.XSHG', 'F004.XSHG', 'F001.XSHG'], str(got))
    check('中性那只被排除', 'F005.XSHG' not in got)
    check('排序：买入>超卖>卖出>超买', got[:1] == ['F003.XSHG'])

    b = R.screen_rsi(pool=list(CASES), signal='buy', verbose=False)
    check('buy 只出 F003', list(b['代码']) == ['F003.XSHG'], str(list(b['代码'])))
    _last = pd.bdate_range('2024-01-01', periods=40)[-1].strftime('%Y-%m-%d')
    check('buy 的信号日期是最后一根',
          len(b) == 1 and list(b['信号日期'])[0] == _last,
          (list(b['信号日期'])[0] if len(b) else '') + ' vs ' + _last)

    s = R.screen_rsi(pool=list(CASES), signal='sell', verbose=False)
    check('sell 只出 F004', list(s['代码']) == ['F004.XSHG'], str(list(s['代码'])))

    w0 = R.screen_rsi(pool=['F006.XSHG'], signal='cross', within=0, verbose=False)
    w1 = R.screen_rsi(pool=['F006.XSHG'], signal='cross', within=1, verbose=False)
    check('within=0 抓不到昨天那根', len(w0) == 0, str(len(w0)))
    check('within=1 能抓到', len(w1) == 1, str(len(w1)))

    z = R.screen_rsi(pool=['F001.XSHG'], signal='overbought', verbose=False)
    check('overbought 抓 F001', len(z) == 1 and z.iloc[0]['状态'] == '超买', '')
finally:
    R.load_bars, R.calc_rsi = _orig_load, _orig_rsi

print()
if FAILED:
    print('FAILED: %s' % FAILED)
    sys.exit(1)
print('全部通过')
