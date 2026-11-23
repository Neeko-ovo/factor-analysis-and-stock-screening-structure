# -*- coding: utf-8 -*-
# ============================================================
# 本地 mock 自测：Part I RSI 体系
#   1) 符号齐全、三个时序信号已注册
#   2) calc_rsi 与 Part H 的 rsi_14 数值一致（min_periods=0 口径）
#   3) 边界：全涨/全跌/横盘；三种 avg 方法（wilder/sma/ema）
#   4) rsi_zone 的 75/25 分区
#   5) 反转事件与状态机（long/short/both/confirm）
#   6) rsi_zone_forward 分区前瞻收益
#   7) run_ts_test('rsi', state_thresholds=[25,75])
#   8) rsi_report 一键报告
#   9) plot_rsi 出图落盘
# 运行：python selftest_rsi_mock.py
# ============================================================
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

# ---- 复用 selftest_pct_mock.py 的 mock 数据层（含假 jqdata 模块注入）----
_MOCK = os.path.join(HERE, 'selftest_pct_mock.py')
with io.open(_MOCK, 'r', encoding='utf-8') as fh:
    _src = fh.read()
_head = _src.split("NS = {'__name__': 'jq_ns'")[0]
_ns = {'__name__': '_mock_data', '__file__': _MOCK}
exec(compile(_head, _MOCK, 'exec'), _ns)
CODES = _ns['CODES']

sys.path.insert(0, HERE)
if 'jq_factor_framework' in sys.modules:
    del sys.modules['jq_factor_framework']
import jq_factor_framework as jf

CODE0 = CODES[0]
START, END = '2019-01-02', '2023-12-29'

print('\n[1] 符号与信号注册')
need = ['calc_rsi', 'rsi_panel', 'rsi_zone', 'rsi_zone_label', 'rsi_latest',
        'rsi_cross_up', 'rsi_cross_down', 'rsi_reversal_events', 'rsi_reversal_state',
        'rsi_zone_forward', 'rsi_zone_spread', 'rsi_report', 'plot_rsi',
        'RSI_OB', 'RSI_OS', 'RSI_DEFAULTS']
missing = [n for n in need if not hasattr(jf, n)]
assert not missing, '缺符号: %s' % missing
for s in ('rsi', 'rsi_rev', 'rsi_rev_hold'):
    assert s in jf.TS_SIGNALS, '信号未注册: %s' % s
assert jf.RSI_OB == 75.0 and jf.RSI_OS == 25.0, '默认阈值应是 75 / 25'
print('    默认超买 %.0f / 超卖 %.0f | 信号 %s'
      % (jf.RSI_OB, jf.RSI_OS, [s for s in ('rsi', 'rsi_rev', 'rsi_rev_hold') if s in jf.TS_SIGNALS]))

print('\n[2] calc_rsi 与 Part H rsi_14 数值对齐')
px = jf.get_price_range([CODE0], START, END, ('close', 'high', 'low', 'money'))
close = px['close'][CODE0].dropna()
r_new = jf.calc_rsi(close, 14, 'wilder', min_periods=0)
sig_old, _ = jf.build_ts_signal('rsi_14', CODE0, START, END)
r_old = sig_old.reindex(close.index)
both = pd.concat([r_new.rename('new'), r_old.rename('old')], axis=1).dropna()
assert len(both) > 800, '可比样本太少'
diff = (both['new'] - both['old']).abs().max()
print('    可比 %d 根，最大绝对差 %.3e，新旧首值 %.4f / %.4f'
      % (len(both), diff, both['new'].iloc[0], both['old'].iloc[0]))
assert diff < 1e-9, '新旧 RSI 数值不一致（%.6f）' % diff
assert r_new.dropna().between(0, 100).all(), 'RSI 应落在 0~100'

print('\n[3] 边界与三种均值方法')
up = pd.Series(np.arange(1.0, 121.0))
dn = pd.Series(np.arange(120.0, 0.0, -1.0))
flat = pd.Series(np.full(120, 10.0))
for meth in ('wilder', 'sma', 'ema'):
    u = float(jf.calc_rsi(up, 14, meth).dropna().iloc[-1])
    d = float(jf.calc_rsi(dn, 14, meth).dropna().iloc[-1])
    f = float(jf.calc_rsi(flat, 14, meth).dropna().iloc[-1])
    print('    %-6s 全涨 %.4f | 全跌 %.4f | 横盘 %.4f' % (meth, u, d, f))
    assert u > 99.0, '%s 全涨应接近 100' % meth
    assert d < 1.0, '%s 全跌应接近 0' % meth
    assert abs(f - 50.0) < 1e-9, '%s 横盘应为 50' % meth
assert jf.calc_rsi(close, 14).iloc[:13].isna().all(), '默认前 n 根应为 NaN'

print('\n[4] 75 / 25 分区')
z = jf.rsi_zone(pd.Series([10, 20, 25, 26, 50, 74, 75, 76, 90]), ob=75, os_=25)
# 严格不等号：恰好 25 / 75 算中性
assert z.tolist() == [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], z.tolist()
zc = jf.rsi_zone(r_new, 75, 25).dropna()
assert set(zc.unique().tolist()) <= {-1.0, 0.0, 1.0}
print('    构造序列分区 OK | 真实数据分布 超卖 %d / 中性 %d / 超买 %d'
      % (int((zc == -1).sum()), int((zc == 0).sum()), int((zc == 1).sum())))
_latest_d, _latest_v, _latest_lab = jf.rsi_latest(r_new)
print('    最新 RSI %.2f（%s，%s）'
      % (_latest_v, _latest_lab, pd.Timestamp(_latest_d).date()))

print('\n[5] 反转事件与状态机')
r_seq = pd.Series([50.0, 20.0, 30.0, 40.0, 80.0, 70.0, 20.0, 30.0, 40.0, 90.0])
ev = jf.rsi_reversal_events(r_seq, 75, 25)
assert len(ev) == 3, '应有 3 个反转事件，实际 %d' % len(ev)
assert ev['dir'].tolist() == [1.0, -1.0, 1.0], ev['dir'].tolist()
st_long = jf.rsi_reversal_state(r_seq, 75, 25, mode='long')
st_short = jf.rsi_reversal_state(r_seq, 75, 25, mode='short')
st_both = jf.rsi_reversal_state(r_seq, 75, 25, mode='both')
st_obj = jf.rsi_reversal_state(r_new, 75, 25, mode='long', confirm=1)
print('    long   %s' % st_long.tolist())
print('    short  %s' % st_short.tolist())
print('    both   %s' % st_both.tolist())
assert set(st_long.unique().tolist()) <= {0.0, 1.0}
assert set(st_short.unique().tolist()) <= {-1.0, 0.0}
assert set(st_both.unique().tolist()) <= {-1.0, 0.0, 1.0}
assert st_long.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], st_long.tolist()
assert st_obj.dropna().isin([0, 1]).all(), 'confirm 口径应仍是 0/1'
print('    真实数据 long 状态：在场 %d 天 / %d 天'
      % (int((st_obj == 1).sum()), len(st_obj)))

print('\n[6] rsi_zone_forward 分区前瞻收益')
tbl = jf.rsi_zone_forward(r_new, close, 75, 25, periods=(5, 20))
assert not tbl.empty, '分区收益表为空'
for c in ('区间', 'zone', '前瞻期(日)', '样本', '平均收益%', '中位数%', '胜率'):
    assert c in tbl.columns, '缺列 %s' % c
sp = jf.rsi_zone_spread(tbl)
print(tbl.to_string(index=False))
print('    极差：%s' % sp.to_string(index=False).replace('\n', ' | '))

print('\n[7] run_ts_test 固定阈值分组')
res = jf.run_ts_test('rsi', CODE0, START, END, period=20, th=50.0,
                     state_thresholds=[25, 75], verbose=False)
assert not res['state_table'].empty, '状态分组为空'
assert res['state_table']['状态'].nunique() == 3, '固定阈值应切出 3 档'
rng_txt = ' '.join(res['state_table']['区间'].astype(str).tolist())
assert '25' in rng_txt and '75' in rng_txt, '区间描述没带阈值: %s' % rng_txt
print('    IC(Spearman) %+.4f | t(NW) %+.2f | 分组极差 %+.3f%%/期'
      % (res['ic_stats']['IC(Spearman)'], res['ic_stats']['t值(Newey-West)'],
         res['state_spread']['spread%']))
print(res['state_table'][['状态', '区间', '观测数', '平均收益%', '胜率']].round(3).to_string(index=False))
bt = res['backtest']
assert isinstance(bt, pd.DataFrame) and len(bt) > 0, '回测结果为空'
print(bt.round(3).to_string())

print('\n[7b] 宽表 calc_rsi 与另两个信号')
px3 = jf.get_price_range(CODES[:3], START, END, ('close',))
pnl = jf.calc_rsi(px3['close'], 14, min_periods=0)
assert pnl.shape == px3['close'].shape, '宽表 RSI 形状不对'
assert (pnl.dropna().stack().between(0, 100)).all(), '宽表 RSI 应落在 0~100'
for s in ('rsi_rev', 'rsi_rev_hold'):
    sg, _ = jf.build_ts_signal(s, CODE0, START, END)
    assert isinstance(sg, pd.Series), '%s 信号类型不对' % s
    assert sg.notna().sum() > 0, '%s 信号全空' % s
    r2 = jf.run_ts_test(s, CODE0, START, END, th=0.5 if s == 'rsi_rev_hold' else 0.0,
                        verbose=False)
    assert not r2['backtest'].empty, '%s 回测为空' % s
    print('    %-12s 有效 %d 个 | 择时夏普 %+.2f'
          % (s, int(sg.notna().sum()), r2['backtest'].loc['策略(择时)', '夏普']))

print('\n[8] rsi_report 一键报告')
rep = jf.rsi_report(CODE0, START, END, verbose=False)
for k in ('code', 'close', 'rsi', 'zone', 'state', 'zone_table', 'zone_spread',
          'events', 'last', 'backtest', 'detail', 'params'):
    assert k in rep, '报告缺字段 %s' % k
assert rep['params']['ob'] == 75.0 and rep['params']['os'] == 25.0
assert len(rep['rsi'].dropna()) > 800
_d, _v, _lab = rep['last']
print('    最新 RSI %.2f（%s，%s）| 事件 %d 个'
      % (_v, _lab, pd.Timestamp(_d).date(), len(rep['events'])))
print(rep['backtest'].round(3).to_string())

print('\n[9] plot_rsi 出图')
out_png = os.path.join(HERE, '_rsi_test.png')
fig = jf.plot_rsi(close=close, rsi=r_new, title='RSI 自测 %s' % CODE0, save=out_png)
assert fig is not None and os.path.exists(out_png) and os.path.getsize(out_png) > 20000
print('    已保存 %s（%.1f KB）' % (out_png, os.path.getsize(out_png) / 1024.0))

print('\n[done] Part I RSI 自测全部通过')
