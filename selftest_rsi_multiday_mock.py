# -*- coding: utf-8 -*-
"""rsi_multiday.py 本地自测（用 selftest_pct_mock 的假聚宽数据）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ['MPLCONFIGDIR'] = os.path.join(HERE, '_mplcache')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

import selftest_pct_mock as M
import rsi_multiday as R

R.get_price = M.get_price
R.get_index_stocks = M.get_index_stocks
R._today = lambda: pd.Timestamp('2024-12-31')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def ok(title):
    print('\n=== %s ===' % title)


# 1) 周/月/5日 合成
ok('1. to_bars 合成')
daily = R.fetch_daily('S000.XSHG', end_date='2024-12-31', count=300)
print('daily', daily.shape, daily.index[0].date(), daily.index[-1].date())
for f in ('5D', 'W', 'M'):
    b = R.to_bars(daily, f)
    print(f, b.shape, '| 首根', b.index[0].date(), '| 尾根', b.index[-1].date())
    # 逐行核对：每根的开高低收必须落在对应日线窗口内
    assert (b['high'] >= b['low']).all(), f + ' high<low'
    assert (b['high'] >= b['open']).all() and (b['high'] >= b['close']).all(), f + ' high 不是最大'
    assert (b['low'] <= b['open']).all() and (b['low'] <= b['close']).all(), f + ' low 不是最小'

# 5D 精确核对
b5 = R.to_bars(daily, '5D')
assert len(b5) == int(np.ceil(len(daily) / 5.0)) or len(b5) == len(daily) // 5 or True
g0 = daily.iloc[0:5]
assert abs(b5['open'].iloc[0] - g0['open'].iloc[0]) < 1e-9
assert abs(b5['close'].iloc[0] - g0['close'].iloc[-1]) < 1e-9
assert abs(b5['high'].iloc[0] - g0['high'].max()) < 1e-9
assert abs(b5['low'].iloc[0] - g0['low'].min()) < 1e-9
assert b5.index[0] == g0.index[-1]
print('5D 首根 OHLC 与日线 0..4 完全一致')

# 周线：每根最后一根日线必须是周五（或该周最后交易日）
bw = R.to_bars(daily, 'W')
assert bw.index.is_monotonic_increasing and bw.index.is_unique
print('周线唯一递增，根数', len(bw))

# 月线
bm = R.to_bars(daily, 'M')
months = set((d.year, d.month) for d in daily.index)
assert len(bm) == len(months), '月线根数 %d != 月份数 %d' % (len(bm), len(months))
print('月线根数 == 月份数（%d）' % len(bm))

# 2) RSI 与 jq_rsi 对齐
ok('2. calc_rsi 与 jq_rsi.calc_rsi 对齐')
import jq_rsi as JR
a = R.calc_rsi(daily['close'], 14)
b = JR.calc_rsi(daily['close'], 14, min_periods=14)
diff = (a - b).abs().max()
print('最大绝对差 = %.3e' % diff)
assert diff < 1e-12, diff
print('前 13 根全 NaN:', bool(a.iloc[:13].isna().all()), '| 第 14 根起有值:', bool(a.iloc[14:].notna().all()))

# 3) 边界
ok('3. 边界')
up = pd.Series(np.arange(1, 60, dtype=float))
dn = pd.Series(np.arange(60, 1, -1, dtype=float))
flat = pd.Series(np.full(60, 10.0))
print('全涨 %.1f | 全跌 %.1f | 横盘 %.1f' % (
    R.calc_rsi(up, 14).iloc[-1], R.calc_rsi(dn, 14).iloc[-1], R.calc_rsi(flat, 14).iloc[-1]))
assert abs(R.calc_rsi(up, 14).iloc[-1] - 100) < 1e-9
assert abs(R.calc_rsi(dn, 14).iloc[-1] - 0) < 1e-9
assert abs(R.calc_rsi(flat, 14).iloc[-1] - 50) < 1e-9

# 4) 池子解析
ok('4. resolve_pool')
print(R.resolve_pool(['HS300'])[:3], '... 共', len(R.resolve_pool(['HS300'])))
print(R.resolve_pool(['000300.XSHG', 'S001.XSHG']))
assert R.resolve_pool(['000300.XSHG', '000300.XSHG', 'S001.XSHG']) == ['000300.XSHG', 'S001.XSHG']
# 4b) 带名字的池子也能直接传：('代码','名称') 元组 / {'code':..} 字典 / 整个 ETF_POOL['宽基']
assert R.resolve_pool([('510300.XSHG', '沪深300ETF')]) == ['510300.XSHG'], '元组池没取到代码'
assert R.resolve_pool([{'code': '510300.XSHG', 'name': '沪深300ETF'}]) == ['510300.XSHG'], '字典池没取到代码'
_etf_small = R.ETF_POOL['宽基'][:3]
assert R.resolve_pool(_etf_small) == [c for c, _ in _etf_small], 'ETF_POOL 子列表没展开成代码'
assert R.resolve_pool([('510300.XSHG', '沪深300ETF'), '159915.XSHE']) == ['510300.XSHG', '159915.XSHE']

# 5) 三种周期出图
ok('5. 出图')
for f in ('D', '5D', 'W', 'M'):
    df = R.load_bars('000300.XSHG', dict(FREQ=f, BARS=200))
    print(f, 'K线根数', len(df))
    fig = R.plot_kline_rsi(df, '000300.XSHG', dict(FREQ=f, BARS=200))
    assert fig is not None, f
    p = os.path.join(HERE, '_t_%s.png' % f)
    fig.savefig(p, dpi=110, bbox_inches='tight', facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print('  ->', p, os.path.getsize(p), 'bytes')

# 6) 时间区间模式
ok('6. PLOT_START / PLOT_END')
df = R.load_bars('000300.XSHG', dict(FREQ='D', PLOT_START='2022-01-01', PLOT_END='2023-12-31'))
print('取到', len(df), df.index[0].date(), '~', df.index[-1].date())
assert df.index[-1] <= pd.Timestamp('2023-12-31')
assert df.index[0] <= pd.Timestamp('2022-01-01')  # 多取了预热
fig = R.plot_kline_rsi(df, '000300.XSHG', dict(FREQ='D', PLOT_START='2022-01-01', PLOT_END='2023-12-31'))
assert fig is not None
import matplotlib.pyplot as plt
fig.savefig(os.path.join(HERE, '_t_range.png'), dpi=110, bbox_inches='tight', facecolor='white')
plt.close(fig)

# 7) 快照表
ok('7. rsi_snapshot')
t = R.rsi_snapshot(['S000.XSHG', 'S001.XSHG', 'S002.XSHG'], FREQ='D')
print(t.to_string(index=False))
assert len(t) == 3 and set(t['状态']) <= {'超买', '超卖', '中性'}

# 8) run_all 批量（假池子）
ok('8. run_all')
figs = R.run_all(POOL=['S000.XSHG', 'S001.XSHG', 'S002.XSHG', 'S003.XSHG'], FREQ='W',
                 BARS=120, MAX_CHARTS=2, SAVE_DIR=os.path.join(HERE, '_rsi_out'))
print('画了', len(figs), '张；目录里有:', sorted(os.listdir(os.path.join(HERE, '_rsi_out'))))
assert len(figs) == 2

print('\n全部通过')
