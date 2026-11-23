# -*- coding: utf-8 -*-
"""viz 双引擎自测：截面 4 张 + 时序 3 张在 plotly 下出图，mpl 老路不破。"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAILED = []


def check(name, cond, extra=''):
    print('  %s %-48s %s' % ('OK  ' if cond else 'FAIL', name, extra))
    if not cond:
        FAILED.append(name)


# ============================================================
print('=' * 74)
print('一、截面可视化（jq_factor_viz）')
print('=' * 74)
import jq_factor_viz as F

check('HAS_PLOTLY', F.HAS_PLOTLY is True)
check('默认引擎是 mpl', F.VIZ_ENGINE == 'mpl', F.VIZ_ENGINE)
check('set_viz_engine 切到 plotly', F.set_viz_engine('plotly') == 'plotly')
check('viz_engine 读得到', F.viz_engine() == 'plotly')
check('乱写的值回落 mpl', F.set_viz_engine('wat') == 'mpl')
check('None 也回落 mpl', F.set_viz_engine(None) == 'mpl')
F.set_viz_engine('plotly')

rs = np.random.RandomState(3)
idx = pd.date_range('2021-01-29', periods=24, freq='ME')
ic = pd.Series(rs.normal(0.02, 0.06, len(idx)), index=idx)
qr = pd.DataFrame(rs.normal(0.004, 0.02, (len(idx), 5)), index=idx,
                  columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
qr = qr.add(pd.Series(np.linspace(-0.012, 0.012, len(idx)), index=idx), axis=0)
RES = {'factor': '测试因子', 'ic': ic, 'quantile_returns': qr,
       'interval': 21.0, 'ratio': 1.0}

f1 = F.plot_ic_series(RES)
check('IC 时序出 plotly 图', isinstance(f1, F._go.Figure))
if isinstance(f1, F._go.Figure):
    names = [str(t.name) for t in f1.data]
    check('IC 时序有逐期柱子 + 累计线', any('逐期' in n for n in names) and
          any('累计' in n for n in names), str(names[:3]))
    check('IC 时序两个子图', getattr(f1.layout, 'yaxis2', None) is not None)

f2 = F.plot_quantile_returns(RES)
check('分层收益出 plotly 图', isinstance(f2, F._go.Figure))
if isinstance(f2, F._go.Figure):
    bar = [t for t in f2.data if t.type == 'bar']
    check('分层收益 5 根柱', len(bar) == 1 and len(bar[0].x) == 5)
    check('柱上写了百分比', all('%' in str(t) for t in bar[0].text), str(list(bar[0].text))[:40])

f3 = F.plot_quantile_equity(RES)
check('分层净值出 plotly 图', isinstance(f3, F._go.Figure))
if isinstance(f3, F._go.Figure):
    check('分层净值 5 条线', len(f3.data) == 5 and all(t.type == 'scatter' for t in f3.data))
    check('图例带年化/夏普', '年化' in str(f3.data[0].name), str(f3.data[0].name))

f4 = F.plot_long_short(RES)
check('多空出 plotly 图', isinstance(f4, F._go.Figure))
if isinstance(f4, F._go.Figure):
    check('多空两个子图', len(f4.data) == 2)
    check('多空有指标注释', any('年化' in (a.text or '') for a in f4.layout.annotations))

F.set_viz_engine('mpl')
m1 = F.plot_ic_series(RES)
check('切回 mpl 还是 matplotlib 图', isinstance(m1, plt.Figure))
plt.close(m1)
m3 = F.plot_quantile_equity(RES)
check('mpl 分层净值还在', isinstance(m3, plt.Figure))
plt.close(m3)

# ============================================================
print()
print('=' * 74)
print('二、时序可视化（jq_ts_viz）')
print('=' * 74)
import jq_ts_viz as T
import selftest_viz_mock as SV

check('TS_HAS_PLOTLY', T.TS_HAS_PLOTLY is True)
check('时序默认 mpl', T.ts_engine() == 'mpl')
check('ts_set_engine 切 plotly', T.ts_set_engine('plotly') == 'plotly')

ns = SV.load_framework()
r = ns['run_ts_test']('ma200', '000300.XSHG', '2019-01-01', '2024-12-31',
                      period=20, th=0.0, decision_freq='W', cost=0.0005,
                      direction=1, verbose=False)
print('  (框架跑完：信号 %d 点)' % len(r['signal']))

g1 = T.plot_price_signal(r, code='000300.XSHG', th=0.0)
check('价格/信号出 plotly 图', isinstance(g1, T._tgo.Figure))
if isinstance(g1, T._tgo.Figure):
    names1 = [str(t.name) for t in g1.data]
    check('价格/信号两条线 + 阈值图例', len(g1.data) == 3 and names1[0] == '收盘价'
          and names1[2].startswith('阈值'), str(names1))
    check('有持仓阴影', any(s.type == 'rect' for s in g1.layout.shapes))

g2 = T.plot_backtest(r, code='000300.XSHG')
check('回测出 plotly 图', isinstance(g2, T._tgo.Figure))
if isinstance(g2, T._tgo.Figure):
    names2 = [str(t.name) for t in g2.data]
    check('回测有策略/基准/仓位', any('策略' in n for n in names2) and
          any('基准' in n for n in names2) and any('仓位' in n for n in names2), str(names2))
    yaxis3 = getattr(g2.layout, 'yaxis3', None)
    check('仓位轴 0~1', yaxis3 is not None and list(yaxis3.range) == [-0.05, 1.05])

d = ns['ts_ic_decay']('mom_120', '000300.XSHG', '2019-01-01', '2024-12-31')
g3 = T.ts_plot_ic_decay(d)
check('IC 衰减出 plotly 图', isinstance(g3, T._tgo.Figure))
if isinstance(g3, T._tgo.Figure):
    check('IC 衰减有柱 + t 线', any(t.type == 'bar' for t in g3.data) and
          any(t.type == 'scatter' for t in g3.data))
    check('x 轴是前瞻期标签', list(g3.data[0].x)[:2] == ['%d' % int(v) for v in d['前瞻期(日)'][:2]])

T.ts_set_engine('mpl')
g4 = T.plot_price_signal(r, code='000300.XSHG', th=0.0)
check('切回 mpl 还是 matplotlib 图', isinstance(g4, plt.Figure))
plt.close(g4)

print()
print('=' * 74)
if FAILED:
    print('FAILED: %s' % FAILED)
    sys.exit(1)
print('全部通过')
sys.exit(0)
