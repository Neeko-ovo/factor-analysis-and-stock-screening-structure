# -*- coding: utf-8 -*-
"""plotly 双轨自测：PLOT_ENGINE 切换 / 图形元素 / 存 html / 回归 mpl 不变。"""
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
import matplotlib.pyplot as plt

import rsi_multiday as R

R._today = lambda: pd.Timestamp('2024-12-31')

FAILED = []


def check(name, cond, extra=''):
    print('  %s %-46s %s' % ('OK  ' if cond else 'FAIL', name, extra))
    if not cond:
        FAILED.append(name)


W = 70


def mk(rows, warm_px=9.1):
    rows = [[warm_px] * 4 + [1.0e6]] * W + [list(r) for r in rows]
    idx = pd.bdate_range('2024-01-01', periods=len(rows))
    a = np.array(rows, dtype=float)
    return pd.DataFrame({'open': a[:, 0], 'high': a[:, 1], 'low': a[:, 2],
                         'close': a[:, 3], 'volume': a[:, 4]}, index=idx)


BULL = [
    [9.1, 9.2, 9.0, 9.1, 1.0e6],
    [9.4, 11.6, 9.1, 11.4, 1.2e6],
    [11.2, 12.0, 11.0, 11.8, 2.0e6],      # 缺口 [9.2, 11.0]
    [12.0, 12.5, 11.5, 12.2, 1.5e6],
    [12.2, 12.4, 11.8, 12.0, 1.1e6],
    [12.0, 12.1, 11.9, 12.0, 1.0e6],
    [10.6, 11.0, 10.2, 10.4, 1.3e6],
    [9.9, 10.6, 9.7, 10.3, 3.0e5],        # 回踩 61.8% 线
]
LINE = 11.0 - 0.618 * 1.8

DF = mk(BULL)
PLAIN = mk([[12.0 + 0.01 * i, 12.1 + 0.01 * i, 11.9 + 0.01 * i, 12.0 + 0.01 * i, 1.0e6]
            for i in range(40)])

print('=' * 70)
print('一、引擎切换')
print('=' * 70)

if not R.HAS_PLOTLY:
    print('  这台机器没装 plotly，跳过 plotly 部分')
    fig_m = R.plot_kline_rsi(DF, 'TEST.BULL', dict(PLOT_ENGINE='plotly', SAVE_DIR=''))
    check('没装 plotly 时自动退回 mpl', isinstance(fig_m, plt.Figure))
    print('FAILED: %s' % FAILED)
    sys.exit(0 if not FAILED else 1)

check('HAS_PLOTLY=True', R.HAS_PLOTLY is True)
check('_norm_engine 识别 plotly', R._norm_engine('plotly') == 'plotly')
check('_norm_engine 识别 PLY', R._norm_engine('PLY') == 'plotly')
check('_norm_engine 识别交互', R._norm_engine('交互') == 'plotly')
check('_norm_engine 认 mpl', R._norm_engine('mpl') == 'mpl')
check('_norm_engine 空值回落 mpl', R._norm_engine(None) == 'mpl')

fig_m = R.plot_kline_rsi(DF, 'TEST.BULL', dict(PLOT_ENGINE='mpl', SAVE_DIR=''))
check('mpl 引擎出 matplotlib 图', isinstance(fig_m, plt.Figure))
check('mpl 图不是 plotly 图', R._is_plotly_fig(fig_m) is False)
plt.close(fig_m)

fig_p = R.plot_kline_rsi(DF, 'TEST.BULL', dict(PLOT_ENGINE='plotly', SAVE_DIR=''))
check('plotly 引擎出 go.Figure', isinstance(fig_p, R.go.Figure))
check('_is_plotly_fig 认得出来', R._is_plotly_fig(fig_p) is True)

print()
print('=' * 70)
print('二、图形内容')
print('=' * 70)

tr = fig_p.data
kinds = [str(t.type) for t in tr]
check('有 K 线', 'candlestick' in kinds)
NALL = W + 8
check('K 线根数 = %d（含 70 根垫底）' % NALL,
      sum(len(t.x) for t in tr if t.type == 'candlestick') == NALL)
check('有 RSI 线', any(t.type == 'scatter' and str(t.name).startswith('RSI') for t in tr))
check('有成交量柱', 'bar' in kinds)
check('有均线', any(str(t.name).startswith('MA') for t in tr))

k = [t for t in tr if t.type == 'candlestick'][0]
check('涨跌配色是红的涨', k.increasing.line.color == R.C_UP, k.increasing.line.color)
check('跌是绿的', k.decreasing.line.color == R.C_DN, k.decreasing.line.color)

rsi_tr = [t for t in tr if t.type == 'scatter' and str(t.name).startswith('RSI')][0]
vals = [v for v in rsi_tr.y if v == v]
check('RSI 值都在 0~100', all(0 <= v <= 100 for v in vals), 'n=%d' % len(vals))

shapes = fig_p.layout.shapes
rects = [s for s in shapes if s.type == 'rect']
lines = [s for s in shapes if s.type == 'line']
check('FVG 方框 >= 1', len(rects) >= 1, 'rect=%d' % len(rects))
check('回踩线存在（#b8860b）',
      any(s.line.color == '#b8860b' for s in lines), 'lines=%d' % len(lines))

ann = fig_p.layout.annotations
txts = [a.text for a in ann]
check('标签写了缺口大小', any('FVG' in t and '(' in t for t in txts))
check('标签写了回踩线价格', any('61.8%' in t for t in txts))
check('标签里有算出来的线价 %s' % R._pxf(LINE),
      any(R._pxf(LINE) in t for t in txts), str(txts[:1]))

ylines = [s for s in lines if s.y0 == s.y1 and s.y0 in (70.0, 30.0)]
check('RSI 70/30 虚线段各一', len([s for s in ylines if s.y0 == 70.0]) >= 1
      and len([s for s in ylines if s.y0 == 30.0]) >= 1, 'n=%d' % len(ylines))

title = fig_p.layout.title.text or ''
check('标题有代码', 'TEST.BULL' in title)
check('标题有 K 线根数', 'K线 %d 根' % NALL in title, title.split('<')[0])

fig_h = fig_p.layout.height
check('高度用 PLOTLY_HEIGHT', fig_h == 840, str(fig_h))

print()
print('=' * 70)
print('三、和 mpl 口径一致')
print('=' * 70)

st = R.fvg_pick(DF, R._merge_cfg(dict(FVG_DIR='bull')))
check('缺口下沿 9.2', abs(st['下沿'] - 9.2) < 1e-9, str(st['下沿']))
check('缺口上沿 11.0', abs(st['上沿'] - 11.0) < 1e-9, str(st['上沿']))
check('回踩线 = %.6f' % LINE, abs(st['回踩线'] - LINE) < 1e-9, str(st['回踩线']))
check('状态是回踩', st['信号'] is True or bool(st['信号']), str(st['信号']))

fig_nb = R.plot_kline_rsi(DF, 'TEST.BULL', dict(PLOT_ENGINE='plotly', FVG_ON=False, SAVE_DIR=''))
check('FVG_ON=False 就没方框',
      len([s for s in fig_nb.layout.shapes if s.type == 'rect' and s.yref == 'y']) == 0)

fig_pb = R.plot_kline_rsi(DF, 'TEST.BULL', dict(PLOT_ENGINE='plotly', PLOT_BARS=4, SAVE_DIR=''))
n_candle = len([t for t in fig_pb.data if t.type == 'candlestick'][0].x)
check('PLOT_BARS=4 只画 4 根', n_candle == 4, str(n_candle))

fig_be = R.plot_kline_rsi(mk([[-9.1 + 0.0 * i, 0, 0, 0, 0]] * 2), 'X', dict()) if False else None
_ = fig_be

fig_nd = R.plot_kline_rsi(PLAIN, 'TEST.PLAIN', dict(PLOT_ENGINE='plotly', SAVE_DIR=''))
check('没缺口也能出图', isinstance(fig_nd, R.go.Figure))
check('没缺口就没标签', len([a for a in fig_nd.layout.annotations if 'FVG' in (a.text or '')]) == 0)

print()
print('=' * 70)
print('四、存文件')
print('=' * 70)

out = os.path.join(HERE, '_plytest')
if not os.path.isdir(out):
    os.makedirs(out)
for f in os.listdir(out):
    try:
        os.remove(os.path.join(out, f))
    except Exception:
        pass

cfg_html = R._merge_cfg(dict(PLOT_ENGINE='plotly', SAVE_DIR=out, PLOTLY_FORMAT='html'))
fig_s = R.plot_kline_rsi(DF, 'TEST.HTML', cfg_html)
p1 = R._maybe_save(fig_s, 'TEST.HTML', cfg_html)
check('html 存出来了', p1 is not None and os.path.isfile(p1), str(p1))
if p1:
    with io.open(p1, 'r', encoding='utf-8') as fh:
        body = fh.read()
    check('html 里有 plotly', 'plotly' in body.lower())
    check('html 里有页脚文字', '61.8%线' in body)

cfg_png = R._merge_cfg(dict(PLOT_ENGINE='plotly', SAVE_DIR=out, PLOTLY_FORMAT='png'))
p2 = R._maybe_save(fig_s, 'TEST.PNG', cfg_png)
check('png 模式有返回路径（没 kaleido 就退成 html）', p2 is not None, str(p2))
if p2:
    check('返回的文件真的存在', os.path.isfile(p2))
    print('      实际落盘: %s' % os.path.basename(p2))

p3 = R._maybe_save(fig_m, 'TEST.MPL', R._merge_cfg(dict(SAVE_DIR=out)))
check('mpl 还是存 png', p3 is not None and p3.endswith('.png') and os.path.isfile(p3), str(p3))
plt.close(fig_m)

print()
print('=' * 70)
print('五、入口函数')
print('=' * 70)

FRAMES = {'TEST.BULL': DF}


def _fake_load(code, cfg=None):
    return FRAMES.get(code)


R.load_bars = _fake_load
R.resolve_pool = lambda pool, date=None: ['TEST.BULL']

called = []
R._show_fig = lambda fig: called.append(fig)

figs = R.run_all(POOL=['TEST.BULL'], MAX_CHARTS=1, PLOT_ENGINE='plotly', SAVE_DIR='')
check('run_all 出一张 plotly 图', len(figs) == 1 and R._is_plotly_fig(figs[0]))
check('run_all 走了 _show_fig', len(called) == 1)

figs2 = R.run_all(POOL=['TEST.BULL'], MAX_CHARTS=1, PLOT_ENGINE='mpl', SAVE_DIR='')
check('run_all 的 mpl 分支还在', len(figs2) == 1 and isinstance(figs2[0], plt.Figure))
plt.close(figs2[0])

f1 = R.plot_one('TEST.BULL', PLOT_ENGINE='plotly', SAVE_DIR='')
check('plot_one 返回 plotly 图', R._is_plotly_fig(f1))

print()
print('=' * 70)
if FAILED:
    print('FAILED: %s' % FAILED)
    sys.exit(1)
print('全部通过')
sys.exit(0)
