# -*- coding: utf-8 -*-
"""FVG（公允价值缺口）+ 斐波那契回踩 + 综合筛选 本地自测（mock 数据）"""
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

import selftest_pct_mock as M
import rsi_multiday as R

R.get_price = M.get_price
R.get_index_stocks = M.get_index_stocks
R.get_trade_days = M.get_trade_days
R._today = lambda: pd.Timestamp('2024-12-31')

FAILED = []


def check(name, cond, extra=''):
    print('  %s %-44s %s' % ('OK  ' if cond else 'FAIL', name, extra))
    if not cond:
        FAILED.append(name)


W = 70      # 前面垫的平盘根数（让 MA60 / RSI 能算）


def mk(rows, warm_px=9.1):
    """前面垫 W 根平盘，再拼上真正的 K 线。平盘不会产生缺口。"""
    rows = [[warm_px] * 4 + [1.0e6]] * W + [list(r) for r in rows]
    idx = pd.bdate_range('2024-01-01', periods=len(rows))
    a = np.array(rows, dtype=float)
    return pd.DataFrame({'open': a[:, 0], 'high': a[:, 1], 'low': a[:, 2],
                         'close': a[:, 3], 'volume': a[:, 4]}, index=idx)


# 多头缺口：第 3 根（i=2）low=11.0 > high[0]=9.2 → 缺口 [9.2, 11.0]
# 61.8% 回踩线 = 11.0 - 0.618 * 1.8 = 9.8876
BULL = [
    [9.1, 9.2, 9.0, 9.1, 1.0e6],
    [9.4, 11.6, 9.1, 11.4, 1.2e6],
    [11.2, 12.0, 11.0, 11.8, 2.0e6],
    [12.0, 12.5, 11.5, 12.2, 1.5e6],
    [12.2, 12.4, 11.8, 12.0, 1.1e6],
    [12.0, 12.1, 11.9, 12.0, 1.0e6],
    [10.6, 11.0, 10.2, 10.4, 1.3e6],
    [9.9, 10.6, 9.7, 10.3, 3.0e5],   # 回踩到位：low 9.7 <= 线，收盘 10.3 > 线，阳线
]
LINE = 11.0 - 0.618 * 1.8

print('[1] find_fvg 多头缺口')
df = mk(BULL)
z = R.find_fvg(df, direction='bull', min_pct=0.3)
print(z.to_string(index=False))
check('只扫到 1 个多头缺口', len(z) == 1, str(len(z)))
check('方向列都是多头', len(z) and set(z['方向']) == {'多头'}, str(set(z['方向'])))
if len(z):
    r0 = z.iloc[0]
    check('下沿 = high[i-2]', abs(r0['下沿'] - 9.2) < 1e-9, r0['下沿'])
    check('上沿 = low[i]', abs(r0['上沿'] - 11.0) < 1e-9, r0['上沿'])
    check('高度 = 1.8', abs(r0['高度'] - 1.8) < 1e-9, r0['高度'])
    check('形成日 = 第 3 根', r0['形成日'] == df.index[W + 2], str(r0['形成日']))
    check('未回补', bool(r0['已回补']) is False, str(r0['已回补']))
    check('最深% = (11-9.7)/1.8*100', abs(r0['最深%'] - 72.22) < 1e-6, r0['最深%'])
check('min_pct 太大被过滤', len(R.find_fvg(df, direction='bull', min_pct=50)) == 0)
check('lookback 太短被过滤', len(R.find_fvg(df, direction='bull', lookback=2)) == 0)
check('数据不足返回空表', len(R.find_fvg(df.head(2))) == 0
      and list(R.find_fvg(df.head(2)).columns) == R._FVG_COLS)
flat = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0,
                     'volume': 1e6}, index=pd.bdate_range('2024-01-01', periods=30))
check('平盘无缺口', len(R.find_fvg(flat)) == 0)

print('\n[2] fvg_state：回踩线 / 距线 / 信号')
st = R.fvg_state(df, ratio=0.618, ref='gap', direction='bull', min_pct=0.3, within=0)
check('有缺口', st['有缺口'])
check('有效（未回补）', st['有效'])
check('回踩线 = 9.8876', abs(st['回踩线'] - LINE) < 1e-6, st['回踩线'])
check('信号 = 回踩', st['信号'] == '回踩', st['信号'])
check('信号日 = 最后一根', st['信号日'] == df.index[-1], str(st['信号日']))
check('距线% = (10.3-line)/line*100',
      abs(st['距线'] - round((10.3 - LINE) / LINE * 100, 2)) < 1e-9, st['距线'])
check('天数 = 5', st['天数'] == 5, st['天数'])

print('\n[3] 各种确认口径 / 窗口')
P = [list(x) for x in BULL]
P[-1] = [10.4, 10.5, 9.7, 9.8, 3.0e5]        # 收阴、收盘在线下
dfp = mk(P)
check('close 口径：收盘在线下 → 无信号', R.fvg_state(dfp, confirm='close')['信号'] == '')
check('touch 口径：碰到就算 → 有信号', R.fvg_state(dfp, confirm='touch')['信号'] == '回踩')
check('bull 口径：阴线 → 无信号', R.fvg_state(dfp, confirm='bull')['信号'] == '')
check('已触及 = True', R.fvg_state(dfp, confirm='touch')['已触及'] is True)

Q = [list(x) for x in BULL]
Q[-2] = [10.4, 10.5, 9.8, 10.2, 3.0e5]       # 倒数第二根回踩到位
Q[-1] = [10.3, 10.4, 10.2, 10.35, 3.0e5]     # 最后一根没碰线
dfq = mk(Q)
check('within=0 抓不到上一根', R.fvg_state(dfq, within=0)['信号'] == '')
check('within=1 能抓到', R.fvg_state(dfq, within=1)['信号'] == '回踩')

print('\n[4] 已回补')
F = [list(x) for x in BULL]
F.append([9.6, 9.7, 9.0, 9.2, 4.0e5])        # low 9.0 <= 9.2 → 补掉
dff = mk(F)
zf = R.find_fvg(dff, direction='bull')
sf = R.fvg_state(dff, within=0)
check('标记已回补', bool(zf.iloc[0]['已回补']) is True, str(zf.iloc[0]['已回补']))
check('回补日 = 最后一根', zf.iloc[0]['回补日'] == dff.index[-1], str(zf.iloc[0]['回补日']))
check('有效 = False', sf['有效'] is False)
check('已回补后不再出信号', sf['信号'] == '', sf['信号'])

print('\n[5] ref=leg / 空头缺口镜像')
st_leg = R.fvg_state(df, ref='leg')
leg_hi = max(x[1] for x in BULL[0:3])
leg_lo = min(x[2] for x in BULL[0:3])
check('leg 线 = leg_hi-0.618*span',
      abs(st_leg['回踩线'] - (leg_hi - 0.618 * (leg_hi - leg_lo))) < 1e-6, st_leg['回踩线'])
BEAR = [
    [12.0, 12.2, 11.8, 12.0, 1.0e6],
    [11.6, 11.7, 9.0, 10.0, 1.2e6],
    [9.8, 10.0, 9.2, 9.4, 2.0e6],            # high 10.0 < low[0] 11.8 → 缺口 [10.0, 11.8]
    [9.5, 9.6, 9.0, 9.1, 1.5e6],
    [9.8, 10.6, 9.7, 10.5, 1.1e6],
    [10.8, 11.6, 10.7, 11.5, 1.3e6],
    [11.2, 11.75, 11.0, 11.1, 3.0e5],        # 反弹到线：high>=11.1124 且收盘在线下
]
dfb = mk(BEAR, warm_px=11.6)
zb = R.find_fvg(dfb, direction='bear')
sb = R.fvg_state(dfb, direction='bear', within=2)
check('扫到 1 个空头缺口', len(zb) == 1, str(len(zb)))
if len(zb):
    check('空头下沿 = high[i]', abs(zb.iloc[0]['下沿'] - 10.0) < 1e-9, zb.iloc[0]['下沿'])
    check('空头上沿 = low[i-2]', abs(zb.iloc[0]['上沿'] - 11.8) < 1e-9, zb.iloc[0]['上沿'])
bl = 10.0 + 0.618 * 1.8
check('空头回踩线 = lo+0.618*size', abs(sb['回踩线'] - bl) < 1e-6, sb['回踩线'])
check('空头反弹出信号', sb['信号'] == '回踩', sb['信号'])
check('空头缺口未回补', sb['有效'] is True)

print('\n[6] 指标插件注册')
check('内置指标齐（fvg/rsi/trend/vol/atr）',
      all(n in R.SCREEN_ORDER for n in ('fvg', 'rsi', 'trend', 'vol', 'atr')),
      str(R.SCREEN_ORDER))
for nm in R.SCREEN_ORDER:
    check('  %s 有描述' % nm, bool(R.SCREEN_INDICATORS[nm]['desc']))

cfg = R._merge_cfg({})
ri = R._ind_fvg(df, cfg)
check('fvg 指标命中 2 分', ri['hit'] and abs(ri['score'] - 2.0) < 1e-9, str(ri['score']))
check('fvg 指标带回踩线', ri['info'].get('回踩线') is not None, str(ri['info']))
rv = R._ind_vol(df, cfg)
check('缩量指标命中', rv['hit'], str(rv['info']))
rt = R._ind_trend(df, cfg)
check('趋势指标能出结果', '趋势' in rt['info'], str(rt['info']))
rr = R._ind_rsi(df, cfg)
check('RSI 指标能出结果', 'RSI状态' in rr['info'], str(rr['info']))

print('\n[7] screen_fvg：各模式')
FRAMES = {
    'A.XSHG': df,             # 回踩到位
    'B.XSHG': dfq,            # 等回踩（上一根碰到过）
    'C.XSHG': dff,            # 已回补
    'D.XSHG': mk(BULL[:3]),   # 缺口刚形成，还没回踩
}
_orig_load = R.load_bars
R.load_bars = lambda code, cfg=None: FRAMES.get(code)
try:
    allv = R.screen_fvg(pool=list(FRAMES), mode='all', verbose=False)
    print(allv.to_string(index=False))
    check('all 列出 4 只', len(allv) == 4, str(len(allv)))
    check('列名齐全', list(allv.columns) == R._FVG_SCREEN_COLS, str(list(allv.columns)))
    check('排序：回踩到位在最前', len(allv) and allv.iloc[0]['状态'] == '回踩到位',
          str(allv.iloc[0]['状态']) if len(allv) else '')
    hitv = R.screen_fvg(pool=list(FRAMES), mode='hit', verbose=False)
    check('hit 排除已回补', len(hitv) == 3 and 'C.XSHG' not in list(hitv['代码']),
          str(list(hitv['代码'])))
    touch = R.screen_fvg(pool=list(FRAMES), mode='touch', verbose=False)
    check('touch 只有 A', list(touch['代码']) == ['A.XSHG'], str(list(touch['代码'])))
    fresh = R.screen_fvg(pool=list(FRAMES), mode='fresh', verbose=False)
    check('fresh 是 B 和 D', sorted(fresh['代码']) == ['B.XSHG', 'D.XSHG'],
          str(list(fresh['代码'])))
    filled = R.screen_fvg(pool=list(FRAMES), mode='filled', verbose=False)
    check('filled 只有 C', list(filled['代码']) == ['C.XSHG'], str(list(filled['代码'])))
    try:
        R.screen_fvg(pool=list(FRAMES), mode='xxx', verbose=False)
        check('未知 mode 报错', False)
    except ValueError:
        check('未知 mode 报错', True)
finally:
    R.load_bars = _orig_load

print('\n[8] screen_stocks：档位')
R.load_bars = lambda code, cfg=None: FRAMES.get(code)
try:
    t1 = R.screen_stocks(pool=list(FRAMES), tier=1, verbose=False)
    print(t1.to_string(index=False))
    check('1 档 = 最近 2 根内有回踩信号的', sorted(t1['代码']) == ['A.XSHG', 'B.XSHG'],
          str(list(t1['代码'])))
    check('1 档 分数 2', len(t1) and abs(t1.iloc[0]['分数'] - 2.0) < 1e-9,
          str(t1.iloc[0]['分数']) if len(t1) else '')
    check('列名齐全', list(t1.columns) == R._STOCK_COLS, str(list(t1.columns)))
    t0 = R.screen_stocks(pool=list(FRAMES), tier=0, verbose=False)
    check('0 档不过滤 → 4 只全在', len(t0) == 4, str(len(t0)))
    t2b = R.screen_stocks(pool=list(FRAMES), tier=2, verbose=False)
    check('2 档 <= 1 档', len(t2b) <= len(t1), '%d <= %d' % (len(t2b), len(t1)))
    t2s = R.screen_stocks(pool=list(FRAMES), tier='2', verbose=False)
    check('字符串档位能跑', isinstance(t2s, pd.DataFrame), str(len(t2s)))
    cu = R.screen_stocks(pool=list(FRAMES), use=['fvg'], need=[], min_score=0, verbose=False)
    check('自定义指标组合', len(cu) == 4, str(len(cu)))
    m3 = R.screen_stocks(pool=list(FRAMES), tier=0, use=['fvg'], need=[], min_score=1,
                         verbose=False)
    check('min_score=1 → A/B/D（D 待回踩 1 分）',
          sorted(m3['代码']) == ['A.XSHG', 'B.XSHG', 'D.XSHG'], str(list(m3['代码'])))
    try:
        R.screen_stocks(pool=list(FRAMES), tier=1, use=['macd'], verbose=False)
        check('未知指标报错', False)
    except ValueError:
        check('未知指标报错', True)
    try:
        R.screen_stocks(pool=list(FRAMES), tier=9, verbose=False)
        check('未知档位报错', False)
    except ValueError:
        check('未知档位报错', True)
finally:
    R.load_bars = _orig_load

print('\n[9] 画图：缺口框 + 回踩线 + 图例')
R._setup_style(False)
fig = R.plot_kline_rsi(df, 'A.XSHG', {'FVG_ON': True, 'FVG_SHOW': 3, 'SHOW_VOLUME': True,
                                      'MA_LIST': [20, 60], 'MARK_EVENTS': True})
check('画完返回 Figure', fig is not None)
leg = fig.axes[0].get_legend()
legs = [t.get_text() for t in leg.get_texts()] if leg else []
check('图例含 FVG', any('FVG' in s for s in legs), str(legs))
check('图例含均线', any(s.startswith('MA') for s in legs), str(legs))
_ax = plt.figure().add_subplot(111)
_ax.set_xlim(-1, len(df))
n_draw = R._draw_fvg(_ax, df, z, R._merge_cfg({'FVG_SHOW': 3}))
check('_draw_fvg 画了 1 个缺口', n_draw == 1, str(n_draw))
check('FVG_ON=False 也能画', R.plot_kline_rsi(df, 'A.XSHG', {'FVG_ON': False}) is not None)
check('无缺口也能画', R.plot_kline_rsi(flat, 'FLAT.XSHG', {'FVG_ON': True}) is not None)
plt.close('all')

print('\n[10] 真实 mock 数据（high==low==close，退化缺口，不该炸）')
h = R.screen_fvg(pool=list(M.CODES[:5]), mode='all', bars=250, verbose=False)
check('能跑出表', list(h.columns) == R._FVG_SCREEN_COLS, str(len(h)))
s = R.screen_stocks(pool=list(M.CODES[:5]), tier=1, bars=250, verbose=False)
check('综合筛选能跑', list(s.columns) == R._STOCK_COLS, str(len(s)))
e = R.screen_fvg(pool=[], verbose=False)
check('空池返回空表', len(e) == 0 and list(e.columns) == R._FVG_SCREEN_COLS)

print()
if FAILED:
    print('FAILED: %s' % FAILED)
    sys.exit(1)
print('全部通过')
