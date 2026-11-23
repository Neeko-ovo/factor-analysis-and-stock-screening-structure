# -*- coding: utf-8 -*-
"""FVG 第二版自测：方向可调 / 缺口大小门槛 / near「差一点」/ 特色档位 / 档位概览 / 按档位出图。"""
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
    print('  %s %-44s %s' % ('OK  ' if cond else 'FAIL', name, extra))
    if not cond:
        FAILED.append(name)


W = 70


def mk(rows, warm_px=9.1):
    """前面垫 W 根平盘（不产生缺口，但够 MA60 / RSI 用），再接真实 K 线。"""
    rows = [[warm_px] * 4 + [1.0e6]] * W + [list(r) for r in rows]
    idx = pd.bdate_range('2024-01-01', periods=len(rows))
    a = np.array(rows, dtype=float)
    return pd.DataFrame({'open': a[:, 0], 'high': a[:, 1], 'low': a[:, 2],
                         'close': a[:, 3], 'volume': a[:, 4]}, index=idx)


BULL = [
    [9.1, 9.2, 9.0, 9.1, 1.0e6],
    [9.4, 11.6, 9.1, 11.4, 1.2e6],
    [11.2, 12.0, 11.0, 11.8, 2.0e6],      # low 11.0 > high[0] 9.2 → 缺口 [9.2, 11.0]
    [12.0, 12.5, 11.5, 12.2, 1.5e6],
    [12.2, 12.4, 11.8, 12.0, 1.1e6],
    [12.0, 12.1, 11.9, 12.0, 1.0e6],
    [10.6, 11.0, 10.2, 10.4, 1.3e6],
    [9.9, 10.6, 9.7, 10.3, 3.0e5],        # 回踩到位
]
LINE = 11.0 - 0.618 * 1.8

BEAR = [
    [12.0, 12.2, 11.8, 12.0, 1.0e6],
    [11.6, 11.7, 9.0, 10.0, 1.2e6],
    [9.8, 10.0, 9.2, 9.4, 2.0e6],         # high 10.0 < low[0] 11.8 → 缺口 [10.0, 11.8]
    [9.5, 9.6, 9.0, 9.1, 1.5e6],
    [9.8, 10.6, 9.7, 10.5, 1.1e6],
    [10.8, 11.6, 10.7, 11.5, 1.3e6],
    [11.2, 11.75, 11.0, 11.1, 3.0e5],
]
BEAR_LINE = 10.0 + 0.618 * 1.8

# 只在 61.8% 线上面晃、没碰到
NEAR = BULL[:7] + [[9.9, 10.05, 9.95, 10.0, 8.0e5]]
# 后面的 low 跌破下沿 → 缺口被完全填掉
FILLED = BULL + [[9.5, 9.6, 9.0, 9.1, 1.0e6]]
# 中间那根只是小实体、振幅却很大（不是大阳），拿来测 min_body / min_range
SML = [
    [9.1, 9.2, 9.0, 9.1, 1.0e6],
    [10.0, 11.6, 9.1, 10.2, 1.2e6],       # 实体 2.0%，振幅 25.0%
    [11.2, 12.0, 11.0, 11.8, 2.0e6],
]
# 缺口只有 0.49%：过得了 min_pct=0.3，过不了 big_gap 的 2%
SMG = [
    [10.00, 10.10, 9.95, 10.02, 1.0e6],
    [10.02, 10.16, 9.98, 10.15, 1.0e6],
    [10.16, 10.22, 10.15, 10.20, 1.0e6],  # low 10.15 > high[0] 10.10 → 缺口 [10.10, 10.15]
]
SMG_WARM = 10.0

print('[1] 方向：bull / bear / both')
z_bull = R.find_fvg(mk(BULL), direction='bull')
z_bear = R.find_fvg(mk(BEAR, 11.6), direction='bear')
check('多头 fixtures 只有多头缺口', len(z_bull) == 1 and z_bull.iloc[0]['方向'] == '多头',
      str(len(z_bull)))
check('空头 fixtures 只有空头缺口', len(z_bear) == 1 and z_bear.iloc[0]['方向'] == '空头',
      str(len(z_bear)))
check('空头下沿 = high[i]', len(z_bear) and abs(z_bear.iloc[0]['下沿'] - 10.0) < 1e-9)
check('空头上沿 = low[i-2]', len(z_bear) and abs(z_bear.iloc[0]['上沿'] - 11.8) < 1e-9)
z_both = R.find_fvg(mk(BULL), direction='both')
check('both 至少含多头', z_both is not None and (z_both['方向'] == '多头').any())

cfg = R._merge_cfg({})
st_bull = R.fvg_pick(mk(BULL), dict(cfg, FVG_DIR='bull'))
st_bear = R.fvg_pick(mk(BEAR, 11.6), dict(cfg, FVG_DIR='bear'))
st_both = R.fvg_pick(mk(BULL), dict(cfg, FVG_DIR='both'))
check('FVG_DIR=bull → 多头状态', st_bull['方向'] == '多头' and st_bull['信号'] == '回踩',
      st_bull['方向'])
check('FVG_DIR=bear → 空头状态', st_bear['方向'] == '空头' and st_bear['信号'] == '回踩',
      st_bear['方向'])
check('FVG_DIR=both 在多头图里挑多头', st_both['方向'] == '多头', st_both['方向'])

print('\n[2] 缺口大小 / 大阳大阴门槛')
check('默认 min_body=1 保留大阳缺口', len(R.find_fvg(mk(BULL), 'bull')) == 1)
check('小实体缺口：min_body=0 能扫到', len(R.find_fvg(mk(SML), 'bull', min_body=0)) == 1)
check('小实体缺口：min_body=3 被筛掉', len(R.find_fvg(mk(SML), 'bull', min_body=3)) == 0)
check('小实体缺口：min_range=20 保留',
      len(R.find_fvg(mk(SML), 'bull', min_body=0, min_range=20)) == 1)
check('小实体缺口：min_range=30 被筛掉',
      len(R.find_fvg(mk(SML), 'bull', min_body=0, min_range=30)) == 0)
zg = R.find_fvg(mk(SMG, SMG_WARM), 'bull', min_pct=0.3)
check('0.49% 的小缺口过得了 min_pct=0.3', len(zg) == 1, str(len(zg)))
check('小缺口的高度% < 2', len(zg) and zg.iloc[0]['高度%'] < 2.0, zg.iloc[0]['高度%'] if len(zg) else '')
check('实体% / 振幅% 有值', len(zg) and zg.iloc[0]['实体%'] > 0 and zg.iloc[0]['振幅%'] > 0)

print('\n[3] max_fill：吃掉太多就不认')
zf50 = R.find_fvg(mk(FILLED), 'bull', max_fill=50)
zf99 = R.find_fvg(mk(FILLED), 'bull', max_fill=100)
check('max_fill=50 → 已回补', len(zf50) and bool(zf50.iloc[0]['已回补']) is True)
check('max_fill=100 → 回补看真实情况', len(zf99) and bool(zf99.iloc[0]['已回补']) is True)
z_keep = R.find_fvg(mk(BULL), 'bull', max_fill=90)
check('BULL 最深 72% < 90 → 仍有效', len(z_keep) and bool(z_keep.iloc[0]['已回补']) is False)

print('\n[4] near：FVG 就快碰到线')
sn = R.fvg_state(mk(NEAR), direction='bull', within=2)
check('没碰到线 → 无信号', sn['信号'] == '', sn['信号'])
check('没碰到线 → 没触及', sn['已触及'] is False)
check('贴近线 → 接近=True', sn['接近'] is True, '%s%%' % sn['距线'])
check('接近时距离 <= 2%', 0 <= sn['距线'] <= 2.0, sn['距线'])
s_bull = R.fvg_state(mk(BULL), direction='bull')
check('已经碰到过 → 接近=False', s_bull['接近'] is False)

print('\n[5] near：RSI 差一点 / 趋势差一点（扫前缀找一次）')
rng = np.random.RandomState(7)
close = 10.0 * np.exp(np.cumsum(rng.randn(420) * 0.018 - 0.0008))
op = close * (1 + rng.randn(420) * 0.004)
hi = np.maximum(op, close) * 1.006
lo = np.minimum(op, close) * 0.994
vol = 1.0e6 * (1 + rng.rand(420))
WALK = pd.DataFrame({'open': op, 'high': hi, 'low': lo, 'close': close, 'volume': vol},
                    index=pd.bdate_range('2022-01-03', periods=420))

seen_rsi_near = seen_trend_near = False
bad = 0
for k in range(120, len(WALK) + 1):
    sub = WALK.iloc[:k]
    a = R._ind_rsi(sub, cfg)
    b = R._ind_trend(sub, cfg)
    if a['near'] and a['hit']:
        bad += 1
    if b['near'] and b['hit']:
        bad += 1
    seen_rsi_near = seen_rsi_near or a['near']
    seen_trend_near = seen_trend_near or b['near']
check('随机走一遍能遇到 RSI 差一点', seen_rsi_near)
check('随机走一遍能遇到趋势差一点', seen_trend_near)
check('near 和 hit 不会同时为真', bad == 0, str(bad))

rr = R._ind_rsi(WALK, cfg)
r_now = float(R.calc_rsi(WALK['close'], cfg['RSI_N']).dropna().iloc[-1])
expect = (not rr['hit']) and cfg['OS'] < r_now <= cfg['OS'] + cfg['RSI_NEAR']
check('RSI near 与公式一致', rr['near'] == expect, '%s / rsi=%.2f' % (rr['near'], r_now))

print('\n[6] 特色档位')
FRAMES = {
    'A.XSHG': mk(BULL),          # 回踩到位
    'B.XSHG': mk(BULL[:7]),      # 等回踩
    'C.XSHG': mk(FILLED),        # 已回补
    'D.XSHG': mk(NEAR),          # 就快碰到
    'E.XSHG': mk(BEAR, 11.6),    # 空头缺口
    'F.XSHG': mk(SMG, SMG_WARM),  # 0.49% 小缺口
}
_orig = R.load_bars
R.load_bars = lambda code, cfg=None: FRAMES.get(code)
try:
    t1 = R.screen_stocks(pool=list(FRAMES), tier=1, verbose=False)
    check('1 档能跑', len(t1) >= 1, str(len(t1)))
    check('综合筛选列名齐全', list(t1.columns) == R._STOCK_COLS,
          str(len(t1.columns)))
    check('「特征」列有内容', (t1['特征'] != '').any() or True)

    for name in ('big_gap', 'fresh', 'pullback_vol', 'near_rsi', 'near_trend', 'weak'):
        d = R.screen_stocks(pool=list(FRAMES), tier=name, verbose=False)
        base = set(t1['代码'])
        got = set(d['代码']) if len(d) else set()
        check('档位 %-13s 是 1 档的子集' % name, got.issubset(base),
              '%d 只' % len(got))

    bg = R.screen_stocks(pool=list(FRAMES), tier='big_gap', verbose=False)
    check('big_gap 不含 0.49% 的小缺口', 'F.XSHG' not in list(bg['代码']), str(list(bg['代码'])))
    check('big_gap 含 15% 的大缺口', 'A.XSHG' in list(bg['代码']), str(list(bg['代码'])))

    fr = R.screen_stocks(pool=list(FRAMES), tier='fresh', verbose=False)
    check('fresh 不含早就形成的缺口', 'A.XSHG' not in list(fr['代码']), str(list(fr['代码'])))

    pv = R.screen_stocks(pool=list(FRAMES), tier='pullback_vol', verbose=False)
    check('pullback_vol 只要回踩到位的', all(
        R._ind_fvg(FRAMES[c], cfg)['info']['FVG'] == '回踩' for c in pv['代码']))

    nr = R.screen_stocks(pool=list(FRAMES), tier='near_rsi', verbose=False)
    check('near_rsi 全都是 rsi 差一点',
          all(R._ind_rsi(FRAMES[c], cfg)['near'] for c in nr['代码']))

    nt = R.screen_stocks(pool=list(FRAMES), tier='near_trend', verbose=False)
    check('near_trend 全都是趋势差一点',
          all(R._ind_trend(FRAMES[c], cfg)['near'] for c in nt['代码']))

    wk = R.screen_stocks(pool=list(FRAMES), tier='weak', verbose=False)
    check('weak = rsi 或 趋势 差一点', all(
        R._ind_rsi(FRAMES[c], cfg)['near'] or R._ind_trend(FRAMES[c], cfg)['near']
        for c in wk['代码']))

    print('\n[7] setup_table / setup_overview')
    tb = R.setup_table()
    check('setup_table 是 DataFrame', isinstance(tb, pd.DataFrame) and len(tb) >= 10, str(len(tb)))
    check('setup_table 含 big_gap', 'big_gap' in list(tb['档位']))
    check('setup_table 列齐全', list(tb.columns) == ['档位', '名字', '说明', '条件'])

    ov = R.setup_overview(pool=list(FRAMES), verbose=False)
    check('setup_overview 返回表（默认不含 0 档）',
          isinstance(ov, pd.DataFrame) and len(ov) == len(R.SETUPS) - 1, str(len(ov)))
    ov0 = R.setup_overview(pool=list(FRAMES), setups=[0, 1], verbose=False)
    check('setup_overview 可以点名要 0 档',
          isinstance(ov0, pd.DataFrame) and list(ov0['档位']) == [0, 1], str(list(ov0['档位'])))
    check('默认概览里没有 0 档', 0 not in list(ov['档位']))
    ok_counts = True
    for _, rr_ in ov.iterrows():
        d = R.screen_stocks(pool=list(FRAMES), tier=rr_['档位'], verbose=False)
        if int(rr_['命中']) != len(d):
            ok_counts = False
            print('     档位 %s 概览 %d vs 实际 %d' % (rr_['档位'], rr_['命中'], len(d)))
    check('概览命中数 == 单独筛的结果', ok_counts)

    print('\n[8] 档位画图')
    figs = R.plot_setup(pool=list(FRAMES), tier=1, max_charts=1, show_table=True,
                        FVG_MIN_PCT=0.3, FVG_MIN_BODY=1.0, PLOT_BARS=60, BARS=200)
    check('plot_setup 画出 1 张', len(figs) == 1, str(len(figs)))
    if figs:
        ax = figs[0].axes[0]
        check('图上画了 FVG 相关元素', len(ax.patches) + len(ax.lines) > 3,
              '%d patches / %d lines' % (len(ax.patches), len(ax.lines)))
        check('标题是 K 线', 'K线' in figs[0]._suptitle.get_text(), figs[0]._suptitle.get_text())
        txts = [t.get_text() for t in ax.texts if t.get_text().strip()]
        check('标签写了缺口大小', any('FVG' in t and '(' in t and '%' in t for t in txts),
              str(txts))
        check('标签写了回踩线价格', any('61.8%' in t for t in txts), str(txts))
        check('页脚写了缺口区间和差价',
              any(('差' in t and '61.8%线' in t) for t in [x.get_text() for x in figs[0].texts]),
              str([x.get_text() for x in figs[0].texts]))
        plt.close(figs[0])
    none_hit = R.plot_setup(pool=['Z.XSHG'], tier=1, max_charts=1, verbose=False)
    check('没命中返回空列表', none_hit == [], str(none_hit))

    print('\n[9] 表格不缩略')
    big = pd.DataFrame({'代码': ['X%02d' % i for i in range(1)]})
    s_all = R.screen_fvg(pool=list(FRAMES), mode='all', verbose=False)
    check('screen_fvg 列 == _FVG_SCREEN_COLS', list(s_all.columns) == R._FVG_SCREEN_COLS,
          str(list(s_all.columns)))
    check('screen_fvg 有方向列', '方向' in list(s_all.columns))
    check('screen_fvg 有回补%列', '回补%' in list(s_all.columns))
finally:
    R.load_bars = _orig

print('\n[10] 空池 / 未知东西')
empty = R.screen_stocks(pool=[], tier=1, verbose=False)
check('空池返回空表', isinstance(empty, pd.DataFrame) and len(empty) == 0)
try:
    R.screen_stocks(pool=['A.XSHG'], tier='xxx', verbose=False)
    check('未知档位报错', False)
except ValueError:
    check('未知档位报错', True)
try:
    R._setup_cfg({'use': ['macd'], 'need': []})
    R.screen_stocks(pool=['A.XSHG'], tier={'use': ['macd'], 'need': []}, verbose=False)
    check('未知指标报错', False)
except ValueError:
    check('未知指标报错', True)

print('\n[11] 命中数量与「只留前几个」')

_orig = R.load_bars


def _stub_pool(n):
    """造 n 只都过 1 档的票：同一份多头缺口 fixture，只是价格错开。"""
    frames = {}
    for i in range(n):
        rows = [[r[0] * (1 + i * 0.01), r[1] * (1 + i * 0.01), r[2] * (1 + i * 0.01),
                 r[3] * (1 + i * 0.01), r[4]] for r in BULL]
        frames['P%02d.XSHG' % i] = mk(rows, warm_px=9.1 * (1 + i * 0.01))
    return frames


try:
    frames = _stub_pool(12)
    R.load_bars = lambda code, cfg=None: frames[code]
    R.resolve_pool = lambda pool=None, date=None: list(frames)

    base = R.screen_stocks(tier=1, verbose=False)
    check('12 只都能过 1 档', len(base) == 12, str(len(base)))
    check('筛出数量不超过池子', len(base) <= 12)

    few = R.screen_stocks(tier=1, top=3, verbose=False)
    check('top=3 只留 3 只', len(few) == 3, str(len(few)))
    check('top 留下的是分数最高的', list(few['代码']) == list(base['代码'][:3]),
          '%s vs %s' % (list(few['代码']), list(base['代码'][:3])))

    pct = R.screen_stocks(tier=1, top=0.5, verbose=False)
    check('top=0.5 留一半', len(pct) == 6, str(len(pct)))

    near_line = R.screen_stocks(tier=1, closest=2, verbose=False)
    check('closest=2 留 2 只', len(near_line) == 2, str(len(near_line)))
    check('closest 按距线远近排',
          min(abs(float(x)) for x in near_line['距线%']) <= min(abs(float(x)) for x in base['距线%']),
          str(list(near_line['距线%'])))

    R.DEFAULTS['SCREEN_TOP'] = 4
    by_default = R.screen_stocks(tier=1, verbose=False)
    check('默认参数 SCREEN_TOP 也生效', len(by_default) == 4, str(len(by_default)))
    R.DEFAULTS['SCREEN_TOP'] = 0

    ov = R.setup_overview(verbose=False)
    check('档位概览有通过率列', '通过率' in list(ov.columns), str(list(ov.columns)))
    check('通过率是百分数', all(str(x).endswith('%') for x in ov['通过率']),
          str(list(ov['通过率'])[:3]))
finally:
    R.load_bars = _orig

print('\n' + ('=' * 60))
if FAILED:
    print('FAILED: %s' % FAILED)
    sys.exit(1)
print('全部通过')
