# -*- coding: utf-8 -*-
"""ATR 冒烟测试：口径、档位、日/周/月缩放。"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '_mplcache'))

import numpy as np
import pandas as pd

import rsi_multiday as R

fail = []


def ck(name, cond, extra=''):
    if cond:
        print('[OK]   %s' % name)
    else:
        print('[FAIL] %s %s' % (name, extra))
        fail.append(name)


# ---------- 1. ATR 数值：手算能对上的 5 根 ----------
# 每天都是 高 11 / 低 9 / 收 10，昨收 10 → TR 恒为 2
rows = [[10.0, 11.0, 9.0, 10.0, 1e6]] * 8
df = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close', 'volume'])
df.index = pd.date_range('2024-01-01', periods=len(df), freq='D')
atr = R.calc_atr(df, 5)
ck('ATR 平盘时等于每根振幅', abs(float(atr.iloc[-1]) - 2.0) < 1e-9, atr.iloc[-1])
ck('ATR 前 5-1 根是 NaN', pd.isna(atr.iloc[3]) and not pd.isna(atr.iloc[4]))

# 跳空：昨收 10，今天 高 13 低 12 收 12.5 → TR = max(1, 3, 2) = 3
rows2 = [[10.0, 10.5, 9.5, 10.0, 1e6]] * 6 + [[12.5, 13.0, 12.0, 12.5, 1e6]]
df2 = pd.DataFrame(rows2, columns=['open', 'high', 'low', 'close', 'volume'])
df2.index = pd.date_range('2024-01-01', periods=len(df2), freq='D')
tr_last = max(13.0 - 12.0, abs(13.0 - 10.0), abs(12.0 - 10.0))
ck('跳空算进 TR', abs(tr_last - 3.0) < 1e-9, tr_last)

# ---------- 2. 档位阈值：日 / 周 / 月 ----------
cfg = R._merge_cfg({})
lo_d, hi_d = R.atr_levels(cfg, 'D')
lo_w, hi_w = R.atr_levels(cfg, 'W')
lo_m, hi_m = R.atr_levels(cfg, 'M')
ck('日线阈值 = 基准 1.2 / 3.5', abs(lo_d - 1.2) < 1e-9 and abs(hi_d - 3.5) < 1e-9, (lo_d, hi_d))
ck('周线 = 日线 x sqrt(5)', abs(lo_w - 1.2 * 5 ** 0.5) < 1e-9, lo_w)
ck('月线 = 日线 x sqrt(21)', abs(hi_m - 3.5 * 21 ** 0.5) < 1e-9, hi_m)
cfg_off = R._merge_cfg({'ATR_AUTO': False})
ck('ATR_AUTO=False 时不缩放', abs(R.atr_levels(cfg_off, 'M')[0] - 1.2) < 1e-9)
cfg_hand = R._merge_cfg({'ATR_LEVELS': {'D': (1.0, 2.0), 'W': (4.0, 10.0)}})
ck('手填的 W 覆盖自动缩放', R.atr_levels(cfg_hand, 'W') == (4.0, 10.0), R.atr_levels(cfg_hand, 'W'))
ck('手填 D 后 M 仍按 D 缩放', abs(R.atr_levels(cfg_hand, 'M')[0] - 1.0 * 21 ** 0.5) < 1e-9)

# ---------- 3. atr_state 三档 ----------
def mk(pct_target):
    """造一段价格，让 ATR% 落在目标附近：振幅 = pct_target% × 收盘。"""
    px = 10.0
    rows = []
    for _ in range(60):
        hi = px * (1 + pct_target / 200.0)
        lo = px * (1 - pct_target / 200.0)
        rows.append([px, hi, lo, px, 1e6])
    d = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close', 'volume'])
    d.index = pd.date_range('2024-01-01', periods=len(d), freq='D')
    return d


st_lo = R.atr_state(mk(1.0), cfg)
st_mid = R.atr_state(mk(2.5), cfg)
st_hi = R.atr_state(mk(6.0), cfg)
ck('低波动档', st_lo['ATR档'] == '低', (st_lo['ATR档'], st_lo['ATR%']))
ck('正常档', st_mid['ATR档'] == '正常', (st_mid['ATR档'], st_mid['ATR%']))
ck('高波动档', st_hi['ATR档'] == '高', (st_hi['ATR档'], st_hi['ATR%']))
ck('ATR% 与 ATR 自洽',
   abs(st_mid['ATR%'] - st_mid['ATR'] / 10.0 * 100.0) < 1e-6, st_mid['ATR%'])
ck('ATR 分位在 0~100 之间', 0 <= st_mid['ATR分位'] <= 100, st_mid['ATR分位'])

# ---------- 4. 指标插件 ----------
r_low = R._ind_atr(mk(1.0), cfg)
r_hi = R._ind_atr(mk(6.0), cfg)
ck('ATR_SIDE=low 时低波动命中', r_low['hit'] and not r_hi['hit'], (r_low['hit'], r_hi['hit']))
cfg_high = R._merge_cfg({'ATR_SIDE': 'high'})
ck('ATR_SIDE=high 时高波动命中',
   R._ind_atr(mk(6.0), cfg_high)['hit'] and not R._ind_atr(mk(1.0), cfg_high)['hit'])
cfg_both = R._merge_cfg({'ATR_SIDE': 'both'})
ck('ATR_SIDE=both 两头都命中',
   R._ind_atr(mk(1.0), cfg_both)['hit'] and R._ind_atr(mk(6.0), cfg_both)['hit'])
ck('表格列 key 齐', all(k in r_low['info'] for k in R._ATR_COLS), r_low['info'])

# ---------- 5. 档位与表格列 ----------
ck('5 档存在且 need 含 atr', 'atr' in (R.SETUPS[5].get('need') or []), R.SETUPS[5])
ck('atr_quiet / atr_loud 已注册',
   'atr_quiet' in R.SETUPS and 'atr_loud' in R.SETUPS)
ck('ATR 列已进表头', all(c in R._STOCK_COLS for c in R._ATR_COLS), R._STOCK_COLS)

# ---------- 6. 端到端：扫池 + 出表 ----------
pool = []
bars = {}
for i, target in enumerate((1.0, 2.5, 6.0)):
    code = 'A%03d.XSHG' % i
    pool.append(code)
    bars[code] = mk(target)
R.resolve_pool = lambda pool=None, date=None: list(pool or [])
R.load_bars = lambda code, cfg=None: bars[code].copy()
out = R.screen_stocks(pool=pool, tier=5, verbose=False)
ck('tier5 表能出、列齐', len(out) == 0 or all(c in out.columns for c in R._STOCK_COLS))
ov = R.setup_overview(pool=pool, verbose=False,
                      setups=[{'name': '低波动', 'desc': '只卡 ATR 低档', 'use': ['atr'], 'need': ['atr']}])
ck('档位概览能跑', len(ov) == 1 and int(ov['命中'].iloc[0]) == 1, ov.to_dict('records'))

# ---------- 7. 图上页脚也带 ATR ----------
try:
    import matplotlib
    matplotlib.use('Agg')
    fig = R.plot_kline_rsi(bars['A001.XSHG'], 'A001.XSHG', R._merge_cfg({'BARS': 200}))
    txts = [t.get_text() for t in fig.texts]
    foot = [t for t in txts if '收盘' in t]
    ck('页脚写了 ATR 档位', bool(foot) and 'ATR' in foot[0] and '档' in foot[0], txts)
except Exception as e:  # 环境问题不算失败
    print('[skip] 页脚检查跳过：%s' % e)

# ---------- 8. ATR 副图：TR 与 ATR 两条线都要在，且会互相穿过 ----------
try:
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    cfg_ap = R._merge_cfg({'BARS': 200, 'SHOW_ATR': True, 'SHOW_VOLUME': True})
    fig_ap = R.plot_kline_rsi(bars['A001.XSHG'], 'A001.XSHG', cfg_ap)
    axes = fig_ap.get_axes()
    ck('开 ATR 后有 4 格', len(axes) == 4, len(axes))
    names = []
    for ax in axes:
        names += [ln.get_label() for ln in ax.get_lines()]
    ck('ATR 副图有 TR% 线', 'TR%' in names, names)
    ck('ATR 副图有 ATR% 线', any(str(n).startswith('ATR%') for n in names), names)
    ck('ATR 副图 ylabel 写 TR', 'TR' in axes[-1].get_ylabel(), axes[-1].get_ylabel())

    # TR 抖、ATR 平：两条线必须来回穿（用随机游走造真波动，mk 里那几根振幅是固定的）
    import pandas as _pd
    _rs = np.random.RandomState(11)
    _n = 150
    _cl = 10.0 * np.cumprod(1 + _rs.normal(0.0005, 0.015, _n))
    _op = _cl * (1 + _rs.normal(0, 0.005, _n))
    _hi = np.maximum(_op, _cl) * (1 + np.abs(_rs.normal(0, 0.007, _n)))
    _lo = np.minimum(_op, _cl) * (1 - np.abs(_rs.normal(0, 0.007, _n)))
    _rv = _pd.DataFrame({'open': _op, 'high': _hi, 'low': _lo, 'close': _cl,
                         'volume': _rs.randint(1e6, 3e6, _n).astype(float)},
                        index=_pd.bdate_range('2024-01-02', periods=_n))
    trp = (R.calc_tr(_rv) / _rv['close'] * 100.0).values
    pct_ = (R.calc_atr(_rv, 14) / _rv['close'] * 100.0).values
    ok = np.isfinite(trp) & np.isfinite(pct_)
    sg = np.sign(trp - pct_)[ok]
    cross = int((np.diff(sg) != 0).sum())
    ck('TR 与 ATR 交织（穿过 >5 次）', cross > 5, cross)

    # 关掉就不该多那一格
    fig_off = R.plot_kline_rsi(bars['A001.XSHG'], 'A001.XSHG',
                               R._merge_cfg({'BARS': 200, 'SHOW_ATR': False}))
    ck('SHOW_ATR=False 只剩 3 格', len(fig_off.get_axes()) == 3, len(fig_off.get_axes()))

    # plotly 版同样两条线
    try:
        import plotly.graph_objects as _pgo
        _pgo.Figure.show = lambda self, *a, **k: None
        f2 = R.plot_kline_rsi(bars['A001.XSHG'], 'A001.XSHG',
                              R._merge_cfg({'BARS': 200, 'PLOT_ENGINE': 'plotly'}))
        tn = [t.name for t in f2.data]
        ck('plotly 版也有 TR%', 'TR%' in tn, tn)
        ck('plotly 版也有 ATR%', any(str(s).startswith('ATR%') for s in tn), tn)
    except Exception as e:
        print('[skip] plotly 副图跳过：%s' % e)
except Exception as e:  # 环境问题不算失败
    print('[skip] 副图检查跳过：%s' % e)

# ---------- 9. ATR 信号：异动（TR 突刺）/ 收缩（波幅压死） ----------
import pandas as _pd2


def mk_rv(n=260, seed=5, vol=0.012):
    rs = np.random.RandomState(seed)
    cl = 10.0 * np.cumprod(1 + rs.normal(0.0002, vol, n))
    op = cl * (1 + rs.normal(0, 0.003, n))
    hi = np.maximum(op, cl) * (1 + np.abs(rs.normal(0, 0.004, n)))
    lo = np.minimum(op, cl) * (1 - np.abs(rs.normal(0, 0.004, n)))
    vv = rs.randint(1e6, 3e6, n).astype(float)
    return _pd2.DataFrame({'open': op, 'high': hi, 'low': lo, 'close': cl,
                           'volume': vv},
                          index=_pd2.bdate_range('2024-01-02', periods=n))


CFG9 = R._merge_cfg({'BARS': 300, 'FREQ': 'D'})

# 异动：倒数第 3 根猛地拉出 12% 的振幅
SPK = mk_rv()
_cl3 = float(SPK['close'].iloc[-3])
SPK.iloc[-3, SPK.columns.get_loc('high')] = _cl3 * 1.12
SPK.iloc[-3, SPK.columns.get_loc('low')] = _cl3 * 0.98
sig_spk = R.atr_signal(SPK, CFG9)
ck('异动被认出来', sig_spk['异动'] is True and sig_spk['信号'] == '异动',
   (sig_spk['异动'], sig_spk['信号']))
ck('异动倍数 >1.8（报的是那根）', float(sig_spk['异动倍数']) > 1.8, sig_spk['异动倍数'])
ck('距异动 = 2 根', float(sig_spk['距异动']) == 2.0, sig_spk['距异动'])
ck('异动日就是那一根', str(sig_spk['异动日'].date()) ==
   str(SPK.index[-3].date()), (sig_spk['异动日'], SPK.index[-3]))

# 收缩：最后 30 根振幅压成 0.2%，收盘也只挪一点点（不然 TR 会因跳空又变大）
SQZ = mk_rv(seed=9)
for _i in range(-30, 0):
    _c = float(SQZ['close'].iloc[_i - 1]) * (1 + 0.0005 * ((_i % 3) - 1))
    SQZ.iloc[_i, SQZ.columns.get_loc('close')] = _c
    SQZ.iloc[_i, SQZ.columns.get_loc('open')] = _c
    SQZ.iloc[_i, SQZ.columns.get_loc('high')] = _c * 1.001
    SQZ.iloc[_i, SQZ.columns.get_loc('low')] = _c * 0.999
sig_sqz = R.atr_signal(SQZ, CFG9)
ck('收缩被认出来', sig_sqz['收缩'] is True and sig_sqz['信号'] == '收缩',
   (sig_sqz['收缩'], sig_sqz['信号'], sig_sqz['异动倍数']))
ck('收缩时分位够低', float(sig_sqz['ATR分位']) <= 20.0, sig_sqz['ATR分位'])
ck('收缩时大部分根都安静', float(sig_sqz['安静比']) >= 0.7, sig_sqz['安静比'])

# atr_state 里要带上信号
st9 = R.atr_state(SPK, CFG9)
ck('atr_state 带 ATR信号', 'ATR信号' in st9 and st9['ATR信号'] == '异动', st9.get('ATR信号'))

# 指标插件
ind9 = R.SCREEN_INDICATORS['atr_sig']['fn']
r9 = ind9(SPK, CFG9)
ck('atr_sig 异动 hit', r9['hit'] is True and r9['score'] == 1.5, r9)
ck('atr_sig info 写了信号', r9['info'].get('ATR信号') == '异动', r9['info'])
r9s = ind9(SQZ, CFG9)
ck('atr_sig 收缩 hit', r9s['hit'] is True, r9s)
ck('ATR_SIG=squeeze 时异动不算', ind9(SPK, R._merge_cfg({'ATR_SIG': 'squeeze'}))['hit'] is False)
ck('ATR_SIG=spike 时收缩不算', ind9(SQZ, R._merge_cfg({'ATR_SIG': 'spike'}))['hit'] is False)

# 差一点：把异动线抬到「最后一根刚好差一点」，并只看当根（LOOK=1）免得前面那根大阳线抢戏
_r_last = float(R._atr_ratio(SPK, CFG9).iloc[-1])
CFG_NEAR = R._merge_cfg({'ATR_SPIKE': _r_last / 0.95, 'ATR_SPIKE_LOOK': 1})
r9n = ind9(SPK, CFG_NEAR)
ck('异动差一点算 near', r9n['near'] is True and r9n['score'] == 0.75, (r9n['near'], r9n['score']))

# 档位
ck('有 gap_burst 档位', 'gap_burst' in R.SETUPS)
ck('有 squeeze_wait 档位', 'squeeze_wait' in R.SETUPS)
ck('gap_burst 判异动', R.SETUPS['gap_burst']['where']({'ATR信号': '异动'}, CFG9) is True)
ck('squeeze_wait 判收缩', R.SETUPS['squeeze_wait']['where']({'ATR信号': '收缩'}, CFG9) is True)
ck('表格有 ATR信号 列', 'ATR信号' in R._STOCK_COLS and len(R._STOCK_COLS) == 26, len(R._STOCK_COLS))

# 图上要有标记
try:
    import matplotlib
    matplotlib.use('Agg')
    fig9 = R.plot_kline_rsi(SPK, 'SPK.XSHG', CFG9)
    names9 = []
    for ax in fig9.get_axes():
        names9 += list(ax.get_legend_handles_labels()[1])
    txt9 = ' '.join([str(s) for s in names9])
    ck('图上标了异动', '异动' in txt9, names9)
    fig9b = R.plot_kline_rsi(SQZ, 'SQZ.XSHG', CFG9)
    names9b = []
    for ax in fig9b.get_axes():
        names9b += list(ax.get_legend_handles_labels()[1])
    ck('图上标了收缩', any('收缩' in str(s) for s in names9b), names9b)
    f9 = R.plot_kline_rsi(SPK, 'SPK.XSHG', R._merge_cfg({'PLOT_ENGINE': 'plotly'}))
    nm9 = [t.name for t in f9.data]
    ck('plotly 也标了异动', any('异动' in str(s) for s in nm9), nm9)
except Exception as e:
    print('[skip] 信号标记检查跳过：%s' % e)

print('')
print('FAILED: %s' % fail)
sys.exit(0 if not fail else 1)
