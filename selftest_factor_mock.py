# -*- coding: utf-8 -*-
# ============================================================
# 本地 mock 自测：不连聚宽，造带真实 alpha 的假行情，跑 Part B 全流程 +
# jq_factor_viz 出图。用来验证画图层和框架接口对得上。
#
# 运行： python selftest_factor_mock.py
# 产物： pic_factor/ 下的 png
# ============================================================
import os
import sys
import types
import datetime
import inspect

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

import numpy as np
import pandas as pd

FRAMEWORK = os.path.join(HERE, 'jq_unified_factor_framework.py')
VIZ = os.path.join(HERE, 'jq_factor_viz.py')
OUT = os.path.join(HERE, 'pic_factor')

# ------------------------------------------------------------
# 一、造数据：AR(1) 隐藏信号 lat，日收益含 alpha，未来收益与 lat 正相关
# ------------------------------------------------------------
TRADE = pd.bdate_range('2018-01-02', '2024-12-31')
N = 60
CODES = ['S%03d.XSHG' % i for i in range(N)]
_T, _POS = len(TRADE), {d: i for i, d in enumerate(TRADE)}


def _make_prices(seed=20240101, alpha=0.0012, noise=0.018):
    rng = np.random.default_rng(seed)
    lat = np.zeros((_T, N))
    lat[0] = rng.standard_normal(N)
    for t in range(1, _T):
        lat[t] = 0.97 * lat[t - 1] + 0.24 * rng.standard_normal(N)
    ret = 0.0002 + alpha * lat + noise * rng.standard_normal((_T, N))
    close = 10.0 * np.exp(np.cumsum(np.log1p(ret), axis=0))
    money = close * 1e7 * np.exp(0.5 * rng.standard_normal((_T, N)))
    return close, money, lat, rng


_CLOSE, _MONEY, _LAT, _RNG = _make_prices()

# ------------------------------------------------------------
# 二、mock 聚宽 API
# ------------------------------------------------------------
def _ts(x):
    return pd.Timestamp(x).normalize()


def get_trade_days(start_date=None, end_date=None, count=None):
    s = TRADE
    if start_date is not None:
        s = s[s >= _ts(start_date)]
    if end_date is not None:
        s = s[s <= _ts(end_date)]
    if count is not None:
        s = s[:int(count)] if start_date is not None else s[-int(count):]
    return pd.DatetimeIndex(s)


def get_price(security, start_date=None, end_date=None, count=None, frequency='daily',
              fields=None, skip_paused=False, fq='pre', panel=False):
    fields = list(fields or ['close'])
    codes = [security] if isinstance(security, str) else list(security)
    if start_date is not None and end_date is not None:
        sub = TRADE[(TRADE >= _ts(start_date)) & (TRADE <= _ts(end_date))]
    elif end_date is not None and count:
        sub = TRADE[TRADE <= _ts(end_date)][-int(count):]
    else:
        sub = TRADE
    if len(sub) == 0:
        return pd.DataFrame(columns=['time', 'code'] + fields)
    pos = [_POS[d] for d in sub]
    idx = [CODES.index(c) for c in codes]
    data = {'close': _CLOSE, 'money': _MONEY}
    series = []
    for f in fields:
        arr = data.get(f, _CLOSE)[np.ix_(pos, idx)]
        df = pd.DataFrame(arr, index=pd.DatetimeIndex(sub, name='time'), columns=codes)
        series.append(df.stack().rename(f))
    out = pd.concat(series, axis=1).reset_index()
    out.columns = ['time', 'code'] + fields
    return out


def get_index_stocks(index_code, date=None):
    return list(CODES)


class _SecInfo(object):
    def __init__(self, code):
        self.code = code
        self.display_name = code.split('.')[0]


def get_security_info(code):
    return _SecInfo(code)


# 让 `from jqdata import *` 能过
_fake_jq = types.ModuleType('jqdata')
for _n in ('get_price', 'get_trade_days', 'get_index_stocks', 'get_security_info'):
    setattr(_fake_jq, _n, globals()[_n])
sys.modules['jqdata'] = _fake_jq

# ------------------------------------------------------------
# 三、装载框架 + 画图层
# ------------------------------------------------------------
NS = {'__name__': 'jq_ns', '__file__': FRAMEWORK}

with open(FRAMEWORK, encoding='utf-8') as fh:
    exec(compile(fh.read(), FRAMEWORK, 'exec'), NS)

with open(VIZ, encoding='utf-8') as fh:
    exec(compile(fh.read(), VIZ, 'exec'), NS)

print('框架函数 %d 个 | 画图函数 %d 个'
      % (len([k for k, v in NS.items() if callable(v) and not k.startswith('_')]),
         len(NS['__all__'])))

_backend, _notes = NS['setup_style'](interactive=False)
print('后端: %s' % _backend)
for _n in _notes:
    print('  %s' % _n)

# ------------------------------------------------------------
# 四、注册 mock 因子（走自定义通道，绕开 get_fundamentals / jqfactor）
# ------------------------------------------------------------
register_factor = NS['register_factor']


def _lat_at(date, noise=0.0, seed=None):
    i = _POS.get(_ts(date))
    if i is None:
        return None
    v = _LAT[i].copy()
    if noise:
        r = np.random.default_rng(seed if seed is not None else i)
        v = v + noise * r.standard_normal(N)
    return pd.Series(v, index=CODES)


@register_factor('lat_strong')
def f_lat_strong(stock_list, date):
    s = _lat_at(date)
    return pd.Series(dtype=float) if s is None else s.reindex(stock_list)


@register_factor('lat_weak')
def f_lat_weak(stock_list, date):
    s = _lat_at(date, noise=2.0, seed=7)
    return pd.Series(dtype=float) if s is None else s.reindex(stock_list)


@register_factor('lat_rev')
def f_lat_rev(stock_list, date):
    s = _lat_at(date)
    return pd.Series(dtype=float) if s is None else (-s).reindex(stock_list)


PW, ST, ED = CODES, '2019-01-01', '2024-06-30'

# ------------------------------------------------------------
# 五、单因子全流程 + 一键出图
# ------------------------------------------------------------
print('\n' + '#' * 78)
print('# 单因子：lat_strong')
print('#' * 78)
r = NS['run_factor_test']('lat_strong', PW, ST, ED, period=20, q=5)

figs = NS['plot_single_all'](r, save=True, outdir=OUT, prefix='strong')

# 换手时序（框架只给标量，这里自己算）
ts = NS['quantile_turnover_series'](r['factor_panel'], q=5, group='ALL')
fig_turn = NS['plot_turnover'](ts, title='lat_strong 逐期换手率（最高层与全层均值）')
if fig_turn is not None:
    fig_turn.savefig(os.path.join(OUT, 'strong_turnover_series.png'), dpi=140, bbox_inches='tight')

print('\n[净值绩效]')
nav, perf = NS['quantile_nav'](r['quantile_returns'], interval=r['interval'], ratio=r['ratio'])
print(perf.round(3).to_string())
print('\n[多空绩效]')
_, ls_nav, ls_perf = NS['long_short_stats'](r['quantile_returns'],
                                            interval=r['interval'], ratio=r['ratio'])
print(ls_perf.round(3).to_string())
print('\n[分年度 IC]')
print(NS['ic_yearly'](r['ic']).round(3).to_string())

# ------------------------------------------------------------
# 六、批量 + 相关矩阵 + 衰减
# ------------------------------------------------------------
print('\n' + '#' * 78)
print('# 批量 3 因子')
print('#' * 78)
df, details, panels, fwd = NS['run_batch'](['lat_strong', 'lat_weak', 'lat_rev'],
                                           PW, ST, ED, period=20, q=5)
f_batch = NS['plot_batch'](df)
if f_batch is not None:
    f_batch.savefig(os.path.join(OUT, 'batch.png'), dpi=140, bbox_inches='tight')
f_corr = NS['plot_corr'](panels)
if f_corr is not None:
    f_corr.savefig(os.path.join(OUT, 'corr.png'), dpi=140, bbox_inches='tight')
f_to = NS['plot_turnover'](df)
if f_to is not None:
    f_to.savefig(os.path.join(OUT, 'batch_turnover.png'), dpi=140, bbox_inches='tight')

print('\n' + '#' * 78)
print('# IC 衰减')
print('#' * 78)
dec = NS['scan_ic_decay']('lat_strong', PW, ST, ED)
f_dec = NS['plot_ic_decay'](dec)
if f_dec is not None:
    f_dec.savefig(os.path.join(OUT, 'ic_decay.png'), dpi=140, bbox_inches='tight')

# ------------------------------------------------------------
# 七、组合
# ------------------------------------------------------------
print('\n' + '#' * 78)
print('# 组合方案')
print('#' * 78)
schemes = {
    '单因子 strong': {'lat_strong': (1.0, 1)},
    '双因子等权': {'lat_strong': (0.5, 1), 'lat_weak': (0.5, 1)},
    '反向对照': {'lat_rev': (1.0, 1)},
}
try:
    out, cpanels, cfwd = NS['run_combo'](schemes, PW, ST, ED, period=20)
    f_combo = NS['plot_combo'](out)
    if f_combo is not None:
        f_combo.savefig(os.path.join(OUT, 'combo.png'), dpi=140, bbox_inches='tight')
except Exception as e:
    print('  run_combo 失败: %s: %s' % (type(e).__name__, e))

print('\n完成，产物在 %s' % OUT)
