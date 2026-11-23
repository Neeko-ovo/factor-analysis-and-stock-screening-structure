# -*- coding: utf-8 -*-
# ============================================================
# 本地 mock 自测：不连聚宽，验证 Part F（历史分位打分因子）整条链路
#   1) 算子语义（分位方向、打分单调性、截面上"最后一行"分位）
#   2) 时序信号：pct_750 / pct_score 跑通 run_ts_test + 四联图
#   3) 截面因子：px_pct_* 跑通 run_factor_test / run_batch
#   4) 估值分位：run_pct_value_test（mock get_fundamentals）
# 运行： python selftest_pct_mock.py
# 产物： pic_pct/
# ============================================================
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

FRAMEWORK = os.path.join(HERE, 'jq_unified_factor_framework.py')
PKGS = [os.path.join(HERE, n) for n in
        ('jq_factor_viz.py', 'jq_ts_viz.py', 'jq_tsa_patch.py', 'jq_pct_factor.py')]
OUT = os.path.join(HERE, 'pic_pct')

TRADE = pd.bdate_range('2018-01-02', '2024-12-31')
N = 60
CODES = ['S%03d.XSHG' % i for i in range(N)]
_T, _POS = len(TRADE), {d: i for i, d in enumerate(TRADE)}

rng = np.random.default_rng(20240101)
lat = np.zeros((_T, N))
lat[0] = rng.standard_normal(N)
for t in range(1, _T):
    lat[t] = 0.97 * lat[t - 1] + 0.24 * rng.standard_normal(N)
ret = 0.0002 + 0.0012 * lat + 0.018 * rng.standard_normal((_T, N))
_CLOSE = 10.0 * np.exp(np.cumsum(np.log1p(ret), axis=0))
_MONEY = _CLOSE * 1e7 * np.exp(0.5 * rng.standard_normal((_T, N)))
# 估值：便宜股（lat 高）给低 PE —— 用来验证"分位低=便宜"的方向
PE = np.clip(30.0 - 12.0 * lat + 2.0 * rng.standard_normal((_T, N)), 3.0, 200.0)
PB = np.clip(3.0 - 1.2 * lat + 0.3 * rng.standard_normal((_T, N)), 0.2, 30.0)

_IDX = {}


def _ts(x):
    return pd.Timestamp(x).normalize()


def _index_series(code):
    if code not in _IDX:
        r = np.random.default_rng(abs(hash(code)) % (2 ** 31))
        d = np.concatenate([np.full(_T // 3, 0.0009),
                            np.full(_T - 2 * (_T // 3), -0.0008),
                            np.full(_T // 3, 0.0007)])[:_T]
        rr = r.normal(0, 0.011, _T) + d
        _IDX[code] = 3000.0 * np.exp(np.cumsum(rr))
    return _IDX[code]


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
    series = []
    for f in fields:
        cols = {}
        for c in codes:
            if c in CODES:
                arr = _CLOSE if f == 'close' else (_MONEY if f == 'money' else _CLOSE)
                cols[c] = arr[np.ix_(pos, [CODES.index(c)])].ravel()
            else:
                v = _index_series(c)[pos]
                if f == 'money':
                    v = v * 1e9
                elif f in ('high', 'low'):
                    v = v * (1.0 if f == 'high' else 0.99)
                cols[c] = v
        df = pd.DataFrame(cols, index=pd.DatetimeIndex(sub, name='time'))
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


# ---- 财务表 mock ----
class _Col(object):
    def __init__(self, name):
        self.name = name

    def in_(self, codes):
        return ('in', list(codes))


class _Table(object):
    def __init__(self, name):
        self.name = name

    def __getattr__(self, item):
        return _Col(item)


valuation = _Table('valuation')
balance = _Table('balance')
cash_flow = _Table('cash_flow')
income = _Table('income')
indicator = _Table('indicator')


class _Q(object):
    def __init__(self, *cols):
        self.cols = cols
        self.codes = list(CODES)

    def filter(self, *args):
        for a in args:
            if isinstance(a, tuple) and a and a[0] == 'in':
                self.codes = a[1]
        return self

    def limit(self, n):
        self.n = n
        return self


def query(*cols):
    return _Q(*cols)


def get_fundamentals(q, date=None):
    if not isinstance(q, _Q):
        return pd.DataFrame(columns=['id', 'code', 'pe_ratio', 'pb_ratio'])
    i = _POS.get(_ts(date)) if date is not None else _T - 1
    if i is None:
        return pd.DataFrame()
    codes = list(q.codes)
    idx = [CODES.index(c) for c in codes if c in CODES]
    codes = [c for c in codes if c in CODES]
    want = [getattr(c, 'name', '') for c in q.cols if isinstance(c, _Col)]
    df = pd.DataFrame({'code': codes})
    if 'pe_ratio' in want or not want:
        df['pe_ratio'] = PE[np.ix_([i], idx)].ravel()
    if 'pb_ratio' in want or not want:
        df['pb_ratio'] = PB[np.ix_([i], idx)].ravel()
    return df


_fake_jq = types.ModuleType('jqdata')
for _n in ('get_price', 'get_trade_days', 'get_index_stocks', 'get_security_info',
           'get_fundamentals', 'query', 'valuation', 'balance', 'cash_flow',
           'income', 'indicator'):
    setattr(_fake_jq, _n, globals()[_n])
sys.modules['jqdata'] = _fake_jq

NS = {'__name__': 'jq_ns', '__file__': FRAMEWORK}
for _p in [FRAMEWORK] + PKGS:
    with open(_p, encoding='utf-8') as fh:
        exec(compile(fh.read(), _p, 'exec'), NS)

print('\n[1] 算子语义检查')
pr, ps, pl = NS['pct_rank'], NS['pct_to_score'], NS['pct_last']
up = pd.Series(np.arange(1.0, 101.0))
dn = up[::-1].reset_index(drop=True)
assert abs(pr(up, 50).iloc[-1] - 1.0) < 1e-9, '单调上行分位应为 1'
assert abs(pr(dn, 50).iloc[-1] - 1.0 / 50) < 1e-9, '单调下行分位应为 1/W（含当日口径）'
assert abs(ps(pr(up, 50), 'linear').iloc[-1] - 0.0) < 1e-9, '最高分位→最贵→0 分'
assert abs(ps(pr(dn, 50), 'linear').iloc[-1] - (100.0 - 100.0 / 50)) < 1e-9, '最低分位→最便宜'
band = ps(pd.Series([0.05, 0.20, 0.40, 0.60, 0.80, 0.95]), 'band')
assert band.tolist() == [100.0, 85.0, 65.0, 45.0, 25.0, 10.0], band.tolist()
dfw = pd.DataFrame({'a': np.arange(1.0, 21.0), 'b': np.arange(20.0, 0.0, -1.0)})
p = pl(dfw, 20)
assert abs(p['a'] - 1.0) < 1e-9 and abs(p['b'] - 1.0 / 20) < 1e-9, p.tolist()
# 未来函数检查：把未来数据改掉，历史分位不应变
s0 = np.random.default_rng(1).normal(size=300)
a1 = pr(pd.Series(s0), 100)
s1 = s0.copy(); s1[200:] += 50.0
a2 = pr(pd.Series(s1), 100)
assert np.allclose(a1.iloc[:200], a2.iloc[:200], equal_nan=True), '未来数据影响了历史分位'
print('    OK：方向/档位/截面/无未来函数 全部通过')

print('\n[2] 时序版：pct_750 / pct_score')
NS['setup_style'](interactive=False)
for name in ('pct_750', 'pct_score', 'pct_dev250', 'pct_dd_750'):
    r = NS['run_ts_test'](name, '000300.XSHG', '2019-01-01', '2024-12-31',
                          period=20, th=60, decision_freq='W', cost=0.0005,
                          verbose=False)
    st = r['ic_stats']
    print('    %-10s IC=%+.4f t(NW)=%+.2f | 策略年化=%.2f%% 夏普=%.2f'
          % (name, st.get('IC(Spearman)', np.nan), st.get('t值(Newey-West)', np.nan),
             r['backtest'].loc['策略(择时)', '年化收益%'],
             r['backtest'].loc['策略(择时)', '夏普']))

rep = NS['pct_report']('000300.XSHG', '2019-01-01', '2024-12-31',
                       signal='pct_score', th=60, decision_freq='W',
                       save=True, outdir=OUT)
print('    四联图已保存: %s' % os.path.join(OUT, 'pct_pct_score_000300.png'))

print('\n[3] 截面版：px_pct_*')
r1 = NS['run_factor_test']('px_pct_750', CODES, '2019-01-01', '2024-06-30',
                           period=20, q=5, verbose=False)
print('    px_pct_750 覆盖率=%.2f IC均值=%+.4f t修正=%+.2f 判定=%s'
      % (r1['factor_panel'].notna().mean().mean(), r1['ic_stats'].get('IC均值', np.nan),
         r1['ic_stats'].get('t值_修正', np.nan), NS['_verdict'](r1['ic_stats'])))
figs = NS['plot_single_all'](r1, save=True, outdir=OUT, prefix='px_pct_750')

names = ['px_pct_750', 'px_pct_500', 'px_pct_dev250', 'px_pct_score']
dfb, details, panels, fwd = NS['run_batch'](names, CODES, '2019-01-01', '2024-06-30',
                                            period=20, q=5, verbose=False)
print(dfb[['因子', '覆盖率', 'IC均值', 'IC_IR', 't值_修正', '判定']].round(4).to_string(index=False))
fb = NS['plot_batch'](dfb)
if fb is not None:
    fb.savefig(os.path.join(OUT, 'batch_pct.png'), dpi=140, bbox_inches='tight')

print('\n[4] 估值分位：pe_ratio（mock 财务表，验证方向 = 低 PE 高分）')
rv = NS['run_pct_value_test']('pe_ratio', CODES, '2019-01-01', '2024-06-30',
                              window=12, period=20, freq='M', q=5, verbose=False)
print('    %s 覆盖率=%.2f IC均值=%+.4f t修正=%+.2f'
      % (rv['name'], rv['summary'].loc[0, '覆盖率'], rv['summary'].loc[0, 'IC均值'],
         rv['summary'].loc[0, 't值_修正']))

print('\n[done] 产物目录 %s' % OUT)
