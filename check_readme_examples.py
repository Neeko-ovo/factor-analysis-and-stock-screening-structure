# -*- coding: utf-8 -*-
"""核验 README 里的代码：①所有 py 代码块能编译 ②附一三个示例能真跑"""
import io, os, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '_mplcache'))
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd

OK = []
def ck(name, cond, extra=''):
    OK.append((name, bool(cond), extra))
    print(('  [OK] ' if cond else '  [FAIL] ') + name + (('  ' + str(extra)) if extra else ''))

# 1) 代码围栏编译
md = open('README.md', encoding='utf-8').read()
blocks = re.findall(r'```python\n(.*?)```', md, re.S)
bad = []
for i, b in enumerate(blocks):
    try:
        compile(b, '<README block %d>' % i, 'exec')
    except SyntaxError as e:
        bad.append((i, str(e).split('\n')[0]))
print('[围栏] 共 %d 个 python 块' % len(blocks))
ck('README 里 python 代码块全部可编译', not bad, bad)

# 2) 名字存在性
import factor_lib as F
import rsi_multiday as R
for n in ('register_factor', 'register_ts_signal', 'get_price_history', 'run_factor_test', 'run_ts_test'):
    ck('F.%s 存在' % n, hasattr(F, n))
for n in ('register_indicator', 'screen_stocks', 'SETUPS', 'load_bars', 'DEFAULTS'):
    ck('R.%s 存在' % n, hasattr(R, n))

# 3) 附录一.1 截面因子：真数据
@F.register_factor('rev_20')
def rev_20(stock_list, date):
    close = F.get_price_history(stock_list, date, 21, ['close']).get('close')
    if close is None or close.empty or len(close) < 3:
        return pd.Series(np.nan, index=list(stock_list), name='rev_20')
    s = (close.iloc[0] / close.iloc[-1] - 1.0).reindex(stock_list)
    s.name = 'rev_20'
    return s

CODES = ['000300.XSHG', '510300.XSHG', '600000.XSHG', '512480.XSHG']
s = rev_20(CODES, '2024-06-28')
ck('rev_20 返回 Series 且索引是代码', isinstance(s, pd.Series) and list(s.index) == CODES, list(s.notna()))
ck('rev_20 有非空取值', int(s.notna().sum()) >= 2, s.round(4).to_dict())

# 4) 附录一.2 时序信号：真数据
@F.register_ts_signal('boll_pos_20')
def boll_pos_20(code, px, params=None):
    n = int((params or {}).get('n', 20))
    c = px['close']
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    return (c - (mid - 2 * sd)) / (4 * sd).replace(0.0, np.nan)

hist = F.get_price_history(['000300.XSHG'], '2024-06-28', 200, ['close'])
px = pd.DataFrame({'close': hist['close']['000300.XSHG']})
sg = boll_pos_20('000300.XSHG', px, {'n': 20})
ck('boll_pos_20 返回 Series(index=日期)', isinstance(sg, pd.Series) and isinstance(sg.index, pd.DatetimeIndex), len(sg.dropna()))
ck('boll_pos_20 取值大致在 0~1 附近', 0 <= float(sg.dropna().iloc[-1]) <= 1.2, round(float(sg.dropna().iloc[-1]), 3))

# 5) 附录一.3 筛股指标：真数据
@R.register_indicator('bw_20', '布林带宽（压死=蓄势）')
def _ind_bw(df, cfg):
    n = int(cfg.get('BW_N', 20))
    c = df['close']
    bw = (c.rolling(n).std() * 4 / c.rolling(n).mean() * 100.0).dropna()
    if bw.empty:
        return {'hit': False, 'score': 0.0, 'near': False, 'info': {'带宽%': None}}
    p = float(bw.iloc[-1])
    q = float((bw <= p).mean() * 100.0)
    lo = float(cfg.get('BW_LOW', 20.0))
    return {'hit': q <= lo, 'score': 1.0 if q <= lo else 0.0,
            'near': lo < q <= lo + 10.0,
            'info': {'带宽%': round(p, 2), '带宽分位': round(q, 1)}}

df = R.load_bars('000300.XSHG', dict(R.DEFAULTS, BARS=300))
cfg = R._merge_cfg({'BW_N': 20, 'BW_LOW': 20.0})
res = _ind_bw(df, cfg)
ck('_ind_bw 返回四键', set(res) == {'hit', 'score', 'near', 'info'}, res)
ck('_ind_bw info 有带宽与分位', res['info'].get('带宽%') is not None, res['info'])
ck('bw_20 已在 SCREEN_INDICATORS', 'bw_20' in R.SCREEN_INDICATORS)
ck('_IND_CN 自动补了中文名', R._IND_CN.get('bw_20') == '布林带宽（压死=蓄势）', R._IND_CN.get('bw_20'))

# 6) 组档位能跑
R.SETUPS['bw_squeeze'] = {'name': '布林挤压 × FVG', 'desc': '缺口回踩 + 带宽压到历史低位',
                          'use': ['fvg', 'bw_20'], 'need': ['fvg', 'bw_20']}
st = R._setup_cfg('bw_squeeze')
ck('_setup_cfg 认自定义档位', st.get('name') == '布林挤压 × FVG', st.get('need'))

bad = [n for n, o, _ in OK if not o]
print('\n=== %d 条断言，%d 失败 %s ===' % (len(OK), len(bad), bad if bad else ''))
sys.exit(1 if bad else 0)
