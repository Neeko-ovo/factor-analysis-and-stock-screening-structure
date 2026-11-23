# -*- coding: utf-8 -*-
"""jq_params.py 自测：参数校验 + 统一入口套参数是否正确。"""
import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, cond, extra=''):
    print('  %s %-46s %s' % ('OK  ' if cond else 'FAIL', name, extra))
    if not cond:
        FAILED.append(name)


CALLS = {}


def _rec(fn):
    def f(*a, **kw):
        CALLS[fn] = (a, kw)
        return {'fn': fn, 'args': a, 'kw': kw}
    return f


ns = {
    'run_factor_test': _rec('cs'),
    'run_batch': _rec('cs_batch'),
    'scan_ic_decay': _rec('cs_decay'),
    'run_ts_test': _rec('ts'),
    'run_ts_batch': _rec('ts_batch'),
    'ts_ic_decay': _rec('ts_decay'),
    'set_viz_engine': lambda v: CALLS.__setitem__('viz', v),
    'ts_set_engine': lambda v: CALLS.__setitem__('tsengine', v),
}
src = open(os.path.join(HERE, 'jq_params.py'), encoding='utf-8').read()
exec(compile(src, 'jq_params.py', 'exec'), ns)

print('=' * 70)
print('一、默认值与校验')
print('=' * 70)
check('默认校验通过', ns['check_params']() is True)
t = ns['params_table']()
check('参数表是 DataFrame', isinstance(t, pd.DataFrame))
check('参数表行数 = PARAM_DOC 条数', len(t) == len(ns['PARAM_DOC']), '%d / %d' % (len(t), len(ns['PARAM_DOC'])))
check('参数表含 Barra 三项', set(['BARRA_MODE', 'BARRA_STYLES', 'BARRA_WINSOR']).issubset(set(t['参数'])))
check('参数表四列', list(t.columns) == ['参数', '当前值', '说明', '规则'])
check('默认 TH=0 / PERIOD=20', ns['TH'] == 0 and ns['PERIOD'] == 20)
check('默认引擎 mpl', ns['PLOT_ENGINE'] == 'mpl')

for bad, why in [(dict(FREQ='X'), 'FREQ 非法'), (dict(DEC_FREQ='Q'), 'DEC_FREQ 非法'),
                 (dict(PERIOD=0), 'PERIOD=0'), (dict(TH='a'), 'TH 非数字'),
                 (dict(START='2025-01-01', END='2019-01-01'), 'START>=END'),
                 (dict(Q=1), 'Q=1'), (dict(RULE='mid'), 'RULE 非法'),
                 (dict(PLOT_ENGINE='svg'), '引擎非法'), (dict(FACTORS=[]), '空清单'),
                 (dict(COST=-1), '负成本')]:
    try:
        ns['check_params'](**bad)
        check('拒绝 %s' % why, False)
    except ValueError:
        check('拒绝 %s' % why, True)

print()
print('=' * 70)
print('二、统一入口套参数')
print('=' * 70)
ns['cs_run']()
a, kw = CALLS['cs']
check('cs_run 传了池子与区间', a[1] == 'HS300' and a[2] == '2019-01-01' and a[3] == '2025-06-30', str(a))
check('cs_run 因子默认取第一个', a[0] == 'px_pct_750', str(a[0]))
check('cs_run period/freq/q', kw['period'] == 20 and kw['freq'] == 'M' and kw['q'] == 5)

ns['cs_run']('vol_ratio', period=60, freq='W')
a, kw = CALLS['cs']
check('cs_run 可覆盖', a[0] == 'vol_ratio' and kw['period'] == 60 and kw['freq'] == 'W')

ns['cs_batch_run']()
a, kw = CALLS['cs_batch']
check('cs_batch 跑 FACTORS', list(a[0]) == ['px_pct_750', 'px_pct_dev250', 'px_pct_score'], str(a[0]))

ns['cs_decay_run']()
a, kw = CALLS['cs_decay']
check('cs_decay 用 CS_PERIODS', tuple(kw['periods']) == (5, 10, 20, 63, 126, 250), str(kw['periods']))

ns['ts_run']('pct_score', th=60)
a, kw = CALLS['ts']
check('ts_run 传标的与区间', a[1] == '000300.XSHG' and a[2] == '2019-01-01', str(a))
check('ts_run 决策频率 W', kw['decision_freq'] == 'W' and kw['th'] == 60)
check('ts_run 成本/方向', kw['cost'] == 0.0005 and kw['direction'] == 1)

ns['ts_run']()
a, kw = CALLS['ts']
check('ts_run 信号默认取第一个', a[0] == 'ma200', str(a[0]))

ns['ts_batch_run']()
a, kw = CALLS['ts_batch']
check('ts_batch 跑 SIGNALS', list(a[0]) == ['ma200', 'mom_120', 'pct_score'], str(a[0]))

ns['ts_decay_run']()
a, kw = CALLS['ts_decay']
check('ts_decay 用 TS_PERIODS', tuple(kw['periods']) == (5, 10, 20, 60, 120), str(kw['periods']))

print()
print('=' * 70)
print('三、改参数 / 引擎 / 保守口径')
print('=' * 70)
t2 = ns['apply_params'](TH=60, PERIOD=60)
check('apply_params 改了全局', ns['TH'] == 60 and ns['PERIOD'] == 60)
check('返回表里也变了', int(t2.loc[t2['参数'] == 'TH', '当前值'].iloc[0]) == 60)
ns['ts_run']()
a, kw = CALLS['ts']
check('ts_run 吃到了新 TH/PERIOD', kw['th'] == 60 and kw['period'] == 60)

try:
    ns['apply_params'](THH=1)
    check('拼错参数名要报错', False)
except KeyError as e:
    check('拼错参数名要报错', 'THH' in str(e))
check('拼错后 TH 没被污染', ns['TH'] == 60)

ns['plot_engine']()
check('plot_engine 同步两个模块', CALLS.get('viz') == 'mpl' and CALLS.get('tsengine') == 'mpl')
ns['plot_engine']('plotly')
check('plot_engine 切 plotly', CALLS.get('viz') == 'plotly' and CALLS.get('tsengine') == 'plotly')
check('plot_engine 也改了参数区', ns['PLOT_ENGINE'] == 'plotly')
try:
    ns['plot_engine']('svg')
    check('引擎写错要报错', False)
except ValueError:
    check('引擎写错要报错', True)

ns['no_lookahead']()
check('no_lookahead 开 STRICT + 非重叠', ns['STRICT'] is True and ns['NON_OVERLAP'] is True)
ns['cs_run']()
a, kw = CALLS['cs']
check('保守口径进了 cs_run', kw['strict'] is True and kw['non_overlap'] is True)

print()
print('=' * 70)
if FAILED:
    print('FAILED: %s' % FAILED)
    sys.exit(1)
print('全部通过')
sys.exit(0)
