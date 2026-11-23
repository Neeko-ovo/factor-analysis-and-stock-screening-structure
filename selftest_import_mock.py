# -*- coding: utf-8 -*-
# ============================================================
# 本地 mock 自测：验证「合并成单文件后 import 就能用」
#   1) import jq_factor_framework 成功、符号齐全、无重名覆盖
#   2) usage() / health_check() 能跑
#   3) 分位算子语义仍然正确
#   4) 截面 px_pct_750 + 时序 pct_score 端到端跑通
#   5) from jq_factor_framework import * 可用
# 运行：python selftest_import_mock.py
# ============================================================
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

# 框架正文里有 ✓ / − 之类的符号，GBK 控制台会 UnicodeEncodeError；测试时强制 utf-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

# ---- 复用 selftest_pct_mock.py 的 mock 数据层（截到 jqdata 注入那一行为止）----
_MOCK = os.path.join(HERE, 'selftest_pct_mock.py')
with io.open(_MOCK, 'r', encoding='utf-8') as fh:
    _src = fh.read()
_head = _src.split("NS = {'__name__': 'jq_ns'")[0]
_ns = {'__name__': '_mock_data', '__file__': _MOCK}
exec(compile(_head, _MOCK, 'exec'), _ns)
CODES = _ns['CODES']

# ---- 真正的测试：import 而不是 exec ----
sys.path.insert(0, HERE)
if 'jq_factor_framework' in sys.modules:
    del sys.modules['jq_factor_framework']
import jq_factor_framework as jf

print('\n[1] 模块导入与符号完整性')
assert hasattr(jf, '__version__'), '缺 __version__'
need = ['TS_SIGNALS', 'CUSTOM_FACTORS', 'run_factor_test', 'run_ts_test', 'run_batch',
        'run_combo', 'scan_ic_decay', 'ts_ic_decay', 'tsa_report', 'tsa_build',
        'pct_rank', 'pct_to_score', 'run_pct_value_test', 'pct_report',
        'plot_single_all', 'plot_batch', 'plot_combo', 'plot_ic_decay',
        'plot_all', 'ts_save_all', 'ts_plot_ic_decay', 'ts_setup_style',
        'setup_style', 'save_all', 'health_check', 'usage', 'preview_pool']
missing = [n for n in need if not hasattr(jf, n)]
assert not missing, '缺符号: %s' % missing
assert jf._tsv_as_series is not jf._as_series, '重名覆盖又出现了（_as_series）'
assert jf._tsv_edge_desc is not jf._edge_desc, '重名覆盖又出现了（_edge_desc）'
assert 'pct_score' in jf.TS_SIGNALS and 'pct_750' in jf.TS_SIGNALS
assert 'px_pct_750' in jf.CUSTOM_FACTORS and 'px_pct_score' in jf.CUSTOM_FACTORS
print('    版本 %s | 时序信号 %d 个 | 自定义因子 %d 个'
      % (jf.__version__, len(jf.TS_SIGNALS), len(jf.CUSTOM_FACTORS)))

print('\n[2] usage() / health_check()')
jf.usage()
hc = jf.health_check('HS300', '2019-01-01', '2024-06-30')
assert hc['ok'], 'health_check 未通过: %s' % hc['messages']

print('\n[3] 分位算子语义')
up = pd.Series(np.arange(1.0, 101.0))
dn = up[::-1].reset_index(drop=True)
assert abs(jf.pct_rank(up, 50).iloc[-1] - 1.0) < 1e-9
assert abs(jf.pct_rank(dn, 50).iloc[-1] - 1.0 / 50) < 1e-9
assert jf.pct_to_score(pd.Series([0.05, 0.20, 0.40, 0.60, 0.80, 0.95]), 'band').tolist() \
    == [100.0, 85.0, 65.0, 45.0, 25.0, 10.0]
print('    OK：方向 / 档位 / 首尾值')

print('\n[4] 端到端：截面 px_pct_750 + 时序 pct_score')
r1 = jf.run_factor_test('px_pct_750', CODES, '2019-01-01', '2024-06-30',
                        period=20, q=5, verbose=False)
print('    截面 IC均值=%+.4f t修正=%+.2f 判定=%s'
      % (r1['ic_stats'].get('IC均值', np.nan), r1['ic_stats'].get('t值_修正', np.nan),
         jf._verdict(r1['ic_stats'])))
t1 = jf.run_ts_test('pct_score', '000300.XSHG', '2019-01-01', '2024-12-31',
                    period=20, th=60, decision_freq='W', verbose=False)
print('    时序 IC=%+.4f t(NW)=%+.2f 策略年化=%.2f%% 夏普=%.2f'
      % (t1['ic_stats'].get('IC(Spearman)', np.nan),
         t1['ic_stats'].get('t值(Newey-West)', np.nan),
         t1['backtest'].loc['策略(择时)', '年化收益%'],
         t1['backtest'].loc['策略(择时)', '夏普']))

print('\n[5] from ... import *')
_g = {}
exec('from jq_factor_framework import *', _g)
assert 'run_factor_test' in _g and 'pct_rank' in _g and 'plot_all' in _g
print('    星号导入 OK（%d 个名字）' % len([k for k in _g if not k.startswith('__')]))

print('\n[6] Part G IC 视角')
df, _det, _pan, _fwd = jf.run_batch(['px_pct_750', 'px_pct_500', 'px_pct_dev250', 'rev_20'],
                                   CODES, '2019-01-01', '2024-06-30', period=20,
                                   verbose=False, show_corr=False)
sc = jf.screen_ic(df, min_abs_ic=0.03, min_abs_ir=0.2, min_t=2.0)
assert '通过' in sc.columns and '方向' in sc.columns
print('    初筛：%d 个因子，通过 %d 个' % (len(sc), int(sc['通过'].sum())))
print(sc[['因子', 'IC均值', 'IC_IR', 't值_修正', '通过']].round(3).to_string())

tb = jf.ic_tables(r1, window=12)
for k in ('summary', 'yearly', 'monthly', 'dist', 'rolling'):
    assert isinstance(tb[k], pd.DataFrame), k
assert len(tb['summary']) == 4
print('    各表：summary %d 行 / yearly %d 行 / monthly %dx%d / dist %d 行 / rolling %d 行'
      % (len(tb['summary']), len(tb['yearly']), tb['monthly'].shape[0], tb['monthly'].shape[1],
         len(tb['dist']), len(tb['rolling'])))

out = jf.ic_report(r1, save=True, outdir=os.path.join(HERE, 'pic_ic'), prefix='sel')
assert len(out['saved']) >= 5, 'IC 图落盘不足: %s' % out['saved']
print('    ic_report 落盘 %d 张' % len(out['saved']))

tt = jf.ts_ic_table(t1)
assert isinstance(tt, pd.DataFrame) and '全段' in tt.index
print('    时序 IC 表：%s' % ' / '.join(tt.index[:4]))

print('\n[7] 老版 matplotlib 三位色值兼容')
import matplotlib.colors as _mc
assert getattr(jf, '_COLOR_COMPAT_DONE', False), '颜色兼容没有装上'
assert _mc.to_rgba('#333') == _mc.to_rgba('#333333')
assert _mc.to_rgba('#888', 0.5) == _mc.to_rgba('#888888', 0.5)
print('    #333 / #888 展开成六位 OK，draw 阶段不会再抛 Invalid RGBA argument')

print('\n[8] scan_ic_decay：调仓日与前向收益期对齐')
dec = jf.scan_ic_decay('px_pct_750', CODES, '2020-01-02', '2023-12-29',
                       periods=(5, 20, 63), verbose=False)
assert not dec.empty, '衰减表为空'
assert (dec['持仓期(日)'] == dec['调仓间隔(日)']).all(), '调仓间隔与前向期没对齐'
assert (dec['重叠倍数'] == 1.0).all(), '对齐后重叠倍数应恒为 1'
assert dec['持仓期(日)'].tolist() == [5, 20, 65], dec['持仓期(日)'].tolist()
print(dec[['请求持仓期', '持仓期(日)', '调仓间隔(日)', '重叠倍数', '评估期数', 'IC均值', 't值_修正']]
      .round(4).to_string(index=False))
dec2 = jf.scan_ic_decay('px_pct_750', CODES, '2020-01-02', '2023-12-29',
                        periods=(20, 60), align=False, verbose=False)
assert (dec2['重叠倍数'] == (dec2['持仓期(日)'] / dec2['调仓间隔(日)']).round(2)).all()
print('    老口径 ratio = 持仓期/调仓间隔：%s' % dec2['重叠倍数'].tolist())

print('\n[9] scan_freq_period / best_cycle：调仓频率 × 持仓期')
grid = jf.scan_freq_period('px_pct_750', CODES, '2020-01-02', '2023-12-29',
                           freqs=('M', 'Q'), periods=(5, 20, 60), q=3, verbose=False)
assert not grid.empty, '二维扫描表为空'
for col in ('频率', '调仓间隔(日)', '持仓期(日)', '重叠倍数', '周期匹配', '资金占用%',
            'IC_IR', 't值_修正', '净年化%', '判定'):
    assert col in grid.columns, '缺列 %s' % col
assert set(grid['频率']) <= {'月频', '季频'}, grid['频率'].unique()
assert (grid['重叠倍数'] >= 1.0).all()
assert ((grid['资金占用%'] >= 0) & (grid['资金占用%'] <= 100)).all()
best = jf.best_cycle(grid, verbose=False)
assert best is not None and '频率' in best, 'best_cycle 没给出结论'
print(grid[['频率', '调仓间隔(日)', '持仓期(日)', '重叠倍数', 'IC_IR', 't值_修正', '净年化%', '判定']]
      .round(3).to_string(index=False))
print('    最佳：%s 调仓 + 持有 %d 日 | IC_IR %.3f' % (best['频率'], best['持仓期(日)'], best['IC_IR']))
fig_grid = jf.plot_freq_period(grid, metric='IC_IR')
assert fig_grid is not None, '热力图没画出来'

print('\n[10] Part H：ATR 信号 / 底部打分 / 仓位状态机 / 连续仓位回测')
CODE0 = CODES[0]
for s in ('atr_pct_20', 'atr_rank_20', 'atr_squeeze', 'vol_ratio', 'bw_rank', 'rsi_14', 'mom_60'):
    sig_i, px_i = jf.build_ts_signal(s, CODE0, '2020-01-02', '2023-12-29')
    assert isinstance(sig_i, pd.Series) and sig_i.notna().sum() > 60, '%s 信号无效' % s
    print('    %-12s 有效 %d 个，最新 %.4f' % (s, int(sig_i.notna().sum()), float(sig_i.dropna().iloc[-1])))
sig_atr, _ = jf.build_ts_signal('atr_rank_20', CODE0, '2020-01-02', '2023-12-29')
assert sig_atr.dropna().between(0, 1).all(), 'atr_rank_20 应落在 0~1'

score_panel = jf.bottom_score_panel(pool=CODES[:20], start='2021-01-04', end='2023-12-29',
                                    freq='Q', window=500, verbose=False)
assert not score_panel.empty and score_panel.shape[1] == 20, '底部打分面板形状不对'
vals = score_panel.values[np.isfinite(score_panel.values)]
assert vals.min() >= 0 and vals.max() <= 100, '打分应落在 0~100'
cand = jf.bottom_candidates(score_panel, top=3, min_score=50)
print('    底部打分 %d 期，分数区间 %.1f~%.1f；有候选 %d 期'
      % (score_panel.shape[0], vals.min(), vals.max(),
         int((cand.notna().sum(axis=1) > 0).sum())))

px1 = jf.get_price_range([CODE0], '2019-01-02', '2023-12-29', ('close',))
cl1 = px1['close'][CODE0].dropna()
cheap1 = jf.cheap_score_series(cl1, window=500)
assert cheap1.notna().sum() > 100, '便宜度序列为空'
w1 = jf.weight_state(cheap1, pd.Series(0.45, index=cl1.index), close=cl1,
                     ma=cl1.rolling(200, min_periods=120).mean())
assert w1.between(0, 1).all(), '仓位应落在 0~1'
assert (w1.diff().abs().dropna() <= 0.2500001).all(), '单次调仓不应超过 0.25'
res_w = jf.run_weight_test(w1, CODE0, '2019-01-02', '2023-12-29', decision_freq='W', verbose=False)
for k in ('年化%', '夏普', '最大回撤%', '平均仓位%', '等暴露年化%', '夏普差', '回撤收窄pp'):
    assert k in res_w['metrics'], '缺指标 %s' % k
print('    状态机仓位：平均 %.0f%%，夏普 %.2f（基准 %.2f），回撤 %.1f%%（基准 %.1f%%）'
      % (res_w['metrics']['平均仓位%'], res_w['metrics']['夏普'], res_w['metrics']['基准夏普'],
         res_w['metrics']['最大回撤%'], res_w['metrics']['基准回撤%']))
fig_w = jf.plot_weight_test(res_w, cheap=cheap1.reindex(cl1.index), atr_rank=sig_atr.reindex(cl1.index))
assert fig_w is not None, '仓位三联图没画出来'
rep = jf.weight_report(CODE0, '2020-01-02', '2023-12-29', plot=False, verbose=False)
assert 'metrics' in rep and np.isfinite(rep['metrics']['夏普']), 'weight_report 失败'
print('\n[done] import 版全部通过')
