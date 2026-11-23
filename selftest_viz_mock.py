# -*- coding: utf-8 -*-
"""
自测：不依赖聚宽，用假数据跑通「原框架 + matplotlib 可视化补丁」整条链路。
运行：python selftest_viz_mock.py
产物：pic/*.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')          # 自测用静态后端，只落 png

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK = os.path.join(HERE, 'jq_ts_factor_framework.py')


# ------------------------------------------------------------
# 一、伪造聚宽主框架的 get_price_range
#     契约：返回 {字段名: DataFrame(index=日期, columns=股票代码)}
# ------------------------------------------------------------
def mock_get_price_range(codes, start, end, fields):
    code = codes[0] if isinstance(codes, (list, tuple)) else codes
    idx = pd.bdate_range(start, end)
    n = len(idx)
    rng = np.random.default_rng(20240101)
    # 造一段带牛熊切换的价格，让均线类信号有东西可抓
    drift = np.concatenate([
        np.full(n // 3, 0.0012),
        np.full(n - 2 * (n // 3), -0.0010),
        np.full(n // 3, 0.0008),
    ])
    drift = drift[:n]
    ret = rng.normal(0.0, 0.0125, n) + drift
    close = 3000.0 * np.exp(np.cumsum(ret))
    out = {}
    for f in fields:
        if f == 'close':
            v = close
        elif f == 'high':
            v = close * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
        elif f == 'low':
            v = close * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
        elif f == 'money':
            v = rng.uniform(1e8, 5e8, n) * (close / 3000.0)
        else:
            v = close
        out[f] = pd.DataFrame({code: v}, index=idx)
    return out


# ------------------------------------------------------------
# 二、把原框架 exec 进命名空间（不改原文件）
# ------------------------------------------------------------
def load_framework():
    with open(FRAMEWORK, 'r', encoding='utf-8') as f:
        src = f.read()
    ns = {'pd': pd, 'np': np, 'get_price_range': mock_get_price_range}
    exec(compile(src, FRAMEWORK, 'exec'), ns)
    return ns


def main():
    ns = load_framework()
    run_ts_test = ns['run_ts_test']
    run_ts_batch = ns['run_ts_batch']
    ts_ic_decay = ns['ts_ic_decay']
    print('[ok] 框架加载完成，已注册信号: %s' % sorted(ns['TS_SIGNALS'].keys()))

    import jq_ts_viz as viz
    backend, notes = viz.ts_setup_style(interactive=False, dpi=130)
    print('[ok] backend=%s' % backend)
    for line in notes:
        print('     %s' % line)

    code = '000300.XSHG'
    start, end = '2019-01-01', '2024-12-31'

    # ---- 单信号全流程 ----
    r = run_ts_test('ma200', code, start, end, period=20, th=0.0,
                    decision_freq='W', cost=0.0005, direction=1)
    print('[ok] run_ts_test 完成: 信号 %d 个点, IC=%.4f'
          % (len(r['signal']), r['ic_stats'].get('IC(Spearman)', float('nan'))))

    outdir = os.path.join(HERE, 'pic')
    saved = viz.ts_save_all(r, outdir=outdir, prefix='ma200', code=code, th=0.0)
    print('[ok] 单信号图:')
    for k, v in saved.items():
        print('     %-14s %s' % (k, v))

    # ---- IC 衰减 ----
    d = ts_ic_decay('mom_120', code, start, end)
    fig = viz.ts_plot_ic_decay(d)
    p = os.path.join(outdir, 'ic_decay_mom120.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    print('[ok] IC衰减图: %s' % p)

    # ---- 多信号对比 ----
    b = run_ts_batch(['ma200', 'ma60', 'mom_120', 'trend_ma'], code, start, end,
                     period=20, th=0.0, decision_freq='W', cost=0.0005, direction=1)
    fig = viz.plot_batch_compare(b)
    p = os.path.join(outdir, 'batch_compare.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    print('[ok] 批量对比图: %s' % p)

    # ---- 负向信号：direction=-1，验证阴影/阈值翻转不崩 ----
    r2 = run_ts_test('volz_20_250', code, start, end, period=20, th=0.0,
                     decision_freq='M', cost=0.0005, direction=-1, verbose=False)
    saved2 = viz.ts_save_all(r2, outdir=outdir, prefix='volz_dm1', code=code,
                          th=0.0, direction=-1)
    print('[ok] 负向信号图: %s'
          % ', '.join(os.path.basename(v) for v in saved2.values()))

    # ---- 全信号扫描：确认可视化对连/离散/负向信号都能出图 ----
    allsig = sorted(ns['TS_SIGNALS'].keys())
    b2 = run_ts_batch(allsig, code, start, end, period=20, th=0.0,
                      decision_freq='M', cost=0.0005, direction=1)
    fig = viz.plot_batch_compare(b2)
    p = os.path.join(outdir, 'batch_all_signals.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    print('[ok] 全信号对比图: %s' % p)

    print('\n[done] 全部产物在 %s' % outdir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
