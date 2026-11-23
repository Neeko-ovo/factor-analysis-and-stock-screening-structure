# -*- coding: utf-8 -*-
"""
自测：不依赖聚宽，用假数据跑通「时序框架 + tsa 有效性补丁」整条链路。
运行：python selftest_tsa_mock.py
产物：pic_tsa/*.png
"""
import os
import sys
import zlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK = os.path.join(HERE, 'jq_ts_factor_framework.py')
PATCH = os.path.join(HERE, 'jq_tsa_patch.py')
OUT = os.path.join(HERE, 'pic_tsa')


# ------------------------------------------------------------
# 一、假数据：一份价格，所有聚宽 API 都从它取，保证信号与 close 对齐
# ------------------------------------------------------------
_PX = {}


def _px_all(code):
    if code in _PX:
        return _PX[code]
    idx = pd.bdate_range('2015-01-01', '2026-12-31')
    rng = np.random.default_rng(zlib.crc32(code.encode()) % (2 ** 32))
    n = len(idx)
    drift = np.concatenate([
        np.full(n // 4, 0.0009),
        np.full(n // 4, -0.0011),
        np.full(n // 4, 0.0006),
        np.full(n - 3 * (n // 4), 0.0008),
    ])[:n]
    ret = rng.normal(0.0, 0.013, n) + drift
    close = 3000.0 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({
        'close': close,
        'high': close * (1.0 + np.abs(rng.normal(0.0, 0.004, n))),
        'low': close * (1.0 - np.abs(rng.normal(0.0, 0.004, n))),
        'money': rng.uniform(1e8, 5e8, n) * (close / 3000.0),
    }, index=idx)
    _PX[code] = df
    return df


def mock_get_price_range(codes, start, end, fields):
    out = {}
    for f in fields:
        cols = {}
        for c in codes:
            cols[c] = _px_all(c)[f].loc[pd.Timestamp(start):pd.Timestamp(end)]
        out[f] = pd.DataFrame(cols)
    return out


def mock_get_price(security, start_date, end_date, frequency='daily',
                   fields=None, fq='pre', skip_paused=False, panel=False):
    code = security[0] if isinstance(security, (list, tuple)) else security
    s = _px_all(code)['close'].loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    return pd.DataFrame({'close': s})


def mock_get_trade_days(start_date=None, end_date=None):
    return pd.bdate_range(start_date, end_date)


# ------------------------------------------------------------
# 二、装进同一个命名空间（模拟粘贴到同一个 notebook）
# ------------------------------------------------------------
def load_all():
    ns = {'pd': pd, 'np': np, 'get_price_range': mock_get_price_range,
          'get_price': mock_get_price, 'get_trade_days': mock_get_trade_days}
    with open(FRAMEWORK, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), FRAMEWORK, 'exec'), ns)
    with open(PATCH, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), PATCH, 'exec'), ns)
    return ns


def build_pullback_w(ns):
    """造一个和用户 pullback_w 同量纲的信号：距 250 日高点的百分比（0~100）。"""
    def _sig(code, px, params=None):
        hi = px['close'].rolling(250, min_periods=60).max()
        return (px['close'] / hi * 100.0).clip(0.0, 100.0)
    ns['TS_SIGNALS']['pullback_w'] = _sig


def hook_show(ns, outdir):
    """拦截 plt.show() → 存 png，便于无界面环境检查。"""
    plt = ns['plt']
    saved = []

    def _show(*args, **kwargs):
        fig = plt.gcf()
        if fig.get_axes():
            n = len(saved) + 1
            ax = fig.get_axes()[0]
            ttl = (ax.get_title() or 'fig').replace(' ', '_')
            ttl = ''.join(ch for ch in ttl if ch.isalnum() or ch in '_-')[:28]
            path = os.path.join(outdir, 'tsa_%02d_%s.png' % (n, ttl or 'fig'))
            fig.savefig(path, dpi=120, bbox_inches='tight')
            saved.append(path)
        plt.close(fig)

    plt.show = _show
    return saved


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    ns = load_all()
    build_pullback_w(ns)
    # 本机没有 SimHei，补一个可用中文字体（用户聚宽上有 SimHei，不受影响）
    ns['plt'].rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Arial Unicode MS', 'sans-serif']
    print('[ok] 已装载: 框架 + tsa 补丁，信号 %s'
          % sorted(k for k in ns['TS_SIGNALS'] if k == 'pullback_w'))

    saved = hook_show(ns, OUT)
    tsa_report = ns['tsa_report']

    codes = ['512890.XSHG', '510880.XSHG']
    S, E = '2019-01-01', '2024-12-31'

    print('\n' + '#' * 90)
    print('# 场景 A：复刻用户用法  period=150, th=60, edges=(45,60,80)')
    print('#' * 90)
    rA = tsa_report('pullback_w', codes, S, E, period=150, th=60,
                    th_list=(50, 60, 80), edges=(45, 60, 80))

    print('\n' + '#' * 90)
    print('# 场景 B：同一信号，period=20（独立样本数够不够，对比就看这里）')
    print('#' * 90)
    rB = tsa_report('pullback_w', codes[:1], S, E, period=20, th=60,
                    th_list=(50, 60, 80), edges=(45, 60, 80))

    print('\n' + '#' * 90)
    print('# 场景 C：0/1 离散信号（ma200），edges 必须手给 (0.5,)')
    print('#' * 90)
    rC = tsa_report('ma200', ['000300.XSHG'], S, E, period=20, th=0.5,
                    th_list=(0.5,), edges=(0.5,), ic_win=250)

    print('\n[ok] 出图 %d 张:' % len(saved))
    for p in saved:
        print('     %s' % os.path.basename(p))

    # 关键中间结果落成表，方便核对
    print('\n[对账] 场景A 汇总表列名: %s' % list(rA['table'].columns))
    print('[对账] 场景A store: %s' % list(rA['store'].keys()))
    print('[对账] 场景A 年度表:\n%s' % rA['yearly'][codes[0]].to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
