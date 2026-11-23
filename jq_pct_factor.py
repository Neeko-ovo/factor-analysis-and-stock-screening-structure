# -*- coding: utf-8 -*-
# ============================================================
# Part F 历史分位打分因子（Percentile Score）
#
#   一句话：把"当前值在过去 N 个观测里排第几"变成 0~100 的便宜度分数，
#           时序上高分→持有（安全边际高），截面上高分→买（低估反转）。
#
#   pct   = 当前值在过去 window 个观测中的分位（0=最低，1=最高）
#   score = 100 * (1 - pct)      100=最便宜，0=最贵
#
#   三个镜头（都是"越低越便宜"，方向天然一致）：
#     px   价格分位        当前价在过去 N 日处于什么位置
#     dev  均线偏离分位    当前价对 MA 的偏离程度在过去 N 日的位置
#     dd   回撤分位        当前价距区间高点的回撤在过去 N 日的位置
#
#   依赖：本文件须在 jq_unified_factor_framework.py（Part A~D）之后运行，
#         因为它要用 register_ts_signal / register_factor / get_price_history /
#         get_price_range / get_rebalance_dates / resolve_universe / run_batch 等。
#
#   注意：分位只用过去数据（rolling rank），不引入未来函数；
#         分位低 ≠ 立刻涨，短周期常是动量占优，判读见 Part F-6 说明。
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Part F-1 核心算子
# ------------------------------------------------------------
# 打分档位：分位越低 → 分数越高（便宜度）
PCT_DEFAULT_EDGES = (0.10, 0.25, 0.50, 0.75, 0.90)
PCT_DEFAULT_SCORES = (100.0, 85.0, 65.0, 45.0, 25.0, 10.0)
# 多镜头默认权重（px 权重最高：可解释性最好；dev/dd 补充"贵在趋势里"与"贵在高位"）
PCT_DEFAULT_WEIGHTS = {'px': 0.5, 'dev': 0.3, 'dd': 0.2}


def pct_rank(x, window=750, min_periods=None):
    """滚动历史分位（0~1，越大越贵）。x 可以是 Series 或 DataFrame。
    t 点的值 = x[t] 在 [t-window+1, t] 这些观测里的百分位（含当日，即"不高于当前值的占比"），
    只看过去、无未来函数。窗口内最大值 → 1.0；最小值 → 1/window（不是 0）。
    min_periods 默认 = 窗口的 60%（不足则返回 NaN，避免新股/新指数样本太少就打分）。"""
    win = max(2, int(window))
    if min_periods is None:
        min_periods = max(20, int(round(win * 0.6)))
    min_periods = max(2, min(int(min_periods), win))
    try:
        return x.rolling(win, min_periods=min_periods).rank(pct=True)
    except (AttributeError, TypeError, ValueError):
        # 老版本 pandas 没有 rolling().rank，退回逐窗循环
        if isinstance(x, pd.DataFrame):
            return x.apply(lambda s: pct_rank(s, win, min_periods))
        arr = np.asarray(x, float)
        out = np.full(arr.shape, np.nan)
        for i in range(len(arr)):
            lo = max(0, i - win + 1)
            w = arr[lo:i + 1]
            w = w[~np.isnan(w)]
            if np.isnan(arr[i]) or len(w) < min_periods:
                continue
            out[i] = float((w <= arr[i]).mean())
        return pd.Series(out, index=getattr(x, 'index', None))


def pct_last(panel, window=750, min_frac=0.6):
    """宽表版：只算"最后一行"在各自列过去 window 个观测中的分位（截面单日用，省算力）。
    panel: index=日期, columns=代码。返回 Series(index=代码)。"""
    if panel is None or len(panel) == 0:
        return pd.Series(dtype=float)
    panel = pd.DataFrame(panel)
    win = panel.iloc[-max(2, int(window)):]
    last = win.iloc[-1]
    cnt = win.notna().sum()
    need = max(20, int(round(len(win) * float(min_frac))))
    le = win.le(last, axis=1).sum()
    pct = (le / cnt.replace(0, np.nan)).clip(0.0, 1.0)
    return pct.where(cnt >= need)


def pct_to_score(pct, mode='linear', edges=None, scores=None):
    """分位 → 0~100 便宜度分（100=最便宜）。
    mode='linear' : score = 100*(1-pct)，连续、无跳变，适合做因子/信号
    mode='band'   : 按分位区间给台阶分，默认 <=10%→100、<=25%→85、<=50%→65、
                    <=75%→45、<=90%→25、>90%→10，适合人看/写策略规则"""
    if mode == 'linear':
        return 100.0 * (1.0 - pct)
    edges = PCT_DEFAULT_EDGES if edges is None else tuple(sorted(float(e) for e in edges))
    scores = PCT_DEFAULT_SCORES if scores is None else tuple(float(s) for s in scores)
    if len(scores) != len(edges) + 1:
        raise ValueError('scores 个数必须 = edges 个数 + 1（%d vs %d）'
                         % (len(scores), len(edges) + 1))
    if isinstance(pct, pd.DataFrame):
        out = pd.DataFrame(np.nan, index=pct.index, columns=pct.columns)
        for c in pct.columns:
            out[c] = pct_to_score(pct[c], 'band', edges, scores)
        return out
    s = pct if isinstance(pct, pd.Series) else pd.Series(np.asarray(pct, float))
    arr = s.astype(float).values
    res = np.full(len(arr), np.nan)
    ok = ~np.isnan(arr)
    v = arr[ok]
    band = np.full(v.shape, scores[-1], float)
    for i in range(len(edges) - 1, -1, -1):
        band[v <= edges[i]] = scores[i]
    res[ok] = band
    return pd.Series(res, index=s.index, name='pct_score')


def _lens_dev(close, ma_n=250):
    """镜头 2：对均线的偏离度（>0 在均线上方，<0 在下方）。"""
    ma = close.rolling(int(ma_n), min_periods=max(20, int(ma_n) // 2)).mean()
    return close / ma - 1.0


def _lens_dd(close, dd_window=750):
    """镜头 3：距区间高点的回撤（<=0，越负回撤越深）。"""
    hi = close.rolling(int(dd_window), min_periods=max(20, int(dd_window) // 2)).max()
    return close / hi - 1.0


def _weighted_score(parts, weights):
    """parts: {镜头名: 面板/序列}，weights: {镜头名: 权重}。逐行按可用镜头加权（缺的不算）。"""
    df = pd.DataFrame(parts)
    wv = pd.Series({k: float(weights.get(k, 0.0)) for k in df.columns})
    keep = wv[wv > 0].index
    if len(keep) == 0:
        raise ValueError('权重全为 0，没有可用镜头')
    df = df[keep]
    wv = wv[keep]
    num = df.mul(wv, axis=1).sum(axis=1, min_count=1)
    den = df.notna().mul(wv, axis=1).sum(axis=1)
    return num / den.where(den > 0)


def pct_composite(close, window=750, ma_n=250, dd_window=750, weights=None,
                  mode='linear', edges=None, scores=None):
    """时序版多镜头合成分（0~100，高=便宜）。close: Series(index=日期)。"""
    w = dict(PCT_DEFAULT_WEIGHTS)
    w.update(weights or {})
    parts = {
        'px': pct_to_score(pct_rank(close, window), mode, edges, scores),
        'dev': pct_to_score(pct_rank(_lens_dev(close, ma_n), window), mode, edges, scores),
        'dd': pct_to_score(pct_rank(_lens_dd(close, dd_window), dd_window), mode, edges, scores),
    }
    out = _weighted_score(parts, w)
    out.name = 'pct_score'
    return out


def pct_composite_last(close, window=750, ma_n=250, dd_window=750, weights=None,
                       mode='linear', edges=None, scores=None):
    """截面版多镜头合成分（0~100，高=便宜）。close: 宽表(index=日期, columns=代码)。"""
    w = dict(PCT_DEFAULT_WEIGHTS)
    w.update(weights or {})
    parts = {
        'px': pct_to_score(pct_last(close, window), mode, edges, scores),
        'dev': pct_to_score(pct_last(_lens_dev(close, ma_n), window), mode, edges, scores),
        'dd': pct_to_score(pct_last(_lens_dd(close, dd_window), dd_window), mode, edges, scores),
    }
    out = _weighted_score(parts, w)
    out.name = 'pct_score'
    return out


# ------------------------------------------------------------
# Part F-2 时序信号（0~100，高=便宜 → 用 run_ts_test(..., th=60)）
# ------------------------------------------------------------
@register_ts_signal('pct_750')
def ts_pct_750(code, px, params=None):
    """价格 750 日历史分位（0~100，高=便宜）。params: {'mode':'linear'|'band','window':750}"""
    p = params or {}
    pct = pct_rank(px['close'], int(p.get('window', 750)))
    return pct_to_score(pct, p.get('mode', 'linear'), p.get('edges'), p.get('scores'))


@register_ts_signal('pct_500')
def ts_pct_500(code, px, params=None):
    """价格 500 日历史分位（窗口短一点，更贴近一轮周期）"""
    p = params or {}
    pct = pct_rank(px['close'], int(p.get('window', 500)))
    return pct_to_score(pct, p.get('mode', 'linear'), p.get('edges'), p.get('scores'))


@register_ts_signal('pct_dev250')
def ts_pct_dev250(code, px, params=None):
    """对 250 日均线偏离度的历史分位（高=偏离度处在历史低位=便宜）"""
    p = params or {}
    pct = pct_rank(_lens_dev(px['close'], int(p.get('ma_n', 250))),
                   int(p.get('window', 750)))
    return pct_to_score(pct, p.get('mode', 'linear'), p.get('edges'), p.get('scores'))


@register_ts_signal('pct_dd_750')
def ts_pct_dd_750(code, px, params=None):
    """回撤深度分位（高=当前回撤处在历史上较深的位置=便宜）"""
    p = params or {}
    pct = pct_rank(_lens_dd(px['close'], int(p.get('dd_window', 750))),
                   int(p.get('window', 750)))
    return pct_to_score(pct, p.get('mode', 'linear'), p.get('edges'), p.get('scores'))


@register_ts_signal('pct_score')
def ts_pct_score(code, px, params=None):
    """多镜头合成分（价格+偏离+回撤，默认权重 0.5/0.3/0.2）。
    params: {'window':750,'ma_n':250,'dd_window':750,'weights':{'px':.5,...},'mode':'linear'}"""
    p = dict(params or {})
    return pct_composite(px['close'],
                         window=int(p.get('window', 750)),
                         ma_n=int(p.get('ma_n', 250)),
                         dd_window=int(p.get('dd_window', 750)),
                         weights=p.get('weights'),
                         mode=p.get('mode', 'linear'),
                         edges=p.get('edges'), scores=p.get('scores'))


# ------------------------------------------------------------
# Part F-3 截面因子（分数越高=越便宜 → 方向 +1）
# ------------------------------------------------------------
def _pct_factor(stock_list, date, window, lens, ma_n=250, dd_window=750,
                mode='linear', name='pct'):
    """截面分位打分通用实现。lens: 'px'/'dev'/'dd'/'all'。"""
    # 回撤镜头要两段历史：前一段算滚动最高价，后一段给回撤序列做分位
    need = int(window) + 1
    if lens in ('dd', 'all'):
        need = max(need, int(dd_window) * 2 + 2)
    close = get_price_history(stock_list, date, need, ['close']).get('close')
    if close is None or close.empty or len(close) < 30:
        return pd.Series(np.nan, index=list(stock_list), name=name)
    if lens == 'px':
        s = pct_to_score(pct_last(close, window), mode)
    elif lens == 'dev':
        s = pct_to_score(pct_last(_lens_dev(close, ma_n), window), mode)
    elif lens == 'dd':
        s = pct_to_score(pct_last(_lens_dd(close, dd_window), dd_window), mode)
    else:
        s = pct_composite_last(close, window=window, ma_n=ma_n,
                               dd_window=dd_window, mode=mode)
    s = s.reindex(stock_list)
    s.name = name
    return s


@register_factor('px_pct_750')
def factor_px_pct_750(stock_list, date):
    """价格 750 日历史分位打分（100=最便宜）。截面：低分位=长期低估，反转/价值回归。"""
    return _pct_factor(stock_list, date, 750, 'px', name='px_pct_750')


@register_factor('px_pct_500')
def factor_px_pct_500(stock_list, date):
    """价格 500 日历史分位打分（窗口短、样本多，换手会高一些）"""
    return _pct_factor(stock_list, date, 500, 'px', name='px_pct_500')


@register_factor('px_pct_dev250')
def factor_px_pct_dev250(stock_list, date):
    """对 MA250 偏离度的历史分位打分（比价格分位更"均值回复"）"""
    return _pct_factor(stock_list, date, 750, 'dev', name='px_pct_dev250')


@register_factor('px_pct_dd750')
def factor_px_pct_dd750(stock_list, date):
    """回撤深度分位打分（高分=当前回撤处在历史较深处）"""
    return _pct_factor(stock_list, date, 750, 'dd', name='px_pct_dd750')


@register_factor('px_pct_score')
def factor_px_pct_score(stock_list, date):
    """多镜头合成分（价格 0.5 / 偏离 0.3 / 回撤 0.2），截面版"""
    return _pct_factor(stock_list, date, 750, 'all', name='px_pct_score')


# ------------------------------------------------------------
# Part F-4 估值分位（股票专属，PE/PB）：逐期快照 → 面板 → 分位 → 打分
#   量价分位是"价格便宜"，估值分位才是"基本面便宜"。
#   聚宽取不到"逐日历史估值"的现成面板，所以自己按调仓日建快照面板。
# ------------------------------------------------------------
def build_valuation_panel(fields=('pe_ratio', 'pb_ratio'), rebal_dates=None,
                          pool='HS300', verbose=True, non_positive_to_nan=True):
    """按调仓日逐期取估值快照，拼成 {字段: 宽表(index=调仓日, columns=代码)}。
    pool 为指数键名时按每期成分取（防未来函数）；也可直接传代码列表。
    非正数（亏损股 PE、净资产为负 PB）置 NaN —— 分位对负值没有意义。"""
    if rebal_dates is None:
        raise ValueError('请传 rebal_dates（用 get_rebalance_dates 生成）')
    rebal_dates = list(rebal_dates)
    _, per_date = resolve_universe(pool, rebal_dates, verbose=verbose)
    fields = list(fields)
    out = {f: {} for f in fields}
    for i, d in enumerate(rebal_dates):
        codes = per_date.get(d) if per_date is not None else get_stock_pool(pool, d)
        if not codes:
            continue
        try:
            q = query(*([valuation.code] + [getattr(valuation, f) for f in fields])) \
                .filter(valuation.code.in_(list(codes)))
            df = get_fundamentals(q, d)
        except Exception as e:
            if verbose:
                print('  [估值] %s 取数失败: %s' % (d, e))
            continue
        if df is None or df.empty:
            continue
        df = df.set_index('code')
        for f in fields:
            s = df[f].astype(float) if f in df.columns else pd.Series(dtype=float)
            if non_positive_to_nan:
                s = s.where(s > 0)
            out[f][d] = s
        if verbose and (i + 1) % 12 == 0:
            print('  [估值] %d/%d 期' % (i + 1, len(rebal_dates)))
    panels = {f: pd.DataFrame(v).T.sort_index() for f, v in out.items()}
    if verbose:
        for f, p in panels.items():
            print('  估值面板 %-8s %d 期 × %d 只（覆盖率 %.1f%%）'
                  % (f, p.shape[0], p.shape[1], 100.0 * p.notna().mean().mean()))
    return panels


def run_pct_value_test(field='pe_ratio', pool='HS300', start='2019-01-01', end='2024-12-31',
                       window=36, period=60, freq='M', rule='first', q=5,
                       mode='linear', min_frac=0.5, verbose=True, panels=None):
    """估值历史分位打分 · 截面一键检验。
    field  : 'pe_ratio'（PE-TTM）/ 'pb_ratio' / 'market_cap'
    window : 分位窗口（期数）。freq='M' + window=36 → 过去 36 个月分位（3年）
    period : 持仓期，估值类建议 60 交易日
    返回 dict: name / summary / details / panels / fwd / value / pct / score
    """
    rebal_dates = get_rebalance_dates(start, end, freq=freq, rule=rule)
    if verbose:
        print('=' * 88)
        print('【估值分位打分】%s | %s | %d 个调仓日 | 分位窗口 %d 期'
              % (field, pool_desc(pool), len(rebal_dates), int(window)))
    val = build_valuation_panel([field], rebal_dates, pool, verbose=verbose)[field]
    pct = pct_rank(val, int(window), min_periods=max(6, int(int(window) * float(min_frac))))
    score = pct_to_score(pct, mode)
    pct.columns = [str(c) for c in pct.columns]
    score.columns = pct.columns
    name = '%s_pct_%dm' % (field.replace('_ratio', ''), int(window))
    use_panels = dict(panels or {})
    use_panels[name] = score
    df, details, out_panels, fwd = run_batch(
        [name], pool, start, end, period=period, freq=freq, rule=rule, q=q,
        rebal_dates=rebal_dates, panels=use_panels, verbose=verbose)
    return {'name': name, 'summary': df, 'details': details, 'panels': out_panels,
            'fwd': fwd, 'value': val, 'pct': pct, 'score': score}


# ------------------------------------------------------------
# Part F-5 一键 + 专用图：价格 / 分位 / 分档 / 滚动IC 四联图
# ------------------------------------------------------------
def run_pct_ts(code='000300.XSHG', start='2019-01-01', end='2024-12-31', signal='pct_score',
               th=60, period=20, decision_freq='W', cost=0.0005, params=None, verbose=True):
    """分位打分的时序一键检验（默认 th=60：过去 40% 分位以内才持有）。"""
    return run_ts_test(signal, code, start, end, period=period, th=th,
                       decision_freq=decision_freq, cost=cost, params=params,
                       verbose=verbose)


def _pct_rolling_ic(sig, fwd, win=120, step=5):
    """滚动 IC（Spearman 的滑动近似：秩相关逐窗算），用于看"什么时候好使"。"""
    df = pd.concat([pd.Series(sig).rename('s'), pd.Series(fwd).rename('r')], axis=1).dropna()
    if len(df) < win + 10:
        return pd.Series(dtype=float)
    idx, vals = [], []
    for i in range(win, len(df) + 1, max(1, int(step))):
        ch = df.iloc[i - win:i]
        c = ch['s'].corr(ch['r'], method='spearman')
        if not pd.isna(c):
            idx.append(ch.index[-1]); vals.append(c)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def plot_pct_ts(result, th=60, title=None, figsize=(12, 9), state_edges=None):
    """分位打分专用四联图：
    (1) 价格 + 持有区间阴影  (2) 打分 0~100 + 阈值 + 便宜/中性/贵色带
    (3) 分档未来收益（固定档位，看单调性）  (4) 滚动 IC（120 窗口）
    result 为 run_ts_test / run_pct_ts 的返回值。"""
    import matplotlib.pyplot as plt

    sig = pd.Series(result.get('signal')).dropna()
    close = pd.Series(result.get('close')).reindex(sig.index).dropna()
    fwd = pd.Series(result.get('fwd')).reindex(sig.index)
    detail = result.get('detail')
    pos = None
    if isinstance(detail, pd.DataFrame) and 'position' in detail.columns:
        pos = detail['position'].reindex(sig.index).ffill().fillna(0.0)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [3.0, 1.5, 1.2, 1.2]})
    ax1, ax2, ax3, ax4 = axes

    # --- (1) 价格 + 持有阴影
    ax1.plot(close.index, close.values, color='#1f3b57', lw=1.1, label='收盘价')
    if pos is not None:
        on = (pos > 0.5).astype(float)
        ax1.fill_between(close.index, close.min(), close.max(), where=on > 0,
                         color='#5aa9e6', alpha=0.20, step='pre', label='持有')
    ax1.set_title(title or '分位打分 · %s' % result.get('signal_name', ''), fontsize=12, loc='left')
    ax1.legend(loc='upper left', fontsize=8, frameon=False)
    ax1.grid(alpha=0.25)

    # --- (2) 打分 + 色带
    ax2.plot(sig.index, sig.values, color='#2a6f97', lw=1.0)
    ax2.axhspan(60, 100, color='#cfe8d5', alpha=0.55, lw=0)
    ax2.axhspan(40, 60, color='#f2ecd6', alpha=0.55, lw=0)
    ax2.axhspan(0, 40, color='#f4d6d6', alpha=0.55, lw=0)
    ax2.axhline(float(th), color='#b03a2e', lw=1.0, ls='--')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('打分(高=便宜)', fontsize=9)
    ax2.grid(alpha=0.25)

    # --- (3) 固定档位分档收益
    edges = PCT_DEFAULT_EDGES if state_edges is None else state_edges
    thr = sorted([100.0 * (1.0 - float(e)) for e in edges])   # 分位边界 → 分数边界（升序）
    tbl, _ = ts_state_analysis(sig, fwd, thresholds=thr, n_states=len(thr) + 1)
    if isinstance(tbl, pd.DataFrame) and not tbl.empty:
        rows = tbl.reset_index(drop=True)
        half = len(rows) / 2.0
        names = ['贵 %s' % rows.loc[i, '区间'] if i < half else '便宜 %s' % rows.loc[i, '区间']
                 for i in range(len(rows))]
        ax3.bar(range(len(tbl)), tbl['平均收益%'].fillna(0).values,
                color=['#c0392b' if v < 0 else '#2e7d32' for v in tbl['平均收益%'].fillna(0).values],
                alpha=0.75)
        ax3.set_xticks(range(len(tbl)))
        ax3.set_xticklabels(names, fontsize=7, rotation=20)
        ax3.axhline(0, color='#555555', lw=0.8)
        ax3.set_ylabel('未来收益%/期', fontsize=9)
        ax3.grid(alpha=0.2, axis='y')

    # --- (4) 滚动 IC
    ric = _pct_rolling_ic(sig, fwd)
    if len(ric):
        ax4.bar(ric.index, ric.values, width=4,
                color=['#c0392b' if v < 0 else '#2e7d32' for v in ric.values], alpha=0.6)
    ax4.axhline(0, color='#555555', lw=0.8)
    ax4.set_ylabel('滚动IC(120)', fontsize=9)
    ax4.grid(alpha=0.2, axis='y')

    fig.tight_layout()
    return fig


def pct_report(code='000300.XSHG', start='2019-01-01', end='2024-12-31',
               signal='pct_score', th=60, period=20, decision_freq='W', params=None,
               plot=True, save=False, outdir='pic_pct', figsize=(12, 9)):
    """一行跑完：时序检验 + 四联图（可选落盘）。返回 result dict。"""
    r = run_pct_ts(code=code, start=start, end=end, signal=signal, th=th,
                   period=period, decision_freq=decision_freq, params=params)
    r['signal_name'] = signal
    if plot:
        fig = plot_pct_ts(r, th=th, title='%s · %s · th=%s' % (signal, code, th),
                          figsize=figsize)
        if save:
            import os
            os.makedirs(outdir, exist_ok=True)
            p = os.path.join(outdir, 'pct_%s_%s.png' % (signal, str(code).split('.')[0]))
            fig.savefig(p, dpi=140, bbox_inches='tight')
            print('  图已保存: %s' % p)
        r['fig'] = fig
    return r


print('[Part F] 历史分位打分因子已就绪：时序 %s | 截面 %s'
      % ([k for k in TS_SIGNALS if k.startswith('pct')],
         [k for k in CUSTOM_FACTORS if 'pct' in k]))
