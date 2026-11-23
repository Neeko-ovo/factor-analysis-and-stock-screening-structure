# -*- coding: utf-8 -*-
"""
Part H 仓位管理（研究环境）
================================
一、底部打分（截面）：低估 + 低位 + 波动收敛 + 量能收缩 → bottom_score_panel
二、仓位状态机（时序）：便宜度 × 波动冷热 → weight_state
三、连续仓位回测：run_weight_test / plot_weight_test / weight_report

依赖：
  jq_unified_factor_framework.py（register_ts_signal / get_price_range / get_price_history /
                                 get_rebalance_dates / resolve_universe / trading_day_shift）
  jq_pct_factor.py（pct_rank / pct_last / pct_composite）
必须放在 jq_pct_factor.py 之后加载。

新增时序信号（可直接 run_ts_test / run_ts_batch）：
  atr_pct_20   ATR20 / close（原始值，越小越"冷"）
  atr_rank_20  ATR% 在过去 N 日的分位（0~1，越大越"热"）
  atr_squeeze  ATR20 / ATR60（<1 = 波动收敛）
  vol_ratio    20 日均成交额 / 250 日均成交额（<1 = 量能收缩）
  bw_rank      布林带宽度（4σ/MA20）在过去 N 日的分位
  rsi_14       RSI(14)
  mom_60       60 日动量
"""

import numpy as np
import pandas as pd


# ============================================================
# Part H-1 基础算子：TR / ATR / 序列分位
# ============================================================

def calc_tr(high, low, close):
    """真实波幅 TR = max(H-L, |H-C_prev|, |L-C_prev|)，支持 Series 与 DataFrame。"""
    prev = close.shift(1)
    a = high - low
    b = (high - prev).abs()
    c = (low - prev).abs()
    return np.maximum(np.maximum(a, b), c)


def calc_atr(close, high, low, n=20, method='ema'):
    """ATR。method='ema' 用 Wilder 平滑（alpha=1/n），'sma' 用简单均值。"""
    tr = calc_tr(high, low, close)
    n = int(n)
    if method == 'sma':
        return tr.rolling(n, min_periods=max(5, n // 2)).mean()
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=max(5, n // 2)).mean()


def atr_pct(close, high, low, n=20):
    """ATR / close：波动率口径，可直接跨标的比较。"""
    return calc_atr(close, high, low, n) / close


def _pos_last_rank(x, window=250, min_frac=0.6):
    """最后一个观测在最近 window 个观测里的分位（0~1），Series / DataFrame 都支持。"""
    if isinstance(x, pd.Series):
        s = x.dropna()
        win = s.tail(int(window))
        need = max(20, int(len(win) * float(min_frac)))
        if len(win) == 0 or len(win) < need:
            return np.nan
        return float((win <= win.iloc[-1]).sum()) / float(len(win))
    return pct_last(x, int(window), float(min_frac))


# ============================================================
# Part H-2 时序信号
# ============================================================

@register_ts_signal('atr_pct_20')
def ts_atr_pct_20(code, px, params=None):
    """ATR(20)/close，越大越"热"。"""
    p = dict(params or {})
    n = int(p.get('n', 20))
    return atr_pct(px['close'], px['high'], px['low'], n)


@register_ts_signal('atr_rank_20')
def ts_atr_rank_20(code, px, params=None):
    """ATR% 在最近 window 日的分位：0=最冷，1=最热。用于仓位缩放。"""
    p = dict(params or {})
    n = int(p.get('n', 20))
    win = int(p.get('window', 250))
    return pct_rank(atr_pct(px['close'], px['high'], px['low'], n), win)


@register_ts_signal('atr_squeeze')
def ts_atr_squeeze(code, px, params=None):
    """ATR20 / ATR60：<1 表示波动在收敛（VCP 的第一条件）。"""
    p = dict(params or {})
    n1 = int(p.get('n1', 20))
    n2 = int(p.get('n2', 60))
    a1 = calc_atr(px['close'], px['high'], px['low'], n1)
    a2 = calc_atr(px['close'], px['high'], px['low'], n2)
    return a1 / a2


@register_ts_signal('vol_ratio')
def ts_vol_ratio(code, px, params=None):
    """20 日均额 / 250 日均额：<1 = 缩量，>1.5 = 放量。"""
    p = dict(params or {})
    n1 = int(p.get('n1', 20))
    n2 = int(p.get('n2', 250))
    m = px['money']
    short = m.rolling(n1, min_periods=max(5, n1 // 2)).mean()
    long = m.rolling(n2, min_periods=max(20, n2 // 2)).mean()
    return short / long.replace(0, np.nan)


@register_ts_signal('bw_rank')
def ts_bw_rank(code, px, params=None):
    """布林带宽度 4σ/MA20 的滚动分位：越小越"憋"，突破前的典型形态。"""
    p = dict(params or {})
    n = int(p.get('n', 20))
    win = int(p.get('window', 250))
    c = px['close']
    ma = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std()
    return pct_rank(4.0 * sd / ma, win)


@register_ts_signal('rsi_14')
def ts_rsi_14(code, px, params=None):
    """RSI(14)，0~100。这里只给连续值；超买>75 / 超卖<25 的分区与反转事件见 Part I
    （rsi_zone / rsi_reversal_state），要改周期请用信号 'rsi' + params={'n': 14}。"""
    p = dict(params or {})
    n = int(p.get('n', 14))
    c = px['close']
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(dn > 0, 100.0)
    rsi = rsi.where(~((dn <= 0) & (up <= 0)), 50.0)
    return rsi


@register_ts_signal('mom_60')
def ts_mom_60(code, px, params=None):
    """60 日动量，用于风格/相对强弱的排序。"""
    p = dict(params or {})
    n = int(p.get('n', 60))
    return px['close'] / px['close'].shift(n) - 1.0


# ============================================================
# Part H-3 截面：底部区域打分
# ============================================================

def _pos_codes_on(per_date, d, all_stocks):
    """resolve_universe 的 per_date 键可能是 date / Timestamp；list 池时 per_date 为 None。"""
    if per_date is None:
        return all_stocks
    for key in (d, pd.Timestamp(d)):
        try:
            if key in per_date:
                return per_date[key]
        except TypeError:
            continue
    return all_stocks


def bottom_score_panel(pool='HS300', start=None, end=None, rebal_dates=None,
                       freq='M', rule='first', every=1,
                       window=750, atr_n=20, vol_n1=20, vol_n2=250,
                       val_panel=None, w_val=0.35, w_px=0.25, w_atr=0.20, w_vol=0.20,
                       min_frac=0.6, verbose=True):
    """底部区域打分面板：0~100，越高越该建仓（低估 + 低位 + 波动收敛 + 量能收缩）。

    参数
    ----
    pool / start / end / freq / rule / every : 同其它面板函数
    rebal_dates : 也可直接传调仓日
    window      : 价格位置/回撤的分位窗口（交易日），默认 750
    atr_n       : ATR 周期，默认 20
    val_panel   : 可选估值面板 {字段: 宽表}，或单个宽表。给了就用「1-估值分位」当低估镜头；
                  不给就用「回撤分位」当代理（1 - dd_rank），并在权重里把它归到 px/atr/vol 三镜头重归一。
    w_val/w_px/w_atr/w_vol : 四镜头权重；缺 val 时按剩余三项重归一。

    返回
    ----
    DataFrame(index=调仓日, columns=代码, values=0~100)
    """
    if rebal_dates is None:
        if start is None or end is None:
            raise ValueError('请给 start/end 或 rebal_dates')
        rebal_dates = get_rebalance_dates(start, end, freq=freq, rule=rule, every=every)
    rebal_dates = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in rebal_dates])
    if len(rebal_dates) == 0:
        return pd.DataFrame()

    all_stocks, per_date = resolve_universe(pool, rebal_dates, verbose=verbose)
    stocks = list(all_stocks)
    need = int(window) + int(max(atr_n, vol_n2)) + 60

    out = pd.DataFrame(index=rebal_dates, columns=stocks, dtype=float)
    use_val = val_panel is not None
    for d in rebal_dates:
        codes = list(_pos_codes_on(per_date, d, stocks))
        if len(codes) < 2:
            continue
        try:
            hist = get_price_history(codes, d, need, ('close', 'high', 'low', 'money'))
        except Exception as e:
            if verbose:
                print('  [底部打分] %s 取数失败：%s' % (pd.Timestamp(d).date(), e))
            continue
        cl = hist.get('close')
        if cl is None or cl.shape[0] < 80:
            continue
        hi = hist.get('high')
        lo = hist.get('low')
        mo = hist.get('money')

        px_rank = pct_last(cl, int(window), float(min_frac))
        dd_win = cl.rolling(int(window), min_periods=max(20, int(window * min_frac))).max()
        dd = cl / dd_win - 1.0
        dd_rank = pct_last(dd, int(window), float(min_frac))

        ap = atr_pct(cl, hi, lo, atr_n) if hi is not None and lo is not None else None
        atr_rank = pct_last(ap, min(250, max(60, ap.shape[0])), float(min_frac)) if ap is not None else None

        if mo is not None:
            vr = (mo.rolling(vol_n1, min_periods=max(5, vol_n1 // 2)).mean()
                  / mo.rolling(vol_n2, min_periods=max(20, vol_n2 // 2)).mean().replace(0, np.nan))
            vr_last = vr.iloc[-1]
            vol_part = 1.0 - np.minimum(vr_last.clip(lower=0), 1.5) / 1.5
        else:
            vol_part = None

        val_ok = False
        parts = {}
        if use_val:
            vp = val_panel
            if isinstance(vp, dict):
                vp = list(vp.values())[0]
            vp = vp.loc[vp.index <= pd.Timestamp(d)]
            if len(vp) >= 6:
                val_rank = vp.rank(pct=True).iloc[-1]
                parts['val'] = (w_val, 1.0 - val_rank)
                val_ok = True
        if atr_rank is not None:
            parts['atr'] = (w_atr, 1.0 - atr_rank)
        if vol_part is not None:
            parts['vol'] = (w_vol, vol_part)
        parts['px'] = (w_px, 1.0 - px_rank)
        if not val_ok:
            parts['dd'] = (w_val, 1.0 - dd_rank)   # 没有估值时，用回撤分位顶替「低估」镜头

        tot = sum(w for w, _ in parts.values() if w > 0)
        if tot <= 0:
            continue
        score = None
        for w, v in parts.values():
            if w <= 0:
                continue
            term = (w / tot) * v
            score = term if score is None else score + term
        out.loc[pd.Timestamp(d)] = (score * 100.0).reindex(out.columns)

    if verbose:
        cov = float(np.isfinite(out.values).sum()) / max(1, out.shape[0] * out.shape[1])
        print('  [底部打分] %d 期 × %d 只，覆盖率 %.1f%%（越高越该建仓；缺估值时 val 镜头自动省略）'
              % (out.shape[0], out.shape[1], cov * 100))
    return out


def bottom_candidates(score_panel, top=5, min_score=60.0, verbose=False):
    """从打分面板里挑每期的候选：分数 ≥ min_score 且排前 top。返回 DataFrame（每期一行）。"""
    rows = {}
    for d, row in score_panel.iterrows():
        s = row.dropna()
        s = s[s >= float(min_score)].sort_values(ascending=False)
        if len(s) == 0:
            rows[d] = {}
            continue
        rows[d] = dict((k, float(v)) for k, v in s.head(int(top)).items())
    if verbose:
        print('  [底部候选] %d 期中 %d 期有候选' % (len(rows), sum(1 for v in rows.values() if v)))
    return pd.DataFrame.from_dict(rows, orient='index')


# ============================================================
# Part H-4 时序：仓位状态机
# ============================================================

def weight_state(cheap, atr_rank, close=None, ma=None, stop=None,
                 cheap_low=30.0, cheap_high=70.0, cold=0.5, hot=0.7,
                 step=0.25, min_change=0.20, w_max=1.0):
    """仓位状态机：便宜度（0~100，越高越便宜）× 波动冷热（0~1）→ 目标仓位，再走步长 + 迟滞。

    档位表（目标仓位）
    ------------------
    便宜(>=cheap_high) × 冷(<cold)      → 1.00      便宜 × 温            → 0.70
    便宜 × 热(>=hot)                    → 0.40      中性 × 冷            → 0.80
    中性 × 温                           → 0.60      中性 × 热            → 0.40
    贵(<=cheap_low) × 冷                → 0.60      贵 × 温              → 0.40    贵 × 热 → 0.20

    覆盖规则
    --------
    close < ma 且 atr_rank >= 0.6      → 最多 0.20（破位降档）
    close < stop                       → 直接 0（ATR 移动止损，跳过步长限制）

    执行规则
    --------
    目标与当前差 >= min_change 才动；单次变动不超过 step；止损直接清零。
    """
    idx = pd.DatetimeIndex(cheap.index)
    c = cheap.reindex(idx).astype(float)
    a = atr_rank.reindex(idx).ffill().astype(float)

    cv = c.values
    av = a.values
    tgt = np.full(len(idx), np.nan)

    m_cheap = cv >= float(cheap_high)
    m_mid = (cv > float(cheap_low)) & (cv < float(cheap_high))
    m_rich = cv <= float(cheap_low)
    m_cold = av < float(cold)
    m_hot = av >= float(hot)
    m_warm = (~m_cold) & (~m_hot)

    tgt[m_cheap & m_cold] = 1.00
    tgt[m_cheap & m_warm] = 0.70
    tgt[m_cheap & m_hot] = 0.40
    tgt[m_mid & m_cold] = 0.80
    tgt[m_mid & m_warm] = 0.60
    tgt[m_mid & m_hot] = 0.40
    tgt[m_rich & m_cold] = 0.60
    tgt[m_rich & m_warm] = 0.40
    tgt[m_rich & m_hot] = 0.20

    if close is not None and ma is not None:
        cls = close.reindex(idx).astype(float)
        mas = ma.reindex(idx).astype(float)
        weak = ((cls < mas) & (a >= 0.6)).fillna(False).values
        tgt = np.where(weak, np.minimum(tgt, 0.20), tgt)
    else:
        weak = np.zeros(len(idx), dtype=bool)

    brk = np.zeros(len(idx), dtype=bool)
    if stop is not None and close is not None:
        st = stop.reindex(idx).astype(float)
        cls = close.reindex(idx).astype(float)
        brk = (cls < st).fillna(False).values

    out = np.zeros(len(idx), dtype=float)
    cur = 0.0
    for i in range(len(idx)):
        if brk[i]:
            cur = 0.0
            out[i] = 0.0
            continue
        t = tgt[i]
        if not np.isfinite(t):
            out[i] = cur
            continue
        t = float(max(0.0, min(w_max, t)))
        diff = t - cur
        if abs(diff) >= float(min_change):
            cur = cur + float(np.clip(diff, -step, step))
        out[i] = cur
    return pd.Series(out, index=idx, name='weight')


def atr_stop_line(close, high, n=22, k=3.0, low=None):
    """ATR 移动止损线 = 近 n 日最高 - k×ATR(n)。收盘跌破就走。"""
    atr = calc_atr(close, high, close if low is None else low, n)
    return high.rolling(int(n), min_periods=max(5, int(n) // 2)).max() - float(k) * atr


def cheap_score_series(close, window=750, ma_n=250, dd_window=750, weights=None):
    """便宜度 0~100（高=便宜）：价格分位 + 偏离分位 + 回撤分位合成，复用 Part F 的 pct_composite。"""
    return pct_composite(close, window=int(window), ma_n=int(ma_n),
                         dd_window=int(dd_window), weights=weights)


# ============================================================
# Part H-5 连续仓位回测
# ============================================================

def _weight_metrics(strat, pos, bench_ret, cost):
    """内部：算收益/波动/夏普/回撤/Calmar/换手/成本。"""
    strat = strat.fillna(0.0)
    n = len(strat)
    nav = (1.0 + strat).cumprod()
    years = max(n / 244.0, 1e-9)
    ann = nav.iloc[-1] ** (1.0 / years) - 1.0 if n > 0 else np.nan
    vol = float(strat.std() * np.sqrt(244.0))
    sharpe = float(strat.mean() / strat.std() * np.sqrt(244.0)) if strat.std() > 0 else np.nan
    mdd = float((nav / nav.cummax() - 1.0).min())
    calmar = float(ann / abs(mdd)) if mdd < 0 else np.nan
    turn = float(pos.diff().abs().sum() / years) if n > 1 else np.nan
    drag = float(turn * float(cost))
    bn = (1.0 + bench_ret.fillna(0.0)).cumprod()
    b_ann = bn.iloc[-1] ** (1.0 / years) - 1.0 if n > 0 else np.nan
    b_mdd = float((bn / bn.cummax() - 1.0).min())
    b_sharpe = (float(bench_ret.mean() / bench_ret.std() * np.sqrt(244.0))
                if bench_ret.std() > 0 else np.nan)
    expo = float(pos.mean())
    eq = (1.0 + expo * bench_ret.fillna(0.0)).cumprod()
    eq_ann = eq.iloc[-1] ** (1.0 / years) - 1.0 if n > 0 else np.nan
    eq_mdd = float((eq / eq.cummax() - 1.0).min())
    eq_sharpe = (float((expo * bench_ret).mean() / (expo * bench_ret).std() * np.sqrt(244.0))
                 if expo > 0 and bench_ret.std() > 0 else np.nan)
    return {
        '年化%': 100.0 * ann, '波动%': 100.0 * vol, '夏普': sharpe,
        '最大回撤%': 100.0 * mdd, 'Calmar': calmar,
        '平均仓位%': 100.0 * expo, '年换手': turn, '成本拖累%/年': 100.0 * drag,
        '单位暴露年化%': (100.0 * ann / expo) if expo > 0 else np.nan,
        '基准年化%': 100.0 * b_ann, '基准夏普': b_sharpe, '基准回撤%': 100.0 * b_mdd,
        '等暴露年化%': 100.0 * eq_ann, '等暴露夏普': eq_sharpe, '等暴露回撤%': 100.0 * eq_mdd,
        '夏普差': sharpe - b_sharpe, '回撤收窄pp': 100.0 * (mdd - b_mdd),
        '等暴露年化差pp': 100.0 * (ann - eq_ann),
        '等暴露回撤收窄pp': 100.0 * (mdd - eq_mdd),
    }


def run_weight_test(weight, code, start, end, cost=0.0005, decision_freq='W',
                    bench_ret=None, verbose=True):
    """连续仓位回测（0~1 仓位，不是 0/1）。

    weight        : Series(index=日期, values 0~1)，缺的日期会 ffill
    decision_freq : 'D' 每日调仓 / 'W' 周频（取每周最后一个值）/ 'M' 月频
    bench_ret     : 外部基准日收益（可选），不给就用标的自身买入持有

    返回 dict：{'nav','bench_nav','eq_nav','strat','pos','weight','metrics','code'}
    """
    px = get_price_range([code], start, end, ('close',))
    cl = px['close'][code].dropna()
    if len(cl) < 40:
        raise ValueError('区间太短或代码取不到行情：%s' % code)

    w = weight.reindex(cl.index).ffill().fillna(0.0).clip(0.0, 1.0)
    if decision_freq == 'W':
        w = w.resample('W').last().reindex(cl.index, method='ffill').fillna(0.0)
    elif decision_freq == 'M':
        w = w.resample('M').last().reindex(cl.index, method='ffill').fillna(0.0)

    pos = w.shift(1).fillna(0.0)          # 次日生效，防未来函数
    ret = cl.pct_change().fillna(0.0)
    dpos = pos.diff().abs()
    if len(dpos):
        dpos.iloc[0] = abs(pos.iloc[0])
    strat = pos * ret - dpos * float(cost)
    bench = ret if bench_ret is None else bench_ret.reindex(cl.index).fillna(0.0)
    eq = (1.0 + pos.mean() * bench).cumprod()

    m = _weight_metrics(strat, pos, bench, cost)
    res = {'code': code, 'weight': w, 'pos': pos, 'strat': strat,
           'nav': (1.0 + strat).cumprod(),
           'bench_nav': (1.0 + bench).cumprod(),
           'eq_nav': eq, 'close': cl, 'metrics': m, 'cost': cost}
    if verbose:
        print('  [仓位回测] %s | 年化 %.1f%% 夏普 %.2f 回撤 %.1f%% 平均仓位 %.0f%%'
              % (code, m['年化%'], m['夏普'], m['最大回撤%'], m['平均仓位%']))
        print('             对比：买入持有 年化 %.1f%% 夏普 %.2f 回撤 %.1f%%；等暴露 年化 %.1f%% 夏普 %.2f 回撤 %.1f%%'
              % (m['基准年化%'], m['基准夏普'], m['基准回撤%'],
                 m['等暴露年化%'], m['等暴露夏普'], m['等暴露回撤%']))
        print('             换手 %.2f 次/年，成本拖累 %.2f%%/年；夏普差 %+.2f，回撤收窄 %+.1fpp'
              % (m['年换手'], m['成本拖累%/年'], m['夏普差'], m['回撤收窄pp']))
    return res


@_styled
def plot_weight_test(res, cheap=None, atr_rank=None, title=None, figsize=(12.5, 9.0)):
    """三联图：价格 + 仓位 / 便宜度与波动冷热 / 净值对比。"""
    cl = res['close']
    pos = res['pos'].reindex(cl.index).ffill()
    nav = res['nav']
    bn = res['bench_nav']
    eq = res['eq_nav']
    code = res.get('code', '')

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [2.0, 1.0, 1.4]})
    ax = axes[0]
    ax.plot(cl.index, cl.values, color='#1f77b4', lw=1.2, label='收盘价')
    ax.set_ylabel('价格')
    ax2 = ax.twinx()
    ax2.fill_between(pos.index, 0, pos.values, color='#2ca02c', alpha=0.22, step='post')
    ax2.plot(pos.index, pos.values, color='#2ca02c', lw=1.0, drawstyle='steps-post')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('仓位')
    ax.set_title('%s 仓位管理回测' % (title or code))
    ax.grid(alpha=0.25)

    ax = axes[1]
    if cheap is not None:
        ax.plot(cheap.index, cheap.values, color='#d62728', lw=1.0, label='便宜度(0~100)')
        ax.axhline(70, color='#d62728', ls='--', lw=0.8, alpha=0.6)
        ax.axhline(30, color='#ff7f0e', ls='--', lw=0.8, alpha=0.6)
    if atr_rank is not None:
        ax2 = ax.twinx()
        ax2.plot(atr_rank.index, atr_rank.values, color='#7f7f7f', lw=0.9, alpha=0.8)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel('波动冷热', color='#7f7f7f')
    ax.set_ylabel('便宜度')
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(nav.index, nav.values, color='#d62728', lw=1.4, label='策略')
    ax.plot(bn.index, bn.values, color='#7f7f7f', lw=1.1, ls='--', label='买入持有')
    ax.plot(eq.index, eq.values, color='#1f77b4', lw=1.0, ls=':', label='等暴露基准')
    ax.set_ylabel('净值')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig


def weight_report(code='000300.XSHG', start='2016-01-01', end='2024-12-31',
                  cost=0.0005, decision_freq='W', window=750, atr_n=20,
                  stop_n=22, stop_k=3.0, plot=True, verbose=True):
    """一键：便宜度 + 波动冷热 → 状态机仓位 → 连续仓位回测。

    这是「指数/ETF 择时仓位」的最小闭环，先在这个函数上调阈值，再谈组合。
    """
    px = get_price_range([code], start, end, ('close', 'high', 'low'))
    cl = px['close'][code].dropna()
    hi = px['high'][code].reindex(cl.index).ffill()
    cheap = cheap_score_series(cl, window=window).reindex(cl.index)
    ap = atr_pct(cl, hi, cl, atr_n)
    arank = pct_rank(ap, 250)
    stop = atr_stop_line(cl, hi, stop_n, stop_k)
    ma = cl.rolling(200, min_periods=120).mean()

    w = weight_state(cheap, arank, close=cl, ma=ma, stop=stop)
    res = run_weight_test(w, code, start, end, cost=cost,
                          decision_freq=decision_freq, verbose=verbose)
    res['cheap'] = cheap
    res['atr_rank'] = arank
    res['stop'] = stop
    if plot:
        res['fig'] = plot_weight_test(res, cheap=cheap, atr_rank=arank, title=code)
    return res


print('[Part H] 仓位管理已就绪：atr_pct_20 / atr_rank_20 / atr_squeeze / vol_ratio / bw_rank / rsi_14 / mom_60'
      ' · bottom_score_panel / bottom_candidates · weight_state / run_weight_test / weight_report')
