# -*- coding: utf-8 -*-
# ============================================================
# 时序有效性分析补丁 v2.2（单标的：走势 / 滚动IC / 状态 / 换仓 / 年度）
#   v2.1 → v2.2 改动（全部为纠错，不改你的判读口径）：
#     [修1] 缓冲段前视：close 取到 end+period*3+40 是为了算前瞻收益，
#           但年度表/换手/净值也用了它 → 多算出未来年份。现在统计全部截到 end。
#     [修2] tsa_build 支持 verbose=False 透传给框架，不再刷屏。
#     [修3] "IC(抽稀)" 门槛由 >10 放宽到 >5，period 大时不再一律 NaN。
#     [修4] 滚动 IC 只算一次（缓存进 store），不再重复计算。
#     [修5] 走势图均线加 min_periods，信号有缺口时不再整段断线。
#     [修6] 状态分组保留 Categorical 排序，负边界不再乱序。
#     [修7] 字体候选把 Microsoft YaHei 提到首位（Windows 常见有、SimHei 常缺）。
#     [修8] 走势图底栏注明"信号状态(未滞后)"，别和回测仓位混。
#     [修9] 所有绘图函数 return fig，可以自己 savefig / 落盘。
#     [修10] 阈值标签统一 %.10g，60 与 60.0 不再混排。
#     [新增] 信号首个有效日比 start 晚太多时给出提醒（start 要留信号窗口）。
#   依赖：时序框架 run_ts_test 已运行；研究环境专用
#   用法：tsa_report('pullback_w', ['512890.XSHG','510880.XSHG'], S, E,
#                    period=150, th=60, th_list=(50,60,80), edges=(45,60,80))
# ============================================================
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
                                   'Noto Sans CJK SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _tsa_th(th):
    """[修10] 阈值统一成可读字符串：60 / 60.0 / 59.5 都不重复小数零。"""
    try:
        return ('%.10g' % float(th))
    except Exception:
        return str(th)

# ==================== 0. 基础工具 ====================
def _tsa_close(code, td):
    """前复权收盘价，对齐到交易日轴。"""
    if not td:
        return pd.Series(dtype=float)
    raw = get_price(code, start_date=str(pd.Timestamp(td[0]).date()),
                    end_date=str(pd.Timestamp(td[-1]).date()), frequency='daily',
                    fields=['close'], fq='pre', skip_paused=False, panel=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    s = raw['close'] if 'close' in raw.columns else raw.iloc[:, 0]
    s.index = pd.to_datetime(s.index).normalize()
    s = s.astype(float)
    s = s[~s.index.duplicated(keep='last')]
    return s.reindex(pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in td]))

def _tsa_nw_t(vals, lag):
    """HAC（Newey-West，Bartlett 核）t 值：检验序列均值是否显著非零。"""
    v = pd.Series(vals).dropna().values.astype(float)
    n = len(v)
    if n < 8:
        return np.nan
    e = v - v.mean()
    L = max(1, min(int(lag), max(1, n // 3)))
    S = (e * e).mean()
    for k in range(1, L + 1):
        S += 2.0 * (1.0 - k / (L + 1.0)) * (e[k:] * e[:-k]).mean()
    return float(v.mean() / math.sqrt(max(S, 1e-14) / n))

def _tsa_pos(d, th):
    """★ 唯一的仓位口径（修复3）：sig(ffill) > th → 次日持有（shift 1）。
    返回 (pos, turn, ret)：pos=持仓0/1，turn=切换（每次=1），ret=当日收益。"""
    close = d.get('close')
    if close is None or close.empty:
        return None, None, None
    ret = close.pct_change().fillna(0.0)
    sig = pd.Series(d['signal']).astype(float).reindex(ret.index).ffill()
    pos = (sig > float(th)).astype(float).shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos)
    return pos, turn, ret

def _tsa_state_ttest(sig, fwd, th, step, lag=None):
    """★ 状态条件均值差检验（修复1）：抽稀→OLS fwd=a+b·1{状态内}→对 b 做 NW 检验。
    返回 (b, t, n)；样本不足返回 (nan, nan, n)。"""
    both = pd.concat([sig, fwd], axis=1).dropna()
    thin = both.iloc[::max(1, int(step))]
    n = len(thin)
    if n < 8:
        return np.nan, np.nan, n
    y = thin.iloc[:, 1].values.astype(float)
    x = (thin.iloc[:, 0].values.astype(float) > float(th)).astype(float)
    if x.std() == 0 or (x == 1).sum() == 0 or (x == 0).sum() == 0:
        return np.nan, np.nan, n
    xc = x - x.mean()
    sxx = (xc * xc).mean()
    a0 = y[x == 0].mean()
    b = float(y[x == 1].mean() - a0)
    e = y - (a0 + b * x)
    z = xc * e
    L = min(max(1, int(lag if lag else step)), max(1, n // 3))
    S = (z * z).mean()
    for k in range(1, L + 1):
        S += 2.0 * (1.0 - k / (L + 1.0)) * (z[k:] * z[:-k]).mean()
    se = math.sqrt(max(S, 1e-18) / n) / sxx if sxx > 0 else np.nan
    return b, (b / se if se == se and se > 0 else np.nan), n

def _tsa_ppf(p):
    """标准正态分位（Acklam 近似，免 scipy）。"""
    a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269,
         -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197,
         -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

# ==================== 1. 数据构建 ====================
def tsa_build(signal, code, start, end, period=20, th=60, params=None,
              decision_freq='D', cost=0.0005, direction=1, verbose=False):
    """取回框架结果（signal/fwd/backtest），补齐 close（含前瞻缓冲段）。

    ★[修1] 关键：close 要往 end 之后再取一段，才够算 period 日前瞻收益；
      但所有绩效统计（年度表/换手/净值）只能到 end，否则就是拿未来数据算历史。
      所以这里拆成两份：
        close     —— 截到 end，供 _tsa_pos / 年度表 / 净值图使用
        close_ext —— 含缓冲段，只给 fwd 用
    ★[修2] verbose 透传给框架的 run_ts_test，tsa_report(verbose=False) 才能真正安静。
    """
    f = globals().get('run_ts_test')
    if f is None:
        raise RuntimeError('未找到 run_ts_test —— 请先运行时序框架主模块，再运行本补丁。')
    r = f(signal, code, start, end, period=period, th=th,
          decision_freq=decision_freq, cost=cost, direction=direction,
          params=params, verbose=verbose)
    sig = pd.Series(r['signal']).astype(float)
    sig.index = pd.to_datetime(sig.index).normalize()
    sig = sig[~sig.index.duplicated(keep='last')].sort_index()
    end_ts = pd.Timestamp(end).normalize()
    td = list(get_trade_days(start_date=pd.Timestamp(start).date(),
              end_date=(end_ts + pd.Timedelta(days=int(period) * 3 + 40)).date()))
    close_ext = _tsa_close(code, td)
    if close_ext is not None and not close_ext.empty:
        sig = sig.reindex(close_ext.index)
        close = close_ext.loc[:end_ts]
    else:
        close = close_ext
    if close is not None and not close.empty:
        sig = sig.loc[:end_ts]

    fwd = r.get('fwd')
    if fwd is not None and len(fwd) > 0:
        fwd = pd.Series(fwd).astype(float)
        fwd.index = pd.to_datetime(fwd.index).normalize()
        fwd = fwd.reindex(close_ext.index) if (close_ext is not None and not close_ext.empty) else fwd
    elif close_ext is not None and not close_ext.empty:
        fwd = close_ext.shift(-int(period)) / close_ext - 1.0
    else:
        fwd = pd.Series(np.nan, index=sig.index)
    if len(fwd) > 0:
        fwd = fwd.loc[:end_ts]          # [修1] 只留 end 之前（尾部 NaN 各函数会自己 dropna）
    return {'signal': sig, 'fwd': fwd, 'close': close, 'close_ext': close_ext, 'code': code,
            'period': int(period), 'th': th, 'cost': float(cost),
            'backtest': r.get('backtest'), 'raw': r}

# ==================== 2. 信号走势 ====================
def tsa_signal_plot(d, th=None, smooth=20, last_n=None, title=None):
    """信号走势 + 阈值线 + 在场阴影（换手来源一眼看清）。[修9] 返回 fig。"""
    sig = pd.Series(d['signal']).dropna()
    if last_n:
        sig = sig.iloc[-int(last_n):]
    th = d['th'] if th is None else th
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    ax = axes[0]
    ax.plot(sig.index, sig.values, lw=0.8, color='#8fb3d9', label='信号')
    # [修5] min_periods：信号有缺口时均线不再整段断掉
    ax.plot(sig.index, sig.rolling(int(smooth), min_periods=max(1, int(smooth) // 3)).mean(),
            lw=1.6, color='seagreen', label='%d期均线' % int(smooth))
    if th is not None:
        ax.axhline(th, color='firebrick', lw=1.0, ls='--', label='th=%s' % _tsa_th(th))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(title or ('%s 信号走势 | 阈值上下摆动越多 → 换手越高' % d['code']))
    ax2 = axes[1]
    if th is not None:
        s2 = sig.ffill()
        pos = (s2 > th).astype(float)      # 注意：这里是"信号状态"，不是回测仓位（回测次日生效）
        ax2.fill_between(s2.index, 0, pos.values, step='post', color='seagreen', alpha=0.45)
        ax2.plot(s2.index, pos.values, lw=0.6, color='seagreen')
    ax2.set_yticks([0, 1])
    # [修8] 明说未滞后，避免和 _tsa_pos 的次日生效仓位差一天被误读
    ax2.set_yticklabels(['空仓(信号)', '在场(信号)'])
    ax2.set_ylim(-0.05, 1.05); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return fig

# ==================== 3. 滚动 IC ====================
def tsa_ic_rolling(sig, fwd, win=250, step=5):
    """滚动窗口 Rank IC（win 天窗、step 天步长）。
    ※ 窗内含前瞻收益 → 只能做事后诊断（看有效性时段），不能当交易信号。"""
    both = pd.concat([sig, fwd], axis=1).dropna()
    if len(both) < int(win) + 20:
        print('  （样本不足，无法算滚动 IC：需 >%d 个观测）' % (int(win) + 20))
        return pd.Series(dtype=float)
    x = both.iloc[:, 0].values; y = both.iloc[:, 1].values; idx = both.index
    out = {}
    for i in range(int(win), len(both) + 1, max(1, int(step))):
        xs = pd.Series(x[i - int(win):i]); ys = pd.Series(y[i - int(win):i])
        if xs.std() == 0 or ys.std() == 0:
            continue
        out[idx[i - 1]] = xs.rank().corr(ys.rank())
    s = pd.Series(out).sort_index(); s.name = 'rolling_IC'
    return s

def tsa_ic_plot(ric, smooth=3, title=None):
    """滚动 IC 曲线：长期 >0 = 持续有效；时正时负 = 周期性。[修9] 返回 fig。"""
    if ric is None or ric.dropna().empty:
        print('  （滚动 IC 为空，跳过）'); return None
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(ric.index, ric.values, lw=1.0, color='#9bb8d9', label='滚动IC')
    ax.plot(ric.index, ric.rolling(int(smooth)).mean(), lw=1.7, color='seagreen',
            label='%d点平滑' % int(smooth))
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axhline(ric.mean(), color='firebrick', lw=0.8, ls=':',
               label='均值 %.3f（正占比 %.0f%%）' % (ric.mean(), 100 * (ric > 0).mean()))
    ax.set_title(title or '滚动窗口 IC（事后诊断线）')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return fig

def tsa_ic_heatmap(ric, title=None):
    """滚动 IC 的月度均值热度图（红=负, 绿=正）。[修9] 返回 fig。"""
    r = ric.dropna() if ric is not None else pd.Series(dtype=float)
    if len(r) < 30:
        print('  （滚动 IC 太少，跳过热度图）'); return None
    df = r.to_frame('ic'); df['y'] = df.index.year; df['m'] = df.index.month
    piv = df.pivot_table(index='y', columns='m', values='ic', aggfunc='mean').reindex(columns=range(1, 13))
    if piv.empty:
        print('  （热度图无数据）'); return None
    v = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1.0
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(piv) + 1.8))
    im = ax.imshow(piv.values, cmap='RdYlGn', vmin=-v, vmax=v, aspect='auto')
    ax.set_xticks(range(12)); ax.set_xticklabels(['%d月' % i for i in range(1, 13)], fontsize=8)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if piv.values[i, j] == piv.values[i, j]:
                ax.text(j, i, '%.2f' % piv.values[i, j], ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title or '滚动 IC 月度均值（窗口跨月仅作定性参考；空白=该月无滚动IC）')
    plt.tight_layout(); plt.show()
    return fig

def tsa_ic_dist(ric, win=250, step=5, bins=25, title=None):
    """滚动 IC 分布直方图 + QQ 图（修复4：重叠比例按实际 win/step 提示）。[修9] 返回 fig。"""
    r = ric.dropna() if ric is not None else pd.Series(dtype=float)
    if len(r) < 20:
        print('  （滚动 IC 太少，跳过分布图）'); return None
    overlap = max(0.0, (int(win) - int(step)) / float(max(1, int(win))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.hist(r.values, bins=int(bins), density=True, color='#9bb8d9', edgecolor='white')
    xs = np.linspace(r.min(), r.max(), 150)
    ax.plot(xs, np.exp(-0.5 * ((xs - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi)),
            'r-', lw=1.4)
    ax.set_title('滚动 IC 分布（红线=正态拟合）'); ax.grid(alpha=0.3)
    ax = axes[1]
    s = r.sort_values(); n = len(s)
    ps = (np.arange(1, n + 1) - 0.5) / n
    theo = s.mean() + s.std() * np.array([_tsa_ppf(p) for p in ps])
    ax.plot(theo, s.values, 'o', ms=3, color='#2b7bba')
    lo, hi = min(theo.min(), s.min()), max(theo.max(), s.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.1)
    ax.set_title('滚动 IC 的 QQ 图'); ax.grid(alpha=0.3)
    fig.suptitle(title or ('★ 相邻滚动窗共享 %.0f%% 样本，分布仅作定性参考'
                           % (100 * overlap)), fontsize=9)
    plt.tight_layout(); plt.show()
    return fig

# ==================== 4. 状态分组 / 净值 ====================
def tsa_state_groups(sig, fwd, edges=None, n_states=4, sample=None):
    """按信号值分档看未来收益。edges=固定边界（离散分数必用，升序）；None=分位等分。"""
    both = pd.concat([sig, fwd], axis=1).dropna()
    both.columns = ['sig', 'fwd']
    if both.empty:
        return pd.DataFrame()
    if sample:
        both = both.iloc[::int(sample)]
    if edges is not None:
        e = sorted([float(x) for x in edges])
        lbl = pd.cut(both['sig'], bins=[-np.inf] + e + [np.inf])
    else:
        lbl = pd.qcut(both['sig'].rank(method='first'), int(n_states), labels=False)
    # [修6] 保留 Categorical 排序（按边界升序），不要 .astype(str) 提前变 object，
    #       否则负数/(-inf) 边界会按字符串排，档位顺序乱掉。最后再把索引转文字。
    g = both.groupby(lbl, observed=True)['fwd']
    tbl = pd.DataFrame({'观测数': g.size(), '平均收益%': 100 * g.mean(),
                        '胜率': g.apply(lambda x: (x > 0).mean())})
    tbl.index = [str(i) for i in tbl.index]
    tbl.index.name = 'sig'
    return tbl.round(4)

def tsa_state_equity(d, th_list, cost=None):
    """各阈值下"只在该状态持有"的净值曲线（分层累计收益的时序对应）。"""
    cost = d.get('cost', 0.0005) if cost is None else cost
    out = {}
    for th in th_list:
        pos, turn, ret = _tsa_pos(d, th)
        if pos is None:
            continue
        pnl = pos * ret - float(cost) * turn
        out[th] = (1.0 + pnl).cumprod()
    return out

def tsa_equity_plot(curves, bench, title=None):
    """状态净值曲线 + 基准（同图对比，对数轴）。[修9] 返回 fig；[修10] 阈值标签统一。"""
    if not curves:
        print('  （无净值曲线）'); return None
    fig, ax = plt.subplots(figsize=(11, 4.0))
    if bench is not None and not bench.empty:
        ax.plot(bench.index, bench.values, lw=1.8, color='gray', label='基准(买入持有)')
    for th, c in curves.items():
        ax.plot(c.index, c.values, lw=1.3, label='只做 sig>%s' % _tsa_th(th))
    ax.set_yscale('log'); ax.set_title(title or '各阈值"只在该状态持有"净值（对数轴）')
    ax.margins(x=0.01)      # 别让 x 轴多吐出数据范围外的年份刻度（易被误读成"算到未来"）
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return fig

# ==================== 5. 自相关 / 换仓 / 年度 ====================
def tsa_acf(sig, lags=(1, 20)):
    """信号时序自相关：越高越稳定、换手越低。"""
    s = pd.Series(sig).dropna()
    out = {}
    for L in lags:
        if len(s) > int(L) + 10:
            out[int(L)] = float(s.autocorr(lag=int(L)))
    return out

def tsa_switch_stats(d, th=None, cost=None):
    """换仓统计（修复2：成本拖累 = 切换次数 × 单边成本 / 年）。"""
    th = d['th'] if th is None else th
    cost = d.get('cost', 0.0005) if cost is None else cost
    pos, turn, ret = _tsa_pos(d, th)
    if pos is None or pos.empty:
        return {}
    switches = int((turn > 0).sum())
    years = max(1e-9, len(pos) / 252.0)
    lens, cur, cnt = [], float(pos.iloc[0]), 0
    for v in pos.values:
        if v == cur:
            cnt += 1
        else:
            if cur > 0:
                lens.append(cnt)
            cur, cnt = v, 1
    if cur > 0:
        lens.append(cnt)
    return {'切换次数': switches,
            '切换次数/年': switches / years,
            '平均持有交易日': float(np.mean(lens)) if lens else np.nan,
            '在场占比%': 100 * float(pos.mean()),
            '年化成本拖累%': 100 * switches * float(cost) / years}

def tsa_hold_hist(d, th=None):
    """持有段长度分布直方图：与自相关一起判断换手是否过高。"""
    th = d['th'] if th is None else th
    pos, turn, ret = _tsa_pos(d, th)
    if pos is None or pos.empty:
        return
    lens, cur, cnt = [], float(pos.iloc[0]), 0
    for v in pos.values:
        if v == cur:
            cnt += 1
        else:
            if cur > 0:
                lens.append(cnt)
            cur, cnt = v, 1
    if cur > 0:
        lens.append(cnt)
    if not lens:
        print('  （没有在场段）'); return None
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.hist(lens, bins=min(30, max(5, len(lens) // 3)), color='#9bb8d9', edgecolor='white')
    ax.axvline(np.mean(lens), color='firebrick', ls='--', label='均值 %.1f 交易日' % np.mean(lens))
    ax.set_xlabel('持有段长度（交易日）'); ax.set_ylabel('段数')
    ax.set_title('持有段长度分布'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return fig

def tsa_yearly_table(d, th=None, cost=None):
    """★ 年度拆解（新增）：每年 在场% / 策略% / 基准% / vs持有pp / 等暴露基准% / 超额pp / 切换 / 成本拖累%。

    ★ 关于两个"超额"列（很容易被其中一个骗）：
      vs持有pp  = 策略% - 基准%          → 含"空仓躲跌"的贡献，全年空仓时会很大
      超额pp    = 策略% - 等暴露基准%    → 扣掉仓位暴露后的真本事，但在场%≈0 或 ≈100 时退化
      两者一起看：vs持有pp 大、超额pp 小 → 赢在"没参与"，不是赢在"选得准"。
    """
    th = d['th'] if th is None else th
    cost = d.get('cost', 0.0005) if cost is None else cost
    pos, turn, ret = _tsa_pos(d, th)
    if pos is None or pos.empty:
        return pd.DataFrame()
    pnl = pos * ret - float(cost) * turn
    df = pd.DataFrame({'pos': pos, 'ret': ret, 'pnl': pnl, 'turn': turn})
    g = df.groupby(df.index.year)
    out = pd.DataFrame({
        '在场%': 100 * g['pos'].mean(),
        '策略%': 100 * g['pnl'].apply(lambda s: (1.0 + s).prod() - 1.0),
        '基准%': 100 * g['ret'].apply(lambda s: (1.0 + s).prod() - 1.0),
        '切换次数': g['turn'].apply(lambda s: int((s > 0).sum())),
    })
    out['vs持有pp'] = out['策略%'] - out['基准%']
    out['等暴露基准%'] = out['基准%'] * out['在场%'] / 100.0
    out['超额pp'] = out['策略%'] - out['等暴露基准%']
    out['成本拖累%'] = 100 * float(cost) * g['turn'].sum()
    return out.round(3)

def tsa_cross_plot(df):
    """★ 多标的横向对比（新增）：超额pp / 夏普差 / 滚动IC为正%。[修9] 返回 fig。"""
    if df is None or df.empty:
        return None
    df = df.copy()
    if '夏普差' not in df.columns:
        df['夏普差'] = df.get('策略夏普', np.nan) - df.get('基准夏普', np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col, ttl in zip(axes, ['超额pp', '夏普差', '滚动IC为正%'],
                            ['超额收益 pp', '夏普差（策略-基准）', '滚动IC为正 %']):
        v = pd.Series(df[col]).astype(float).values
        colors = ['seagreen' if x == x and x > 0 else 'indianred' for x in v]
        ax.bar(range(len(v)), v, color=colors)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([str(c) for c in df['标的']], rotation=30, fontsize=8)
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.show()
    return fig

# ==================== 6. 一键报告 ====================
def tsa_report(signal, codes, start, end, period=20, th=60, th_list=None, params=None,
               decision_freq='D', cost=0.0005, direction=1,
               ic_win=250, ic_step=5, edges=None, n_states=4,
               thin_step=None, yearly=True, do_plots=True, plot_code=None, verbose=True):
    """一键：逐标的构建 → 汇总表（含正确的 NW 检验）+ 年度表 + 图表。
    thin_step：抽稀步长（默认=period，保证近似独立样本）。"""
    if isinstance(codes, str):
        codes = [codes]
    th_list = list(th_list) if th_list else [th]
    thin = int(thin_step) if thin_step else max(1, int(period))
    rows, store, yearly_store = [], {}, {}
    for c in codes:
        try:
            d = tsa_build(signal, c, start, end, period=period, th=th, params=params,
                          decision_freq=decision_freq, cost=cost, direction=direction,
                          verbose=False)          # [修2] 框架细节不刷屏，只要这里的汇总
        except Exception as e:
            print('  %s 构建失败：%s' % (c, e)); continue
        store[c] = d
        ric = tsa_ic_rolling(d['signal'], d['fwd'], win=ic_win, step=ic_step)
        d['ric'] = ric                              # [修4] 缓存，画图时不再重算
        b_st, t_st, n_st = _tsa_state_ttest(d['signal'], d['fwd'], th, thin, lag=thin)
        sw = tsa_switch_stats(d, th)
        ac = tsa_acf(d['signal'], lags=(1, 20))
        both = pd.concat([d['signal'], d['fwd']], axis=1).dropna()
        thinb = both.iloc[::thin]
        # [修3] 门槛从 >10 放宽到 >5：period 大时抽稀后样本本来就少，别一律给 NaN
        ic_all = (thinb.iloc[:, 0].rank().corr(thinb.iloc[:, 1].rank())
                  if len(thinb) > 5 else np.nan)
        # [新增] 信号前段是否因历史不够而缺失 —— 这是"回测期莫名短了一截"的常见原因
        fv = d['signal'].first_valid_index()
        if verbose and fv is not None:
            gap = (fv - pd.Timestamp(start)).days
            if gap > 60:
                print('  [!] %s 信号首个有效日 %s（比 start 晚 %d 天）：'
                      '请把 start 再往前留足信号窗口，否则回测被白白截短'
                      % (c, pd.Timestamp(fv).date(), gap))
        bt = d.get('backtest')
        def _get(row, col):
            try:
                return float(bt.loc[row, col])
            except Exception:
                return np.nan
        r = {'标的': c,
             'IC(抽稀)': ic_all,
             '滚动IC均值': ric.mean() if len(ric) else np.nan,
             '滚动IC为正%': 100 * (ric > 0).mean() if len(ric) else np.nan,
             '状态差%/期': 100 * b_st if b_st == b_st else np.nan,
             '状态差NW_t': t_st,
             '独立样本数': n_st,
             '自相关(1)': ac.get(1, np.nan),
             '切换/年': sw.get('切换次数/年', np.nan),
             '平均持有日': sw.get('平均持有交易日', np.nan),
             '在场%': sw.get('在场占比%', np.nan),
             '成本拖累%/年': sw.get('年化成本拖累%', np.nan),
             '策略年化%': _get('策略(择时)', '年化收益%'),
             '基准年化%': _get('基准(买入持有)', '年化收益%'),
             '策略夏普': _get('策略(择时)', '夏普'),
             '基准夏普': _get('基准(买入持有)', '夏普'),
             '策略回撤%': _get('策略(择时)', '最大回撤%'),
             '基准回撤%': _get('基准(买入持有)', '最大回撤%')}
        r['等暴露基准%'] = (r['基准年化%'] * r['在场%'] / 100.0
                            if r['在场%'] == r['在场%'] else np.nan)
        r['超额pp'] = r['策略年化%'] - r['等暴露基准%']
        r['夏普差'] = r['策略夏普'] - r['基准夏普']
        rows.append(r)
        if yearly:
            yearly_store[c] = tsa_yearly_table(d, th=th, cost=cost)

    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        cols = ['标的', 'IC(抽稀)', '滚动IC均值', '滚动IC为正%', '状态差%/期', '状态差NW_t',
                '独立样本数', '切换/年', '平均持有日', '在场%', '成本拖累%/年',
                '策略年化%', '等暴露基准%', '超额pp', '策略夏普', '基准夏普', '夏普差',
                '策略回撤%', '基准回撤%']
        print('=' * 118)
        print('【时序有效性汇总】%s | 前瞻%d日 | th=%s | 决策=%s | 抽稀步长=%d日'
              % (signal, period, th, decision_freq, thin))
        print(df[cols].round(3).to_string(index=False))
        print('  判读: ① 状态差NW_t ≥ 2 才算"分状态有效"（独立样本数少时谨慎）')
        print('        ② 超额pp>0 且 夏普差≥0 → 有择时价值；超额pp>0 但 夏普差<0 → 只是降beta')
        print('        ③ 成本拖累%/年 相对超额pp 的大小，决定这条信号"值不值得做"')
        print('=' * 118)

    use = plot_code or (list(store)[0] if store else None)
    if verbose and yearly and use and use in yearly_store:
        print('【年度拆解】%s' % use)
        print(yearly_store[use].to_string())

    if do_plots and store and use is not None:
        d = store[use]
        print('（图表针对 %s）' % use)
        tsa_signal_plot(d, th=th)
        ric = d.get('ric')
        if ric is None:
            ric = tsa_ic_rolling(d['signal'], d['fwd'], win=ic_win, step=ic_step)
        tsa_ic_plot(ric)
        tsa_ic_heatmap(ric)
        tsa_ic_dist(ric, win=ic_win, step=ic_step)
        tbl = tsa_state_groups(d['signal'], d['fwd'], edges=edges, n_states=n_states)
        if not tbl.empty:
            print('【状态分组】%s（%s）' % (use, '固定边界' if edges else '分位等分'))
            print(tbl.to_string())
            try:
                print('  最高档-最低档: %.3f%%/期（描述性，重叠样本不做显著性）'
                      % (tbl['平均收益%'].iloc[-1] - tbl['平均收益%'].iloc[0]))
            except Exception:
                pass
        curves = tsa_state_equity(d, th_list, cost=cost)
        ret = d['close'].pct_change().fillna(0.0) if d['close'] is not None and not d['close'].empty else None
        bench = (1.0 + ret).cumprod() if ret is not None else None
        tsa_equity_plot(curves, bench)
        tsa_hold_hist(d, th=th)
        if len(df) > 1:
            tsa_cross_plot(df)
    return {'table': df, 'store': store, 'yearly': yearly_store}
