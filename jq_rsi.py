# -*- coding: utf-8 -*-
"""
Part I RSI 相对强弱体系（研究环境）
================================================

RSI 回答的问题：最近 n 天里，上涨的力量占总波动的多大比例？→ 0~100

    RSI = 100 - 100 / (1 + RS)，RS = 平均涨幅 / 平均跌幅

默认读数（按用户设定）：**75 以上超买，25 以下超卖**。阈值全部可改。

一个容易踩的坑：
  超买/超卖本身不是买卖点。RSI 在 80 上方停留两个月很常见（趋势行情）。
  真正可检验的事件是「从极端区回到正常区」——即上穿 25 / 下穿 75 的穿越点。

四件事，按顺序用：
  1) 算    calc_rsi / rsi_panel
  2) 分区  rsi_zone / rsi_zone_label / rsi_latest
  3) 事件  rsi_cross_up / rsi_cross_down / rsi_reversal_events / rsi_reversal_state
  4) 检验  rsi_zone_forward / rsi_zone_spread / rsi_report / plot_rsi

已注册的时序信号（可直接 run_ts_test / run_ts_batch）：
  rsi          RSI(n) 连续值 0~100      params={'n':14,'method':'wilder'}
  rsi_rev      反转事件 -1/0/+1        超卖区上穿 os → +1；超买区下穿 ob → -1
  rsi_rev_hold 反转持有 0/1            上穿 os 建仓，持有到 RSI >= ob 离场

多空决策的扩展点：
  rsi_reversal_state(mode='long'|'short'|'both') 是状态机骨架，
  后续要加确认根数 / 止损 / 背离，都在这一个函数里长，不用动上面三层。

依赖：
  取数用框架的 get_price_range；回测用 ts_backtest。两者都做了软依赖，
  单独把这个文件粘进聚宽研究环境也能算 RSI（只是 report 的回测段会跳过）。
"""

import numpy as np
import pandas as pd

# ============================================================
# Part I-0 默认参数
# ============================================================

RSI_OB = 75.0          # 超买阈值
RSI_OS = 25.0          # 超卖阈值
RSI_DEFAULTS = {'n': 14, 'method': 'wilder', 'ob': RSI_OB, 'os': RSI_OS}

_RSI_METHODS = {
    'wilder': 'Wilder 平滑 alpha=1/n（默认，与多数行情软件一致）',
    'sma':    '简单均值（Cutler 版）',
    'ema':    'EMA span=n',
}


def _rsi_cfg(params=None, **override):
    """合并默认参数，并校验阈值次序。params 里给 None 的键会被忽略。"""
    cfg = dict(RSI_DEFAULTS)
    if params:
        cfg.update({k: v for k, v in dict(params).items() if v is not None})
    cfg.update({k: v for k, v in override.items() if v is not None})
    cfg['n'] = int(cfg['n'])
    cfg['method'] = str(cfg['method']).lower()
    cfg['ob'] = float(cfg['ob'])
    cfg['os'] = float(cfg['os'])
    if cfg['n'] < 2:
        raise ValueError('RSI 周期 n 至少为 2：%s' % cfg['n'])
    if cfg['os'] >= cfg['ob']:
        raise ValueError('超卖阈值必须小于超买阈值：os=%.1f ob=%.1f' % (cfg['os'], cfg['ob']))
    if cfg['method'] not in _RSI_METHODS:
        raise ValueError('未知 method=%s（可选：%s）' % (cfg['method'], sorted(_RSI_METHODS)))
    return cfg


# ============================================================
# Part I-1 计算
# ============================================================

def _rsi_avg(x, n, method='wilder', min_periods=None):
    """涨幅/跌幅的平均。x 可以是 Series 或 DataFrame。"""
    n = int(n)
    mp = int(n) if min_periods is None else int(min_periods)
    mp = max(1, mp)
    m = str(method).lower()
    if m in ('sma', 'simple', 'cutler'):
        return x.rolling(n, min_periods=n).mean()
    if m in ('ema', 'span'):
        return x.ewm(span=n, adjust=False, min_periods=mp).mean()
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=mp).mean()


def _rsi_from_avgs(up, dn):
    """由平均涨幅、平均跌幅合成 RSI，并处理三种边界：
    dn==0 且 up>0 → 100；dn==0 且 up==0 → 50（横盘）；up==0 且 dn>0 → 0。"""
    rs = up / dn.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(dn > 0, 100.0)
    rsi = rsi.where(~((dn <= 0) & (up <= 0)), 50.0)
    valid = up.notna() & dn.notna()
    rsi = rsi.where(valid, np.nan)
    return rsi.clip(0.0, 100.0)


def calc_rsi(close, n=14, method='wilder', min_periods=None):
    """RSI（相对强弱指数），0~100。

    参数
    ----
    close       : Series(index=日期) 或 DataFrame(index=日期, columns=代码)
    n           : 周期，默认 14
    method      : 'wilder'（默认，alpha=1/n）/ 'sma'（简单均值）/ 'ema'（span=n）
    min_periods : 有效观测下限。None → n（严谨：前 n 根为 NaN）
                  传 0 或 1 → 复现 ewm 默认口径，从第 2 根起就有值

    返回
    ----
    与 close 同形状的 RSI。close 是宽表时，返回同形状宽表（逐列独立算）。
    """
    if not isinstance(close, (pd.Series, pd.DataFrame)):
        close = pd.Series(close, dtype=float)
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    up_avg = _rsi_avg(up, n, method, min_periods)
    dn_avg = _rsi_avg(dn, n, method, min_periods)
    return _rsi_from_avgs(up_avg, dn_avg)


def rsi_panel(close_panel, n=14, method='wilder', min_periods=None):
    """宽表版 RSI：index=日期, columns=代码 → 同形状。只是 calc_rsi 的别名，图个语义清楚。"""
    return calc_rsi(close_panel, n=n, method=method, min_periods=min_periods)


# ============================================================
# Part I-2 分区
# ============================================================

def rsi_zone(rsi, ob=RSI_OB, os_=RSI_OS):
    """三档分区：1=超买(>ob) / 0=中性 / -1=超卖(<os)。NaN 保持 NaN。"""
    if isinstance(rsi, pd.DataFrame):
        z = rsi.astype(float) * 0.0
    else:
        z = pd.Series(rsi, dtype=float) * 0.0
    z = z.mask(rsi > float(ob), 1.0)
    z = z.mask(rsi < float(os_), -1.0)
    return z


def rsi_zone_label(value, ob=RSI_OB, os_=RSI_OS):
    """把一个 RSI 数值翻译成中文区间。"""
    if value is None or pd.isna(value):
        return '—'
    v = float(value)
    if v > float(ob):
        return '超买'
    if v < float(os_):
        return '超卖'
    return '中性'


def rsi_latest(rsi, ob=RSI_OB, os_=RSI_OS):
    """最新一根的 (日期, RSI 值, 区间文字)；没有数据时返回 (None, nan, '无数据')。"""
    s = pd.Series(rsi, dtype=float).dropna()
    if len(s) == 0:
        return None, np.nan, '无数据'
    v = float(s.iloc[-1])
    return s.index[-1], v, rsi_zone_label(v, ob=ob, os_=os_)


# ============================================================
# Part I-3 事件与状态机
# ============================================================

def rsi_cross_up(rsi, level):
    """上穿 level（前一根 <= level，当根 > level）。"""
    s = pd.Series(rsi, dtype=float)
    return (s > float(level)) & (s.shift(1) <= float(level))


def rsi_cross_down(rsi, level):
    """下穿 level（前一根 >= level，当根 < level）。"""
    s = pd.Series(rsi, dtype=float)
    return (s < float(level)) & (s.shift(1) >= float(level))


def rsi_reversal_events(rsi, ob=RSI_OB, os_=RSI_OS):
    """反转事件表（只保留事件日）。

    返回 DataFrame(index=事件日, columns=['dir', 'rsi', 'kind'])
      dir  +1 = 从超卖区上穿 os（潜在做多）
           -1 = 从超买区下穿 ob（潜在减仓/做空）
    """
    s = pd.Series(rsi, dtype=float)
    up = rsi_cross_up(s, os_)
    dn = rsi_cross_down(s, ob)
    ev = pd.DataFrame({'dir': 0.0, 'rsi': s}, index=s.index)
    ev.loc[up, 'dir'] = 1.0
    ev.loc[dn, 'dir'] = -1.0
    ev = ev.loc[ev['dir'] != 0].copy()
    if len(ev):
        ev['kind'] = np.where(ev['dir'] > 0, '超卖反转(多)', '超买反转(空)')
    return ev


def _rsi_confirm_cross(v, i, level, k, kind):
    """第 i 根是否构成「穿越 level 且之后连续 k 根都在该侧」的确认事件。"""
    if i - k < 1:
        return False
    if kind == 'up':
        if v[i - k] > level:
            return False
        return bool(np.all(v[i - k + 1:i + 1] > level))
    if v[i - k] < level:
        return False
    return bool(np.all(v[i - k + 1:i + 1] < level))


def rsi_reversal_state(rsi, ob=RSI_OB, os_=RSI_OS, mode='long', confirm=0):
    """RSI 反转状态机（多空决策的骨架，后续扩展都往这里加）。

    参数
    ----
    mode    : 'long' 只做多 / 'short' 只做空 / 'both' 双向
    confirm : 确认根数。0 = 穿越当天生效；k>0 = 穿越后连续 k 根留在该侧才生效
              （用来压掉假突破，代价是进场更晚）

    规则
    ----
    做多：空仓时 RSI 从超卖区上穿 os → 持多；RSI 回到 ob → 平多
    做空：空仓时 RSI 从超买区下穿 ob → 持空；RSI 回到 os → 平空
    单边持仓，不叠加。

    返回
    ----
    Series(index=RSI 的日期, values=-1/0/1)

    后续可加的（先留接口不实现）：
      · 止损：跌破建仓日的价格 / RSI 再破 os-n
      · 背离：价格新低而 RSI 不新低，作为额外确认
      · 过滤：只在 ma200 上方做多（趋势过滤），避免下跌趋势里接飞刀
    """
    s = pd.Series(rsi, dtype=float).dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    v = s.values
    mm = str(mode).lower()
    long_on = mm in ('long', 'l', 'both', 'b', '多')
    short_on = mm in ('short', 's', 'both', 'b', '空')
    if not (long_on or short_on):
        raise ValueError("mode 只能是 'long' / 'short' / 'both'：%s" % mode)
    k = max(0, int(confirm))

    pos = np.zeros(len(v))
    state = 0.0
    for i in range(1, len(v)):
        x = v[i]
        if state == 0.0:
            if long_on:
                hit = (v[i - 1] <= os_ and x > os_) if k == 0 \
                    else _rsi_confirm_cross(v, i, os_, k, 'up')
                if hit:
                    state = 1.0
            if state == 0.0 and short_on:
                hit = (v[i - 1] >= ob and x < ob) if k == 0 \
                    else _rsi_confirm_cross(v, i, ob, k, 'down')
                if hit:
                    state = -1.0
        if state > 0 and x >= ob:
            state = 0.0
        elif state < 0 and x <= os_:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=s.index)


# ============================================================
# Part I-4 分区检验
# ============================================================

def rsi_zone_forward(rsi, close, ob=RSI_OB, os_=RSI_OS, periods=(5, 20, 60),
                     strict=False):
    """把 RSI 分三档，看每档之后 N 个交易日的收益。

    返回长表：列 = 区间 / zone / 前瞻期(日) / 样本 / 平均收益% / 中位数% / 胜率
    strict=True 时前瞻收益从 t+1 起算（与框架 strict 口径一致）。
    """
    c = pd.Series(close, dtype=float)
    s = pd.Series(rsi, dtype=float).reindex(c.index)
    z = rsi_zone(s, ob=ob, os_=os_)
    zones = [
        ('超买(>%g)' % ob, 1.0),
        ('中性(%g~%g)' % (os_, ob), 0.0),
        ('超卖(<%g)' % os_, -1.0),
    ]
    rows = []
    for p in periods:
        p = int(p)
        if p <= 0:
            continue
        if strict:
            fwd = c.shift(-(p + 1)) / c.shift(-1) - 1.0
        else:
            fwd = c.shift(-p) / c - 1.0
        for name, code in zones:
            vals = fwd.loc[z == code].dropna()
            rows.append({
                '区间': name,
                'zone': code,
                '前瞻期(日)': p,
                '样本': int(len(vals)),
                '平均收益%': 100.0 * float(vals.mean()) if len(vals) else np.nan,
                '中位数%': 100.0 * float(vals.median()) if len(vals) else np.nan,
                '胜率': float((vals > 0).mean()) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def rsi_zone_spread(tbl):
    """从 rsi_zone_forward 的表里提「超卖 - 超买」的每期收益差（反转强度）。"""
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame()
    pv = tbl.pivot_table(index='前瞻期(日)', columns='zone', values='平均收益%')
    if -1.0 not in pv.columns or 1.0 not in pv.columns:
        return pd.DataFrame()
    out = pd.DataFrame({'超卖-超买(%/期)': pv[-1.0] - pv[1.0]})
    out['样本(超卖)'] = tbl.loc[tbl['zone'] == -1.0].set_index('前瞻期(日)')['样本']
    out['样本(超买)'] = tbl.loc[tbl['zone'] == 1.0].set_index('前瞻期(日)')['样本']
    return out


# ============================================================
# Part I-5 绘图
# ============================================================

def _rsi_style():
    """中文字体与负号。框架里有 ts_setup_style 就复用，没有就自己设一份。"""
    try:
        ts_setup_style(interactive=False)
        return
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        names = set(f.name for f in fm.fontManager.ttflist)
        for cand in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
                     'WenQuanYi Micro Hei', 'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS'):
            if cand in names:
                plt.rcParams['font.sans-serif'] = [cand] + list(
                    plt.rcParams.get('font.sans-serif', []))
                break
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


def plot_rsi(close=None, rsi=None, n=14, ob=RSI_OB, os_=RSI_OS, title='',
             ma=(20, 60), figsize=(13.0, 7.0), mark_events=True, save=None):
    """价格 + RSI 上下两联图。close 为空时只画 RSI 单图。返回 fig。"""
    if rsi is None:
        raise ValueError('plot_rsi 需要 rsi（可用 calc_rsi 算）')
    _rsi_style()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    s = pd.Series(rsi, dtype=float)
    if close is None:
        fig, ax_r = plt.subplots(figsize=(figsize[0], figsize[1] * 0.55))
    else:
        c = pd.Series(close, dtype=float).reindex(s.index)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.08)
        ax_p = fig.add_subplot(gs[0])
        ax_r = fig.add_subplot(gs[1], sharex=ax_p)
        ax_p.plot(c.index, c.values, color='#2c3e50', lw=1.1, label='close')
        for m in (ma or ()):
            ax_p.plot(c.rolling(int(m)).mean(), lw=0.8, alpha=0.75, label='MA%d' % int(m))
        ax_p.set_ylabel('价格')
        ax_p.legend(loc='upper left', fontsize=9, ncol=3, frameon=False)
        ax_p.grid(alpha=0.25)
        if title:
            ax_p.set_title(title, fontsize=12)
        plt.setp(ax_p.get_xticklabels(), visible=False)

    ax_r.plot(s.index, s.values, color='#8e44ad', lw=1.0)
    ax_r.axhline(float(ob), color='#c0392b', ls='--', lw=0.9)
    ax_r.axhline(50.0, color='#999999', ls=':', lw=0.7)
    ax_r.axhline(float(os_), color='#27ae60', ls='--', lw=0.9)
    ax_r.fill_between(s.index, float(ob), 100.0, color='#c0392b', alpha=0.07)
    ax_r.fill_between(s.index, 0.0, float(os_), color='#27ae60', alpha=0.07)
    ax_r.set_ylim(0.0, 100.0)
    ax_r.set_yticks([0, float(os_), 50, float(ob), 100])
    ax_r.set_ylabel('RSI(%d)' % int(n))
    ax_r.grid(alpha=0.25)

    if mark_events:
        ev = rsi_reversal_events(s, ob=ob, os_=os_)
        if len(ev):
            buy = ev.loc[ev['dir'] > 0]
            sell = ev.loc[ev['dir'] < 0]
            ax_r.scatter(buy.index, buy['rsi'].values, marker='^', s=34,
                         color='#27ae60', zorder=5, label='超卖反转')
            ax_r.scatter(sell.index, sell['rsi'].values, marker='v', s=34,
                         color='#c0392b', zorder=5, label='超买反转')
            ax_r.legend(loc='upper right', fontsize=8.5, ncol=2, frameon=False,
                        handletextpad=0.3, columnspacing=1.2)

    if save:
        fig.savefig(save, dpi=130, bbox_inches='tight')
    return fig


# ============================================================
# Part I-6 一键报告
# ============================================================

def rsi_report(code='000300.XSHG', start='2019-01-01', end='2024-12-31',
               n=14, method='wilder', ob=RSI_OB, os_=RSI_OS,
               periods=(5, 20, 60), mode='long', confirm=0,
               decision_freq='W', cost=0.0005, plot=False, verbose=True):
    """RSI 一键体检：当前分区 + 分区未来收益 + 反转事件 + 反转持有回测。

    返回 dict: code / close / rsi / zone / state / zone_table / zone_spread /
               events / last / backtest / detail
    """
    if 'get_price_range' not in globals():
        raise ValueError('rsi_report 需要框架的 get_price_range；'
                         '只算 RSI 用 calc_rsi(close)，别调这个函数')
    px = get_price_range([code], start, end, ('close',))
    if not px or 'close' not in px or px['close'].empty or code not in px['close'].columns:
        raise ValueError('取不到价格数据: %s' % code)
    close = px['close'][code].dropna()

    # 前 n 根是冷启动（首根会被算成 100 或 0），直接丢掉，图上也干净
    rsi = calc_rsi(close, n=n, method=method, min_periods=n).reindex(close.index)
    z = rsi_zone(rsi, ob=ob, os_=os_)
    last_d, last_v, last_lab = rsi_latest(rsi, ob=ob, os_=os_)
    tbl = rsi_zone_forward(rsi, close, ob=ob, os_=os_, periods=periods)
    spd = rsi_zone_spread(tbl)
    ev = rsi_reversal_events(rsi, ob=ob, os_=os_)
    state = rsi_reversal_state(rsi, ob=ob, os_=os_, mode=mode, confirm=confirm)

    out = {'code': code, 'close': close, 'rsi': rsi, 'zone': z, 'state': state,
           'zone_table': tbl, 'zone_spread': spd, 'events': ev,
           'last': (last_d, last_v, last_lab), 'backtest': pd.DataFrame(),
           'detail': pd.DataFrame(), 'params': {'n': n, 'method': method,
                                                'ob': ob, 'os': os_,
                                                'periods': tuple(periods),
                                                'mode': mode, 'confirm': confirm}}

    if verbose:
        print('=' * 84)
        print('【RSI 体检】%s | %s ~ %s' % (code, start, end))
        print('参数: n=%s | method=%s | 超买>%g | 超卖<%g | 分区=%s'
              % (n, method, ob, os_, _RSI_METHODS.get(str(method).lower(), method)))
        if last_d is not None:
            print('最新: %s  RSI=%.2f  → %s'
                  % (str(pd.Timestamp(last_d).date()), last_v, last_lab))

        if len(tbl):
            print('\n[分区未来收益] 每个档位往后 N 日')
            show = tbl.copy()
            show['平均收益%'] = show['平均收益%'].round(3)
            show['中位数%'] = show['中位数%'].round(3)
            show['胜率'] = show['胜率'].round(3)
            print(show[['区间', '前瞻期(日)', '样本', '平均收益%', '中位数%', '胜率']]
                  .to_string(index=False))
            if len(spd):
                print('  反转强度（超卖区收益 - 超买区收益）：')
                print(spd.round({'超卖-超买(%/期)': 3}).to_string())

        if len(ev):
            print('\n[反转事件] 最近 8 次')
            tail = ev.tail(8).copy()
            tail['rsi'] = tail['rsi'].round(2)
            tail.index = [str(pd.Timestamp(d).date()) for d in tail.index]
            print(tail.to_string())
            print('  共 %d 次（超卖反转 %d / 超买反转 %d）'
                  % (len(ev), int((ev['dir'] > 0).sum()), int((ev['dir'] < 0).sum())))
        else:
            print('\n[反转事件] 区间内没有发生穿越')

    if 'ts_backtest' in globals():
        bt, detail = ts_backtest(state, close, th=0.5, decision_freq=decision_freq,
                                 cost=cost)
        out['backtest'] = bt
        out['detail'] = detail
        if verbose and not bt.empty:
            print('\n[反转持有回测] mode=%s confirm=%d | RSI 上穿 %g 建仓，回到 %g 离场'
                  % (mode, int(confirm), os_, ob))
            print(bt.round(3).to_string())
            print('  判读: 夏普提升 + 回撤收窄 才算择时价值；只收益提升不算数')
    elif verbose:
        print('\n[回测] 当前环境没有 ts_backtest，跳过（只做统计）')

    if plot:
        out['fig'] = plot_rsi(close=close, rsi=rsi, n=n, ob=ob, os_=os_,
                              title='%s  RSI(%s)' % (code, n))
    return out


# ============================================================
# Part I-7 注册到时序信号表（框架在就注册，不在就跳过）
# ============================================================

try:
    register_ts_signal
except NameError:
    TS_SIGNALS = {}

    def register_ts_signal(name):
        """独立使用时的兜底注册器（框架加载后会用框架的实现覆盖本表）。"""
        def deco(func):
            TS_SIGNALS[name] = func
            return func
        return deco


@register_ts_signal('rsi')
def ts_rsi(code, px, params=None):
    """RSI 连续值 0~100。params: {'n':14,'method':'wilder'}"""
    cfg = _rsi_cfg(params)
    return calc_rsi(px['close'], n=cfg['n'], method=cfg['method'])


@register_ts_signal('rsi_rev')
def ts_rsi_rev(code, px, params=None):
    """反转事件信号：超卖区上穿 os → +1，超买区下穿 ob → -1，其余 0。
    配 run_ts_test(..., th=0) 只吃多头反转（signal>0 持有）。"""
    cfg = _rsi_cfg(params)
    rsi = calc_rsi(px['close'], n=cfg['n'], method=cfg['method'])
    s = pd.Series(0.0, index=rsi.index)
    s = s.mask(rsi_cross_up(rsi, cfg['os']), 1.0)
    s = s.mask(rsi_cross_down(rsi, cfg['ob']), -1.0)
    return s


@register_ts_signal('rsi_rev_hold')
def ts_rsi_rev_hold(code, px, params=None):
    """反转持有信号 0/1：上穿 os 建多，RSI 回到 ob 平多。
    params: {'n':14,'method':'wilder','ob':75,'os':25,'confirm':0}"""
    cfg = _rsi_cfg(params)
    rsi = calc_rsi(px['close'], n=cfg['n'], method=cfg['method'])
    p = dict(params or {})
    return rsi_reversal_state(rsi, ob=cfg['ob'], os_=cfg['os'],
                              mode='long', confirm=int(p.get('confirm', 0)))


print('[Part I] RSI 体系已就绪：calc_rsi / rsi_zone / rsi_reversal_state / '
      'rsi_zone_forward / rsi_report / plot_rsi | 阈值 %g/%g'
      % (RSI_OB, RSI_OS))
