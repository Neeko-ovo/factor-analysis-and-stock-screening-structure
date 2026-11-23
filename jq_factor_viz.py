# -*- coding: utf-8 -*-
# ============================================================
# 截面因子可视化层 · jq_factor_viz.py
# 贴在 jq_unified_factor_framework.py 之后运行（同目录直接 import 或 exec 进同一命名空间）
#
# 只读框架的返回值，不改框架任何一行：
#   run_factor_test(...) -> result dict
#   run_batch(...)       -> (df, details, panels, fwd)
#   run_combo(...)       -> (out, panels, fwd)
#   scan_ic_decay(...)   -> df
#
# 补掉框架缺的分析口径：
#   · 分层累计净值（框架只有各层平均收益，看不到路径）
#   · 多空组合 Qn-Q1 净值 / 回撤 / 夏普
#   · IC 累计曲线 + 分年度 IC + 滚动 IC
#   · 换手率时序（框架只给最高组一个标量）
#   · 因子相关热力图（框架只 print 矩阵）
#
# 净值换算口径（重要）：
#   qr 每期代表"持仓 period 天"的收益，调仓间隔 interval 天，ratio = period/interval。
#   直接 cumprod 会在 ratio>1 时重复计算重叠窗口，故每期收益先开 ratio 次方：
#       nav = (1 + qr).pow(1/ratio).cumprod()
#   ratio 由 run_factor_test 的返回值直接给出，无需手算。
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import functools

try:
    from matplotlib import font_manager as _fm
except Exception:
    _fm = None

__all__ = [
    'setup_style', 'explain_viz',
    # 计算
    'quantile_nav', 'long_short_stats', 'quantile_turnover_series',
    'ic_yearly', 'ic_rolling',
    'alpha_beta_table', 'factor_weighted_returns', 'factor_autocorr',
    # 单因子
    'plot_ic_series', 'plot_ic_dist', 'plot_ic_yearly', 'plot_quantile_returns',
    'plot_quantile_equity', 'plot_long_short', 'plot_buckets', 'plot_turnover',
    'plot_return_summary', 'plot_spread_daily', 'plot_factor_weighted_ls',
    'plot_ic_heatmap', 'plot_ic_qq', 'plot_autocorr',
    'plot_single_all',
    # 批量 / 组合 / 衰减
    'plot_corr', 'plot_batch', 'plot_combo', 'plot_ic_decay', 'plot_freq_period',
    # 落盘
    'save_all', 'show',
]

ANN = 244  # A股年化交易日

_FONT_CANDIDATES = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Hiragino Sans GB',
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC',
    'Arial Unicode MS',
]


# ============================================================
# 零、样式与后端
# ============================================================
def _pick_font():
    """挑一个本机真有 CJK 字形的字体，挑不到返回 None。"""
    if _fm is None:
        return None
    try:
        installed = {f.name for f in _fm.fontManager.ttflist}
    except Exception:
        return None
    for name in _FONT_CANDIDATES:
        if name in installed:
            return name
    return None


_COLOR_COMPAT_DONE = False


def install_color_compat(verbose=False):
    """兼容老版 matplotlib 不认三位十六进制的问题（'#333' → '#333333'）。

    聚宽研究环境是 python3.6 + 老 matplotlib，draw 阶段会抛
    ValueError: Invalid RGBA argument: '#333'。这里在 colors.to_rgba 的入口
    兜一层，三位色值先展开成六位，画图代码一行都不用改。
    """
    global _COLOR_COMPAT_DONE
    if _COLOR_COMPAT_DONE:
        return True
    try:
        import matplotlib.colors as _mc
        _orig = _mc.to_rgba

        def to_rgba(c, alpha=None):
            if isinstance(c, str) and len(c) == 4 and c[0] == '#':
                c = '#' + c[1] * 2 + c[2] * 2 + c[3] * 2
            return _orig(c, alpha)

        try:
            to_rgba.__doc__ = _orig.__doc__
            to_rgba.__name__ = 'to_rgba'
        except Exception:
            pass
        _mc.to_rgba = to_rgba
        _COLOR_COMPAT_DONE = True
        if verbose:
            print('  已启用三位十六进制颜色兼容（#abc → #aabbcc）')
        return True
    except Exception:
        return False


def setup_style(interactive=True, dpi=110, figsize=(11.5, 4.6),
                font_size=11, grid=True):
    """设置中文字体/负号/网格。返回 (backend_name, notes 列表)。

    interactive=True 时尝试切到 ipympl（可缩放/拖拽/取点）；
    ipympl 没装就如实回报当前后端，不假装能交互。
    """
    notes = []
    install_color_compat()
    # twinx + colorbar 的图会触发 tight_layout 兼容性警告，纯噪声，压掉
    import warnings
    warnings.filterwarnings('ignore', message='.*not compatible with tight_layout.*')
    if interactive:
        try:
            matplotlib.use('module://ipympl.backend_nbagg', force=True)
            notes.append('ipympl 已启用（图形可缩放/拖拽）')
        except Exception as e:
            notes.append('ipympl 不可用（%s），沿用当前后端：静态图' % type(e).__name__)

    font = _pick_font()
    if font:
        plt.rcParams['font.sans-serif'] = [font] + _FONT_CANDIDATES
        notes.append('中文字体：%s' % font)
    else:
        notes.append('未找到中文字体，中文会变方块')

    plt.rcParams.update({
        'axes.unicode_minus': False,      # 负号用 ASCII，避免 U+2212 缺字形
        'font.size': font_size,
        'figure.dpi': dpi,
        'figure.figsize': figsize,
        'axes.grid': bool(grid),
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
    })
    return matplotlib.get_backend(), notes


def explain_viz():
    print("""\
---------------------------------------------------------------
截面因子可视化层：读什么 / 看什么
  plot_ic_series      IC 逐期柱 + 累计 IC。累计线往上=方向稳定，来回摆=噪声。
  plot_ic_dist        IC 分布。均值离 0 远 + 峰型集中=好；双峰=换过风格。
  plot_ic_yearly      分年度 IC。某年突然翻负=因子失效或样本期特例。
  plot_quantile_returns 各层平均收益。Q1→Qn 单调递增=因子有区分度。
  plot_quantile_equity  各层累计净值（核心图）。看的是"分层是不是一直分得开"。
  plot_long_short     多空净值 + 回撤。年化/夏普/回撤三项定生死。
  plot_buckets        离散因子各分值的相对收益 + 胜率。
  plot_turnover       换手率时序。换手 >50% 时成本会吃掉大半 IC。
  plot_corr           因子相关热力图。>0.7 视为重复，组合前先剔除。
  plot_batch          多因子横向对比（IC_IR / t / 价差 / 换手）。
  plot_combo          组合方案的训练段 vs 检验段 IC。
  plot_ic_decay       不同持仓期的 IC，用来定调仓频率。

  判读口径：
    |IC均值| > 0.03 且 |t值_修正| >= 2 才有资格进下一步；
    IC_IR > 0.3 算稳定；分层单调 + 多空夏普 > 1 才算能落地。
    只看 IC 不看换手 = 纸上富贵。
---------------------------------------------------------------""")

# ---- 首次画图自动套样式（省得用户忘了先调 setup_style） ----
_STYLE_DONE = False


def _ready():
    global _STYLE_DONE
    if not _STYLE_DONE:
        try:
            setup_style(interactive=False)
        except Exception:
            pass
        _STYLE_DONE = True


def _styled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _ready()
        return fn(*args, **kwargs)
    return wrapper



# ============================================================
# 〇、出图引擎：matplotlib（默认，静态）/ plotly（可缩放、可悬停）
# ============================================================
try:
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots
    HAS_PLOTLY = True
except Exception:                      # 老环境没有 plotly 也能照常用
    _go = None
    _make_subplots = None
    HAS_PLOTLY = False

VIZ_ENGINE = 'mpl'


def set_viz_engine(v='mpl'):
    """切出图引擎：'mpl' 静态图 / 'plotly' 交互图。返回实际生效的那个。

    没装 plotly 就自动留在 mpl，不会报错。
    """
    global VIZ_ENGINE
    want = str(v or 'mpl').strip().lower()
    if want in ('plotly', 'ply', 'interactive', '交互', 'i'):
        if not HAS_PLOTLY:
            print('  [提示] 这个环境没装 plotly，继续用 matplotlib（pip install plotly 之后再切）')
            VIZ_ENGINE = 'mpl'
        else:
            VIZ_ENGINE = 'plotly'
    else:
        VIZ_ENGINE = 'mpl'
    return VIZ_ENGINE


def viz_engine():
    """现在用的是哪个引擎。"""
    return VIZ_ENGINE


def _ply():
    return bool(HAS_PLOTLY) and VIZ_ENGINE == 'plotly'


def _pl_hline(fig, y, row=1, color='#333333', dash='solid', width=1.0):
    """子图里的横线（老版本 plotly 没有 add_hline，这里手搓 shape）。"""
    fig.add_shape(type='line',
                  xref=('x domain' if int(row) == 1 else 'x%d domain' % int(row)),
                  yref=('y' if int(row) == 1 else 'y%d' % int(row)),
                  x0=0, x1=1, y0=float(y), y1=float(y),
                  line=dict(color=color, width=width, dash=dash), layer='below')


def _pl_layout(fig, title, height=460):
    fig.update_layout(
        title=dict(text='<b>%s</b>' % title, x=0.01, xanchor='left', font=dict(size=13)),
        height=height, template='plotly_white',
        margin=dict(l=64, r=20, t=58, b=42), hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=10)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#e6e6e6', zeroline=False)
    return fig


def _pl_cmap(n):
    """和 matplotlib 那边同一套 RdYlGn 取色，保证两个引擎看着是同一张图。"""
    try:
        cmap = plt.get_cmap('RdYlGn')
        return ['#%02x%02x%02x' % tuple(int(255 * v) for v in cmap(i / max(1, n - 1))[:3])
                for i in range(n)]
    except Exception:
        return ['#1f77b4'] * n


def _ply_ic_series(x, title=None, window=12):
    ic = _as_series(x)
    if ic.empty:
        print('  plot_ic_series: 没有可用 IC')
        return None
    name = _pick(x, 'factor', '') or ''
    cum = ic.cumsum()
    fig = _make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[1.0, 0.85])
    fig.add_trace(_go.Bar(x=list(ic.index), y=ic.values, name='逐期 IC', opacity=0.8,
                          marker=dict(color=['#d62728' if v < 0 else '#2ca02c' for v in ic.values]),
                          hovertemplate='IC %{y:.4f}<extra></extra>'), row=1, col=1)
    m = float(ic.mean())
    _pl_hline(fig, 0, 1, '#333333', 'solid', 1.0)
    _pl_hline(fig, m, 1, '#1f77b4', 'dash', 1.1)
    fig.add_trace(_go.Scatter(x=[None], y=[None], mode='lines', name='IC 均值 %.4f' % m,
                              line=dict(color='#1f77b4', dash='dash', width=1.1)), row=1, col=1)
    fig.update_yaxes(title_text='逐期 IC', row=1, col=1)

    fig.add_trace(_go.Scatter(x=list(cum.index), y=cum.values, name='累计 IC %.2f' % cum.iloc[-1],
                              mode='lines', fill='tozeroy', line=dict(color='#1f77b4', width=1.4),
                              fillcolor='rgba(31,119,180,0.22)'), row=2, col=1)
    _pl_hline(fig, 0, 2, '#333333', 'solid', 1.0)
    fig.update_yaxes(title_text='累计 IC', row=2, col=1)
    fig.update_xaxes(title_text='调仓日', row=2, col=1)
    return _pl_layout(fig, title or ('%s  IC 时序（Rank IC）' % name), height=520)


def _ply_quantile_returns(x, title=None):
    qr = _pick(x, 'quantile_returns')
    if qr is None or getattr(qr, 'empty', True):
        print('  plot_quantile_returns: 没有分层收益')
        return None
    name = _pick(x, 'factor', '') or ''
    cols = _qcols(qr)
    m = qr[cols].mean().values * 100
    fig = _go.Figure()
    fig.add_trace(_go.Bar(x=cols, y=m, marker=dict(color=_pl_cmap(len(cols)), opacity=0.88),
                          text=['%.2f%%' % v for v in m], textposition='outside',
                          textfont=dict(size=9),
                          hovertemplate='%{x} 平均 %{y:.2f}%<extra></extra>'))
    _pl_hline(fig, 0, 1, '#333333', 'solid', 1.0)
    mono = ('单调递增' if np.all(np.diff(m) > 0)
            else ('单调递减' if np.all(np.diff(m) < 0) else '非单调'))
    fig.update_yaxes(title_text='平均每期收益 (%)')
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=64, r=20, t=58, b=42))
    return _pl_layout(fig, title or ('%s  分层平均收益（%s，价差 %.2f%%）'
                                     % (name, mono, m[-1] - m[0])), height=430)


def _ply_quantile_equity(x, title=None):
    qr = _pick(x, 'quantile_returns')
    interval = _pick(x, 'interval', 21.0)
    ratio = _pick(x, 'ratio', 1.0)
    name = _pick(x, 'factor', '') or ''
    nav, perf = quantile_nav(qr, interval=interval, ratio=ratio)
    if nav.empty:
        print('  plot_quantile_equity: 没有分层收益')
        return None
    cols = _qcols(nav)
    cs = _pl_cmap(len(cols))
    fig = _go.Figure()
    for i, c in enumerate(cols):
        lab = c
        if c in perf.index:
            lab = '%s  年化 %.1f%%  夏普 %.2f' % (c, perf.loc[c, '年化%'], perf.loc[c, '夏普'])
        fig.add_trace(_go.Scatter(x=list(nav.index), y=nav[c].values, mode='lines', name=lab,
                                  line=dict(color=cs[i], width=1.5),
                                  hovertemplate='%{y:.3f}<extra>' + str(c) + '</extra>'))
    _pl_hline(fig, 1.0, 1, '#888888', 'dot', 0.9)
    fig.update_yaxes(title_text='累计净值')
    r = _norm_ratio(ratio)
    sub = '' if r <= 1.02 else ('　ratio=%.2f 已按重叠摊平' % r)
    return _pl_layout(fig, title or ('%s  分层累计净值（间隔 %.0f 日）%s' % (name, interval, sub)),
                      height=480)


def _ply_long_short(x, title=None):
    qr = _pick(x, 'quantile_returns')
    interval = _pick(x, 'interval', 21.0)
    ratio = _pick(x, 'ratio', 1.0)
    name = _pick(x, 'factor', '') or ''
    _, nav, perf = long_short_stats(qr, interval=interval, ratio=ratio)
    if nav.empty:
        print('  plot_long_short: 分层不足 2 层')
        return None
    dd = nav / nav.cummax() - 1.0
    txt = ('年化 %.1f%%　夏普 %.2f　回撤 %.1f%%　胜率 %.1f%%'
           % (perf.get('年化%', np.nan), perf.get('夏普', np.nan),
              perf.get('最大回撤%', np.nan), 100 * perf.get('胜率', np.nan)))
    fig = _make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                         row_heights=[1.0, 0.5])
    fig.add_trace(_go.Scatter(x=list(nav.index), y=nav.values, name='多空净值', mode='lines',
                              line=dict(color='#1f77b4', width=1.6)), row=1, col=1)
    _pl_hline(fig, 1.0, 1, '#888888', 'dot', 0.9)
    fig.add_annotation(x=0.985, y=0.05, xref='x domain', yref='y domain', showarrow=False,
                       xanchor='right', yanchor='bottom', text=txt, font=dict(size=9.5),
                       bgcolor='rgba(255,255,255,0.92)', bordercolor='#bbbbbb', borderwidth=0.8)
    fig.update_yaxes(title_text='多空净值', row=1, col=1)

    fig.add_trace(_go.Scatter(x=list(dd.index), y=dd.values * 100, name='回撤', mode='lines',
                              fill='tozeroy', line=dict(color='#d62728', width=1.0),
                              fillcolor='rgba(214,39,40,0.28)'), row=2, col=1)
    _pl_hline(fig, 0, 2, '#333333', 'solid', 1.0)
    fig.update_yaxes(title_text='回撤 (%)', row=2, col=1)
    fig.update_xaxes(title_text='调仓日', row=2, col=1)
    return _pl_layout(fig, title or ('%s  多空组合 Qn-Q1' % name), height=520)


# ============================================================
# 一、取值辅助（兼容"传 result" 和 "传裸数据"两种写法）
# ============================================================
def _pick(x, key, default=None):
    """从 result dict 里取值；传的是裸数据就返回 default（避免 Series 真假判断炸掉）。"""
    if isinstance(x, dict):
        return x.get(key, default)
    return default


def _as_series(x, name=None):
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, dict):
        x = x.get('ic')
    if x is None:
        return pd.Series(dtype=float)
    s = pd.Series(x).dropna()
    if name and not s.empty:
        s.name = name
    return s


def _qcols(qr):
    """按 Q1..Qn 的数字顺序排好列（避免 Q10 排在 Q2 前面）。"""
    cols = [c for c in qr.columns if isinstance(c, str) and c.startswith('Q')]
    cols.sort(key=lambda c: int(c[1:]) if c[1:].isdigit() else 0)
    return cols


# ============================================================
# 二、计算：净值 / 绩效 / 换手时序 / 分年度 IC
# ============================================================
def _norm_ratio(ratio):
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return 1.0
    return r if r > 0 else 1.0


def quantile_nav(qr, interval=21.0, ratio=1.0, ann=ANN):
    """把"每期收益"折成累计净值。

    qr: index=调仓日, columns=Q1..Qn（quantile_returns 的返回值）
    interval: 相邻调仓日的交易日间隔
    ratio:    period/interval，重叠倍数；每期收益开 ratio 次方后按 interval 天推进

    返回 (nav, perf)：nav 从 1.0 起，perf 各层年化/波动/夏普/回撤/累计。
    """
    if qr is None or (hasattr(qr, 'empty') and qr.empty):
        return pd.DataFrame(), pd.DataFrame()
    qr = pd.DataFrame(qr).astype(float).dropna(how='all')
    if qr.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = _qcols(qr) or list(qr.columns)
    qr = qr[cols]
    ratio = _norm_ratio(ratio)
    interval = float(interval) if interval and float(interval) > 0 else 21.0

    # 每期（interval 天）的实际增长；ratio>1 时开方摊平重叠
    per = np.power(np.maximum(1.0 + qr.values, 1e-8), 1.0 / ratio)
    per = pd.DataFrame(per, index=qr.index, columns=cols)

    nav = per.cumprod()
    base = pd.DataFrame(1.0, index=[nav.index[0] - pd.Timedelta(days=1)], columns=cols)
    nav = pd.concat([base, nav]).sort_index()

    n = len(per)
    years = max(n * interval / float(ann), 1e-9)
    ppy = float(ann) / interval          # 每年的调仓期数
    rows = {}
    for c in cols:
        s = nav[c].dropna()
        # per 是"每期的增长倍数"（已按 ratio 摊平），减 1 才是该期有效收益。
        # 夏普必须用这个超额部分算；直接拿 (1+r) 的均值会把夏普虚高几十倍。
        pe = (per[c].dropna() - 1.0)
        if s.empty or pe.empty:
            continue
        total = float(s.iloc[-1])
        ann_ret = total ** (1.0 / years) - 1.0 if total > 0 else np.nan
        sd = float(pe.std(ddof=1)) if len(pe) > 1 else np.nan
        sharpe = (float(pe.mean()) / sd * np.sqrt(ppy)) if sd and sd > 0 else np.nan
        mdd = float((s / s.cummax() - 1.0).min())
        rows[c] = {
            '累计%': 100.0 * (total - 1.0),
            '年化%': 100.0 * ann_ret,
            '波动%': 100.0 * sd * np.sqrt(ppy) if sd == sd else np.nan,
            '夏普': sharpe,
            '最大回撤%': 100.0 * mdd,
            '胜率': float((pe > 0).mean()),
        }
    perf = pd.DataFrame(rows).T
    if not perf.empty:
        perf.index.name = '分层'
    return nav, perf


def long_short_stats(qr, interval=21.0, ratio=1.0, ann=ANN):
    """多空组合 = 最高层 - 最低层。返回 (ls_ret Series, nav, perf Series)。"""
    empty = (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))
    if qr is None or (hasattr(qr, 'empty') and qr.empty):
        return empty
    qr = pd.DataFrame(qr).astype(float)
    cols = _qcols(qr)
    if len(cols) < 2:
        return empty
    ls = (qr[cols[-1]] - qr[cols[0]]).dropna()
    if ls.empty:
        return empty
    nav, perf = quantile_nav(ls.to_frame('LS'), interval=interval, ratio=ratio, ann=ann)
    if nav.empty:
        return empty
    nav = nav['LS']
    perf = perf.loc['LS'] if not perf.empty else pd.Series(dtype=float)
    perf['换手%'] = np.nan
    return ls, nav, perf


def quantile_turnover_series(factor_panel, q=5, group=None, min_each=2):
    """逐期换手率（框架只给一个均值标量，这里给时序）。

    换手 = 1 - |本期与上期分组交集| / |上期分组|
    group=None 取最高组；group='ALL' 返回各层平均。
    """
    if factor_panel is None or getattr(factor_panel, 'empty', True):
        return pd.Series(dtype=float)

    # 复刻框架的 _eff_q 自适应组数
    cov = [len(factor_panel.loc[d].dropna()) for d in factor_panel.index]
    med = int(np.median(cov)) if cov else 0
    if med < min_each * 2:
        return pd.Series(dtype=float)
    q_eff = int(min(q, max(2, med // min_each)))

    mem = {}
    for d in factor_panel.index:
        f = factor_panel.loc[d].dropna()
        if len(f) < q_eff * min_each:
            continue
        try:
            grp = pd.qcut(f.rank(method='first'), q_eff, labels=False) + 1
        except Exception:
            continue
        mem[pd.Timestamp(d)] = {int(g): set(grp[grp == g].index) for g in range(1, q_eff + 1)}

    ds = sorted(mem)
    if len(ds) < 2:
        return pd.Series(dtype=float)

    targets = range(1, q_eff + 1) if group == 'ALL' else [
        q_eff if group is None else min(int(group), q_eff)]
    out = {}
    for i in range(1, len(ds)):
        prev, curr = mem[ds[i - 1]], mem[ds[i]]
        vals = []
        for g in targets:
            a, b = prev.get(g, set()), curr.get(g, set())
            if a:
                vals.append(1.0 - len(a & b) / float(len(a)))
        if vals:
            out[ds[i]] = float(np.mean(vals))
    s = pd.Series(out, name='换手率').sort_index()
    return s


def ic_yearly(ic):
    """分年度 IC 汇总：均值 / IR / 正占比 / 期数。"""
    ic = _as_series(ic)
    if ic.empty:
        return pd.DataFrame()
    g = ic.groupby(ic.index.year)
    rows = []
    for y, sub in g:
        sd = float(sub.std(ddof=1)) if len(sub) > 1 else np.nan
        rows.append({
            '年度': int(y), '期数': int(len(sub)),
            'IC均值': float(sub.mean()),
            'IC_IR': float(sub.mean() / sd) if sd and sd > 0 else np.nan,
            '正占比': float((sub > 0).mean()),
            '累计IC': float(sub.sum()),
        })
    return pd.DataFrame(rows).set_index('年度')


def ic_rolling(ic, window=12):
    """滚动 IC 均值（默认 12 期≈一年），看因子是不是"这几年才灵"。"""
    ic = _as_series(ic)
    if ic.empty:
        return pd.Series(dtype=float)
    w = max(2, int(window))
    return ic.rolling(w, min_periods=max(2, w // 2)).mean().dropna()


# ============================================================
# 三、单因子图
# ============================================================
@_styled
def plot_ic_series(x, title=None, figsize=None, window=12):
    """IC 逐期柱 + 累计 IC 面积 + 0 轴。"""
    if _ply():
        return _ply_ic_series(x, title=title, window=window)
    ic = _as_series(x)
    if ic.empty:
        print('  plot_ic_series: 没有可用 IC'); return None
    name = _pick(x, 'factor', '') or ''
    fig, axes = plt.subplots(2, 1, figsize=figsize or (11.5, 5.4), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 0.85], 'hspace': 0.12})
    ax1, ax2 = axes

    neg = ic < 0
    # 柱宽按实际时间跨度算：固定宽度在长区间上会细成一条线，读不出东西
    _span = max((ic.index[-1] - ic.index[0]).days, 1)
    _bw = max(1.0, _span / float(max(1, len(ic))) * 0.7)
    ax1.bar(ic.index, ic.values, width=_bw,
            color=np.where(neg, '#d62728', '#2ca02c'), alpha=0.78, linewidth=0)
    ax1.axhline(0, color='#333333', lw=1.0)
    m = float(ic.mean())
    ax1.axhline(m, color='#1f77b4', ls='--', lw=1.1, label='IC 均值 %.4f' % m)
    ax1.set_ylabel('逐期 IC')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(title or ('%s  IC 时序（Rank IC）' % name), fontsize=12)

    cum = ic.cumsum()
    ax2.fill_between(cum.index, 0, cum.values, color='#1f77b4', alpha=0.28)
    ax2.plot(cum.index, cum.values, color='#1f77b4', lw=1.4, label='累计 IC %.2f' % cum.iloc[-1])
    ax2.axhline(0, color='#333333', lw=1.0)
    # 累计线的斜率看着舒服就行，绝对值别当收益
    ax2.set_ylabel('累计 IC')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlabel('调仓日')
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig


@_styled
def plot_ic_dist(x, title=None, figsize=None, bins=30):
    """IC 分布 + 正态拟合 + 偏度，看是单峰还是换过风格。"""
    ic = _as_series(x)
    if len(ic) < 5:
        print('  plot_ic_dist: IC 期数不足（<5）'); return None
    name = _pick(x, 'factor', '') or ''
    mu, sd = float(ic.mean()), float(ic.std(ddof=1))
    skew = float(ic.skew()) if len(ic) > 2 else np.nan

    fig, ax = plt.subplots(figsize=figsize or (9.6, 4.6))
    ax.hist(ic.values, bins=min(bins, max(8, len(ic) // 5)),
            color='#4c78a8', alpha=0.72, edgecolor='white', linewidth=0.6)
    if sd and sd > 0:
        xs = np.linspace(ic.min(), ic.max(), 200)
        pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        n, bw = len(ic), (ic.max() - ic.min()) / min(bins, max(8, len(ic) // 5))
        ax.plot(xs, pdf * n * bw, color='#d62728', lw=1.6, label='正态拟合')
    ax.axvline(0, color='#333333', lw=1.0)
    ax.axvline(mu, color='#d62728', ls='--', lw=1.2,
               label='均值 %.4f  标准差 %.4f  偏度 %.2f' % (mu, sd, skew))
    win = float((ic > 0).mean())
    ax.set_title(title or ('%s  IC 分布（正占比 %.1f%%，n=%d）' % (name, 100 * win, len(ic))),
                 fontsize=12)
    ax.set_xlabel('Rank IC'); ax.set_ylabel('期数')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    return fig


@_styled
def plot_ic_yearly(x, title=None, figsize=None):
    """分年度 IC 均值柱 + 正占比折线，看稳定性。"""
    ic = _as_series(x)
    tbl = ic_yearly(ic)
    if tbl.empty:
        print('  plot_ic_yearly: 没有可用 IC'); return None
    name = _pick(x, 'factor', '') or ''
    fig, ax = plt.subplots(figsize=figsize or (9.6, 4.2))
    xs = np.arange(len(tbl))
    ax.bar(xs, tbl['IC均值'].values, color=np.where(tbl['IC均值'] < 0, '#d62728', '#2ca02c'),
           alpha=0.8, width=0.6, label='IC 均值')
    ax.axhline(0, color='#333333', lw=1.0)
    ax.set_xticks(xs); ax.set_xticklabels([str(i) for i in tbl.index], rotation=0)
    ax.set_ylabel('IC 均值')
    ax2 = ax.twinx()
    ax2.plot(xs, tbl['正占比'].values * 100, 'o--', color='#ff7f0e', lw=1.4,
             label='IC>0 占比 %')
    ax2.set_ylim(0, 105); ax2.set_ylabel('正占比 (%)')
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    _lo, _hi = ax.get_ylim()
    ax.set_ylim(_lo, _hi + (_hi - _lo) * 0.20)   # 顶部留白给图例，别压着折线
    ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=9, ncol=2, framealpha=0.95)
    ax.set_title(title or ('%s  分年度 IC' % name), fontsize=12)
    fig.tight_layout()
    return fig


@_styled
def plot_quantile_returns(x, title=None, figsize=None):
    """各层平均收益柱 + 单调性标注。"""
    if _ply():
        return _ply_quantile_returns(x, title=title)
    qr = _pick(x, 'quantile_returns')
    if qr is None or getattr(qr, 'empty', True):
        print('  plot_quantile_returns: 没有分层收益'); return None
    name = _pick(x, 'factor', '') or ''
    cols = _qcols(qr)
    m = qr[cols].mean().values * 100
    fig, ax = plt.subplots(figsize=figsize or (9.6, 4.4))
    cmap = plt.get_cmap('RdYlGn')
    colors = [cmap(i / max(1, len(cols) - 1)) for i in range(len(cols))]
    bars = ax.bar(range(len(cols)), m, color=colors, alpha=0.85, width=0.62)
    for b, v in zip(bars, m):
        ax.text(b.get_x() + b.get_width() / 2, v, '%.2f%%' % v,
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.axhline(0, color='#333333', lw=1.0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_ylabel('平均每期收益 (%)')
    mono = '单调递增' if np.all(np.diff(m) > 0) else ('单调递减' if np.all(np.diff(m) < 0) else '非单调')
    ax.set_title(title or ('%s  分层平均收益（%s，价差 %.2f%%）'
                           % (name, mono, m[-1] - m[0])), fontsize=12)
    fig.tight_layout()
    return fig


@_styled
def plot_quantile_equity(x, title=None, figsize=None):
    """各层累计净值（核心图）。ratio>1 时已按重叠倍数摊平。"""
    if _ply():
        return _ply_quantile_equity(x, title=title)
    qr = _pick(x, 'quantile_returns')
    interval = _pick(x, 'interval', 21.0)
    ratio = _pick(x, 'ratio', 1.0)
    name = _pick(x, 'factor', '') or ''
    nav, perf = quantile_nav(qr, interval=interval, ratio=ratio)
    if nav.empty:
        print('  plot_quantile_equity: 没有分层收益'); return None
    cols = _qcols(nav)
    fig, ax = plt.subplots(figsize=figsize or (11.5, 4.8))
    cmap = plt.get_cmap('RdYlGn')
    for i, c in enumerate(cols):
        ax.plot(nav.index, nav[c].values, lw=1.5,
                color=cmap(i / max(1, len(cols) - 1)),
                label='%s  年化 %.1f%%  夏普 %.2f' % (
                    c, perf.loc[c, '年化%'], perf.loc[c, '夏普']) if c in perf.index else c)
    ax.axhline(1.0, color='#888888', lw=0.9, ls=':')
    ax.set_ylabel('累计净值')
    r = _norm_ratio(ratio)
    sub = '' if r <= 1.02 else ('　ratio=%.2f 已按重叠摊平' % r)
    ax.set_title(title or ('%s  分层累计净值（间隔 %.0f 日）%s' % (name, interval, sub)),
                 fontsize=12)
    ax.legend(loc='upper left', fontsize=8.5, ncol=1)
    fig.tight_layout()
    return fig


@_styled
def plot_long_short(x, title=None, figsize=None):
    """多空组合 Qn-Q1：净值 + 回撤。"""
    if _ply():
        return _ply_long_short(x, title=title)
    qr = _pick(x, 'quantile_returns')
    interval = _pick(x, 'interval', 21.0)
    ratio = _pick(x, 'ratio', 1.0)
    name = _pick(x, 'factor', '') or ''
    _, nav, perf = long_short_stats(qr, interval=interval, ratio=ratio)
    if nav.empty:
        print('  plot_long_short: 分层不足 2 层'); return None

    fig, axes = plt.subplots(2, 1, figsize=figsize or (11.5, 5.2), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 0.5], 'hspace': 0.1})
    ax1, ax2 = axes
    ax1.plot(nav.index, nav.values, color='#1f77b4', lw=1.6, label='多空净值')
    ax1.axhline(1.0, color='#888888', lw=0.9, ls=':')
    txt = ('年化 %.1f%%　夏普 %.2f　回撤 %.1f%%　胜率 %.1f%%'
           % (perf.get('年化%', np.nan), perf.get('夏普', np.nan),
              perf.get('最大回撤%', np.nan), 100 * perf.get('胜率', np.nan)))
    ax1.text(0.985, 0.05, txt, transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=9, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#bbbbbb', alpha=0.92))
    ax1.set_ylabel('多空净值')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(title or ('%s  多空组合 %s-%s'
                            % (name, 'Qn', 'Q1')), fontsize=12)

    dd = nav / nav.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values * 100, 0, color='#d62728', alpha=0.32)
    ax2.plot(dd.index, dd.values * 100, color='#d62728', lw=1.0)
    ax2.axhline(0, color='#333333', lw=1.0)
    ax2.set_ylabel('回撤 (%)'); ax2.set_xlabel('调仓日')
    fig.tight_layout()
    return fig


@_styled
def plot_buckets(x, title=None, figsize=None):
    """离散因子分档：相对收益柱 + 胜率折线。"""
    bt = _pick(x, 'buckets')
    if bt is None or getattr(bt, 'empty', True):
        print('  plot_buckets: 没有分档表（连续因子按分位切，用 plot_quantile_returns）')
        return None
    name = _pick(x, 'factor', '') or ''
    d = bt.copy()
    col_v = '平均相对收益%' if '平均相对收益%' in d.columns else d.columns[0]
    xs = np.arange(len(d))
    fig, ax = plt.subplots(figsize=figsize or (9.8, 4.4))
    ax.bar(xs, d[col_v].values, color=np.where(d[col_v] < 0, '#d62728', '#2ca02c'),
           alpha=0.82, width=0.6, label=col_v)
    ax.axhline(0, color='#333333', lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in d['分值'].values] if '分值' in d.columns else xs,
                       rotation=45 if len(d) > 10 else 0)
    ax.set_xlabel('因子分值'); ax.set_ylabel('平均相对收益 (%)')
    if '胜率' in d.columns:
        ax2 = ax.twinx()
        ax2.plot(xs, d['胜率'].values * 100, 'o--', color='#ff7f0e', lw=1.3, label='胜率 %')
        ax2.set_ylim(0, 105); ax2.set_ylabel('胜率 (%)'); ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=9, ncol=2)
    else:
        ax.legend(loc='best', fontsize=9)
    ax.set_title(title or ('%s  分档相对收益' % name), fontsize=12)
    fig.tight_layout()
    return fig


@_styled
def plot_turnover(x, title=None, figsize=None):
    """换手率时序（传 Series）或换手标量对比（传 DataFrame 的 '换手%' 列）。"""
    if isinstance(x, pd.Series) and not x.empty:
        s = x.dropna()
        fig, ax = plt.subplots(figsize=figsize or (11.0, 3.8))
        ax.plot(s.index, s.values * 100, color='#9467bd', lw=1.4)
        ax.fill_between(s.index, 0, s.values * 100, color='#9467bd', alpha=0.22)
        ax.axhline(s.mean() * 100, color='#d62728', ls='--', lw=1.0,
                   label='均值 %.1f%%' % (s.mean() * 100))
        ax.set_ylim(bottom=0)
        ax.set_ylabel('换手率 (%)'); ax.set_xlabel('调仓日')
        ax.set_title(title or '换手率时序', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        fig.tight_layout()
        return fig

    if isinstance(x, pd.DataFrame) and '换手%' in x.columns:
        d = x[['因子', '换手%']].dropna()
        fig, ax = plt.subplots(figsize=figsize or (9.6, 4.2))
        xs = np.arange(len(d))
        ax.bar(xs, d['换手%'].values, color='#9467bd', alpha=0.82, width=0.6)
        for i, v in enumerate(d['换手%'].values):
            ax.text(i, v, '%.1f' % v, ha='center', va='bottom', fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(d['因子'].values, rotation=30, ha='right')
        ax.set_ylabel('换手率 (%)')
        ax.set_title(title or '各因子调仓换手率', fontsize=12)
        fig.tight_layout()
        return fig

    print('  plot_turnover: 参数类型不支持（要 Series 或带「换手%」列的 DataFrame）')
    return None


def plot_single_all(x, save=False, outdir='pic_factor', prefix=None, dpi=140):
    """单因子一键出图（13 张），返回 {名字: fig}。"""
    name = (_pick(x, 'factor', '') or 'factor').replace('/', '_')
    figs = {}
    specs = [
        ('ic', plot_ic_series), ('ic_dist', plot_ic_dist),
        ('ic_yearly', plot_ic_yearly), ('quantile_returns', plot_quantile_returns),
        ('quantile_equity', plot_quantile_equity), ('long_short', plot_long_short),
        ('buckets', plot_buckets),
        # 聚宽教程口径补充：1.3 速览表 / 1.8 收益差σ带 / 1.9 因子值加权
        # / 2.3 月度热度 / 2.6 QQ / 3.4 自相关
        ('return_summary', plot_return_summary),
        ('spread_daily', plot_spread_daily),
        ('factor_weighted_ls', plot_factor_weighted_ls),
        ('ic_heatmap', plot_ic_heatmap), ('ic_qq', plot_ic_qq),
        ('autocorr', plot_autocorr),
    ]
    for key, fn in specs:
        try:
            figs[key] = fn(x)
        except Exception as e:
            print('  %s 画图失败: %s' % (key, e))
    if save:
        import os
        os.makedirs(outdir, exist_ok=True)
        for key, fig in figs.items():
            if fig is None:
                continue
            p = os.path.join(outdir, '%s_%s.png' % (prefix or name, key))
            fig.savefig(p, dpi=dpi, bbox_inches='tight')
            print('  已存 %s' % p)
    return figs


# ============================================================
# 四、批量 / 组合 / 衰减
# ============================================================
@_styled
def plot_corr(panels, title=None, figsize=None, annot=True):
    """因子相关热力图。高相关说明信息重复，组合前先剔除。"""
    if not panels:
        print('  plot_corr: 没有面板'); return None
    names = list(panels.keys())
    if len(names) < 2:
        print('  plot_corr: 至少 2 个因子'); return None
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    stack = {n: panels[n].stack().dropna() for n in names}
    for a in names:
        for b in names:
            try:
                mat.loc[a, b] = stack[a].corr(stack[b], method='spearman')
            except Exception:
                mat.loc[a, b] = np.nan
    fig, ax = plt.subplots(figsize=figsize or (max(6.0, 1.0 * len(names) + 3.0),
                                                max(5.0, 0.8 * len(names) + 2.4)))
    im = ax.imshow(mat.values.astype(float), cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    if annot and len(names) <= 14:
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=8,
                            color='white' if abs(v) > 0.55 else '#222222')
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    ax.set_title(title or '因子相关性（Spearman）｜>0.7 视为重复', fontsize=12)
    fig.tight_layout()
    return fig


@_styled
def plot_batch(df, title=None, figsize=None):
    """多因子横向对比：|IC_IR| / t值_修正 / 分层价差 / 换手。"""
    if df is None or getattr(df, 'empty', True):
        print('  plot_batch: 没有汇总表'); return None
    d = df.copy()
    if 'IC_IR' in d.columns:
        d = d.reindex(d['IC_IR'].abs().sort_values(ascending=False).index)
    names = [str(v) for v in d['因子'].values] if '因子' in d.columns else list(d.index)
    fig, axes = plt.subplots(2, 2, figsize=figsize or (13.0, 6.6))
    specs = [('IC_IR', 1.0, 'IC_IR'), ('t值_修正', 2.0, 't值（修正）'),
             ('分层价差%', 0.0, '分层价差 (%)'), ('换手%', 0.0, '换手率 (%)')]
    for ax, (col, thresh, label) in zip(axes.ravel(), specs):
        if col not in d.columns:
            ax.axis('off'); continue
        vals = pd.to_numeric(d[col], errors='coerce').values
        xs = np.arange(len(vals))
        ax.bar(xs, vals, color=np.where(vals < 0, '#d62728', '#4c78a8'), alpha=0.85, width=0.6)
        if thresh:
            for t in (thresh, -thresh):
                ax.axhline(t, color='#ff7f0e', ls='--', lw=1.0)
        ax.axhline(0, color='#333333', lw=1.0)
        for i, v in enumerate(vals):
            if pd.notna(v):
                ax.text(i, v, '%.2f' % v, ha='center',
                        va='bottom' if v >= 0 else 'top', fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8.5)
        ax.set_ylabel(label, fontsize=9.5)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            pad = (hi - lo) * 0.16
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle(title or '多因子对比｜|t|>=2 且 |IC_IR|>=0.3 才算有效', fontsize=12.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig


@_styled
def plot_combo(out, title=None, figsize=None):
    """组合方案对比：训练段 vs 检验段 IC（都为正才算没拟合历史）。"""
    if out is None or getattr(out, 'empty', True):
        print('  plot_combo: 没有组合结果'); return None
    d = out.copy()
    names = [str(v) for v in d['方案'].values] if '方案' in d.columns else list(d.index)
    pairs = [('训练IC', '检验IC', 'IC'), ('训练IR', '检验IR', 'IC_IR')]
    fig, axes = plt.subplots(1, 2, figsize=figsize or (13.0, 4.6))
    for ax, (a, b, lab) in zip(axes, pairs):
        if a not in d.columns or b not in d.columns:
            ax.axis('off'); continue
        xs = np.arange(len(d)); w = 0.36
        va, vb = pd.to_numeric(d[a], errors='coerce').values, pd.to_numeric(d[b], errors='coerce').values
        ax.bar(xs - w / 2, va, width=w, color='#4c78a8', alpha=0.88, label='训练段')
        ax.bar(xs + w / 2, vb, width=w, color='#f58518', alpha=0.88, label='检验段')
        ax.axhline(0, color='#333333', lw=1.0)
        ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8.5)
        ax.set_ylabel(lab, fontsize=10)
        ax.legend(loc='best', fontsize=9)
        lo = np.nanmin(np.r_[va, vb]); hi = np.nanmax(np.r_[va, vb])
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            pad = (hi - lo) * 0.18
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle(title or '组合方案：训练段 vs 检验段（检验段塌掉=过拟合）', fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


@_styled
def plot_ic_decay(df, title=None, figsize=None, thresh=2.0):
    """不同持仓期的 IC 衰减：定调仓频率用。"""
    if df is None or getattr(df, 'empty', True):
        print('  plot_ic_decay: 没有衰减表'); return None
    d = df.copy()
    xs = np.arange(len(d))
    # 优先用实际持仓期当横轴标签：scan_ic_decay 对齐后 请求持仓期 与 实际 可能差一点
    xcol = '持仓期(日)' if '持仓期(日)' in d.columns else d.columns[0]
    lbl = [str(v) for v in d[xcol].values]

    fig, axes = plt.subplots(2, 1, figsize=figsize or (10.5, 5.6), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 0.85], 'hspace': 0.12})
    ax1, ax2 = axes
    if 'IC均值' in d.columns:
        v = pd.to_numeric(d['IC均值'], errors='coerce').values
        ax1.bar(xs, v, color=np.where(v < 0, '#d62728', '#2ca02c'), alpha=0.85, width=0.55)
        for i, val in enumerate(v):
            if pd.notna(val):
                ax1.text(i, val, '%.4f' % val, ha='center',
                         va='bottom' if val >= 0 else 'top', fontsize=9)
        ax1.axhline(0, color='#333333', lw=1.0)
        for t in (0.03, -0.03):
            ax1.axhline(t, color='#2ca02c', ls=':', lw=1.0)
        ax1.set_ylabel('IC 均值')
    ax1.set_title(title or 'IC 衰减：t 仍 >=2 且 IC 不塌的那一档，就是该配的持仓期', fontsize=12)

    if 't值_修正' in d.columns:
        t = pd.to_numeric(d['t值_修正'], errors='coerce').values
        ax2.plot(xs, t, 'o-', color='#4c78a8', lw=1.6, label='t 值（修正）')
        for tt in (thresh, -thresh):
            ax2.axhline(tt, color='#d62728', ls='--', lw=1.1)
        ax2.axhline(0, color='#333333', lw=1.0)
        ax2.set_ylabel('t 值（修正）')
        lo = np.nanmin(t) if len(t) else -1; hi = np.nanmax(t) if len(t) else 1
        ax2.set_ylim(min(lo, -thresh) - 0.4, max(hi, thresh) + 0.4)
        ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.set_xticks(xs); ax2.set_xticklabels(lbl, fontsize=9)
    ax2.set_xlabel('持仓期（交易日）')
    fig.tight_layout()
    return fig


@_styled
def plot_freq_period(df, metric='IC_IR', title=None, figsize=None, annotate=True):
    """调仓频率 × 持仓期 热力图（配合 scan_freq_period）。

    行 = 调仓频率（日/周/月/季），列 = 持仓期（交易日）。
    metric 用 'IC_IR' / 't值_修正' 看信号强度，用 '净年化%' 看扣成本后的收益，
    用 '资金占用%' 看钱有多少时间在场上。绿=好、红=差；IC 类指标以 0 为中心。
    """
    if df is None or getattr(df, 'empty', True):
        print('  plot_freq_period: 扫描表是空的'); return None
    d = df.copy()
    if metric not in d.columns:
        print('  plot_freq_period: 表里没有 %s 列；可选 %s'
              % (metric, [c for c in d.columns if c != '判定']))
        return None
    d[metric] = pd.to_numeric(d[metric], errors='coerce')
    piv = d.pivot_table(index='频率', columns='持仓期(日)', values=metric, aggfunc='mean')
    order = [v for v in ('日频', '周频', '月频', '季频', '年频') if v in piv.index]
    piv = piv.reindex(order if order else list(piv.index))
    if piv.empty:
        print('  plot_freq_period: 交叉表为空'); return None

    vals = piv.values.astype(float)
    if metric in ('IC均值', 'IC_IR', 't值_修正', '净年化%', '折年价差%'):
        lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
        lim = float(lim) if lim and lim > 0 else 1.0
        vmin, vmax, cmap = -lim, lim, 'RdYlGn'
    else:
        vmin, vmax, cmap = np.nanmin(vals), np.nanmax(vals), 'Reds'

    fig, ax = plt.subplots(figsize=figsize or (8.4, 0.85 * len(piv) + 2.2))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto')
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([str(c) for c in piv.columns], fontsize=9)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([str(i) for i in piv.index], fontsize=10)
    ax.set_xlabel('持仓期（交易日）')
    ax.set_ylabel('调仓频率')
    ax.set_title(title or ('%s · 调仓频率 × 持仓期' % metric), fontsize=12, pad=10)
    if annotate:
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = vals[i, j]
                if not np.isfinite(v):
                    continue
                txt = ('%.1f' % v) if metric.endswith('%') else ('%.3f' % v)
                ax.text(j, i, txt, ha='center', va='center', fontsize=8.5,
                        color='#222222')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(metric, fontsize=9)
    # 画出 |IC_IR| 最强的那一格
    if metric in ('IC_IR', 'IC均值', 't值_修正') and np.isfinite(vals).any():
        k = int(np.nanargmax(np.abs(vals)))
        i, j = divmod(k, piv.shape[1])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='#1f4e79', lw=2.0))
    fig.tight_layout()
    return fig



# ============================================================
# 四·五、聚宽《单因子有效性分析》口径补充
#   教程分三块：收益分析 / 信息分析 / 换手分析。这里补上前面没覆盖的口径：
#     1.3  收益分析速览表   Ann.Alpha / Beta / Top / Bottom / Spread(bps)
#     1.8  最高层减最低层每日收益差（1σ 带 + 月度均线）
#     1.9  因子值加权多空组合（用因子值当权重，不只取两端两层）
#     2.3  月度 IC 热度图（红=IC 小，绿=IC 大）
#     2.6  IC QQ 图（越贴对角线越接近正态）
#     3.2  因子 rank 自相关（Mean Factor Rank Autocorrelation）
#     3.4  因子自相关时序
#   暂未做：1.6 各行业平均收益 / 2.4 行业分组 IC —— 要 get_industry，
#   ETF 池没有行业分类，只在股票池上有意义，需要时说一声再加。
# ============================================================
def _daily_ret(per, interval=21.0, ratio=1.0):
    """每期收益 -> 日均收益：先按 ratio 摊掉重叠窗口，再按 interval 几何摊平。"""
    ser = pd.Series(per, dtype=float)
    if ser.empty:
        return ser
    try:
        itv = float(interval)
    except (TypeError, ValueError):
        itv = 21.0
    itv = itv if itv > 0 else 21.0
    total = _norm_ratio(ratio) * itv
    return np.power(np.maximum(1.0 + ser, 1e-8), 1.0 / total) - 1.0


def _win_by_month(result):
    """把"12 个月"折算成调仓期数：interval 天一期，21 交易日≈1 个月。"""
    try:
        itv = float(_pick(result, 'interval', 21.0))
    except (TypeError, ValueError):
        itv = 21.0
    itv = itv if itv > 0 else 21.0
    return max(3, int(round(12.0 * itv / 21.0)))


def _ols_alpha_beta(y, x):
    """OLS y = alpha + beta*x，返回 (alpha_daily, beta)。样本不足给 nan。"""
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    m = np.isfinite(yy) & np.isfinite(xx)
    yy, xx = yy[m], xx[m]
    if len(yy) < 10 or np.std(xx) == 0:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(xx)), xx])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return float(coef[0]), float(coef[1])


def alpha_beta_table(qr, interval=21.0, ratio=1.0, ann=ANN):
    """收益分析速览表（教程 1.3）。

    市场基准取"当期全部样本股等权平均收益"（教程 1.4 的取法），
    这样 Alpha 才是扣掉样本内平均涨跌之后的独立收益。

    返回 DataFrame（行=分层）：
        Ann.Alpha%   年化 Alpha，越大越好；负得很大也说明反向因子强
        Beta         对样本等权组合的敏感度，没有单调好坏
        Top/bps      最高层日均收益（bps=万分之一）
        Bottom/bps   最低层日均收益
        Spread/bps   Top - Bottom，衡量因子收益能力的核心
    """
    if qr is None or not isinstance(qr, pd.DataFrame) or qr.empty:
        return pd.DataFrame()
    cols = _qcols(qr)
    if len(cols) < 2:
        return pd.DataFrame()
    qr = qr[cols].astype(float)
    daily = pd.DataFrame({c: _daily_ret(qr[c], interval, ratio) for c in cols},
                         index=qr.index)
    mkt = _daily_ret(qr.mean(axis=1), interval, ratio)
    rows = {}
    for c in cols:
        a, b = _ols_alpha_beta(daily[c].values, mkt.values)
        rows[c] = {'Ann.Alpha%': 100.0 * a * float(ann), 'Beta': b}
    tbl = pd.DataFrame(rows).T
    tbl['Top/bps'] = 10000.0 * float(daily[cols[-1]].mean())
    tbl['Bottom/bps'] = 10000.0 * float(daily[cols[0]].mean())
    tbl['Spread/bps'] = tbl['Top/bps'] - tbl['Bottom/bps']
    tbl.index.name = '分层'
    return tbl


@_styled
def plot_return_summary(result, title=None, figsize=None):
    """收益分析速览表（教程 1.3）—— 先看整体，再决定要不要往下看细节。"""
    qr = _pick(result, 'quantile_returns')
    tbl = alpha_beta_table(qr, interval=_pick(result, 'interval', 21.0),
                           ratio=_pick(result, 'ratio', 1.0))
    if tbl.empty:
        raise ValueError('quantile_returns 为空或不足 2 层，先跑 run_factor_test')
    name = _pick(result, 'factor', '')
    fmt = {'Ann.Alpha%': '%.2f', 'Beta': '%.3f', 'Top/bps': '%.2f',
           'Bottom/bps': '%.2f', 'Spread/bps': '%.2f'}
    # Alpha / Beta 是每层各算一个；Top / Bottom / Spread 是全样本的汇总指标，
    # 摊进每一行会变成 5 行重复值，所以单独压成表尾一行。
    body = []
    for i, r in tbl.iterrows():
        body.append([str(i),
                     '—' if pd.isna(r['Ann.Alpha%']) else fmt['Ann.Alpha%'] % r['Ann.Alpha%'],
                     '—' if pd.isna(r['Beta']) else fmt['Beta'] % r['Beta'],
                     '—', '—', '—'])
    body.append(['全样本', '—', '—',
                 fmt['Top/bps'] % tbl['Top/bps'].iloc[-1],
                 fmt['Bottom/bps'] % tbl['Bottom/bps'].iloc[-1],
                 fmt['Spread/bps'] % tbl['Spread/bps'].iloc[-1]])
    fig, ax = plt.subplots(figsize=figsize or (10.2, 0.55 * len(tbl) + 1.5))
    ax.axis('off')
    tb = ax.table(cellText=body, colLabels=['分层'] + list(tbl.columns),
                  loc='center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1.0, 1.55)
    for (r0, _c0), cell in tb.get_celld().items():
        if r0 == 0:
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#cccccc')
    ax.set_title(title or ('%s  收益分析速览（市场基准=样本等权）' % name), fontsize=12, pad=16)
    fig.tight_layout()
    return fig


@_styled
def plot_spread_daily(result, title=None, figsize=None, win=None, band=True):
    """最高层减最低层的每日收益差（教程 1.8）。

    细线 = 逐期收益差折成日均（bps）
    阴影 = ±1 个滚动标准差
    红线 = 月度移动均线
    读法：红线长期在 0 上方 = 持续有效；正负交替 = 周期性有效，得去找周期原因。
    """
    qr = _pick(result, 'quantile_returns')
    cols = _qcols(qr) if isinstance(qr, pd.DataFrame) else []
    if len(cols) < 2:
        raise ValueError('quantile_returns 不足 2 层')
    spread = (qr[cols[-1]] - qr[cols[0]]).astype(float).dropna()
    if spread.empty:
        raise ValueError('多空价差序列为空')
    d = 10000.0 * _daily_ret(spread, _pick(result, 'interval', 21.0),
                             _pick(result, 'ratio', 1.0))
    w = max(2, int(win)) if win is not None else _win_by_month(result)
    mp = max(2, w // 3)
    ma = d.rolling(w, min_periods=mp).mean()
    sd = d.rolling(w, min_periods=mp).std()

    fig, ax = plt.subplots(figsize=figsize)
    if band:
        ax.fill_between(d.index, (ma - sd).values, (ma + sd).values,
                        color='#4c78a8', alpha=0.16, lw=0, label='±1σ')
    ax.plot(d.index, d.values, color='#7fbf7f', lw=0.9, alpha=0.85, label='逐期收益差')
    ax.plot(ma.index, ma.values, color='#c0392b', lw=1.8, label='%d 期均线' % w)
    ax.axhline(0.0, color='k', lw=1.0)
    pos = float((ma.dropna() > 0).mean() * 100.0) if ma.notna().any() else np.nan
    ax.set_ylabel('日均收益差 (bps)')
    ax.set_title(title or ('%s  最高层-最低层 每日收益差（均线在0上方占 %.0f%%）'
                           % (_pick(result, 'factor', ''), pos)), fontsize=12)
    ax.legend(loc='upper left', fontsize=9, ncol=3)
    fig.tight_layout()
    return fig


def factor_weighted_returns(factor_panel, fwd_panel, winsor=3.0, min_n=5):
    """因子值加权多空组合（教程 1.9）。

    w_i = (f_i - 均值) / 标准差，截尾到 ±winsor
    组合收益 = Σ(w_i * r_i) / Σ|w_i|
    因子值越大做多越重、越小做空越重，比只取最高/最低两层用了更多信息。
    """
    empty = pd.Series(dtype=float)
    if factor_panel is None or getattr(factor_panel, 'empty', True):
        return empty
    if fwd_panel is None or getattr(fwd_panel, 'empty', True):
        return empty
    out = {}
    for d in factor_panel.index:
        t = pd.Timestamp(d)
        if t not in fwd_panel.index:
            continue
        f = factor_panel.loc[d].dropna()
        r = fwd_panel.loc[t].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_n:
            continue
        f2 = f[common].astype(float)
        r2 = r[common].astype(float)
        sd = float(f2.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        w = (f2 - f2.mean()) / sd
        if winsor:
            w = w.clip(-abs(float(winsor)), abs(float(winsor)))
        den = float(w.abs().sum())
        if den <= 0:
            continue
        out[t] = float((w * r2).sum() / den)
    return pd.Series(out, name='因子值加权多空').sort_index()


@_styled
def plot_factor_weighted_ls(result, title=None, figsize=None, winsor=3.0):
    """因子值加权多空组合累计净值（教程 1.9）+ 回撤。"""
    r = factor_weighted_returns(_pick(result, 'factor_panel'),
                                _pick(result, 'fwd_panel'), winsor=winsor)
    if r.empty:
        raise ValueError('因子值加权组合算不出来，检查 factor_panel / fwd_panel')
    interval = _pick(result, 'interval', 21.0)
    ratio = _pick(result, 'ratio', 1.0)
    nav, perf = quantile_nav(r.to_frame('FW'), interval=interval, ratio=ratio)
    if nav.empty:
        raise ValueError('净值换算失败')
    nav = nav['FW']
    perf = perf.loc['FW']
    fig, axes = plt.subplots(2, 1, figsize=figsize or (11.5, 5.6), sharex=True,
                             gridspec_kw={'height_ratios': [2.6, 1.0]})
    ax1, ax2 = axes
    ax1.plot(nav.index, nav.values, color='#2b6cb0', lw=1.5, label='因子值加权多空')
    ax1.axhline(1.0, color='k', lw=0.9, ls=':')
    note = '累计 %.1f%% | 年化 %.1f%% | 夏普 %.2f | 回撤 %.1f%%' % (
        perf['累计%'], perf['年化%'], perf['夏普'], perf['最大回撤%'])
    ax1.text(0.015, 0.96, note, transform=ax1.transAxes, fontsize=9.5,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cccccc', alpha=0.9))
    ax1.set_ylabel('累计净值')
    ax1.set_title(title or ('%s  因子值加权多空（截尾±%gσ）'
                            % (_pick(result, 'factor', ''), winsor)), fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    dd = (nav / nav.cummax() - 1.0) * 100.0
    ax2.fill_between(dd.index, dd.values, 0, color='#e06c75', alpha=0.55, lw=0)
    ax2.set_ylabel('回撤 (%)')
    fig.tight_layout()
    return fig


@_styled
def plot_ic_heatmap(x, title=None, figsize=None, annot=True):
    """月度 IC 热度图（教程 2.3）。红=IC 小（反向），绿=IC 大（正向），空白=该月无数据。"""
    ic = _as_series(x)
    if ic.empty:
        raise ValueError('IC 序列为空')
    tmp = pd.DataFrame({'y': ic.index.year, 'm': ic.index.month, 'ic': ic.values})
    piv = tmp.pivot_table(index='y', columns='m', values='ic', aggfunc='mean')
    piv = piv.reindex(columns=list(range(1, 13)))
    if piv.empty:
        raise ValueError('按年月透视后为空')
    mat = piv.values.astype(float)
    mx = np.nanmax(np.abs(mat))
    lim = float(mx) if np.isfinite(mx) and mx > 0 else 1.0
    fig, ax = plt.subplots(figsize=figsize or (9.4, 0.62 * len(piv) + 1.5))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=-lim, vmax=lim)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['%d月' % m for m in range(1, 13)], fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(y) for y in piv.index], fontsize=9)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(piv.index), 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.2)
    ax.tick_params(which='minor', length=0)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=8,
                            color='black' if abs(v) < lim * 0.6 else 'white')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('IC 均值', fontsize=9)
    ax.set_title(title or '月度 IC 热度（红=IC小，绿=IC大，空白=无数据）',
                 fontsize=12, pad=12)
    fig.tight_layout()
    return fig


def _norm_ppf(p):
    """标准正态分位函数（Acklam 近似）。

    不用 statistics.NormalDist：那是 python3.8 才有的，
    聚宽那套 python3.6 一 import 就报 cannot import name 'NormalDist'。
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError('p 必须在 (0, 1) 之间')
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    lo, hi = 0.02425, 1.0 - 0.02425
    if p < lo:
        q = np.sqrt(-2.0 * np.log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    if p > hi:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    q = p - 0.5
    r = q * q
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))


@_styled
def plot_ic_qq(x, title=None, figsize=None):
    """IC 的 QQ 图（教程 2.6）。点越贴红色对角线，IC 越接近正态。"""
    ic = _as_series(x)
    if len(ic) < 8:
        raise ValueError('IC 样本 < 8，QQ 图没意义')
    sd = float(ic.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('IC 标准差为 0，画不了 QQ 图')
    z = np.sort(((ic - ic.mean()) / sd).values)
    n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = np.array([_norm_ppf(float(v)) for v in p])
    fig, ax = plt.subplots(figsize=figsize or (5.8, 5.0))
    ax.scatter(q, z, s=24, color='#2b6cb0', alpha=0.72, label='IC 分位点 (n=%d)' % n)
    lo = min(float(q.min()), float(z.min()))
    hi = max(float(q.max()), float(z.max()))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#c0392b', lw=1.5,
            ls='--', label='正态参考线')
    ax.set_xlabel('理论分位数（标准正态）')
    ax.set_ylabel('IC 实际分位数（标准化）')
    ax.set_title(title or 'IC QQ 图', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def factor_autocorr(factor_panel, lag=1, min_n=5):
    """逐期因子 rank 自相关（教程 3.2/3.4 的 Mean Factor Rank Autocorrelation）。

    lag=1 就是相邻两次调仓之间因子排名的稳定性。
    数值低 = 因子换得快 = 换手高 = 交易成本高。
    """
    if factor_panel is None or getattr(factor_panel, 'empty', True):
        return pd.Series(dtype=float)
    lag = max(1, int(lag))
    df = factor_panel
    try:                                  # 统一成 DatetimeIndex，顺便排序
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in df.index])
        if not idx.equals(df.index):
            df = df.copy()
            df.index = idx
            df = df.sort_index()
    except Exception:
        df = df.copy()
        df = df.sort_index()
    out = {}
    # 用 iloc 定位：面板 index 有时是 datetime.date（老 pandas 下 .loc[Timestamp] 会 KeyError）
    for i in range(lag, len(df)):
        a = df.iloc[i - lag].dropna()
        b = df.iloc[i].dropna()
        common = a.index.intersection(b.index)
        if len(common) < min_n:
            continue
        c = a[common].astype(float).corr(b[common].astype(float), method='spearman')
        if pd.notna(c):
            out[df.index[i]] = float(c)
    return pd.Series(out, name='因子自相关').sort_index()


@_styled
def plot_autocorr(result, title=None, figsize=None, lag=1):
    """因子自相关时序（教程 3.4）。低自相关 = 高换手，成本吃掉收益。"""
    s = factor_autocorr(_pick(result, 'factor_panel'), lag=lag)
    if s.empty:
        raise ValueError('因子自相关算不出，检查 factor_panel')
    mu = float(s.mean())
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, color='#41a05e', lw=1.3, marker='o', ms=2.5,
            label='逐期自相关')
    ax.axhline(mu, color='#c0392b', lw=1.6, ls='--', label='均值 %.3f' % mu)
    ax.axhline(0.0, color='k', lw=0.9)
    ax.set_ylabel('Rank 自相关 (lag=%d)' % int(lag))
    ax.set_title(title or ('%s  因子自相关（均值 %.3f，越低换手越高）'
                           % (_pick(result, 'factor', ''), mu)), fontsize=12)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    return fig


# ============================================================
# 五、落盘 / 显示
# ============================================================
def save_all(figs, outdir='pic_factor', prefix='', dpi=140):
    """把 {名字: fig} 批量存 png。"""
    import os
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for key, fig in (figs or {}).items():
        if fig is None:
            continue
        p = os.path.join(outdir, ('%s%s.png' % (prefix + '_' if prefix else '', key)))
        try:
            fig.savefig(p, dpi=dpi, bbox_inches='tight')
            saved.append(p)
        except Exception as e:
            print('  存 %s 失败: %s' % (p, e))
    if saved:
        print('  已存 %d 张到 %s' % (len(saved), outdir))
    return saved


def show(note=None):
    """弹图（非交互后端下 no-op 提示）。"""
    backend = matplotlib.get_backend().lower()
    if 'agg' in backend and 'nbagg' not in backend:
        print('  当前后端 %s 不能弹窗，图已生成可直接 savefig' % backend)
        return False
    plt.show()
    return True


# ---- 导入即装：老版 matplotlib 遇到 '#333' 这类三位色值不再在 draw 时炸 ----
install_color_compat()
