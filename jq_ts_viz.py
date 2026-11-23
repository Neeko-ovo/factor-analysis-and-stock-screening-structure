# -*- coding: utf-8 -*-
"""
时序因子（择时）检验框架 · matplotlib 可视化补丁
================================================
配套 jq_ts_factor_framework.py 使用：不修改原框架任何逻辑，
只把 run_ts_test / run_ts_batch / ts_ic_decay 的返回值画成图。

关于"交互"
----------
matplotlib 本身是静态绘图库，交互能力来自后端。在研究环境（Jupyter）里，
先执行下面任意一行，再 import 本模块：

    %matplotlib widget      # 推荐，需 ipympl（缩放/平移/悬停）
    %matplotlib notebook    # 老版 Jupyter 内置，无 ipympl 时用
    %matplotlib inline      # 静态出图，无交互，只能 savefig

本模块会尝试自动切到 ipympl；失败则保持当前后端并通过 ts_setup_style() 回报，
不会假装交互可用。

用法
----
    import jq_ts_viz as viz
    viz.ts_setup_style()

    r = run_ts_test('ma200', '000300.XSHG', '2018-01-01', '2025-01-01')
    viz.plot_all(r, code='000300.XSHG')          # 一键出全套
    viz.ts_save_all(r, 'pic', 'ma200_000300')       # 落到 png

    d = ts_ic_decay('mom_120', '000300.XSHG', '2018-01-01', '2025-01-01')
    viz.ts_plot_ic_decay(d)                          # 单张：IC 衰减

    b = run_ts_batch(['ma200', 'ma60'], '000300.XSHG', '2018-01-01', '2025-01-01')
    viz.plot_batch_compare(b)                     # 单张：多信号对比
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker as mticker

_TSV_FONT_CANDIDATES = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'PingFang SC',
    'Hiragino Sans GB', 'Arial Unicode MS',
]

_STATE = {'figsize': (13.0, 7.0), 'dpi': 110, 'style_ready': False}

_FONT_CACHE = {'names': None}


# ============================================================
# 一、基础设置
# ============================================================
def _tsv_pyplot():
    """延迟导入 pyplot：交互后端必须在 pyplot 导入前设置才生效。"""
    import matplotlib.pyplot as plt
    return plt


def _tsv_available_fonts():
    if _FONT_CACHE['names'] is None:
        try:
            from matplotlib import font_manager as fm
            _FONT_CACHE['names'] = set(f.name for f in fm.fontManager.ttflist)
        except Exception:
            _FONT_CACHE['names'] = set()
    return _FONT_CACHE['names']


def ts_setup_style(interactive=True, dpi=110, figsize=(13.0, 7.0),
                font_size=11, grid=True):
    """设置中文字体、尺寸、网格，并尝试切到交互后端。

    返回 (backend_name, notes)；notes 是给用户看的说明列表。
    必须在任何绘图之前调用一次。
    """
    notes = []
    backend = matplotlib.get_backend()

    if interactive:
        if 'matplotlib.pyplot' in sys.modules or 'pyplot' in sys.modules:
            notes.append('pyplot 已导入，后端锁定为 %s（如需交互请先 %matplotlib widget 再 import）'
                         % backend)
        else:
            try:
                matplotlib.use('module://ipympl.backend_nbagg', force=True)
                backend = matplotlib.get_backend()
                notes.append('已切到交互后端：ipympl')
            except Exception as e:
                notes.append('ipympl 不可用（%s），保持后端 %s' % (e.__class__.__name__, backend))

    plt = _tsv_pyplot()

    avail = _tsv_available_fonts()
    picked = [f for f in _TSV_FONT_CANDIDATES if f in avail]
    if picked:
        plt.rcParams['font.sans-serif'] = picked + [
            f for f in plt.rcParams['font.sans-serif'] if f not in picked]
        notes.append('中文字体：%s' % picked[0])
    else:
        notes.append('未找到候选中文字体，中文可能显示为方块（装一个 SimHei / Noto Sans CJK 即可）')

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.grid'] = bool(grid)
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = font_size - 1
    plt.rcParams['axes.titlesize'] = font_size + 2
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['savefig.bbox'] = 'tight'

    _STATE['figsize'] = tuple(figsize)
    _STATE['dpi'] = dpi
    _STATE['style_ready'] = True
    return backend, notes


def _tsv_figsize(figsize):
    if figsize is not None:
        return tuple(figsize)
    return _STATE['figsize']


# ============================================================
# 二、数据整理小工具
# ============================================================
def _tsv_as_series(x, name='value'):
    """任意输入 → 干净的 float Series + DatetimeIndex（能转就转）。"""
    s = pd.Series(x).copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    try:
        s = s.astype(float)
    except Exception:
        pass
    if isinstance(s.index, pd.DatetimeIndex) and not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s


def _tsv_pct_fmt(v, digits=1):
    if v is None or not np.isfinite(v):
        return '—'
    return ('%%.%df%%%%' % digits) % v


def _tsv_num_fmt(v, digits=2):
    if v is None or not np.isfinite(v):
        return '—'
    return ('%%.%df' % digits) % v


def _tsv_mdd(cum):
    cum = pd.Series(cum)
    return cum / cum.cummax() - 1.0


def _tsv_fill_position(ax, index, pos, color='C0', alpha=0.13, label='持仓区间'):
    """把 0/1 仓位画成背景阴影（阶梯）。"""
    if pos is None or len(pos) == 0:
        return
    p = pd.Series(pos, index=index).astype(float).fillna(0.0)
    y0, y1 = ax.get_ylim()
    ax.fill_between(p.index, y0, y1, where=(p.values > 0.5),
                    step='post', color=color, alpha=alpha, lw=0, label=label)
    ax.set_ylim(y0, y1)


# ============================================================
# 二·五、出图引擎：matplotlib（默认，静态）/ plotly（可缩放、可悬停）
# ============================================================
try:
    import plotly.graph_objects as _tgo
    from plotly.subplots import make_subplots as _tsubs
    TS_HAS_PLOTLY = True
except Exception:
    _tgo = None
    _tsubs = None
    TS_HAS_PLOTLY = False

TS_ENGINE = 'mpl'


def ts_set_engine(v='mpl'):
    """切时序图的引擎：'mpl' 静态 / 'plotly' 交互。没装 plotly 就留在 mpl。"""
    global TS_ENGINE
    want = str(v or 'mpl').strip().lower()
    if want in ('plotly', 'ply', 'interactive', '交互', 'i'):
        if not TS_HAS_PLOTLY:
            print('  [提示] 这个环境没装 plotly，继续用 matplotlib（pip install plotly 之后再切）')
            TS_ENGINE = 'mpl'
        else:
            TS_ENGINE = 'plotly'
    else:
        TS_ENGINE = 'mpl'
    return TS_ENGINE


def ts_engine():
    return TS_ENGINE


def _tply():
    return bool(TS_HAS_PLOTLY) and TS_ENGINE == 'plotly'


def _t_hline(fig, y, row=1, color='#333333', dash='solid', width=1.0):
    fig.add_shape(type='line',
                  xref=('x domain' if int(row) == 1 else 'x%d domain' % int(row)),
                  yref=('y' if int(row) == 1 else 'y%d' % int(row)),
                  x0=0, x1=1, y0=float(y), y1=float(y),
                  line=dict(color=color, width=width, dash=dash), layer='below')


def _t_bands(fig, index, pos, row=1, color='#1f77b4', alpha=0.13, name='持仓区间'):
    """o/1 仓位画成背景竖条（和 mpl 那条阶梯阴影同一件事）。"""
    if pos is None or len(pos) == 0:
        return 0
    p = pd.Series(pos, index=index).astype(float).fillna(0.0)
    idx = list(p.index)
    vals = p.values > 0.5
    n = 0
    i = 0
    while i < len(vals):
        if not vals[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(vals) and vals[j + 1]:
            j += 1
        fig.add_shape(type='rect', xref='x', yref='y domain', x0=idx[i], x1=idx[min(j + 1, len(idx) - 1)],
                      y0=0, y1=1, fillcolor=color, opacity=alpha, line=dict(width=0),
                      layer='below', row=row, col=1)
        n += 1
        i = j + 1
    return n


def _t_layout(fig, title, height=560):
    fig.update_layout(
        title=dict(text='<b>%s</b>' % title, x=0.01, xanchor='left', font=dict(size=13)),
        height=int(height), template='plotly_white',
        margin=dict(l=64, r=20, t=58, b=42), hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, font=dict(size=10)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#e6e6e6', zeroline=False)
    return fig


def _tply_price_signal(result, code='', th=None, direction=1, title=None):
    close = _tsv_as_series(result['close'], 'close')
    sig = _tsv_as_series(result['signal'], 'signal')
    detail = result.get('detail')
    name = str(sig.name or 'signal')
    pos = None
    if isinstance(detail, pd.DataFrame) and 'position' in detail.columns:
        pos = detail['position']

    if th is None:
        th = 0.0
        th_note = 'th=0（未显式传入，仅示意）'
    else:
        th_note = 'th=%g' % th

    fig = _tsubs(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                 row_heights=[2.3, 1.0])
    fig.add_trace(_tgo.Scatter(x=list(close.index), y=close.values, mode='lines', name='收盘价',
                               line=dict(color='#1f77b4', width=1.2),
                               hovertemplate='%{y:.3f}<extra>价格</extra>'), row=1, col=1)
    _t_bands(fig, close.index, pos, row=1, color='#1f77b4' if direction >= 0 else '#2ca02c')
    fig.update_yaxes(title_text='价格', row=1, col=1)

    fig.add_trace(_tgo.Scatter(x=list(sig.index), y=sig.values, mode='lines', name=name,
                               line=dict(color='#ff7f0e', width=1.0),
                               hovertemplate='%{y:.4f}<extra>' + name + '</extra>'), row=2, col=1)
    _t_hline(fig, th, 2, '#d62728', 'dash', 0.9)
    fig.add_trace(_tgo.Scatter(x=[None], y=[None], mode='lines', name='阈值 %g' % th,
                               line=dict(color='#d62728', dash='dash', width=0.9)), row=2, col=1)
    fig.update_yaxes(title_text=name, row=2, col=1)
    return _t_layout(fig, title or ('%s 价格 / 信号 / 持仓   %s   %s'
                                    % (code, th_note, '方向+1' if direction >= 0 else '方向-1')),
                     height=560)


def _tply_backtest(result, code='', title=None):
    detail = result.get('detail')
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise ValueError('result["detail"] 为空，无法回测作图')
    d = detail.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        try:
            d.index = pd.to_datetime(d.index)
        except Exception:
            pass
    d = d[d.index.notna()]
    market = d['market'].astype(float).fillna(0.0)
    strat = d['strategy'].astype(float).fillna(0.0)
    pos = d['position'].astype(float).fillna(0.0)
    cum_m = (1.0 + market).cumprod()
    cum_s = (1.0 + strat).cumprod()
    dd_m = _tsv_mdd(cum_m)
    dd_s = _tsv_mdd(cum_s)

    fig = _tsubs(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                 row_heights=[2.4, 1.0, 0.7])
    fig.add_trace(_tgo.Scatter(x=list(cum_s.index), y=cum_s.values, mode='lines', name='策略(择时)',
                               line=dict(color='#1f77b4', width=1.4),
                               hovertemplate='%{y:.3f}<extra>策略</extra>'), row=1, col=1)
    fig.add_trace(_tgo.Scatter(x=list(cum_m.index), y=cum_m.values, mode='lines', name='基准(买入持有)',
                               line=dict(color='#777777', width=1.1),
                               hovertemplate='%{y:.3f}<extra>基准</extra>'), row=1, col=1)
    fig.update_yaxes(title_text='累计净值', row=1, col=1)

    fig.add_trace(_tgo.Scatter(x=list(dd_s.index), y=dd_s.values * 100, mode='lines', name='策略回撤',
                               fill='tozeroy', line=dict(color='#1f77b4', width=0.9),
                               fillcolor='rgba(31,119,180,0.32)'), row=2, col=1)
    fig.add_trace(_tgo.Scatter(x=list(dd_m.index), y=dd_m.values * 100, mode='lines', name='基准回撤',
                               line=dict(color='#777777', width=0.9)), row=2, col=1)
    fig.update_yaxes(title_text='回撤 (%)', row=2, col=1)

    fig.add_trace(_tgo.Scatter(x=list(pos.index), y=pos.values, mode='lines', name='仓位',
                               line=dict(color='#2ca02c', width=1.0, shape='hv'),
                               fill='tozeroy', fillcolor='rgba(44,160,44,0.28)'), row=3, col=1)
    fig.update_yaxes(title_text='仓位', range=[-0.05, 1.05], dtick=1, row=3, col=1)
    fig.update_xaxes(title_text='日期', row=3, col=1)

    note = _tsv_perf_note(result.get('backtest'))
    if note:
        fig.add_annotation(x=0.985, y=0.035, xref='x domain', yref='y domain', showarrow=False,
                           xanchor='right', yanchor='bottom', text=note.replace('\n', '<br>'),
                           font=dict(size=9.5), bgcolor='rgba(255,255,255,0.9)',
                           bordercolor='#bbbbbb', borderwidth=0.8)
    return _t_layout(fig, title or ('%s 择时回测' % code), height=640)


def _tply_ic_decay(df, title=None, thresh=2.0):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('IC 衰减表为空，先跑 ts_ic_decay()')
    labels = ['%s' % int(v) for v in df['前瞻期(日)']]
    ic = df['IC(Spearman)'].astype(float).values
    tnw = df['t(NW)'].astype(float).values
    ir = df['分段IC_IR'].astype(float).values if '分段IC_IR' in df.columns else None
    has_ir = ir is not None and np.isfinite(ir).any()
    nrow = 3 if has_ir else 2
    fig = _tsubs(rows=nrow, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                 row_heights=[1.15, 1.0, 0.82] if has_ir else [1.15, 1.0])

    fig.add_trace(_tgo.Bar(x=labels, y=ic, name='IC(Spearman)', opacity=0.86,
                           marker=dict(color=['#1f77b4' if v >= 0 else '#d62728' for v in ic]),
                           text=['%.4f' % v if np.isfinite(v) else '' for v in ic],
                           textposition='outside', textfont=dict(size=8.5),
                           hovertemplate='%{x} 日 IC %{y:.4f}<extra></extra>'), row=1, col=1)
    _t_hline(fig, 0, 1, '#666666', 'solid', 0.9)
    _t_hline(fig, 0.05, 1, '#2ca02c', 'dot', 0.9)
    _t_hline(fig, -0.05, 1, '#2ca02c', 'dot', 0.9)
    fig.update_yaxes(title_text='IC (Spearman)', row=1, col=1)

    fig.add_trace(_tgo.Scatter(x=labels, y=tnw, mode='lines+markers', name='t(Newey-West)',
                               line=dict(color='#1f77b4', width=1.2),
                               hovertemplate='%{x} 日 t= %{y:.2f}<extra></extra>'), row=2, col=1)
    _t_hline(fig, thresh, 2, '#d62728', 'dash', 0.9)
    _t_hline(fig, -thresh, 2, '#d62728', 'dash', 0.9)
    _t_hline(fig, 0, 2, '#666666', 'solid', 0.9)
    fig.update_yaxes(title_text='t(NW)', row=2, col=1)

    if has_ir:
        fig.add_trace(_tgo.Scatter(x=labels, y=ir, mode='lines+markers', name='分段IC_IR',
                                   line=dict(color='#9467bd', width=1.1, dash='dash'),
                                   hovertemplate='%{x} 日 IR %{y:.2f}<extra></extra>'), row=3, col=1)
        _t_hline(fig, 0, 3, '#666666', 'solid', 0.9)
        fig.update_yaxes(title_text='分段IC_IR', row=3, col=1)
    fig.update_xaxes(title_text='前瞻期（交易日）', row=nrow, col=1)
    return _t_layout(fig, title or 'IC 衰减：t(NW) 仍 ≥2 且 分段IC_IR 最高的前瞻期 = 该配的持有期',
                     height=520 if has_ir else 420)


# ============================================================
# 三、悬停十字准星（静态后端下自动无害）
# ============================================================
def crosshair(ax, series_list, time_fmt='%Y-%m-%d', decimals=None,
              value_fmt=None, fontsize=9):
    """给一个轴加"悬停显示最近读数"的十字准星。

    series_list: [(标签, Series), ...]，Series 需 DatetimeIndex 且单调。
    value_fmt: 可选，统一的格式化函数 f(v) -> str。
    返回 disconnect 函数。
    """
    fig = ax.figure
    prepared = []
    for item in series_list:
        label, s = item[0], _tsv_as_series(item[1], item[0])
        if len(s) == 0:
            continue
        prepared.append((label, s))
    if not prepared:
        return lambda: None

    vline = ax.axvline(np.nan, color='0.35', lw=0.9, ls='--', alpha=0.8,
                       visible=False, zorder=6)
    dots = []
    for label, s in prepared:
        d, = ax.plot([], [], 'o', ms=5.5, zorder=8, color='C3',
                     label='_nolegend_')
        dots.append(d)

    txt = ax.text(0.012, 0.975, '', transform=ax.transAxes, va='top', ha='left',
                  fontsize=fontsize, zorder=10, visible=False,
                  bbox=dict(boxstyle='round,pad=0.35', fc='white',
                            ec='0.75', alpha=0.9))

    def _fmt(v):
        if value_fmt is not None:
            return value_fmt(v)
        if decimals is not None:
            return ('%%.%df' % decimals) % v
        return '%g' % v

    def _on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            if vline.get_visible():
                vline.set_visible(False)
                txt.set_visible(False)
                for d in dots:
                    d.set_visible(False)
                fig.canvas.draw_idle()
            return
        ts = pd.Timestamp(event.xdata)
        lines = ['时间 %s' % ts.strftime(time_fmt)]
        for (label, s), d in zip(prepared, dots):
            try:
                v = s.asof(ts)
            except Exception:
                v = np.nan
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                lines.append('%s —' % label)
                d.set_visible(False)
                continue
            try:
                lines.append('%s %s' % (label, _fmt(float(v))))
            except Exception:
                lines.append('%s %s' % (label, v))
            d.set_data([ts], [float(v)])
            d.set_visible(True)
        vline.set_xdata([ts, ts])
        vline.set_visible(True)
        txt.set_text('\n'.join(lines))
        txt.set_visible(True)
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('motion_notify_event', _on_move)

    def _disconnect():
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass

    return _disconnect


# ============================================================
# 四、图 1：价格 + 信号 + 持仓
# ============================================================
def plot_price_signal(result, code='', th=None, direction=1, title=None,
                      figsize=None, cursor=True, ax=None):
    """价格曲线 + 信号曲线（含阈值线）+ 持仓区间阴影。

    result: run_ts_test 的返回 dict。
    th: 触发阈值；不传则画 0 并在标题里标注未指定。
    """
    if _tply() and ax is None:
        return _tply_price_signal(result, code=code, th=th, direction=direction, title=title)
    plt = _tsv_pyplot()
    close = _tsv_as_series(result['close'], 'close')
    sig = _tsv_as_series(result['signal'], 'signal')
    detail = result.get('detail')
    name = str(sig.name or 'signal')

    pos = None
    if isinstance(detail, pd.DataFrame) and 'position' in detail.columns:
        pos = detail['position']

    if th is None:
        th = 0.0
        th_note = 'th=0（未显式传入，仅示意）'
    else:
        th_note = 'th=%g' % th

    if ax is None:
        fig, axes = plt.subplots(
            2, 1, figsize=_tsv_figsize(figsize), sharex=True,
            gridspec_kw={'height_ratios': [2.3, 1.0], 'hspace': 0.12})
        ax1, ax2 = axes
    else:
        fig = ax.figure
        ax1 = ax
        ax2 = ax.twinx()

    ax1.plot(close.index, close.values, lw=1.15, color='C0', label='收盘价')
    ax1.set_ylabel('价格')
    _tsv_fill_position(ax1, close.index, pos,
                   color='C0' if direction >= 0 else 'C2')

    ax2.plot(sig.index, sig.values, lw=1.0, color='C1', label=name)
    ax2.axhline(th, color='C3', ls='--', lw=0.9, label='阈值')
    ax2.set_ylabel(name)
    ax2.grid(False)

    ax1.set_title(title or ('%s 价格 / 信号 / 持仓   %s   %s'
                            % (code, th_note, '方向+1' if direction >= 0 else '方向-1')))
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', ncol=3)

    if cursor:
        items = [('价格', close)]
        if pos is not None:
            items.append(('仓位', pos))
        crosshair(ax1, items, decimals=3)
    fig.autofmt_xdate(rotation=0, ha='center')
    return fig


# ============================================================
# 五、图 2：回测净值 + 回撤 + 仓位 + 指标
# ============================================================
def plot_backtest(result, code='', title=None, figsize=None, cursor=True):
    """策略 vs 基准：累计净值、回撤、仓位的三联图 + 指标表。"""
    if _tply():
        return _tply_backtest(result, code=code, title=title)
    plt = _tsv_pyplot()
    detail = result.get('detail')
    if not isinstance(detail, pd.DataFrame) or detail.empty:
        raise ValueError('result["detail"] 为空，无法回测作图')

    d = detail.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        try:
            d.index = pd.to_datetime(d.index)
        except Exception:
            pass
    d = d[d.index.notna()]
    market = d['market'].astype(float).fillna(0.0)
    strat = d['strategy'].astype(float).fillna(0.0)
    pos = d['position'].astype(float).fillna(0.0)

    cum_m = (1.0 + market).cumprod()
    cum_s = (1.0 + strat).cumprod()
    dd_m = _tsv_mdd(cum_m)
    dd_s = _tsv_mdd(cum_s)

    fig, axes = plt.subplots(
        3, 1, figsize=_tsv_figsize(figsize), sharex=True,
        gridspec_kw={'height_ratios': [2.4, 1.0, 0.7], 'hspace': 0.12})
    ax1, ax2, ax3 = axes

    ax1.plot(cum_s.index, cum_s.values, lw=1.35, color='C0', label='策略(择时)')
    ax1.plot(cum_m.index, cum_m.values, lw=1.15, color='0.45', label='基准(买入持有)')
    ax1.set_ylabel('累计净值')
    ax1.legend(loc='upper left', ncol=2)

    ax2.fill_between(dd_s.index, dd_s.values, 0, color='C0', alpha=0.35, lw=0, label='策略回撤')
    ax2.plot(dd_m.index, dd_m.values, lw=0.9, color='0.45', label='基准回撤')
    ax2.set_ylabel('回撤')
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: '%.0f%%' % (100.0 * v)))
    ax2.legend(loc='lower left', ncol=2)

    ax3.fill_between(pos.index, 0, pos.values, step='post', color='C2', alpha=0.45, lw=0)
    ax3.set_ylabel('仓位')
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_yticks([0, 1])
    ax3.set_xlabel('日期')

    note = _tsv_perf_note(result.get('backtest'))
    if note:
        ax1.text(0.985, 0.035, note, transform=ax1.transAxes, va='bottom', ha='right',
                 fontsize=9, zorder=10,
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.75', alpha=0.9))

    ax1.set_title(title or ('%s 择时回测' % code))

    if cursor:
        crosshair(ax1, [('策略', cum_s), ('基准', cum_m)], decimals=3)
        crosshair(ax2, [('策略回撤', dd_s)], value_fmt=lambda v: '%.2f%%' % (100.0 * v))
    return fig


def _tsv_perf_note(bt):
    """把 run_ts_test 返回的性能表压成两行文本。"""
    if not isinstance(bt, pd.DataFrame) or bt.empty:
        return ''
    rows = list(bt.index)
    try:
        s = bt.iloc[0]
        b = bt.iloc[1] if len(bt) > 1 else None
    except Exception:
        return ''

    def _g(series, key):
        try:
            return float(series.get(key, np.nan))
        except Exception:
            return np.nan

    lines = ['%s  年化 %s  夏普 %s  回撤 %s  在场 %s'
             % (rows[0], _tsv_pct_fmt(_g(s, '年化收益%')), _tsv_num_fmt(_g(s, '夏普')),
                _tsv_pct_fmt(_g(s, '最大回撤%')), _tsv_pct_fmt(_g(s, '在场占比%')))]
    if b is not None:
        lines.append('%s  年化 %s  夏普 %s  回撤 %s'
                     % (rows[1], _tsv_pct_fmt(_g(b, '年化收益%')), _tsv_num_fmt(_g(b, '夏普')),
                        _tsv_pct_fmt(_g(b, '最大回撤%'))))
    return '\n'.join(lines)


# ============================================================
# 六、图 3：IC 衰减（多前瞻期）
# ============================================================
def ts_plot_ic_decay(df, title=None, figsize=None, thresh=2.0):
    """ts_ic_decay 的返回表 → IC 柱状 + t(NW) 折线（带 |t|=2 参考线）。"""
    if _tply():
        return _tply_ic_decay(df, title=title, thresh=thresh)
    plt = _tsv_pyplot()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('IC 衰减表为空，先跑 ts_ic_decay()')

    x = np.arange(len(df))
    labels = ['%s' % int(v) for v in df['前瞻期(日)']]
    ic = df['IC(Spearman)'].astype(float).values
    tnw = df['t(NW)'].astype(float).values
    ir = df['分段IC_IR'].astype(float).values if '分段IC_IR' in df.columns else None

    has_ir = ir is not None and np.isfinite(ir).any()
    nrow = 3 if has_ir else 2
    heights = [1.15, 1.0, 0.82] if has_ir else [1.15, 1.0]
    fig, axes = plt.subplots(
        nrow, 1, figsize=_tsv_figsize(figsize), sharex=True, squeeze=False,
        gridspec_kw={'height_ratios': heights, 'hspace': 0.14})
    axes = axes[:, 0]
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_ir else None

    colors = ['C0' if (v >= 0) else 'C3' for v in ic]
    bars = ax1.bar(x, ic, width=0.55, color=colors, alpha=0.85)
    ax1.axhline(0, color='0.4', lw=0.9)
    ax1.axhline(0.05, color='C2', ls=':', lw=0.9)
    ax1.axhline(-0.05, color='C2', ls=':', lw=0.9)
    ax1.set_ylabel('IC (Spearman)')
    for b, v in zip(bars, ic):
        if np.isfinite(v):
            ax1.annotate('%.4f' % v, (b.get_x() + b.get_width() / 2.0, v),
                         textcoords='offset points',
                         xytext=(0, 4 if v >= 0 else -12), ha='center', fontsize=8)

    ax2.plot(x, tnw, 'o-', lw=1.2, color='C0', label='t(Newey-West)')
    ax2.axhline(thresh, color='C3', ls='--', lw=0.9, label='|t|=%g' % thresh)
    ax2.axhline(-thresh, color='C3', ls='--', lw=0.9)
    ax2.axhline(0, color='0.4', lw=0.9)
    ax2.set_ylabel('t(Newey-West)')
    ax2.legend(loc='lower right', ncol=3)

    if ax3 is not None:
        ax3.plot(x, ir, 's--', lw=1.1, color='C4')
        ax3.axhline(0, color='0.4', lw=0.9)
        ax3.set_ylabel('分段IC_IR\n（只看符号）')
    axes[-1].set_xlabel('前瞻期（交易日）')

    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(labels)
    ax1.set_title(title or 'IC 衰减：t(NW) 仍 ≥2 且 分段IC_IR 最高的前瞻期 = 该配的持有期')
    return fig


# ============================================================
# 七、图 4：状态分组收益
# ============================================================
def _tsv_edge_desc(k, edges):
    lo = '-inf' if k == 0 else ('%.4g' % edges[k - 1])
    hi = 'inf' if k >= len(edges) else ('%.4g' % edges[k])
    return '(%s, %s]' % (lo, hi)


def _tsv_welch_t(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else np.nan


def state_table_fixed(result, thresholds=None, n_states=2, min_n=20):
    """修正版状态分组，补原框架 ts_state_analysis 对离散信号的塌档问题。

    原框架拿 np.unique(sig.quantile([0.5])) 当切点，而分档判定是 v > e。
    0/1 信号若多数时间取 1，分位值就是 1.0，最大值永远进不去下一档，
    结果只剩「状态0」一档、spread 变 nan。
    这里：离散信号（唯一取值 <= 10）用相邻取值的中点当切点，连续信号才用分位。
    返回 (表, spread)，列与 run_ts_test 返回的 state_table 一致。
    """
    sig = result.get('signal')
    fwd = result.get('fwd')
    if sig is None or fwd is None:
        raise ValueError('result 里缺少 signal / fwd，先跑 run_ts_test()')
    df = pd.concat([pd.Series(sig).rename('sig'), pd.Series(fwd).rename('ret')],
                   axis=1).dropna()
    if len(df) < min_n:
        return pd.DataFrame(), {'spread%': np.nan, 't(welch)': np.nan, 'note': '样本不足'}

    if thresholds is not None:
        edges = sorted(float(e) for e in thresholds)
    else:
        uniq = np.sort(pd.unique(df['sig'].values.astype(float)))
        uniq = uniq[np.isfinite(uniq)]
        if 1 < len(uniq) <= 10:
            edges = list((uniq[:-1] + uniq[1:]) / 2.0)
        else:
            qs = np.linspace(0.0, 1.0, int(n_states) + 1)[1:-1]
            edges = list(np.unique(df['sig'].quantile(qs).values))

    def _label(v):
        k = 0
        for e in edges:
            if v > e:
                k += 1
        return k

    df = df.copy()
    df['state'] = df['sig'].apply(_label)
    rows = []
    for s in sorted(df['state'].unique()):
        sub = df.loc[df['state'] == s, 'ret']
        cnt = len(sub)
        if cnt < min_n:
            rows.append({'状态': int(s), '区间': _tsv_edge_desc(int(s), edges), '观测数': cnt,
                         '平均收益%': np.nan, '折年化%': np.nan, '胜率': np.nan})
            continue
        m = float(sub.mean())
        rows.append({'状态': int(s), '区间': _tsv_edge_desc(int(s), edges), '观测数': cnt,
                     '平均收益%': 100.0 * m,
                     '折年化%': 100.0 * ((1.0 + m) ** 244.0 - 1.0) if m > -1 else np.nan,
                     '胜率': float((sub > 0).mean())})
    tbl = pd.DataFrame(rows)
    lo, hi = int(tbl['状态'].min()), int(tbl['状态'].max())
    if hi == lo:
        return tbl, {'spread%': np.nan, 't(welch)': np.nan, 'note': '只有一个状态'}
    a = df.loc[df['state'] == lo, 'ret'].values
    b = df.loc[df['state'] == hi, 'ret'].values
    return tbl, {'spread%': float(100.0 * (np.mean(b) - np.mean(a))),
                 't(welch)': _tsv_welch_t(b, a),
                 'note': 'spread = 最高信号状态 - 最低信号状态'}


def plot_state_table(result, title=None, figsize=None, fixed=True):
    """状态分组表 → 各档平均收益横向条形 + 胜率/观测数标注。

    fixed=True 时，若原表只有一档，自动换成 state_table_fixed() 的离散感知分档。
    """
    plt = _tsv_pyplot()
    tbl = result.get('state_table')
    spread = result.get('state_spread') or {}
    if fixed:
        try:
            ftbl, fspread = state_table_fixed(result)
            n_old = (tbl['状态'].nunique()
                     if isinstance(tbl, pd.DataFrame) and not tbl.empty else 0)
            if (isinstance(ftbl, pd.DataFrame) and not ftbl.empty
                    and ftbl['状态'].nunique() > n_old):
                tbl, spread = ftbl, fspread
        except Exception:
            pass
    if not isinstance(tbl, pd.DataFrame) or tbl.empty:
        raise ValueError('result["state_table"] 为空')

    t = tbl.copy()
    t['状态'] = t['状态'].astype(int)
    t = t.sort_values('状态')
    y = np.arange(len(t))
    vals = t['平均收益%'].astype(float).values
    colors = ['C0' if (v >= 0) else 'C3' for v in vals]

    fig, ax = plt.subplots(figsize=(_tsv_figsize(figsize)[0], max(3.2, 1.05 * len(t) + 2.0)))
    bars = ax.barh(y, vals, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color='0.4', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(['状态%d %s' % (s, rng) for s, rng in zip(t['状态'], t['区间'])])
    ax.set_xlabel('平均未来收益（%）')

    for b, (v, n, w) in zip(bars, zip(vals, t['观测数'].values, t['胜率'].values)):
        if not np.isfinite(v):
            continue
        ax.annotate('%.3f%%  n=%d  胜率 %s' % (v, int(n), _tsv_pct_fmt(100.0 * w if np.isfinite(w) else np.nan)),
                    (v, b.get_y() + b.get_height() / 2.0),
                    textcoords='offset points',
                    xytext=(6 if v >= 0 else -6, 0),
                    ha='left' if v >= 0 else 'right', va='center', fontsize=9)

    sp = spread or {}
    sub = []
    if sp:
        sub = ['极差(最高-最低) %s/期' % _tsv_num_fmt(sp.get('spread%'), 3)]
        if sp.get('t(welch)') is not None:
            sub.append('Welch t=%s' % _tsv_num_fmt(sp.get('t(welch)')))
    ax.set_title(title or '状态分组：不同信号强度下的未来收益', loc='left')
    if sub:
        ax.set_title('   '.join(sub), loc='right', fontsize=9)
    return fig


# ============================================================
# 八、图 5：分段 IC
# ============================================================
def plot_ic_segments(result, title=None, figsize=None):
    """分段 IC 时序柱状：看方向在历史上稳不稳。"""
    plt = _tsv_pyplot()
    seg = result.get('ic_segments')
    if not isinstance(seg, pd.Series) or len(seg) == 0:
        raise ValueError('result["ic_segments"] 为空（样本不足或未分段）')

    seg = _tsv_as_series(seg, 'IC')
    fig, ax = plt.subplots(figsize=(_tsv_figsize(figsize)[0], 3.8))
    colors = ['C0' if v >= 0 else 'C3' for v in seg.values]
    if len(seg) > 1:
        span = float((seg.index[-1] - seg.index[0]).days)
        w = min(60.0, max(10.0, span / (len(seg) * 4.0)))
    else:
        w = 30.0
    ax.bar(seg.index, seg.values, width=w, color=colors, alpha=0.85)
    ax.axhline(0, color='0.4', lw=0.9)
    mean = float(seg.mean())
    ax.axhline(mean, color='C2', ls='--', lw=1.0)
    ax.set_ylabel('分段 IC (Spearman)')

    parts = ['均值 %s' % _tsv_num_fmt(mean, 4), 'n=%d段' % len(seg)]
    if len(seg) < 4:
        parts.append('少于默认 4 段：其余段信号恒定/样本不足已被 ts_ic_stats 丢弃')
    ax.set_title(title or '分段 IC：全为正/全为负才说明方向稳定', loc='left')
    ax.set_title('   '.join(parts), loc='right', fontsize=8.5)
    return fig


# ============================================================
# 九、图 6：多信号对比
# ============================================================
def plot_batch_compare(df, title=None, figsize=None):
    """run_ts_batch 的返回表 → 年化/夏普/回撤/在场 四宫格（策略 vs 基准）。"""
    plt = _tsv_pyplot()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('批量对比表为空，先跑 run_ts_batch()')

    t = df.copy().reset_index(drop=True)
    names = [str(v) for v in t['信号']]
    x = np.arange(len(names))
    w = 0.36

    fig, axes = plt.subplots(2, 2, figsize=_tsv_figsize(figsize), squeeze=False)
    panels = [
        (axes[0][0], '策略年化%', '基准年化%', '年化收益（%）', True),
        (axes[0][1], '策略夏普', '基准夏普', '夏普比率', False),
        (axes[1][0], '策略回撤%', '基准回撤%', '最大回撤（%）', True),
        (axes[1][1], '在场占比%', None, '在场占比（%）', True),
    ]

    handles = None
    for ax, scol, bcol, ylab, as_pct in panels:
        sv = t[scol].astype(float).values if scol in t.columns else np.full(len(t), np.nan)
        ax.bar(x - (w / 2.0 if bcol else 0), sv, width=w if bcol else 0.55,
               color='C0', alpha=0.88, label='策略')
        if bcol and bcol in t.columns:
            bv = t[bcol].astype(float).values
            ax.bar(x + w / 2.0, bv, width=w, color='0.6', alpha=0.88, label='基准')
            for xi, a, b in zip(x, sv, bv):
                if np.isfinite(a) and np.isfinite(b):
                    both_neg = (a <= 0 and b <= 0)
                    anchor = min(a, b) if both_neg else max(a, b)
                    ax.annotate('Δ%+.2f' % (a - b), (xi, anchor),
                                textcoords='offset points',
                                xytext=(0, -11 if both_neg else 3),
                                ha='center', fontsize=7.5, color='0.25')
            if handles is None:
                handles = ax.get_legend_handles_labels()[0]
        else:
            for xi, a in zip(x, sv):
                if np.isfinite(a):
                    ax.annotate('%.1f' % a, (xi, a), textcoords='offset points',
                                xytext=(0, 3), ha='center', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(ylab)
        ax.axhline(0, color='0.4', lw=0.8)
        lo, hi = ax.get_ylim()
        pad = (hi - lo) * 0.14
        ax.set_ylim(lo - pad, hi + pad * 0.5)

    if handles:
        fig.legend(handles, ['策略', '基准'], loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title or '多信号对比：只看收益提升不算数，看夏普提升 + 回撤收窄', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# ============================================================
# 十、一键入口
# ============================================================
def plot_all(result, code='', th=None, direction=1, cursor=True):
    """一键出全套（价格/回测/状态/分段IC），返回 {名字: figure}。"""
    figs = {}
    figs['price_signal'] = plot_price_signal(result, code=code, th=th,
                                             direction=direction, cursor=cursor)
    figs['backtest'] = plot_backtest(result, code=code, cursor=cursor)
    try:
        figs['state_table'] = plot_state_table(result)
    except Exception as e:
        figs['state_table_error'] = str(e)
    try:
        figs['ic_segments'] = plot_ic_segments(result)
    except Exception as e:
        figs['ic_segments_error'] = str(e)
    return figs


def ts_save_all(result, outdir='pic', prefix='ts', code='', th=None,
             direction=1, dpi=140):
    """把 plot_all 的每张图存成 png，返回 {名字: 路径}。"""
    plt = _tsv_pyplot()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    saved = {}
    for key, obj in plot_all(result, code=code, th=th, direction=direction).items():
        if key.endswith('_error'):
            continue
        try:
            fig = obj[0] if isinstance(obj, tuple) else obj
            path = os.path.join(outdir, '%s_%s.png' % (prefix, key))
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            saved[key] = path
        except Exception as e:
            saved[key] = 'ERROR: %s' % e
    return saved


def ts_show(backend_note=None):
    """统一显示；在 notebook 里返回的 fig 已自动显示，这里仅兜底。"""
    plt = _tsv_pyplot()
    if matplotlib.get_backend().lower().endswith('agg'):
        print('[提示] 当前后端 %s 是静态的，plt.show() 不会弹窗；用 ts_save_all() 落图，'
              '或先 %%matplotlib widget' % matplotlib.get_backend())
    plt.show()
