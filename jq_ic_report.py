# -*- coding: utf-8 -*-
# ============================================================
# Part G  IC 视角（单因子研究 / 有效性检验专用）
#
#   工作流定位：
#     初次筛选   →  run_batch(...) + screen_ic(df)       看一批因子谁值得细看
#     单因子研究 →  ic_tables(r) + ic_plots(r)            只看 IC 的各张表/图
#     有效性确认 →  ic_report(r, decay=scan_ic_decay(...)) 一次出全套
#
#   为什么单独做一层：
#     run_factor_test 的一致性/分层/换手是"结果层"，判因子好坏第一眼看的是 IC：
#       · IC 均值与 t(NW)/t修正 —— 方向与显著性
#       · 分年度 IC —— 是不是只有某一年灵
#       · 滚动 IC —— 是不是这几年才灵
#       · 月度 IC 热力 —— 哪几个月系统性反向
#       · IC 分布/QQ —— 有没有被少数极端期带跑
#       · IC 自相关 —— 换手与成本
#
#   依赖：本文件须在 jq_unified_factor_framework.py 与 jq_factor_viz.py 之后运行
#         （用 ic_yearly / ic_rolling / plot_ic_* / factor_autocorr / save_all）
# ============================================================
import numpy as np
import pandas as pd

IC_CHART_KEYS = ('ic', 'ic_dist', 'ic_yearly', 'ic_heatmap', 'ic_qq', 'autocorr')


# ------------------------------------------------------------
# Part G-1 取 IC 序列
# ------------------------------------------------------------
def _ic_of(x):
    """从 run_factor_test 的 result（或裸 Series/DataFrame）里取干净的 IC 序列。"""
    s = x.get('ic') if isinstance(x, dict) else x
    if s is None:
        return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0] if s.shape[1] else pd.Series(dtype=float)
    s = pd.Series(s).astype(float).dropna()
    if s.empty:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
        except Exception:
            pass
    return s.sort_index()


def _ratio_of(x):
    try:
        r = float((x or {}).get('ratio', 1.0))
    except Exception:
        r = 1.0
    return r if r > 0 else 1.0


def _stat_row(ic, label, ratio=1.0):
    """一段 IC 的统计行（t 值按重叠倍数折过）。"""
    n = int(len(ic))
    if n == 0:
        return None
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else np.nan
    eff = n / ratio
    if std and std > 0 and eff > 0:
        t_adj = mean / (std / np.sqrt(eff))
    else:
        t_adj = np.nan
    return {
        '区间': label, '期数': n,
        'IC均值': mean, 'IC标准差': std,
        'IC_IR': (mean / std) if (std and std > 0) else np.nan,
        't值_修正': t_adj,
        '正占比': float((ic > 0).mean()),
        '|IC|>0.03占比': float((ic.abs() > 0.03).mean()),
    }


# ------------------------------------------------------------
# Part G-2 IC 各表
# ------------------------------------------------------------
def _monthly_matrix(ic):
    if ic.empty:
        return pd.DataFrame()
    tmp = pd.DataFrame({'y': ic.index.year, 'm': ic.index.month, 'ic': ic.values})
    piv = tmp.pivot_table(index='y', columns='m', values='ic', aggfunc='mean')
    return piv.reindex(columns=list(range(1, 13)))


def _dist_table(ic):
    """IC 分布表：分位点 + 形态 + 极端期。"""
    if len(ic) < 3:
        return pd.DataFrame()
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    d = {'值': [float(ic.quantile(q)) for q in qs]}
    out = pd.DataFrame(d, index=['P%d' % int(q * 100) for q in qs])
    extra = pd.DataFrame({
        '值': [float(ic.mean()), float(ic.std(ddof=1)), float(ic.skew()),
              float(ic.kurt()), float(ic.min()), float(ic.max()),
              float((ic > 0).mean()), float((ic.abs() > 0.05).mean())]
    }, index=['均值', '标准差', '偏度', '峰度', '最差期', '最好期',
              '为正占比', '|IC|>0.05占比'])
    return pd.concat([out, extra])


def _rolling_table(ic, window=12):
    """滚动 IC 摘要：最近一段强不强，是不是后来才灵。"""
    if len(ic) < 4:
        return pd.DataFrame()
    r = ic_rolling(ic, window)
    if r is None or len(r) == 0:
        return pd.DataFrame()
    return pd.DataFrame({'值': [
        float(r.iloc[-1]), float(r.mean()), float(r.min()), float(r.max()),
        float((r > 0).mean()), float(r.std(ddof=1)),
    ]}, index=['最新滚动IC', '滚动IC均值', '最差窗口', '最好窗口',
               '滚动为正占比', '滚动IC标准差'])


def ic_tables(r, window=12, decay=None):
    """把单因子的 IC 分析拆成几张表（只读 result，不改框架）。

    参数
      r      : run_factor_test 的返回值（也接受裸 IC Series）
      window : 滚动 IC 与「近期」窗口，默认 12 期
      decay  : 可选的 scan_ic_decay 返回值，原样放进结果里

    返回 dict：
      summary  全段 / 前半 / 后半 / 近 N 期 的 IC 统计对比
      yearly   分年度 IC（均值 / IR / 正占比 / 累计）
      monthly  年 × 月 IC 均值矩阵
      dist     IC 分布：分位点 + 偏度峰度 + 极端期
      rolling  滚动 IC 摘要
      decay    传入的衰减表（没传就是空）
    """
    ic = _ic_of(r)
    ratio = _ratio_of(r)
    out = {'summary': pd.DataFrame(), 'yearly': pd.DataFrame(),
           'monthly': pd.DataFrame(), 'dist': pd.DataFrame(),
           'rolling': pd.DataFrame(), 'decay': decay if decay is not None else pd.DataFrame()}
    if ic.empty:
        print('  ic_tables: IC 序列为空（检查 factor_panel / fwd_panel 是否有交集）')
        return out

    n = len(ic)
    half = n // 2
    blocks = [('全段', ic, ratio),
              ('前半段', ic.iloc[:half], ratio),
              ('后半段', ic.iloc[half:], ratio),
              ('近%d期' % int(window), ic.tail(max(2, int(window))), 1.0)]
    rows = [x for x in (_stat_row(sub, lab, ra) for lab, sub, ra in blocks) if x]
    out['summary'] = pd.DataFrame(rows).set_index('区间')
    out['yearly'] = ic_yearly(ic)
    out['monthly'] = _monthly_matrix(ic)
    out['dist'] = _dist_table(ic)
    out['rolling'] = _rolling_table(ic, window)
    return out


# ------------------------------------------------------------
# Part G-3 只画 IC 那几张图
# ------------------------------------------------------------
def ic_plots(r, window=12):
    """IC 六图：逐期IC / 分布 / 分年度 / 月度热力 / QQ / 自相关。

    与 plot_single_all 的区别：只留 IC 相关，一张分层净值都不画。
    返回 {名字: fig}，画失败的那张记成 None 并在图上打印原因。
    """
    figs = {}
    jobs = [
        ('ic', lambda: plot_ic_series(r, window=window)),
        ('ic_dist', lambda: plot_ic_dist(r)),
        ('ic_yearly', lambda: plot_ic_yearly(r)),
        ('ic_heatmap', lambda: plot_ic_heatmap(r)),
        ('ic_qq', lambda: plot_ic_qq(r)),
    ]
    for key, fn in jobs:
        try:
            figs[key] = fn()
        except Exception as e:
            figs[key] = None
            print('  %s 画图失败: %s' % (key, e))

    panel = r.get('factor_panel') if isinstance(r, dict) else None
    if panel is not None and getattr(panel, 'shape', (0, 0))[0] >= 2:
        try:
            figs['autocorr'] = plot_autocorr(r)
        except Exception as e:
            figs['autocorr'] = None
            print('  autocorr 画图失败: %s' % e)
    return figs


# ------------------------------------------------------------
# Part G-4 一页纸：IC 报告
# ------------------------------------------------------------
def _print_tables(tables, name=''):
    """把 ic_tables 的几张表按顺序打出来。"""
    order = [('summary', '① IC 分段统计（全段 / 前后半 / 近期）'),
             ('yearly', '② 分年度 IC'),
             ('rolling', '③ 滚动 IC 摘要'),
             ('monthly', '④ 月度 IC 均值矩阵'),
             ('dist', '⑤ IC 分布'),
             ('decay', '⑥ IC 衰减（多前瞻期）')]
    for key, title in order:
        df = tables.get(key)
        if df is None or getattr(df, 'empty', True):
            continue
        print('\n[%s] %s' % (name or '', title) if name else '\n%s' % title)
        try:
            print(df.round(4).to_string())
        except Exception:
            print(df)


def ic_report(r, decay=None, window=12, save=False, outdir='pic_ic', prefix=None,
              print_tables=True):
    """单因子 IC 全套：打印 6 张表 + 画 6 张图（可选落盘）。

    典型用法：
        r = run_factor_test('px_pct_750', 'HS300', '2019-01-01', '2024-12-31', period=20)
        out = ic_report(r, save=True, outdir='pic_ic')

    返回 {'tables': {...}, 'figs': {...}, 'saved': [...]}
    """
    name = r.get('factor', '') if isinstance(r, dict) else ''
    tables = ic_tables(r, window=window, decay=decay)
    if print_tables:
        _print_tables(tables, name=name)
    figs = ic_plots(r, window=window)
    saved = []
    if save:
        saved = save_all(figs, outdir=outdir, prefix=prefix or name or 'factor')
    return {'tables': tables, 'figs': figs, 'saved': saved}


# ------------------------------------------------------------
# Part G-5 初筛：按 IC 口径过一遍 run_batch 的汇总表
# ------------------------------------------------------------
def screen_ic(df, min_abs_ic=0.03, min_abs_ir=0.3, min_t=2.0,
              min_coverage=0.8, max_turnover=None, top=None):
    """从 run_batch 的汇总表里按 IC 口径挑因子（初筛用）。

    负 IC 不算坏因子：方向由符号决定，这里只看绝对值，另加一列「方向」。
    返回新表（原表不动），按 |t值_修正| 从大到小排，列 通过=True/False。
    """
    if df is None or getattr(df, 'empty', True):
        return pd.DataFrame()
    d = df.copy()

    def col(name, default=np.nan):
        return pd.to_numeric(d[name], errors='coerce') if name in d.columns \
            else pd.Series(default, index=d.index, dtype=float)

    ic = col('IC均值')
    ir = col('IC_IR')
    t = col('t值_修正')
    cov = col('覆盖率')
    tno = col('换手%')

    d['方向'] = np.where(ic >= 0, 1, -1)
    d['|IC|'] = ic.abs()
    d['|IR|'] = ir.abs()
    ok = (d['|IC|'] >= min_abs_ic) & (d['|IR|'] >= min_abs_ir) & (t.abs() >= min_t)
    if min_coverage is not None:
        ok = ok & (cov >= min_coverage)
    if max_turnover is not None:
        ok = ok & (tno <= max_turnover)
    d['通过'] = ok.fillna(False)
    d = d.sort_values('|IR|', ascending=False)
    if top:
        d = d.head(int(top))
    return d


# ------------------------------------------------------------
# Part G-6 时序版的 IC 表（择时信号，口径不同，单独给一张）
# ------------------------------------------------------------
def ts_ic_table(r):
    """时序 result 的 IC 摘要表：整体 + 分段 + NW t（择时信号判读用）。

    注意 ic_segments 是 Series（index=各段末日期, values=该段 IC），不是 DataFrame。
    """
    if not isinstance(r, dict):
        return pd.DataFrame()
    st = r.get('ic_stats')
    if st is None or len(st) == 0:
        return pd.DataFrame()
    rows = [{
        '区间': '全段',
        '样本数': st.get('样本数', np.nan),
        'IC(Spearman)': st.get('IC(Spearman)', np.nan),
        'IC(Pearson)': st.get('IC(Pearson)', np.nan),
        't(OLS)': st.get('t值(OLS)', np.nan),
        't(NW)': st.get('t值(Newey-West)', np.nan),
        'NW滞后': st.get('NW滞后', np.nan),
        '分段IC均值': st.get('分段IC均值', np.nan),
        '分段IC_IR': st.get('分段IC_IR', np.nan),
    }]
    seg = r.get('ic_segments')
    if isinstance(seg, pd.Series) and len(seg):
        for idx, val in seg.items():
            lab = idx.strftime('%Y-%m') if hasattr(idx, 'strftime') else str(idx)
            rows.append({'区间': '段@' + lab, '样本数': np.nan,
                         'IC(Spearman)': float(val), 'IC(Pearson)': np.nan,
                         't(OLS)': np.nan, 't(NW)': np.nan, 'NW滞后': np.nan,
                         '分段IC均值': np.nan, '分段IC_IR': np.nan})
    return pd.DataFrame(rows).set_index('区间')


print('[Part G] IC 视角已就绪：screen_ic / ic_tables / ic_plots / ic_report / ts_ic_table')
