# -*- coding: utf-8 -*-
# ============================================================
# 统一因子研究框架（聚宽投资研究环境）
#   Part A 通用层：池配置 / 数据读取 / 三通道因子
#   Part B 截面研究：Rank IC / 分层 / 换手 / 分档 / 批量 / 组合 / 衰减
#   Part C 时序研究：择时信号 / NW修正IC / 状态分析 / 择时回测
#   Part D 回测衔接：可交易过滤
# 依赖：from jqdata import * 后的环境（get_price 等平台函数）
# ============================================================
import datetime
import inspect
import numpy as np
import os
import pandas as pd

# 数据源：聚宽平台 / 本地 jq_io（baostock + 缓存）/ jqdatasdk
#   想强制指定就设环境变量 JQ_DATA_SOURCE = jq | local | sdk
JQ_DATA_SOURCE = (os.environ.get('JQ_DATA_SOURCE') or 'auto').lower()
_source_ok = ''
if globals().get('LOCAL_MODE') and ('get_price' in globals()):
    _source_ok = 'local(notebook 内联 jq_io)'          # notebook 里 jq_io 已在前面的单元跑过
if (not _source_ok) and JQ_DATA_SOURCE in ('auto', 'jq'):
    try:
        from jqdata import *                            # 聚宽研究环境
        _source_ok = 'jqdata(聚宽平台)'
    except Exception:
        pass
if (not _source_ok) and JQ_DATA_SOURCE in ('auto', 'local'):
    try:
        from jq_io import *                             # 本地数据层（baostock）
        _source_ok = 'jq_io(baostock)'
    except Exception:
        pass
if (not _source_ok) and JQ_DATA_SOURCE in ('auto', 'sdk'):
    try:
        from jqdatasdk import *                         # 本地聚宽 SDK（需 auth）
        _source_ok = 'jqdatasdk'
    except Exception:
        pass
print('[数据源] %s' % (_source_ok or '没找到，请确认 jq_io.py 与本文件同目录'))

# ============================================================
# Part A-1 池配置
# ============================================================
ETF_POOL = {
    '宽基': {'510300.XSHG': '沪深300ETF', '510500.XSHG': '中证500ETF',
             '159915.XSHE': '创业板ETF', '588000.XSHG': '科创50ETF'},
    '防御': {'512890.XSHG': '红利低波ETF', '511360.XSHG': '短融ETF', '511010.XSHG': '国债ETF'},
    '行业': {'512010.XSHG': '医药ETF', '159928.XSHE': '消费ETF', '512660.XSHG': '军工ETF',
             '512480.XSHG': '半导体ETF', '512880.XSHG': '证券ETF', '512400.XSHG': '有色ETF',
             '512800.XSHG': '银行ETF', '515790.XSHG': '光伏ETF'},
    '港股': {'513180.XSHG': '恒生科技ETF', '513050.XSHG': '中概互联网ETF'},
    '海外': {'513100.XSHG': '纳指ETF', '513500.XSHG': '标普500ETF'},
    '商品': {'518880.XSHG': '黄金ETF'},
}

INDEX_POOL = {
    'HS300': '000300.XSHG', 'ZZ500': '000905.XSHG', 'ZZ800': '000906.XSHG',
    'ZZ1000': '000852.XSHG', 'SZ50': '000016.XSHG', 'CYB': '399006.XSHE',
}

def get_etf_pool(types=None, verbose=False):
    """ETF 池（固定列表）。types=None 全池；'宽基' 或 ['宽基','防御']。"""
    if types is None:
        types = list(ETF_POOL)
    if isinstance(types, str):
        types = [types]
    codes = []
    for t in types:
        codes.extend(ETF_POOL.get(t, {}).keys())
    if verbose:
        for t in types:
            print('  [%s] %s' % (t, '、'.join(ETF_POOL.get(t, {}).values())))
    return codes

def get_stock_pool(key, date=None):
    """股票池。key: 指数键名('HS300'...) / 指数代码 / 列表(原样返回)。"""
    if isinstance(key, (list, tuple, set)):
        return list(key)
    code = INDEX_POOL.get(key, key)
    try:
        return get_index_stocks(code, date)
    except Exception as e:
        raise ValueError('股票池解析失败 %s: %s' % (key, e))

def pool_desc(pool):
    """池的一句话描述。"""
    if isinstance(pool, str):
        return '股票池[%s] 动态成分' % pool
    return '固定列表 %d 只' % len(pool)

def preview_pool(pool, date=None, n=8):
    """快速查看池成分。"""
    if isinstance(pool, str):
        codes = get_stock_pool(pool, date)
        print('股票池 %s @ %s: %d 只' % (pool, str(date), len(codes)))
    else:
        codes = list(pool)
        print('固定池: %d 只' % len(codes))
    for c in codes[:n]:
        info = get_security_info(c)
        print('  %s %s' % (c, info.display_name if info else '查询不到'))
    if len(codes) > n:
        print('  ... 共 %d 只' % len(codes))
    return codes

def resolve_universe(pool, rebal_dates, verbose=True):
    """池 -> 历史并集（价格面板用）+ 每期成分明细。
    str: 逐期取成分（防未来函数）；list: 原样。"""
    if not isinstance(pool, str):
        return list(pool), None
    per_date = {}
    for d in rebal_dates:
        try:
            per_date[d] = get_stock_pool(pool, d)
        except Exception:
            per_date[d] = []
    non_empty = [v for v in per_date.values() if v]
    if not non_empty:
        raise ValueError('股票池 %s 在区间内取不到成分' % pool)
    all_stocks = sorted(set().union(*[set(v) for v in non_empty]))
    if verbose:
        sizes = [len(v) for v in non_empty]
        print('股票池 %s：每期成分 %d~%d 只，历史并集 %d 只'
              % (pool, min(sizes), max(sizes), len(all_stocks)))
    return all_stocks, per_date

# ============================================================
# Part A-2 数据读取（统一宽表 {字段: DataFrame(index=日期, columns=代码)}）
# ============================================================
def _to_wide(raw, fields):
    """get_price(panel=False) -> {字段: 宽表}。兼容长表/MultiIndex/宽表三种结构。"""
    if raw is None or raw.empty:
        return {}
    out = {}
    if ('time' in raw.columns) and ('code' in raw.columns):
        for f in fields:
            if f in raw.columns:
                out[f] = raw.pivot(index='time', columns='code', values=f)
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for f in fields:
            try:
                out[f] = raw.xs(f, axis=1, level=1)
            except Exception:
                out[f] = raw.xs(f, axis=1, level=0)
        return out
    out[fields[0]] = raw
    return out

def get_price_history(stock_list, date, count, fields=('close', 'money'), fq='pre'):
    """点位读取：截至 date 最近 count 根日线（因子计算用，股票/ETF 通用）。"""
    if isinstance(date, (datetime.datetime, datetime.date)):
        date = date.strftime('%Y-%m-%d')
    raw = get_price(list(stock_list), end_date=date, count=count, frequency='daily',
                    fields=list(fields), skip_paused=False, fq=fq, panel=False)
    return _to_wide(raw, fields)

def get_price_range(stock_list, start_date, end_date, fields=('close', 'money'), fq='pre'):
    """区间读取（收益面板/时序信号用）。"""
    raw = get_price(list(stock_list), start_date=start_date, end_date=end_date,
                    frequency='daily', fields=list(fields), skip_paused=False,
                    fq=fq, panel=False)
    return _to_wide(raw, fields)

# ============================================================
# Part A-3 因子三通道（自定义 > 聚宽因子库 > 财务表）
# ============================================================
CUSTOM_FACTORS = {}

def register_factor(name):
    """注册截面因子。签名 (stock_list, date) -> Series(index=代码, name=因子名)。"""
    def decorator(func):
        CUSTOM_FACTORS[name] = func
        return func
    return decorator

val = bal = cf = inc = ind = None

def _ensure_table_columns():
    """探测财务表字段（懒加载+缓存）。"""
    global val, bal, cf, inc, ind
    if val is not None:
        return
    def cols_of(table):
        df = get_fundamentals(query(table).limit(1))
        if df is None or df.empty:
            return []
        return [c for c in df.columns if c not in ('id', 'code')]
    val = cols_of(valuation); bal = cols_of(balance); cf = cols_of(cash_flow)
    inc = cols_of(income);    ind = cols_of(indicator)

_library_names = None

def library_factor_names():
    """聚宽因子库全部因子名（懒加载+缓存）。本地数据源没有因子库，直接空集。"""
    global _library_names
    if _library_names is not None:
        return _library_names
    if globals().get('LOCAL_MODE'):
        _library_names = set()
        return _library_names
    names = set()
    try:
        from jqfactor import get_all_factors
        raw = get_all_factors()          # 注意必须带括号
        if isinstance(raw, pd.DataFrame):
            col = 'factor' if 'factor' in raw.columns else raw.columns[0]
            names = set(raw[col].astype(str))
        else:
            names = set(raw)
    except Exception as e:
        print('因子库清单获取失败，使用手动清单：', e)
        names = {'Skewness60', 'TVSTD60', 'MAD60', 'BETA60', 'BIAS20', 'PSY', 'VMACD'}
    _library_names = names
    return _library_names

def get_library_factor_series(factor, stock_list, date):
    """因子库单日截面 -> Series。"""
    d = date.strftime('%Y-%m-%d') if isinstance(date, (datetime.datetime, datetime.date)) else str(date)
    from jqfactor import get_factor_values
    raw = get_factor_values(securities=list(stock_list), factors=[factor],
                            start_date=d, end_date=d)
    df = raw[factor]
    if df is None or df.empty:
        return pd.Series(np.nan, index=list(stock_list), name=factor)
    s = df.iloc[-1].reindex(stock_list)
    s.name = factor
    return s

def get_factor_data(factor, stock_list, date):
    """三通道统一入口。注意：重名时高优先级静默覆盖；
    因子库/财务通道仅支持股票，ETF 只能用自定义通道。"""
    stock_list = list(stock_list)
    if factor in CUSTOM_FACTORS:
        return CUSTOM_FACTORS[factor](stock_list, date)
    if globals().get('LOCAL_MODE'):
        raise ValueError('因子 %r 不在自定义因子里。本地数据源只有行情，没有因子库和财务表，'
                         '请用 @register_factor 自己写（Part A-4 有模板）' % (factor,))
    if factor in library_factor_names():
        try:
            return get_library_factor_series(factor, stock_list, date)
        except Exception as e:
            raise RuntimeError('因子库取值失败 %s: %s' % (factor, e))
    _ensure_table_columns()
    if factor in val:
        q = query(valuation.code, getattr(valuation, factor)).filter(valuation.code.in_(stock_list))
    elif factor in bal:
        q = query(balance.code, getattr(balance, factor)).filter(balance.code.in_(stock_list))
    elif factor in cf:
        q = query(cash_flow.code, getattr(cash_flow, factor)).filter(cash_flow.code.in_(stock_list))
    elif factor in inc:
        q = query(income.code, getattr(income, factor)).filter(income.code.in_(stock_list))
    elif factor in ind:
        q = query(indicator.code, getattr(indicator, factor)).filter(indicator.code.in_(stock_list))
    else:
        raise ValueError('因子 %s 不在 自定义/因子库/财务表 中' % factor)
    df = get_fundamentals(q, date)
    if df is None or df.empty:
        return pd.Series(np.nan, index=stock_list, name=factor)
    df.index = df['code']
    return df[factor]

# ============================================================
# Part A-4 内置示例因子（照写法添加自己的）
# ============================================================
@register_factor('mom_20')
def factor_mom_20(stock_list, date):
    """20日动量（越大越强，方向 +1）"""
    close = get_price_history(stock_list, date, 21, ['close']).get('close')
    if close is None or close.empty or len(close) < 3:
        return pd.Series(np.nan, index=list(stock_list), name='mom_20')
    s = (close.iloc[-1] / close.iloc[0] - 1.0).reindex(stock_list); s.name = 'mom_20'
    return s

@register_factor('mom_60')
def factor_mom_60(stock_list, date):
    """60日动量"""
    close = get_price_history(stock_list, date, 61, ['close']).get('close')
    if close is None or close.empty or len(close) < 3:
        return pd.Series(np.nan, index=list(stock_list), name='mom_60')
    s = (close.iloc[-1] / close.iloc[0] - 1.0).reindex(stock_list); s.name = 'mom_60'
    return s

@register_factor('rev_20')
def factor_rev_20(stock_list, date):
    """20日反转（用方向 -1 即"买跌"）"""
    close = get_price_history(stock_list, date, 21, ['close']).get('close')
    if close is None or close.empty or len(close) < 3:
        return pd.Series(np.nan, index=list(stock_list), name='rev_20')
    s = (close.iloc[-1] / close.iloc[0] - 1.0).reindex(stock_list); s.name = 'rev_20'
    return s

@register_factor('vol_20')
def factor_vol_20(stock_list, date):
    """20日波动率（方向一般 -1）"""
    close = get_price_history(stock_list, date, 21, ['close']).get('close')
    if close is None or close.empty or len(close) < 3:
        return pd.Series(np.nan, index=list(stock_list), name='vol_20')
    s = close.pct_change().std().reindex(stock_list); s.name = 'vol_20'
    return s

@register_factor('ma_score')
def factor_ma_score(stock_list, date):
    """均线排列离散打分：完全多头 +4 / 完全空头 -4 / 其余按净多头关系数(-3~+3)"""
    need = {}
    for n in (5, 10, 20, 40):
        h = get_price_history(stock_list, date, n, ['close']).get('close')
        if h is None or h.empty:
            return pd.Series(np.nan, index=list(stock_list), name='ma_score')
        need[n] = h.mean()
    ma5, ma10, ma20, ma40 = need[5], need[10], need[20], need[40]
    net = ((ma5 > ma10).astype(int) + (ma10 > ma20).astype(int) + (ma20 > ma40).astype(int)
           - (ma5 < ma10).astype(int) - (ma10 < ma20).astype(int) - (ma20 < ma40).astype(int))
    score = net.astype(float)
    score[(ma5 > ma10) & (ma10 > ma20) & (ma20 > ma40)] = 4.0
    score[(ma5 < ma10) & (ma10 < ma20) & (ma20 < ma40)] = -4.0
    score = score.reindex(stock_list); score.name = 'ma_score'
    return score

@register_factor('shrink_rev')
def factor_shrink_rev(stock_list, date):
    """缩量抄底打分 0~3：下跌 + 波动收敛 + 缩量，各1分"""
    hist = get_price_history(stock_list, date, 21, ['close', 'money'])
    close, money = hist.get('close'), hist.get('money')
    if close is None or money is None or close.empty or money.empty:
        return pd.Series(np.nan, index=list(stock_list), name='shrink_rev')
    ret20 = close.iloc[-1] / close.iloc[0] - 1.0
    vol5, vol20 = close.pct_change().iloc[-5:].std(), close.pct_change().std()
    amt5, amt20 = money.iloc[-5:].mean(), money.mean()
    s = ((ret20 < 0).astype(float)
         + (vol5 / vol20.replace(0, np.nan) < 0.8).astype(float)
         + (amt5 / amt20.replace(0, np.nan) < 0.8).astype(float)).reindex(stock_list)
    s.name = 'shrink_rev'
    return s

@register_factor('price_pos')
def factor_price_pos(stock_list, date):
    """价格分位（3年750日）：当前价在区间位置 0~1，低=便宜（方向 -1）"""
    close = get_price_history(stock_list, date, 750, ['close']).get('close')
    if close is None or close.empty or len(close) < 100:
        return pd.Series(np.nan, index=list(stock_list), name='price_pos')
    lo, hi = close.min(), close.max()
    s = ((close.iloc[-1] - lo) / (hi - lo).replace(0, np.nan)).reindex(stock_list)
    s.name = 'price_pos'
    return s

# ============================================================
# Part B-1 调仓日控制
# ============================================================
def get_rebalance_dates(start, end, freq='M', rule='first', every=1):
    """调仓日序列。
    freq : 'D' 每 every 个交易日 | 'W'/'M'/'Q'/'Y' 每 every 周/月/季/年
    rule : 'first' 每期首个交易日 / 'last' 每期最后交易日
    every: 间隔倍数（D=交易日数；W/M/Q/Y=周期数）
    想完全自定义：直接把日期列表传给入口函数的 rebal_dates 参数。"""
    days = list(get_trade_days(start_date=start, end_date=end))
    if not days:
        return []
    every = max(1, int(every))
    if freq in ('D', 'd'):
        return days[::every]
    d = {}
    for x in days:
        if freq in ('Y', 'y', 'A', 'a'):
            key = x.year
        elif freq in ('Q', 'q'):
            key = (x.year, (x.month - 1) // 3)
        elif freq in ('M', 'm'):
            key = (x.year, x.month)
        else:
            key = (x.year, x.isocalendar()[1])
        if rule == 'last':
            d[key] = x
        else:
            d.setdefault(key, x)
    return sorted(d.values())[::every]

def _median_interval_days(rebal_dates):
    """调仓日中位间隔（交易日）：周频≈5，月频≈21，季频≈63。"""
    if rebal_dates is None or len(rebal_dates) < 2:
        return None
    first = pd.Timestamp(rebal_dates[0]).date()
    last = pd.Timestamp(rebal_dates[-1]).date()
    cal = list(get_trade_days(start_date=first, end_date=last))
    # [修] 统一 key 成 datetime.date：get_trade_days 有的环境返回 date、有的返回 Timestamp，
    #      混用时 pos.get(date) 全部落空 -> gaps 为空 -> 静默退化成 1 日间隔（ratio 直接算错）。
    pos = {pd.Timestamp(d).date(): i for i, d in enumerate(cal)}
    idx = [pos.get(pd.Timestamp(d).date()) for d in rebal_dates]
    gaps = [idx[i] - idx[i - 1] for i in range(1, len(idx))
            if idx[i] is not None and idx[i - 1] is not None]
    return int(np.median(gaps)) if gaps else None

def preview_rebalance_dates(start, end, freq='M', rule='first', every=1, n=6):
    """跑检验前先确认调仓日设置。"""
    dates = get_rebalance_dates(start, end, freq=freq, rule=rule, every=every)
    interval = _median_interval_days(dates)
    print('freq=%s rule=%s every=%s -> %d 个调仓日 | 间隔中位 %s 交易日'
          % (freq, rule, every, len(dates), interval))
    if dates:
        print('  前%d: %s' % (min(n, len(dates)), [str(x) for x in dates[:n]]))
        print('  后%d: %s' % (min(n, len(dates)), [str(x) for x in dates[-n:]]))
    return dates

def trading_day_shift(d, n):
    return get_trade_days(start_date=d, count=n + 1)[-1]

# ============================================================
# Part B-2 面板 / 前向收益 / 统一准备
# ============================================================
def build_factor_panel(factor, rebal_dates, pool):
    """多期因子面板。pool=str 每期取当期成分（防未来函数）；list 固定池。"""
    rows, err, first_err = {}, 0, None
    for d in rebal_dates:
        codes = get_stock_pool(pool, d) if isinstance(pool, str) else list(pool)
        try:
            s = pd.Series(get_factor_data(factor, codes, d)).reindex(codes)
        except Exception as e:
            if first_err is None:
                first_err = '%s: %s' % (type(e).__name__, e)
            s = pd.Series(np.nan, index=list(codes)); err += 1
        rows[d] = s
    panel = pd.DataFrame(rows).T.sort_index()
    if err:
        print('  提示：%s 有 %d/%d 期取值失败 | 首个错误：%s'
              % (factor, err, len(rebal_dates), first_err))
    return panel

def build_forward_returns(codes, start, price_end, period, strict=False):
    """fwd(t)=close(t+period)/close(t)-1；strict=True 从 t+1 起算（保守）。"""
    if pd.Timestamp(price_end) <= pd.Timestamp(start):
        return pd.DataFrame()
    px = get_price_range(codes, start, price_end, ('close',)).get('close')
    if px is None or px.empty:
        return pd.DataFrame()
    px = px.sort_index()
    if strict:
        return px.shift(-(period + 1)) / px.shift(-1) - 1.0
    return px.shift(-period) / px - 1.0

def prepare_data(factors, pool, start, end, period=20, freq='M', rule='first', every=1,
                 rebal_dates=None, strict=False, non_overlap=False, panels=None, verbose=True):
    """统一准备流水线：截断日期 -> 调仓日 -> (抽稀) -> 池解析 -> 因子面板 -> 收益面板。"""
    today = datetime.date.today()
    end_eff = end
    try:
        if pd.Timestamp(end).date() > today:
            end_eff = today
            if verbose:
                print('提示：end(%s) 超出今天，自动截断到 %s' % (str(end), end_eff))
    except Exception:
        pass
    if rebal_dates is None:
        rebal_dates = get_rebalance_dates(start, end_eff, freq=freq, rule=rule, every=every)
    if len(rebal_dates) < 4:
        raise ValueError('调仓日太少(%d)，请扩大区间或放宽频率' % len(rebal_dates))
    interval0 = _median_interval_days(rebal_dates) or 1
    step = 1
    if non_overlap:
        step = int(np.ceil(float(period) / float(interval0)))
        if step > 1:
            rebal_dates = rebal_dates[::step]
    interval = _median_interval_days(rebal_dates) or interval0
    ratio = max(1.0, float(period) / float(interval))

    all_stocks, per_date = resolve_universe(pool, rebal_dates, verbose)
    if verbose:
        print('调仓日 %d 个 | 间隔中位 %d 日 | 持仓 %d 日 | 重叠倍数 %.2f%s'
              % (len(rebal_dates), interval, period, ratio,
                 ('（已抽稀 step=%d，样本不重叠）' % step) if step > 1 else ''))

    panels_out = {} if panels is None else panels
    for f in factors:
        if f not in panels_out:
            panels_out[f] = build_factor_panel(f, rebal_dates, pool)

    price_end = trading_day_shift(rebal_dates[-1], period + 5)
    if pd.Timestamp(price_end).date() > today:
        price_end = today
    fwd = build_forward_returns(all_stocks, rebal_dates[0], price_end, period, strict)
    usable = 0
    if not fwd.empty:
        for d in rebal_dates:
            t = pd.Timestamp(d)
            if t in fwd.index and fwd.loc[t].notna().any():
                usable += 1
    if verbose and usable < len(rebal_dates):
        print('提示：%d/%d 期有前向收益（末尾 %d 期数据不足，属正常）'
              % (usable, len(rebal_dates), len(rebal_dates) - usable))
    return {'rebal_dates': rebal_dates, 'panels': panels_out, 'fwd': fwd,
            'interval': interval, 'ratio': ratio, 'step': step, 'usable': usable,
            'universe': all_stocks, 'per_date_pool': per_date}

# ============================================================
# Part B-3 截面检验核心
# ============================================================
def factor_ic_stats(factor_panel, fwd_panel, min_n=3, overlap_ratio=1.0):
    """逐期 Rank IC（Spearman）。overlap_ratio>1 输出有效期数与 t值_修正。"""
    dates, ics = [], []
    for d in factor_panel.index:
        t = pd.Timestamp(d)
        if t not in fwd_panel.index:
            continue
        f = factor_panel.loc[d].dropna()
        r = fwd_panel.loc[t].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_n:
            continue
        ic = f[common].astype(float).corr(r[common].astype(float), method='spearman')
        if pd.isna(ic):
            continue
        dates.append(t); ics.append(ic)
    ic = pd.Series(ics, index=pd.DatetimeIndex(dates), name='rank_ic')
    if len(ic) < 3:
        return ic, pd.Series(dtype=float)
    n = len(ic)
    mean, std = ic.mean(), ic.std(ddof=1)
    ratio = max(1.0, float(overlap_ratio))
    stats = pd.Series({
        '期数': float(n),
        '有效期数': n / ratio,
        'IC均值': mean,
        'IC标准差': std,
        'IC_IR': mean / std if std > 0 else np.nan,
        't值': mean / (std / np.sqrt(n)) if std > 0 else np.nan,
        't值_修正': mean / (std / np.sqrt(n / ratio)) if std > 0 else np.nan,
        'IC>0占比': float((ic > 0).mean()),
        'IC偏度': float(ic.skew()) if n > 2 else np.nan,
    })
    return ic, stats

def _eff_q(factor_panel, q, min_each=2):
    cov = [len(factor_panel.loc[d].dropna()) for d in factor_panel.index]
    med = int(np.median(cov)) if cov else 0
    if med < min_each * 2:
        return 0
    return int(min(q, max(2, med // min_each)))

def quantile_returns(factor_panel, fwd_panel, q=5, min_each=2):
    """分层收益：自适应组数（小池子自动降组），返回 index=期, columns=Q1..Qn。"""
    q_eff = _eff_q(factor_panel, q, min_each)
    if q_eff < 2:
        return pd.DataFrame()
    rows = {}
    for d in factor_panel.index:
        t = pd.Timestamp(d)
        if t not in fwd_panel.index:
            continue
        f = factor_panel.loc[d].dropna()
        r = fwd_panel.loc[t].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < q_eff * min_each:
            continue
        f2, r2 = f[common], r[common]
        try:
            grp = pd.qcut(f2.rank(method='first'), q_eff, labels=False) + 1
        except Exception:
            continue
        ser = r2.groupby(grp).mean()
        # [修] groupby 出来的 index 是整数 1..N，必须显式改成 'Q1'..'QN'，
        #      否则下面 reindex(columns=['Q1'..]) 找不到列，整张表变 NaN。
        ser.index = ['Q%d' % int(g) for g in ser.index]
        rows[t] = ser
    out = pd.DataFrame(rows).T
    if out.empty:
        return pd.DataFrame()
    return out.reindex(columns=['Q%d' % i for i in range(1, q_eff + 1)])

def quantile_turnover(factor_panel, q=5, group=None, min_each=2):
    """最高（或指定）因子分组的平均换手率。"""
    q_eff = _eff_q(factor_panel, q, min_each)
    if q_eff < 2:
        return np.nan
    g = q_eff if group is None else min(group, q_eff)
    mem = {}
    for d in factor_panel.index:
        f = factor_panel.loc[d].dropna()
        if len(f) < q_eff * min_each:
            continue
        grp = pd.qcut(f.rank(method='first'), q_eff, labels=False) + 1
        mem[d] = set(grp[grp == g].index)
    ds = sorted(mem); tos = []
    for i in range(1, len(ds)):
        prev, curr = mem[ds[i - 1]], mem[ds[i]]
        if prev:
            tos.append(1 - len(prev & curr) / len(prev))
    return float(np.mean(tos)) if tos else np.nan

def score_bucket_analysis(factor_panel, fwd_panel, demean=True, max_buckets=15,
                          min_periods=3, verbose=True):
    """离散因子分档诊断：按原始分值精确分组，算各档 平均相对收益/t值/胜率/档内均只数。
    demean=True 每期先减全池均值（剥离大盘涨跌，不影响 IC）。"""
    if factor_panel is None or factor_panel.empty or fwd_panel is None or fwd_panel.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    uniq = []
    for d in factor_panel.index:
        row = factor_panel.loc[d].dropna()
        if len(row):
            uniq.append(row.round(6).nunique())
    nun = int(np.median(uniq)) if uniq else 0
    if nun == 0 or nun > max_buckets:
        if verbose:
            print('  提示：该因子每期约 %d 个不同取值，分档表不适用（看 IC 与分层收益）' % nun)
        return pd.DataFrame(), pd.Series(dtype=float)
    per, sizes, mono = {}, {}, []
    for d in factor_panel.index:
        t = pd.Timestamp(d)
        if t not in fwd_panel.index:
            continue
        f = factor_panel.loc[d].dropna()
        r = fwd_panel.loc[t].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 3:
            continue
        f2, r2 = f[common].astype(float), r[common].astype(float)
        try:
            mono.append(f2.corr(r2, method='spearman'))
        except Exception:
            pass
        if demean:
            r2 = r2 - r2.mean()
        for s, sub in r2.groupby(f2):
            per.setdefault(s, {})[t] = sub.mean()
            sizes.setdefault(s, []).append(len(sub))
    rows = []
    for s in sorted(per):
        ser = pd.Series(per[s]).dropna()
        n = len(ser)
        avg_n = float(np.mean(sizes[s])) if s in sizes else np.nan
        if n < min_periods:
            rows.append({'分值': s, '期数': n, '平均相对收益%': np.nan,
                         't值': np.nan, '胜率': np.nan, '档内均只数': avg_n})
            continue
        m, sd = ser.mean(), ser.std(ddof=1)
        rows.append({'分值': s, '期数': n, '平均相对收益%': 100 * m,
                     't值': m / (sd / np.sqrt(n)) if sd > 0 else np.nan,
                     '胜率': float((ser > 0).mean()), '档内均只数': avg_n})
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        vals = np.asarray(tbl['分值'], dtype=float)
        if np.allclose(vals, np.round(vals)):
            tbl['分值'] = tbl['分值'].astype(int)
    return tbl, pd.Series(mono).dropna()

def diagnose_momentum_reversal(bucket_table, ic_mean, ic_th=0.02):
    """由分档表 + IC 均值判定动量/反转。"""
    if bucket_table is None or bucket_table.empty or pd.isna(ic_mean):
        return '样本不足，无法判定'
    b = bucket_table.dropna(subset=['平均相对收益%', '分值'])
    if len(b) < 2:
        return '样本不足，无法判定'
    corr = b['分值'].corr(b['平均相对收益%'], method='spearman')
    spread = b['平均相对收益%'].iloc[-1] - b['平均相对收益%'].iloc[0]
    if ic_mean > ic_th:
        v = '高分档跑赢 → 动量占优（高分可追）'
    elif ic_mean < -ic_th:
        v = '高分档跑输 → 反转占优（高分应反向/低分可买）'
    else:
        v = '|IC均值| < %.2f，方向不明确' % ic_th
    return '%s；分档单调性=%.2f；最高档-最低档=%.2f%%/期' % (v, corr, spread)

def factor_corr_matrix(panels):
    """多因子截面相关性（堆叠所有期后算 Spearman）。"""
    flat = {}
    for k, p in panels.items():
        flat[k] = p.stack()
    return pd.DataFrame(flat).corr(method='spearman')

# ============================================================
# Part B-3b Barra 风格中性化 / 归因（可选开关）
# ============================================================
# 【这一节解决什么问题】
#   上面的截面检验默认把你的因子当「纯 alpha」：IC 高、分层单调就算有效。
#   但如果这个因子其实只是「小市值」「高 beta」「低波动」换了个写法，
#   那赚的就是风格的钱 —— 风格一反转，因子跟着崩，还以为是自己 alpha 失效。
#   Barra 模式只回答两件事：
#     (1) 我的因子跟哪些风格高度相关？      -> 风格暴露表
#     (2) 把风格剥掉以后，还剩多少 IC？      -> 中性化后 IC / 纯因子收益 λf
#
# 【四个开关，随时切】
#   BARRA_MODE='off'     : 什么都不做（默认，跑出来和以前完全一样）
#   BARRA_MODE='neutral' : 事前中性化。每期把因子对风格做横截面回归取残差，
#                          再用残差跑 IC / 分层。适合检验「非风格因子」。
#   BARRA_MODE='attr'    : 事后归因。因子不打磨，把前向收益对「因子+风格」回归，
#                          看控制风格之后因子还有没有解释力（λf、t、R2）。
#   BARRA_MODE='both'    : 两个都做（慢一点，但一次看全）。
#   临时切换：run_factor_test(..., barra='neutral')，不改全局也行。
#
# 【重要：这里用的是「价量代理风格」，不是真 Barra】
#   真 Barra 有 10 个风格 + 行业哑变量，其中 B/P、E/P、成长、杠杆必须靠财务数据。
#   本地数据层目前只拉了行情，所以这里只能用行情算出来的 8 个描述符：
#     beta / size / momentum / resvol / nlsize / liquidity / volat / reversal
#   方向和量级都跟真 Barra 对得上，判断「你的因子是不是风格的替身」已经够用。
#   以后拿到财务数据，在 barra_exposures 里往 out 加一列、再进 BARRA_STYLES 即可。
#
# 【两个必须知道的坑】
#   1) 用过去 252 日窗口算风格暴露 -> 至少要有 253 根日线，样本开头几期会被跳过；
#      这是对的，别用不足的窗口硬算，否则 beta 全是噪音。
#   2) 中性化后 IC 会普遍变小。这不是变差，是去掉风格水分之后的真实值。
#      如果中性化前后差不多，说明你的因子本来就干净（好事）。
# ============================================================

BARRA_MODE = 'off'
BARRA_STYLES = ('beta', 'size', 'momentum', 'resvol', 'nlsize',
                'liquidity', 'volat', 'reversal')
BARRA_WINSOR = 0.01          # 两端各砍 1% 的极值，避免一只妖股把回归带偏
BARRA_MIN_N = 20             # 一期至少这么多只才做横截面回归
BARRA_INDEX = '000300.XSHG'  # Beta / ResidVol 的市场基准
BARRA_DATA = {}              # 行情缓存（同一个池子反复跑时只取一次数）

STYLE_CN = {'beta': 'Beta', 'size': 'Size', 'momentum': 'Momentum',
            'resvol': 'ResidVol', 'nlsize': 'NonLinSize', 'liquidity': 'Liquidity',
            'volat': 'Volatility', 'reversal': 'Reversal'}

# 每个风格一句话说明 —— 看暴露表时对着这一列读
STYLE_NOTE = {
    'beta': '跟着大盘涨跌的敏感度，>1 放大，<1 钝化',
    'size': '流通市值取对数：大=大盘股，小=小盘股',
    'momentum': '过去 12 个月、剔除最近 1 个月的涨幅',
    'resvol': '扣掉大盘之后的残差波动，高=特质风险大',
    'nlsize': 'size 标准化后的立方，抓「极端大盘/极端小盘」的非线性',
    'liquidity': '近 21 日平均成交额取对数，高=好进好出',
    'volat': '近 60 日日收益标准差',
    'reversal': '近 21 日涨幅，高=刚涨过（短期反转的另一面）',
}

_BARRA_ALIAS = {'none': 'off', '0': 'off', 'false': 'off', 'off': 'off', '关': 'off',
                'n': 'neutral', 'neutral': 'neutral', '中性': 'neutral', '中性化': 'neutral',
                'a': 'attr', 'attr': 'attr', '归因': 'attr', '回归': 'attr',
                'b': 'both', 'both': 'both', 'all': 'both', '都做': 'both'}


def barra_mode_of(mode=None):
    """把各种写法统一成 off / neutral / attr / both；写错直接报错，不要静默放行。"""
    if mode is None:
        return BARRA_MODE
    m = str(mode).strip().lower()
    if not m:
        return 'off'
    if m not in _BARRA_ALIAS:
        raise ValueError("BARRA_MODE 只认 off / neutral / attr / both，收到 %r" % (mode,))
    return _BARRA_ALIAS[m]


def set_barra_mode(mode='neutral', styles=None, winsor=None, verbose=True):
    """改全局 Barra 开关（参数区里改完调一下，后面全部跟着走）。"""
    global BARRA_MODE, BARRA_STYLES, BARRA_WINSOR
    BARRA_MODE = barra_mode_of(mode)
    if styles:
        bad = [s for s in styles if s not in STYLE_NOTE]
        if bad:
            raise ValueError('未知风格 %r，可选：%s' % (bad, ', '.join(STYLE_NOTE)))
        BARRA_STYLES = tuple(styles)
    if winsor is not None:
        BARRA_WINSOR = float(winsor)
    if verbose:
        print('[Barra] 模式=%s | 风格=%d 个（%s）| 缩尾=%.3f'
              % (BARRA_MODE, len(BARRA_STYLES), ', '.join(BARRA_STYLES), BARRA_WINSOR))
    return BARRA_MODE


def barra_data_note():
    """Size 算不出来时给一句能照做的话。"""
    print('[Barra] 本地缓存缺 turn（换手率）字段，Size / NonLinSize 本轮跳过。'
          '要补：from jq_io import upgrade_cache; upgrade_cache(股票列表) —— 只重抓同一区间，补上 turn。')


def _b_winsor(s):
    """两端缩尾：最极端的 BARRA_WINSOR 比例压到分位值上。"""
    s = pd.to_numeric(pd.Series(s), errors='coerce').astype(float)
    v = s.dropna()
    if len(v) < 5 or BARRA_WINSOR <= 0:
        return s
    lo, hi = v.quantile(BARRA_WINSOR), v.quantile(1.0 - BARRA_WINSOR)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return s
    return s.clip(lower=lo, upper=hi)


def _b_z(s):
    """横截面标准化（减均值除标准差）。回归前不做这一步，系数量纲就没法比。"""
    s = pd.to_numeric(pd.Series(s), errors='coerce').astype(float)
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return s * np.nan
    return (s - s.mean()) / sd


def _b_prep(s):
    """缩尾 + 标准化，一步到位：风格值、回归输入都用它。"""
    return _b_z(_b_winsor(s))


def build_barra_data(codes, start, end, index_code=None, fq='pre', verbose=True):
    """一次把 Barra 要用的行情全部拉齐，后面每期只做切片，不再重复取数。

    取的是三条宽表（index=日期, columns=代码）：close / money / turn，
    再加一条指数收盘（Beta 和 ResidVol 要用）。turn 只有 baostock 新缓存才有，
    老缓存没有的话 Size 会被跳过（不是错，是数据没到）。
    """
    index_code = index_code or BARRA_INDEX
    codes = list(codes)
    key = (tuple(sorted(codes)), str(start), str(end), index_code)
    if key in BARRA_DATA:
        return BARRA_DATA[key]
    if verbose:
        print('[Barra] 取数 %d 只 × %s ~ %s（风格描述符要往前回看 252 日）'
              % (len(codes), start, end))
    px = get_price_range(codes, start, end, ('close', 'money', 'turn'), fq=fq) or {}
    close = px.get('close')
    if close is None or close.empty:
        raise ValueError('[Barra] 取不到收盘价，没法算风格暴露')
    money = px.get('money')
    turn = px.get('turn')
    if turn is not None and not turn.empty and turn.notna().sum().sum() == 0:
        turn = None                      # 全是空列等于没取到
    ix = None
    try:
        ic_ = get_price_range([index_code], start, end, ('close',), fq='none') or {}
        ic_ = ic_.get('close')
        if ic_ is not None and not ic_.empty:
            ix = ic_.iloc[:, 0].sort_index()
    except Exception as e:
        if verbose:
            print('[Barra] 指数 %s 取不到（%s），Beta 用等权市场代替' % (index_code, e))
    data = {'close': close.sort_index(), 'money': money, 'turn': turn,
            'index': ix, 'index_code': index_code,
            'codes': list(close.columns), 'has_turn': turn is not None}
    if verbose:
        print('[Barra] 行情 %d 行 × %d 列 | 基准 %s | 换手字段 %s'
              % (close.shape[0], close.shape[1], index_code,
                 '有' if data['has_turn'] else '无'))
    BARRA_DATA[key] = data
    return data


def barra_exposures(data, date, styles=None, verbose=False):
    """某一天的横截面风格暴露：index=代码，columns=风格名（已缩尾 + 标准化）。

    只用 date 当天及之前的行情（含当天），不碰未来一根 K 线。
    """
    styles = list(styles or BARRA_STYLES)
    close = data['close']
    t = pd.Timestamp(date)
    pos = close.index.searchsorted(t, side='right')   # side='right' 表示把当天算进来
    if pos < 40:
        return pd.DataFrame()
    hist = close.iloc[:pos]
    ret = hist.pct_change()
    n = len(hist)

    # ---- 市场收益：有指数用指数，没有就退化成等权市场
    ix = data.get('index')
    if ix is not None and len(ix):
        rm = ix.reindex(hist.index).ffill().bfill().pct_change()
    else:
        rm = ret.mean(axis=1)

    out = {}
    # ---- Beta / ResidVol（252 日窗口）
    if n > 61:
        w = min(252, n - 1)
        r_w, m_w = ret.iloc[-w:], rm.iloc[-w:]
        var = m_w.var(ddof=1)
        if np.isfinite(var) and var > 0:
            beta = r_w.apply(lambda col: col.cov(m_w) / var)
            out['beta'] = beta
            # 残差 = 个股收益 - beta × 市场收益；用 np.outer 而不是 Series 相乘，
            # 因为 beta 的 index 是代码、m_w 的 index 是日期，直接相乘会按标签乱对齐。
            resid = r_w.values - np.outer(m_w.values, beta.values)
            out['resvol'] = pd.Series(resid.std(axis=0, ddof=1), index=r_w.columns)
    # ---- Momentum（12 个月剔最近 1 个月）/ Reversal（近 1 个月）
    if n > 253:
        out['momentum'] = hist.iloc[-22] / hist.iloc[-253] - 1.0
    if n > 22:
        out['reversal'] = hist.iloc[-1] / hist.iloc[-22] - 1.0
    # ---- Volatility（60 日）
    if n > 61:
        out['volat'] = ret.iloc[-60:].std(ddof=1)
    # ---- Liquidity（21 日平均成交额，取对数）
    money = data.get('money')
    if money is not None and not money.empty:
        m21 = money.reindex(hist.index).iloc[-21:].mean()
        out['liquidity'] = np.log(m21.replace(0, np.nan))
    # ---- Size（流通市值）：换手率反推流通股本
    #      turn = 成交量 / 流通股本 × 100  =>  流通股本 = money/close × 100 / turn
    if data.get('has_turn'):
        turn = data['turn']
        vol21 = (money.reindex(hist.index) / hist).iloc[-21:] if money is not None else None
        if vol21 is not None:
            shares = vol21.mul(100.0).div(turn.reindex(hist.index).iloc[-21:])
            shares = shares.median()          # 中位数，抗单日异常
            out['size'] = np.log((hist.iloc[-1] * shares).replace(0, np.nan))
    # ---- NonLinSize：size 立方（需要先标准化）
    if 'size' in out:
        out['nlsize'] = _b_z(out['size']) ** 3

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    keep = [s for s in styles if s in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].apply(_b_prep)              # 每列都缩尾 + 标准化
    df = df.dropna(how='all')
    if verbose:
        print('[Barra] %s 暴露 %d 只 × %d 个风格' % (pd.Timestamp(date).date(), len(df), len(keep)))
    return df


def build_exposure_panel(data, dates, styles=None, min_n=None, verbose=True):
    """逐调仓日算暴露 -> {日期: DataFrame}。样本不够的那几期直接跳过。"""
    min_n = min_n or BARRA_MIN_N
    out, skip = {}, 0
    for d in dates:
        try:
            e = barra_exposures(data, d, styles=styles)
        except Exception as ex:
            e, err = pd.DataFrame(), ex
            if verbose:
                print('[Barra] %s 暴露算失败：%s: %s'
                      % (pd.Timestamp(d).date(), type(err).__name__, err))
        if e is None or e.empty or len(e) < min_n:
            skip += 1
            continue
        out[pd.Timestamp(d)] = e
    if verbose:
        if skip:
            print('[Barra] %d 期暴露不足（样本 < %d 或历史不满 252 日），已跳过'
                  % (skip, min_n))
        print('[Barra] 有效暴露期数 %d/%d' % (len(out), len(list(dates))))
    return out


def _b_ols(y, X):
    """带常数的横截面最小二乘：y = a + X·b + e。

    返回 (b, resid, r2)。样本不够、或者矩阵奇异时返回 (None, None, nan)，
    调用方跳过这一期就行 —— 宁可少一期，不要拿不可靠的回归去污染结论。
    """
    y = pd.Series(y).astype(float)
    X = pd.DataFrame(X).astype(float)
    df = pd.concat([y.rename('_y'), X], axis=1).dropna()
    if len(df) < max(8, X.shape[1] + 3):
        return None, None, np.nan
    yy = df['_y'].values.astype(float)
    Xm = np.column_stack([np.ones(len(df)), df[X.columns].values.astype(float)])
    try:
        b = np.linalg.lstsq(Xm, yy, rcond=None)[0]
    except Exception:
        return None, None, np.nan
    resid = yy - Xm.dot(b)
    ss = ((yy - yy.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss if ss > 0 else np.nan
    return pd.Series(b[1:], index=X.columns), pd.Series(resid, index=df.index), r2


def neutralize_panel(panel, expo_panel, min_n=None, verbose=True):
    """因子对风格做横截面回归取残差 -> 新的因子面板。

    残差的含义：「在同市值、同 beta、同动量的一堆股票里，这只还高出多少分」。
    中性化后的 IC 才是纯 alpha；如果 IC 掉得很厉害，说明原因子主要是风格暴露。
    """
    min_n = min_n or BARRA_MIN_N
    rows, r2s = {}, []
    for d in panel.index:
        t = pd.Timestamp(d)
        e = expo_panel.get(t)
        if e is None or e.empty:
            continue
        f = panel.loc[d].dropna()
        if len(f) < min_n:
            continue
        b, resid, r2 = _b_ols(f, e.reindex(f.index))
        if resid is None:
            continue
        rows[t] = _b_z(resid)          # 残差再标准化，跟原因子放同一个量纲上
        if np.isfinite(r2):
            r2s.append(r2)
    out = pd.DataFrame(rows).T.sort_index() if rows else pd.DataFrame()
    info = {'期数': len(rows),
            '风格R2': float(np.mean(r2s)) if r2s else np.nan}
    if verbose and rows:
        print('[Barra 中性化] %d 期做了回归 | 风格平均解释度 R2=%.3f'
              % (info['期数'], info['风格R2']))
        print('  R2 的意思是「这个因子有多少比例的横截面差异可以被风格解释掉」。'
              'R2>0.3 就要留意：你的因子可能有相当一部分是风格的替身。')
    return out, info


def factor_style_loadings(panel, expo_panel, min_n=None):
    """因子自己的风格画像：每期把标准化后的因子对风格回归，取平均系数。

    两边都标准化过，所以载荷≈相关系数：|载荷|>0.5 基本就等于「这个因子就是那个风格」。
    返回 DataFrame(index=风格, columns=['载荷','相关','同向期占比'])。
    """
    min_n = min_n or BARRA_MIN_N
    coefs, cors = [], []
    for d in panel.index:
        t = pd.Timestamp(d)
        e = expo_panel.get(t)
        if e is None or e.empty:
            continue
        f = _b_z(panel.loc[d].dropna())
        if len(f) < min_n:
            continue
        b, resid, r2 = _b_ols(f, e.reindex(f.index))
        if b is None:
            continue
        coefs.append(b)
        ee = e.reindex(f.index).dropna(how='all')
        common = f.index.intersection(ee.index)
        if len(common) >= min_n:
            cors.append(ee.loc[common].apply(lambda col: col.corr(f.loc[common])))
    if not coefs:
        return pd.DataFrame()
    C = pd.DataFrame(coefs)
    R = pd.DataFrame(cors) if cors else pd.DataFrame(index=C.index, columns=C.columns)
    out = pd.DataFrame({
        '载荷': C.mean(),
        '相关': R.mean() if len(R) else np.nan,
        '同向期占比': (np.sign(C) == np.sign(C.mean())).mean(),
    })
    return out.reindex([s for s in BARRA_STYLES if s in out.index])


def barra_regress(panel, fwd, expo_panel, min_n=None, verbose=True):
    """事后归因：逐期做 r = a + λf·z(因子) + Σ λk·风格 + e。

    跟中性化相反 —— 因子不打磨，只是在回归里「控制住」风格，直接问：
    控制风格之后，这个因子还能解释收益吗？λf 的均值就是纯因子收益。
    """
    min_n = min_n or BARRA_MIN_N
    lams, lam_dates, r2s, style_coefs, used = [], [], [], [], 0
    for d in panel.index:
        t = pd.Timestamp(d)
        e = expo_panel.get(t)
        if e is None or e.empty or t not in fwd.index:
            continue
        f = _b_z(panel.loc[d].dropna())
        r = fwd.loc[t].reindex(f.index).dropna()
        common = f.index.intersection(r.index).intersection(e.index)
        if len(common) < min_n:
            continue
        X = e.loc[common].copy()
        X['zfac'] = f.loc[common]
        b, resid, r2 = _b_ols(r.loc[common], X)
        if b is None:
            continue
        lams.append(b['zfac'])
        lam_dates.append(t)
        style_coefs.append(b.drop('zfac'))
        if np.isfinite(r2):
            r2s.append(r2)
        used += 1
    if not lams:
        return {}
    lam = pd.Series(lams, index=pd.DatetimeIndex(lam_dates))
    mu, sd = lam.mean(), lam.std(ddof=1)
    n = len(lam)
    reg = {'期数': used, '纯因子收益λf': mu, 'IR': mu / sd if sd > 0 else np.nan,
           't值': mu / (sd / np.sqrt(n)) if (sd > 0 and n > 1) else np.nan,
           '风格R2': float(np.mean(r2s)) if r2s else np.nan,
           'λf序列': lam,
           '风格平均系数': pd.DataFrame(style_coefs).mean() if style_coefs else pd.Series(dtype=float)}
    if verbose and used:
        print('[Barra 归因] %d 期横截面回归（收益 ~ 因子 + 风格）' % used)
        print('  纯因子收益 λf   : %+.4f / 期（t=%.2f，IR=%.2f）'
              % (reg['纯因子收益λf'], reg['t值'], reg['IR']))
        print('  风格解释度 R2   : %.3f（收益变动里有多少是风格造成的）' % reg['风格R2'])
    return reg


def barra_report(loadings, neutral_info=None, reg=None, factor=''):
    """把 Barra 的三张结果表打印成人能读的样子：每个数字后面跟一句「怎么看」。"""
    print('-' * 88)
    print('【Barra 视角】%s' % (factor or ''))
    if loadings is not None and not loadings.empty:
        print('\n[1] 因子风格暴露（回归载荷，|载荷|>0.5 基本等于「这就是那个风格」）')
        tbl = loadings.copy()
        tbl.index = [STYLE_CN.get(i, i) for i in tbl.index]
        tbl['说明'] = [STYLE_NOTE.get(i, '') for i in loadings.index]
        print(tbl.round(3).to_string())
        top = loadings['载荷'].abs().sort_values(ascending=False)
        if len(top) and np.isfinite(top.iloc[0]):
            k = top.index[0]
            v = loadings.loc[k, '载荷']
            print('  最大暴露：%s %.3f —— %s' % (STYLE_CN.get(k, k), v, STYLE_NOTE.get(k, '')))
            if abs(v) >= 0.5:
                print('  [!] 这一项已经够大，你的因子很大程度上就是 %s 的另一种写法' % STYLE_CN.get(k, k))
            else:
                print('  各项都不算大，说明因子本身比较独立（不是风格的替身）')
    if neutral_info:
        print('\n[2] 中性化：风格平均解释度 R2=%.3f，%d 期有效'
              % (neutral_info.get('风格R2', np.nan), neutral_info.get('期数', 0)))
        print('  R2 高 = 因子横截面差异大部分能被风格解释 -> 中性化后 IC 会明显掉，属正常；')
        print('  R2 低 = 因子本来就干净 -> 中性化前后 IC 接近，说明 alpha 不是风格借来的。')
    if reg:
        print('\n[3] 归因回归（收益 ~ 因子 + 风格）：控制风格后的纯因子收益')
        print('  λf=%+.4f/期  t=%.2f  IR=%.2f  风格R2=%.3f'
              % (reg['纯因子收益λf'], reg['t值'], reg['IR'], reg['风格R2']))
        if abs(reg['t值']) >= 2:
            print('  λf 显著 -> 控制风格之后因子仍然有效，这个 alpha 是自己的。')
        else:
            print('  λf 不显著 -> 控制风格之后因子失效，之前那点 IC 很可能是风格借来的。')
    print('-' * 88)


def barra_analyze(factor, panel, fwd, rebal_dates, universe, mode=None,
                  styles=None, index_code=None, verbose=True):
    """Barra 全家桶：取数 -> 暴露 -> 中性化 / 归因 -> 打印。

    返回 dict：mode / data / expo_panel / loadings / neutral_panel / neutral_info /
              reg / ic_raw / ic_neutral
    """
    mode = barra_mode_of(mode)
    if mode == 'off':
        return {'mode': 'off'}
    t0 = datetime.datetime.now()
    dates = [pd.Timestamp(d) for d in rebal_dates]
    # 回看窗口：风格要 252 个交易日，按日历日给 420 天（约 290 个交易日）留足余量
    start = dates[0] - pd.Timedelta(days=420)
    data = build_barra_data(universe, start, dates[-1], index_code=index_code,
                            verbose=verbose)
    if not data.get('has_turn'):
        barra_data_note()
    expo_panel = build_exposure_panel(data, dates, styles=styles, verbose=verbose)
    res = {'mode': mode, 'data': data, 'expo_panel': expo_panel}
    if not expo_panel:
        print('[Barra] 一期暴露都没算出来，跳过（检查区间长度 / 是否有 turn 字段）')
        return res

    res['loadings'] = factor_style_loadings(panel, expo_panel)
    if mode in ('neutral', 'both'):
        npanel, ninfo = neutralize_panel(panel, expo_panel, verbose=verbose)
        res['neutral_panel'], res['neutral_info'] = npanel, ninfo
        if not npanel.empty:
            try:
                res['ic_raw'] = factor_ic_stats(panel, fwd)[1]['IC均值']
                res['ic_neutral'] = factor_ic_stats(npanel, fwd)[1]['IC均值']
            except Exception:
                pass
    if mode in ('attr', 'both'):
        res['reg'] = barra_regress(panel, fwd, expo_panel, verbose=verbose)

    if verbose:
        barra_report(res.get('loadings'), res.get('neutral_info'), res.get('reg'), factor)
        if res.get('ic_raw') is not None and res.get('ic_neutral') is not None:
            a, b = res['ic_raw'], res['ic_neutral']
            keep = (1.0 - abs(b) / abs(a)) * 100.0 if abs(a) > 1e-12 else np.nan
            print('[Barra 对比] IC 中性化前 %+.4f -> 后 %+.4f（掉了 %.0f%%）'
                  % (a, b, keep if np.isfinite(keep) else 0.0))
            if np.isfinite(keep) and keep > 40:
                print('  掉这么多，说明原 IC 有一大半来自风格暴露；做组合时按残差因子更安全。')
            elif np.isfinite(keep) and keep < 15:
                print('  基本没掉，因子跟这些风格关系不大，单独用问题不大。')
        print('[Barra] 用时 %.1fs'
              % (datetime.datetime.now() - t0).total_seconds())
    return res

# ============================================================
# Part B-4 截面入口函数
# ============================================================
def explain_metrics():
    print("""\
---------------------------------------------------------------
截面研究关键指标：
  期数       参与统计的调仓期数。
  有效期数   重叠修正后的独立样本数 = 期数 / 重叠倍数。
  重叠倍数   持仓期 / 调仓间隔；>1.5 时显著性以 t值_修正 为准。
  IC均值     每期 Rank IC 平均。|IC|>0.02 有效，>0.05 较强；正负=方向。
  IC标准差   各期 IC 波动，越小越稳（IR 的分母）。
  IC_IR      = IC均值/IC标准差：>0.3 可用，>0.5 优秀；负=反向因子。
  t值        = IC均值/标准误，|t|>=2 显著；t值_修正 = t/√重叠倍数。
  IC>0占比   方向一致性；明显>50% 好，≈50% 方向随机。
  分层价差   最高组-最低组平均收益差（%/期），重点看单调性。
  换手%      最高组每期成分更换比例（成本代理）。
  分档表     离散因子按原始分值分组：各档平均相对收益/t值 → 动量 or 反转。
  Barra 模式 开了以后多出三样东西（详见 explain_barra()）：
             风格暴露表 / 中性化后 IC / 归因 λf —— 用来区分「真 alpha」和「风格的替身」。
---------------------------------------------------------------""")

def explain_barra():
    print("""\
---------------------------------------------------------------
Barra 模式怎么看（run_factor_test(..., barra='both') 或参数区 BARRA_MODE='both'）
  1) 风格暴露表（载荷）
     每期把「标准化后的因子」对 8 个价量风格回归，取平均系数。
     两边都标准化过 -> 载荷≈相关系数。
     |载荷| < 0.3   因子跟这个风格关系不大
     |载荷| 0.3~0.5 有一点关联，可接受
     |载荷| > 0.5   基本就是这个风格换了个写法
  2) 中性化（barra='neutral'）
     每期把因子对风格回归、取残差，再用残差跑 IC/分层。
     残差的意思是「同市值、同 beta、同动量里，这只还高出多少分」。
     中性化后 IC 明显变小 -> 原来那部分不是 alpha，是风格给的。
     IC 基本没变        -> 因子干净，可以放心当 alpha 用。
  3) 归因回归（barra='attr'）
     r = a + λf·z(因子) + Σ λk·风格，逐期做，看 λf 的均值与 t。
     λf 显著(t>=2) -> 控制风格后因子还有效，alpha 是自己的。
     λf 不显著     -> 控制风格后失效，之前的 IC 是风格借来的。
  4) 风格平均解释度 R2
     这个因子横截面差异里有多少比例能被风格解释。
     R2>0.3 要留意；R2>0.5 基本可以认定它不是独立的 alpha。

  注意：这里是「价量代理风格」（beta/size/momentum/resvol/nlsize/liquidity/volat/reversal），
  真 Barra 还要 B/P、E/P、成长、杠杆（财务数据）和行业哑变量，本地数据层暂时没有。
---------------------------------------------------------------""")

def print_ic_stats(stats, name=''):
    head = ('[Rank IC 统计] %s' % name) if name else '[Rank IC 统计]'
    print('\n%s（Spearman 秩相关：因子排名 vs 未来收益排名）' % head)
    if stats is None or stats.empty:
        print('  有效期数不足'); return
    for k in ('期数', '有效期数', 'IC均值', 'IC标准差', 'IC_IR',
              't值', 't值_修正', 'IC>0占比', 'IC偏度'):
        if k in stats.index:
            v = stats[k]
            if k == '期数':
                print('  %-9s : %d' % (k, v))
            elif k == '有效期数':
                print('  %-9s : %.1f' % (k, v))
            else:
                print('  %-9s : %.4f' % (k, v))

def _verdict(r):
    tv = r.get('t值_修正', r.get('t值', np.nan))
    ir = r.get('IC_IR', np.nan)
    if pd.isna(tv) or pd.isna(ir):
        return '样本不足'
    if abs(tv) >= 2 and abs(ir) >= 0.3:
        return '★有效'
    if abs(tv) >= 2:
        return '显著但IR偏低'
    if abs(ir) >= 0.3:
        return 'IR高但不显著'
    return '不显著'

def run_factor_test(factor, pool, start, end, period=20, freq='M', rule='first', every=1,
                    rebal_dates=None, strict=False, q=5, non_overlap=False, verbose=True,
                    barra=None):
    """单因子截面检验。
    factor      : 因子名（三通道：财务字段 / 因子库因子 / 已注册自定义因子）
    pool        : str=股票指数池（动态成分）；list=固定池（ETF/自选股）
    start/end   : 区间；end 超过今天会自动截断
    period      : 持仓期（交易日），与调仓频率解耦。财务因子60+；量价20；ETF短信号5
    freq        : 调仓频率 'D'/'W'/'M'/'Q'/'Y'
    rule        : 每期取首日('first')/末日('last')交易日
    every       : 间隔倍数（freq='M', every=2 → 双月）
    rebal_dates : 直接传调仓日列表（覆盖 freq/rule/every）
    strict      : 前向收益是否从 t+1 起算（更保守）
    q           : 期望分层组数（池子小会自动降组）
    non_overlap : 抽稀调仓日使评估窗口不重叠（长线建议开启）
    barra       : None=跟随全局 BARRA_MODE；'neutral'=先用风格中性化再检验；
                  'attr'=只做归因（不改因子）；'both'=都做；'off'=关
    返回 dict: factor_panel / fwd_panel / ic / ic_stats / quantile_returns /
              turnover / buckets / mono / rebal_dates / interval / ratio"""
    print('=' * 88)
    print('【单因子检验】%s | %s | %s ~ %s | 持仓 %d 交易日'
          % (factor, pool_desc(pool), start, end, period))
    pre = prepare_data([factor], pool, start, end, period=period, freq=freq, rule=rule,
                       every=every, rebal_dates=rebal_dates, strict=strict,
                       non_overlap=non_overlap, verbose=verbose)
    panel, fwd = pre['panels'][factor], pre['fwd']
    # ---- Barra 开关：barra=None 用全局 BARRA_MODE；传 'neutral'/'attr'/'both' 临时切
    panel_raw, barra_res = panel, None
    try:
        _bmode = barra_mode_of(barra)
    except Exception as e:
        print('[Barra] %s，本轮按 off 处理' % e)
        _bmode = 'off'
    if _bmode != 'off':
        try:
            barra_res = barra_analyze(factor, panel, fwd, pre['rebal_dates'],
                                      pre['universe'], mode=_bmode, verbose=verbose)
            npnl = barra_res.get('neutral_panel')
            if npnl is not None and not npnl.empty:
                panel = npnl
                if verbose:
                    print('\n[Barra] 下面的 IC / 分层 / 换手 / 分档 全部基于「中性化后的因子残差」')
        except Exception as e:
            print('[Barra] 跳过（%s: %s）' % (type(e).__name__, e))
    ic, stats = factor_ic_stats(panel, fwd, overlap_ratio=pre['ratio'])
    try:                       # 让读表的人一眼看出这是哪个因子
        ic.name = factor
        stats.name = factor
    except Exception:
        pass
    print_ic_stats(stats, name=factor)
    qr = quantile_returns(panel, fwd, q)
    if not qr.empty:
        print('\n[分层收益] %s 各分位组平均收益（%%/期），Q1=因子最小' % factor)
        print((100 * qr.mean()).round(3).to_string())
        m = qr.mean()
        sp = m.iloc[-1] - m.iloc[0]
        print('  多空价差(Q%d-Q1): %.3f%%/期（折年约 %.2f%%）'
              % (qr.shape[1], 100 * sp, 100 * sp * 244.0 / period))
    to = quantile_turnover(panel, q)
    print('\n[换手] %s 最高因子分组平均换手: %s'
          % (factor, ('%.1f%%' % (100 * to)) if not np.isnan(to) else '数据不足'))
    bt, mono = score_bucket_analysis(panel, fwd)
    if not bt.empty:
        print('\n[分档诊断] %s 各分值档相对收益（已剔除大盘涨跌）' % factor)
        print(bt.round(3).to_string(index=False))
        print('  结论: %s' % diagnose_momentum_reversal(
            bt, stats['IC均值'] if not stats.empty else np.nan))
    return {'factor': factor, 'factor_panel': panel, 'factor_panel_raw': panel_raw,
            'fwd_panel': fwd, 'ic': ic,
            'ic_stats': stats, 'quantile_returns': qr, 'turnover': to,
            'buckets': bt, 'mono': mono, 'rebal_dates': pre['rebal_dates'],
            'interval': pre['interval'], 'ratio': pre['ratio'], 'barra': barra_res}

def run_batch(factors, pool, start, end, period=20, freq='M', rule='first', every=1,
              rebal_dates=None, strict=False, q=5, non_overlap=False,
              panels=None, show_corr=True, verbose=True, barra=None):
    """批量单因子：共享前向收益面板，输出一张汇总表（按 |IC_IR| 降序）+ 明细 + 相关矩阵。
    参数同 run_factor_test；show_corr 是否打印因子间相关矩阵。
    返回 (汇总表, 明细dict, 面板dict, 前向收益表)"""
    print('=' * 88)
    print('【批量单因子检验】%s | %s ~ %s | 持仓 %d 日 | %d 个因子'
          % (pool_desc(pool), start, end, period, len(factors)))
    pre = prepare_data(factors, pool, start, end, period=period, freq=freq, rule=rule,
                       every=every, rebal_dates=rebal_dates, strict=strict,
                       non_overlap=non_overlap, panels=panels, verbose=verbose)
    fwd, ratio = pre['fwd'], pre['ratio']
    # ---- Barra 开关：批量模式下共享一份风格暴露，逐个因子中性化（只打一行，不逐因子出报告）
    try:
        _bmode = barra_mode_of(barra)
    except Exception as e:
        print('[Barra] %s，本轮按 off 处理' % e)
        _bmode = 'off'
    expo_panel = {}
    if _bmode in ('neutral', 'both'):
        try:
            _dates = [pd.Timestamp(d) for d in pre['rebal_dates']]
            _data = build_barra_data(pre['universe'],
                                     _dates[0] - pd.Timedelta(days=420), _dates[-1],
                                     verbose=verbose)
            if not _data.get('has_turn'):
                barra_data_note()
            expo_panel = build_exposure_panel(_data, _dates, verbose=verbose)
            print('[Barra] 批量模式：因子先做风格中性化再检验（归因请对单个因子跑 run_factor_test）')
        except Exception as e:
            print('[Barra] 批量跳过（%s: %s）' % (type(e).__name__, e))
            expo_panel = {}
    rows, details = [], {}
    for f in factors:
        panel = pre['panels'][f]
        if expo_panel:
            try:
                _np, _ = neutralize_panel(panel, expo_panel, verbose=False)
                if not _np.empty:
                    panel = _np
            except Exception:
                pass
        ic, stats = factor_ic_stats(panel, fwd, overlap_ratio=ratio)
        qr = quantile_returns(panel, fwd, q)
        to = quantile_turnover(panel, q)
        bt, mono = score_bucket_analysis(panel, fwd, verbose=False)
        spread = np.nan
        if not qr.empty:
            m = qr.mean(); spread = 100 * (m.iloc[-1] - m.iloc[0])
        row = {'因子': f, '覆盖率': panel.notna().mean().mean(),
               '期数': stats.get('期数', np.nan),
               'IC均值': stats.get('IC均值', np.nan),
               'IC标准差': stats.get('IC标准差', np.nan),
               'IC_IR': stats.get('IC_IR', np.nan),
               't值_修正': stats.get('t值_修正', np.nan),
               'IC>0占比': stats.get('IC>0占比', np.nan),
               '分层价差%': spread,
               '换手%': (100 * to) if not np.isnan(to) else np.nan}
        row['判定'] = _verdict(row)
        rows.append(row)
        details[f] = {'ic': ic, 'stats': stats, 'qr': qr, 'turnover': to,
                      'buckets': bt, 'mono': mono}
    df = pd.DataFrame(rows)
    if 'IC_IR' in df.columns:
        df = df.reindex(df['IC_IR'].abs().sort_values(ascending=False).index)
    show = df.copy()
    for c in ('覆盖率', 'IC均值', 'IC标准差', 'IC_IR', 't值_修正', 'IC>0占比', '分层价差%', '换手%'):
        if c in show.columns:
            show[c] = show[c].astype(float).round(4)
    print('\n[汇总表] 按 |IC_IR| 降序（IC_IR 为负=反向因子）')
    print(show.to_string(index=False))
    if show_corr and len(factors) >= 2:
        print('\n[因子相关性] 高相关(>0.7)信息重复，组合前先去除')
        try:
            print(factor_corr_matrix(pre['panels']).round(2).to_string())
        except Exception as e:
            print('  相关矩阵计算失败:', e)
    return df, details, pre['panels'], fwd

def make_composite_score(panels, cfg):
    """合成打分。cfg = {'因子': (权重, 方向)}，方向 +1=越大越好，-1=越小越好。
    先逐期百分位排名(pct)再加权，消除量纲差异。"""
    total = None
    for name, (w, direction) in cfg.items():
        if name not in panels:
            raise ValueError('缺面板 %s' % name)
        r = panels[name].rank(axis=1, pct=True)
        piece = w * (r if direction >= 0 else 1 - r)
        total = piece if total is None else total + piece
    return total

def run_combo(schemes, pool, start, end, period=20, freq='M', rule='first', every=1,
              rebal_dates=None, strict=False, non_overlap=False, train_frac=0.6,
              panels=None, verbose=True):
    """多因子组合：输出 全段/训练段/检验段 三段 IC 指标。
    schemes    : {'方案名': {因子: (权重, 方向)}, ...}
    train_frac : 训练段占比（前 60% 调参，后 40% 只看一次）
    返回 (汇总表, 面板dict, 前向收益表)"""
    print('=' * 88)
    print('【多因子组合检验】%s | %s ~ %s | 持仓 %d 日'
          % (pool_desc(pool), start, end, period))
    need = set()
    for cfg in schemes.values():
        need.update(cfg.keys())
    pre = prepare_data(sorted(need), pool, start, end, period=period, freq=freq, rule=rule,
                       every=every, rebal_dates=rebal_dates, strict=strict,
                       non_overlap=non_overlap, panels=panels, verbose=verbose)
    fwd, ratio = pre['fwd'], pre['ratio']
    idx = sorted(fwd.index)
    split = idx[int(len(idx) * train_frac)]
    print('训练段 <= %s | 检验段 > %s（训练段调参，检验段只看一次）'
          % (str(split)[:10], str(split)[:10]))

    def _summ(s):
        s = s.dropna()
        if len(s) < 3:
            return (np.nan, np.nan, np.nan)
        m, sd = s.mean(), s.std(ddof=1)
        return (m, (m / sd if sd > 0 else np.nan),
                (m / (sd / np.sqrt(len(s))) if sd > 0 else np.nan))

    rows = []
    for name, cfg in schemes.items():
        score = make_composite_score(pre['panels'], cfg)
        ic, _ = factor_ic_stats(score, fwd, overlap_ratio=ratio)
        a, tr, te = _summ(ic), _summ(ic[ic.index < split]), _summ(ic[ic.index >= split])
        rows.append({'方案': name,
                     '全段IC': a[0], '全段IR': a[1], '全段t': a[2],
                     '训练IC': tr[0], '训练IR': tr[1],
                     '检验IC': te[0], '检验IR': te[1], '权重': str(cfg)})
    out = pd.DataFrame(rows).round(4)
    print('\n[组合汇总] 检验IR 与 训练IR 接近→稳健；检验IR 大幅缩水→过拟合')
    print(out.to_string(index=False))
    return out, pre['panels'], fwd

def _trade_day_index(start, end):
    """交易日序列（DatetimeIndex，升序，去重）。"""
    cal = list(get_trade_days(start_date=start, end_date=end))
    idx = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in cal])
    return idx.sort_values().unique()

def scan_ic_decay(factor, pool, start, end, periods=(5, 10, 20, 63, 126, 250),
                  freq='M', rule='first', every=1, rebal_dates=None,
                  strict=False, non_overlap=False, align=True, verbose=True):
    """IC 衰减扫描：同一因子、多个持仓期，用来选持有时间。

    align=True（默认，推荐）：每个持仓期各配一套调仓日，间隔 = 持仓期（交易日），
        「调仓日」和「前向收益日」是同一个周期，重叠倍数恒为 1，t 值不用打折。
        代价：要按最细的那个持仓期取一遍因子值，比 align=False 慢。
        请求的持仓期不是细网格整数倍时，会就近吸附，实际持仓期见「持仓期(日)」列。
    align=False：所有持仓期共用 freq 定出的同一套调仓日（老的扫描口径）。
        持仓期和调仓间隔不一致，重叠倍数 = 持仓期 ÷ 调仓间隔（都按交易日），
        t 值按这个倍数折算；只适合「固定调仓频率下比较预测力」，不能直接当持有期用。
    non_overlap=True 只在 align=False 下有意义（把调仓日再抽稀一档，样本独立）。
    """
    periods = [int(p) for p in periods if int(p) > 0]
    if not periods:
        return pd.DataFrame()
    if rebal_dates is None:
        rebal_dates = get_rebalance_dates(start, end, freq=freq, rule=rule, every=every)
    if rebal_dates is None or len(rebal_dates) < 3:
        print('  调仓日不足（%s 个），无法扫描' % (0 if rebal_dates is None else len(rebal_dates)))
        return pd.DataFrame()
    anchor = pd.Timestamp(rebal_dates[0]).normalize()
    today = datetime.date.today()
    rows = []

    if align:
        step = min(periods)
        cal = _trade_day_index(start, end)
        cal = cal[cal >= anchor]
        if len(cal) < step * 3:
            print('  可用交易日太少（%d 天），缩短区间或减少持仓期' % len(cal))
            return pd.DataFrame()
        grid = cal[::step]
        if verbose:
            print('  细网格 %d 个调仓日（步长 %d 交易日），因子取一遍较慢，稍等' % (len(grid), step))
        all_stocks, _ = resolve_universe(pool, grid, verbose=False)
        panel = build_factor_panel(factor, grid, pool)
        for p in periods:
            k = max(1, int(round(float(p) / float(step))))
            rd = grid[::k]
            gap = int(step * k)                    # 调仓间隔 = 前向收益期，完全对齐
            if len(rd) < 3:
                continue
            price_end = trading_day_shift(rd[-1], gap + 5)
            if pd.Timestamp(price_end).date() > today:
                price_end = today
            fwd = build_forward_returns(all_stocks, rd[0], price_end, gap, strict)
            sub = panel.loc[panel.index.isin(rd)]
            ic, stats = factor_ic_stats(sub, fwd, overlap_ratio=1.0)
            if stats.empty:
                continue
            rows.append({'请求持仓期': p, '持仓期(日)': gap, '调仓间隔(日)': gap,
                         '重叠倍数': 1.0, '评估期数': int(stats['期数']),
                         '有效期数': stats['有效期数'],
                         'IC均值': stats['IC均值'], 'IC_IR': stats['IC_IR'],
                         't值': stats['t值'], 't值_修正': stats['t值_修正'],
                         'IC>0占比': stats['IC>0占比']})
        note = '口径：每个持仓期一套调仓日，调仓间隔=持仓期，重叠=1（可直接比 IC/IR/t）'
    else:
        interval0 = _median_interval_days(rebal_dates) or 1
        all_stocks, _ = resolve_universe(pool, rebal_dates, verbose=False)
        panel = build_factor_panel(factor, rebal_dates, pool)
        for p in periods:
            rd = rebal_dates
            if non_overlap:
                st = int(np.ceil(float(p) / float(interval0)))
                if st > 1:
                    rd = rebal_dates[::st]
            interval = _median_interval_days(rd) or interval0
            ratio = float(p) / float(interval)      # 不再夹到 1：小于 1 就是前向窗口没覆盖一个调仓期
            price_end = trading_day_shift(rd[-1], p + 5)
            if pd.Timestamp(price_end).date() > today:
                price_end = today
            fwd = build_forward_returns(all_stocks, rd[0], price_end, p, strict)
            sub = panel.loc[panel.index.isin(rd)]
            ic, stats = factor_ic_stats(sub, fwd, overlap_ratio=max(1.0, ratio))
            if stats.empty:
                continue
            rows.append({'请求持仓期': p, '持仓期(日)': p, '调仓间隔(日)': int(interval),
                         '重叠倍数': round(ratio, 2), '评估期数': int(stats['期数']),
                         '有效期数': stats['有效期数'],
                         'IC均值': stats['IC均值'], 'IC_IR': stats['IC_IR'],
                         't值': stats['t值'], 't值_修正': stats['t值_修正'],
                         'IC>0占比': stats['IC>0占比']})
        note = '口径：共用一套调仓日，重叠倍数=持仓期÷调仓间隔（交易日），t值已折算；<1 表示窗口没走完一个调仓期'

    df = pd.DataFrame(rows)
    if df.empty:
        print('  没有可用结果（因子值或价格不足）')
        return df
    df = df.round(4)
    print('【IC衰减扫描】%s | %s | %s ~ %s' % (factor, pool_desc(pool), start, end))
    print(note)
    print(df.to_string(index=False))
    print('判读: 先看 |IC| 随持仓期是否衰减；选持仓期看 IC_IR 和 t值_修正（重叠已折过的才算数）')
    return df

# ============================================================
# Part B-4b 调仓频率 × 持仓期 二维扫描（定「调仓周期 + 持有周期」）
# ============================================================
FREQ_NAMES = {'D': '日频', 'W': '周频', 'M': '月频', 'Q': '季频', 'Y': '年频'}

def _fwd_from_px(px, period, strict=False):
    """从一张全区间收盘价表里切出 period 日前向收益（各持仓期共用一次取价）。"""
    if px is None or px.empty:
        return pd.DataFrame()
    if strict:
        return px.shift(-(period + 1)) / px.shift(-1) - 1.0
    return px.shift(-period) / px - 1.0

def scan_freq_period(factor, pool, start, end, freqs=('W', 'M', 'Q'),
                     periods=(5, 10, 20, 40, 60, 120, 250), rule='first', every=1,
                     strict=False, q=5, cost=0.0005, min_n=6, verbose=True):
    """调仓频率 × 持仓期 的二维扫描：单个因子找「多久调一次仓 + 每次拿多久」。

    为什么单列一个函数：选持仓周期和选调仓周期是两件事。
        · 调仓频率决定「信号更新多快、成本付几次」；
        · 持仓期决定「一次预测多远、IC 有多强」；
        · 两个乘起来才是真实暴露：period > interval 时持仓重叠，t 值要折。
    本函数对每个 (freq, period) 组合各算一次 IC，并给出成本与资金占用：
        · 重叠倍数 = 持仓期 ÷ 调仓间隔，>1 的 t 值已用 NW 口径折过；
        · 周期匹配 = [OK] 表示持仓期≈调仓间隔（0.8~1.25 倍），调仓日与前向收益期严丝合缝；
        · 资金占用% = min(100, 持仓期/调仓间隔×100)，低了说明大把钱闲着；
        · 年化成本拖累% = 2×单期换手×cost×每年调仓次数（双边、粗估）；
        · 折年价差% = 分层价差×244/最长(持仓期, 调仓间隔)；
        · 净年化% = 折年价差 - 年化成本拖累（只做多空参考，不是实盘收益）。
    返回长表，一行一个 (freq, period)；配合 best_cycle() 出结论、plot_freq_period() 画热力图。
    """
    today = datetime.date.today()
    label = pool_desc(pool)
    rows = []
    for fq in freqs:
        try:
            rd = get_rebalance_dates(start, end, freq=fq, rule=rule, every=every)
        except Exception as e:
            print('  %s 调仓日生成失败: %s' % (fq, e))
            continue
        if len(rd) < min_n:
            if verbose:
                print('  跳过 %s：调仓日只有 %d 个（< %d），区间太短' % (fq, len(rd), min_n))
            continue
        interval = _median_interval_days(rd) or 1
        try:
            all_stocks, _ = resolve_universe(pool, rd, verbose=False)
            panel = build_factor_panel(factor, rd, pool)
        except Exception as e:
            print('  %s 面板构建失败: %s' % (fq, e))
            continue
        if panel is None or panel.empty:
            continue
        price_end = trading_day_shift(rd[-1], int(max(periods)) + 5)
        try:
            if pd.Timestamp(price_end).date() > today:
                price_end = today
        except Exception:
            pass
        px = get_price_range(all_stocks, rd[0], price_end, ('close',)).get('close')
        if px is None or px.empty:
            print('  %s 取价为空，跳过' % fq)
            continue
        px = px.sort_index()
        for p in sorted(set(int(x) for x in periods if x and float(x) > 0)):
            fwd = _fwd_from_px(px, p, strict)
            ratio = max(1.0, float(p) / float(interval))
            ic, stats = factor_ic_stats(panel, fwd, overlap_ratio=ratio)
            base = {'频率': FREQ_NAMES.get(fq, fq), '调仓间隔(日)': int(interval),
                    '持仓期(日)': p, '重叠倍数': round(ratio, 2),
                    '周期匹配': '[OK]' if 0.8 <= float(p) / float(interval) <= 1.25 else '',
                    '资金占用%': round(100.0 * min(1.0, float(p) / float(interval)), 1)}
            if stats.empty or int(stats.get('期数', 0)) < min_n:
                base.update({'期数': 0, '有效期数': 0.0, 'IC均值': np.nan, 'IC_IR': np.nan,
                             't值_修正': np.nan, 'IC>0占比': np.nan, '分层价差%/期': np.nan,
                             '折年价差%': np.nan, '换手%/期': np.nan,
                             '年化成本拖累%': np.nan, '净年化%': np.nan, '判定': '样本不足'})
                rows.append(base)
                continue
            qr = quantile_returns(panel, fwd, q)
            spread = np.nan
            if qr is not None and not qr.empty:
                m = qr.mean()
                spread = float(m.iloc[-1] - m.iloc[0])
            to = quantile_turnover(panel, q)
            cycles = 244.0 / float(interval)
            cost_drag = 2.0 * (float(to) if np.isfinite(to) else np.nan) * cost * cycles
            ann_spread = spread * 244.0 / max(float(p), float(interval))
            t, ir = float(stats['t值_修正']), float(stats['IC_IR'])
            if not np.isfinite(t) or not np.isfinite(ir):
                verdict = '数据不足'
            elif abs(t) >= 2.0 and abs(ir) >= 0.3:
                verdict = '★'
            elif abs(t) >= 2.0:
                verdict = '△显著但IR低'
            elif abs(ir) >= 0.3:
                verdict = '○IR高不显著'
            else:
                verdict = '×'
            base.update({'期数': int(stats['期数']), '有效期数': float(stats['有效期数']),
                         'IC均值': float(stats['IC均值']), 'IC_IR': ir, 't值_修正': t,
                         'IC>0占比': float(stats['IC>0占比']),
                         '分层价差%/期': 100.0 * spread, '折年价差%': 100.0 * ann_spread,
                         '换手%/期': 100.0 * float(to) if np.isfinite(to) else np.nan,
                         '年化成本拖累%': 100.0 * cost_drag,
                         '净年化%': 100.0 * (ann_spread - cost_drag), '判定': verdict})
            rows.append(base)
        if verbose:
            print('  %s 完成（间隔 %d 日，%d 个持仓期）' % (fq, interval, len(periods)))

    df = pd.DataFrame(rows)
    if df.empty:
        print('【调仓×持仓扫描】%s | %s 没有结果（调仓日或数据不足）' % (factor, label))
        return df
    for c in ('IC均值', 'IC_IR', 't值_修正', 'IC>0占比', '分层价差%/期',
              '折年价差%', '换手%/期', '年化成本拖累%', '净年化%'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').round(4)
    order = {v: i for i, v in enumerate(FREQ_NAMES.values())}
    df['_o'] = df['频率'].map(order).fillna(99)
    df = df.sort_values(['_o', '持仓期(日)']).drop(columns='_o').reset_index(drop=True)
    print('【调仓×持仓扫描】%s | %s | %s ~ %s' % (factor, label, start, end))
    print('口径：每个 (频率, 持仓期) 各算一次 IC；重叠倍数>1 的 t 值已折；'
          '净年化 = 折年价差 - 成本拖累（多空参考口径，不是实盘收益）')
    print(df.to_string(index=False))
    print('判读：先看 ★ 格子；想「调仓日与前向收益期匹配」就挑 周期匹配=[OK]；再看 资金占用 与 净年化（负了就是 IC 不够付手续费）')
    return df

def best_cycle(df, prefer='ir', verbose=True):
    """从 scan_freq_period 的结果里挑「最佳调仓周期 + 持有周期」。

    prefer='ir' ：先要显著（★ / △ 且净年化为正），再按 |IC_IR| 排序 —— 稳健优先；
    prefer='net'：不看判定，直接按净年化% 排序 —— 收益优先，容易踩到单期极端值。
    返回 dict（频率 / 调仓间隔 / 持仓期 / IC_IR / t值_修正 / 净年化% / row），没有可用格子返回 None。
    """
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    for c in ('IC_IR', 't值_修正', '净年化%'):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[np.isfinite(d['IC_IR'])]
    if d.empty:
        return None
    if prefer == 'net':
        cand = d[np.isfinite(d['净年化%'])]
        cand = cand.sort_values('净年化%', ascending=False)
    else:
        # 稳健优先：先只要显著的格子（★/△），内部按 |IC_IR| 排；
        # 一个显著格子都没有，才退回「净年化为正」的里挑 IR 最大的
        sig = d[d['判定'].astype(str).str.startswith(('★', '△'))]
        if sig.empty:
            pos = d[np.isfinite(d['净年化%']) & (d['净年化%'] > 0)]
            sig = pos if not pos.empty else d
        cand = sig.reindex(sig['IC_IR'].abs().sort_values(ascending=False).index)
    if cand.empty:
        return None
    row = cand.iloc[0]
    out = {'频率': row['频率'], '调仓间隔(日)': int(row['调仓间隔(日)']),
           '持仓期(日)': int(row['持仓期(日)']), '重叠倍数': float(row['重叠倍数']),
           'IC均值': float(row['IC均值']) if np.isfinite(row['IC均值']) else np.nan,
           'IC_IR': float(row['IC_IR']), 't值_修正': float(row['t值_修正']),
           '净年化%': float(row['净年化%']) if np.isfinite(row['净年化%']) else np.nan,
           '判定': str(row['判定']), 'row': row}
    if verbose:
        print('【最佳周期】%s 调仓（间隔约 %d 交易日）+ 持有 %d 交易日 | IC_IR %.3f | t %.2f | 净年化 %.1f%% | %s'
              % (out['频率'], out['调仓间隔(日)'], out['持仓期(日)'], out['IC_IR'],
                 out['t值_修正'], out['净年化%'], out['判定']))
        print('  注意：这是「同一段历史里挑出来的最优格子」，换区间要重扫；'
              '真正落地还要看换手成本、容量和这格子的邻居是否也稳定')
        if np.isfinite(out['净年化%']) and out['净年化%'] <= 0:
            print('  [!] 这格子扣完成本是负的：IC 不够付手续费，别直接上，先想清楚换手怎么降')
    return out

# ============================================================
# Part C 时序研究（择时）
# ============================================================
TS_SIGNALS = {}

def register_ts_signal(name):
    """注册时序信号。签名可选两种：
       (code, px) 或 (code, px, params=None)
       px: index=日期, columns=close/high/low/money（单标的全历史）
       返回: Series(index=日期)，长度可与 px 相同（NaN 会被丢弃）"""
    def deco(func):
        TS_SIGNALS[name] = func
        return func
    return deco

@register_ts_signal('ma200')
def ts_ma200(code, px):
    """收盘价在200日均线上→1，否则→0"""
    return (px['close'] > px['close'].rolling(200).mean()).astype(float)

@register_ts_signal('ma60')
def ts_ma60(code, px):
    """60日均线趋势"""
    return (px['close'] > px['close'].rolling(60).mean()).astype(float)

@register_ts_signal('mom_120')
def ts_mom_120(code, px):
    """120日动量（连续）"""
    return px['close'] / px['close'].shift(120) - 1.0

@register_ts_signal('dev_ma250')
def ts_dev_ma250(code, px):
    """对250日均线偏离度（连续）"""
    return px['close'] / px['close'].rolling(250).mean() - 1.0

@register_ts_signal('volz_20_250')
def ts_volz_20_250(code, px):
    """20日波动率的250日z分数（高=动荡，方向用 -1）"""
    vol = px['close'].pct_change().rolling(20).std()
    return (vol - vol.rolling(250).mean()) / vol.rolling(250).std()

@register_ts_signal('px_pos_750')
def ts_px_pos_750(code, px):
    """价格近750日区间分位（0~1）"""
    lo = px['close'].rolling(750, min_periods=250).min()
    hi = px['close'].rolling(750, min_periods=250).max()
    return (px['close'] - lo) / (hi - lo)

@register_ts_signal('trend_ma')
def ts_trend_ma(code, px, params=None):
    """参数化趋势：收盘价 > N日均线 → 1。params: {'n': 200}"""
    n = int((params or {}).get('n', 200))
    return (px['close'] > px['close'].rolling(n).mean()).astype(float)

def _call_ts_signal(func, code, px, params):
    """兼容带 params 与不带 params 两种写法。"""
    try:
        if 'params' in inspect.signature(func).parameters:
            return func(code, px, params=params)
    except (TypeError, ValueError):
        pass
    return func(code, px)

def build_ts_signal(name, code, start, end, params=None):
    """取价格 + 算时序信号。返回 (信号Series, 价格宽表)。"""
    if name not in TS_SIGNALS:
        raise ValueError('时序信号未注册: %s（已注册: %s）'
                         % (name, sorted(TS_SIGNALS.keys())))
    px = get_price_range([code], start, end, ('close', 'high', 'low', 'money'))
    if not px or 'close' not in px or px['close'].empty:
        raise ValueError('取不到价格数据: %s' % code)
    df = pd.DataFrame({f: w[code] for f, w in px.items() if code in w.columns})
    if 'close' not in df.columns:
        raise ValueError('价格缺少 close 列: %s' % code)
    sig = pd.Series(_call_ts_signal(TS_SIGNALS[name], code, df, params)).dropna()
    sig.name = name
    return sig, df

def ts_forward_return(close, period=20, strict=False):
    """同一标的前瞻收益。strict=True 从 t+1 起算（保守）。"""
    if strict:
        return close.shift(-(period + 1)) / close.shift(-1) - 1.0
    return close.shift(-period) / close - 1.0

def _ols_nw(x, y, lags=0):
    """OLS y=a+b*x，返回 b、OLS t、Newey-West t。
    NW 在方差估计中加入 1~L 阶自协方差（Bartlett 权重），修正自相关/重叠窗口的虚高。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 10 or np.std(x) == 0:
        return {'beta': np.nan, 't_ols': np.nan, 't_nw': np.nan, 'lags': 0, 'n': n}
    X = np.column_stack([np.ones(n), x])
    A_inv = np.linalg.inv(X.T @ X)
    beta = A_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(1, n - 2)
    s2 = (resid @ resid) / dof
    se_ols = np.sqrt(max(s2 * A_inv[1, 1], 0.0))
    Xe = X * resid[:, None]
    B = Xe.T @ Xe
    for k in range(1, int(lags) + 1):
        w = 1.0 - k / (float(lags) + 1.0)
        G = Xe[k:].T @ Xe[:-k]
        B = B + w * (G + G.T)
    V = A_inv @ B @ A_inv
    se_nw = np.sqrt(max(V[1, 1], 0.0))
    return {'beta': float(beta[1]),
            't_ols': float(beta[1] / se_ols) if se_ols > 0 else np.nan,
            't_nw': float(beta[1] / se_nw) if se_nw > 0 else np.nan,
            'lags': int(lags), 'n': n}

def ts_ic_stats(signal, fwd_ret, nw_lags=None, overlap_hint=1, sub_periods=4, min_n=30):
    """时序 IC + NW 修正 t + 分段 IC。overlap_hint=前瞻天数（NW滞后至少覆盖它）。"""
    df = pd.concat([signal.rename('sig'), fwd_ret.rename('ret')], axis=1).dropna()
    if len(df) < min_n:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    n = len(df)
    ic_sp = df['sig'].corr(df['ret'], method='spearman')
    ic_pe = df['sig'].corr(df['ret'])
    if nw_lags is None:
        nw_lags = max(int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), int(overlap_hint))
    nw = _ols_nw(df['sig'].values, df['ret'].values, lags=nw_lags)
    seg = pd.Series(dtype=float)
    if sub_periods and sub_periods >= 2:
        edges = np.linspace(0, n, sub_periods + 1).astype(int)
        idxs, vals = [], []
        for i in range(sub_periods):
            chunk = df.iloc[edges[i]:edges[i + 1]]
            if len(chunk) >= max(10, min_n // 2):
                c = chunk['sig'].corr(chunk['ret'], method='spearman')
                if not pd.isna(c):
                    idxs.append(chunk.index[-1]); vals.append(c)
        if vals:
            seg = pd.Series(vals, index=pd.DatetimeIndex(idxs))
    stats = pd.Series({
        '样本数': float(n),
        'IC(Spearman)': ic_sp,
        'IC(Pearson)': ic_pe,
        '斜率(OLS)': nw['beta'],
        't值(OLS)': nw['t_ols'],
        't值(Newey-West)': nw['t_nw'],
        'NW滞后': float(nw['lags']),
        '分段IC均值': float(seg.mean()) if len(seg) else np.nan,
        '分段IC_IR': float(seg.mean() / seg.std(ddof=1))
                     if len(seg) > 1 and seg.std(ddof=1) > 0 else np.nan,
    })
    return stats, seg

def _edge_desc(k, edges):
    lo = '-inf' if k == 0 else ('%.4g' % edges[k - 1])
    hi = 'inf' if k >= len(edges) else ('%.4g' % edges[k])
    return '(%s, %s]' % (lo, hi)

def _welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else np.nan

def ts_state_analysis(signal, fwd_ret, thresholds=None, n_states=2, min_n=20):
    """按信号状态分组看未来收益。thresholds=None 用分位切 n_states 档。"""
    df = pd.concat([signal.rename('sig'), fwd_ret.rename('ret')], axis=1).dropna()
    if len(df) < min_n:
        return pd.DataFrame(), {'spread%': np.nan, 't(welch)': np.nan, 'note': '样本不足'}
    if thresholds is None:
        qs = np.linspace(0.0, 1.0, n_states + 1)[1:-1]
        edges = list(np.unique(df['sig'].quantile(qs).values))
    else:
        edges = sorted(list(thresholds))

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
            rows.append({'状态': int(s), '区间': _edge_desc(int(s), edges), '观测数': cnt,
                         '平均收益%': np.nan, '折年化%': np.nan, '胜率': np.nan})
            continue
        m, sd = sub.mean(), sub.std(ddof=1)
        rows.append({'状态': int(s), '区间': _edge_desc(int(s), edges), '观测数': cnt,
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
                 't(welch)': _welch_t(b, a),
                 'note': 'spread = 最高信号状态 - 最低信号状态'}

def _perf(daily, ann=244, pos=None):
    """单标的绩效：年化/波动/夏普/最大回撤/日胜率/在场占比。"""
    daily = daily.dropna()
    if len(daily) < 2:
        return {'年化收益%': np.nan, '年化波动%': np.nan, '夏普': np.nan,
                '最大回撤%': np.nan, '日胜率': np.nan, '在场占比%': np.nan}
    cum = (1.0 + daily).cumprod()
    years = len(daily) / float(ann)
    total = float(cum.iloc[-1])
    ann_ret = total ** (1.0 / years) - 1.0 if total > 0 else np.nan
    vol = float(daily.std(ddof=1)) * np.sqrt(ann)
    sharpe = ann_ret / vol if vol > 0 else np.nan
    mdd = float((cum / cum.cummax() - 1.0).min())
    return {'年化收益%': 100.0 * ann_ret, '年化波动%': 100.0 * vol, '夏普': sharpe,
            '最大回撤%': 100.0 * mdd, '日胜率': float((daily > 0).mean()),
            '在场占比%': (float(pos.mean()) * 100.0) if pos is not None else np.nan}

def ts_backtest(signal, close, th=0.0, decision_freq='D', cost=0.0005,
                direction=1, ann=244):
    """signal > th 持有，否则空仓（现金按 0 收益）。
    th           : 触发阈值（信号单位）
    decision_freq: 'D' 每日决策 | 'W' 每周 | 'M' 每月（期初定仓，期内不变）
    cost         : 每次仓位切换成本
    direction    : -1 翻转信号（波动率类"低才持有"）
    信号次日生效（shift 1），防未来函数。返回 (指标表, 明细表)"""
    sig = pd.Series(signal).dropna()
    if len(sig) < 60:
        return pd.DataFrame(), pd.DataFrame()
    close = close.loc[sig.index[0]:]
    ret = close.pct_change().fillna(0.0)
    s = sig.reindex(ret.index).ffill()
    if int(direction) < 0:
        s = -s
        th = -float(th)
    pos = (s > float(th)).astype(float)
    if decision_freq in ('W', 'w', 'M', 'm'):
        per = s.index.to_period('W' if decision_freq in ('W', 'w') else 'M')
        pos = pos.groupby(per).transform('first')
    pos = pos.shift(1).fillna(0.0)
    dpos = pos.diff()
    if len(dpos):
        dpos.iloc[0] = pos.iloc[0]
    tc = dpos.abs() * float(cost)
    strat = pos * ret - tc
    tbl = pd.DataFrame({
        '策略(择时)': _perf(strat, ann=ann, pos=pos),
        '基准(买入持有)': _perf(ret, ann=ann, pos=pd.Series(1.0, index=ret.index)),
    }).T
    detail = pd.DataFrame({'market': ret, 'position': pos, 'strategy': strat})
    return tbl, detail

def explain_ts_metrics():
    print("""\
---------------------------------------------------------------
时序（择时）研究关键指标：
  IC(Spearman)    信号值 vs 未来收益的秩相关。|IC|>0.05 有含义；符号=方向。
  t值(OLS)        普通 t；自相关+重叠窗口会虚高，别单独信。
  t值(Newey-West) 修正 t，判显著以它为准（|t|>=2）。
  NW滞后          自动 ≈ max(经验规则, 前瞻期)。
  分段IC / IR     历史等分4段各算 IC：均值=方向稳定，IR=跨段一致。
  状态分组        信号分档后各档未来收益/胜率/折年化；极差=最强-最弱。
  择时回测        策略(信号>阈值持有) vs 基准(买入持有)：
                  年化/波动/夏普/最大回撤/日胜率/在场占比。
  判读            择时价值 = 夏普提升 + 回撤收窄；只看收益提升不算数。
---------------------------------------------------------------""")

def run_ts_test(signal_name, code, start, end,
                period=20,            # 前瞻期（交易日，IC/状态分析用，不影响回测）
                th=0.0,               # 触发阈值：signal > th 持有
                decision_freq='D',    # 决策频率：'D'/'W'/'M'
                cost=0.0005,          # 每次仓位切换成本
                direction=1,          # 1 正向；-1 翻转（波动率类）
                params=None,          # 信号构造参数，如 {'n': 200}
                strict=False,         # 前瞻收益是否从 t+1 起算
                state_thresholds=None,  # 状态分组用固定阈值，如 RSI 的 [25, 75]；None=按分位切
                verbose=True):
    """单信号时序（择时）检验。返回 dict: signal / px / close / fwd /
    ic_stats / ic_segments / state_table / state_spread / backtest / detail"""
    sig, px = build_ts_signal(signal_name, code, start, end, params=params)
    close = px['close']
    fwd = ts_forward_return(close, period, strict)
    stats, seg = ts_ic_stats(sig, fwd, overlap_hint=period)
    state_tbl, spread = ts_state_analysis(sig, fwd, thresholds=state_thresholds)
    bt, detail = ts_backtest(sig, close, th=th, decision_freq=decision_freq,
                             cost=cost, direction=direction)
    if verbose:
        print('=' * 84)
        print('【时序检验】%s | %s | %s ~ %s' % (signal_name, code, start, end))
        print('参数: params=%s | 前瞻=%d日 | 触发 th=%s | 决策=%s | 成本=%.4f | 方向=%+d'
              % (params, period, th, decision_freq, cost, direction))
        print('\n[时序 IC] %s @ %s | signal_t vs 未来%d日收益' % (signal_name, code, period))
        if stats.empty:
            print('  样本不足')
        else:
            print('  IC(Spearman) %.4f | t(OLS) %.2f | t(NW) %.2f | 分段IC均值 %.4f | 分段IC_IR %.2f'
                  % (stats['IC(Spearman)'], stats['t值(OLS)'], stats['t值(Newey-West)'],
                     stats['分段IC均值'], stats['分段IC_IR']))
            print('  判读: |IC|>0.05 且 |t(NW)|>=2 才有含义')
        if not state_tbl.empty:
            print('\n[状态分组] %s 不同信号强度的未来收益%s'
                  % (signal_name, '（固定阈值 %s）' % (state_thresholds,)
                     if state_thresholds is not None else '（按分位切档）'))
            print(state_tbl[['状态', '区间', '观测数', '平均收益%', '胜率']].round(3).to_string(index=False))
            print('  极差(最高-最低): %.3f%%/期' % spread['spread%'])
        if not bt.empty:
            print('\n[择时回测] %s | signal > %s 持有，否则空仓 | 决策=%s | 成本 %.4f | 次日生效'
                  % (signal_name, th, decision_freq, cost))
            print(bt.round(3).to_string())
            print('  判读: 看“夏普提升 + 回撤收窄”；只收益提升不算数')
    return {'signal': sig, 'px': px, 'close': close, 'fwd': fwd,
            'ic_stats': stats, 'ic_segments': seg,
            'state_table': state_tbl, 'state_spread': spread,
            'backtest': bt, 'detail': detail}

def run_ts_batch(signal_names, code, start, end, period=20,
                 th=0.0, decision_freq='D', cost=0.0005, direction=1,
                 state_thresholds=None):
    """多信号同标的、同一口径对比。参数同 run_ts_test。
    返回对比表（含策略/基准的年化、夏普、回撤、在场占比）。"""
    rows = []
    for name in signal_names:
        try:
            r = run_ts_test(name, code, start, end, period=period, th=th,
                            decision_freq=decision_freq, cost=cost,
                            direction=direction, state_thresholds=state_thresholds,
                            verbose=False)
            st, bt = r['ic_stats'], r['backtest']
            rows.append({'信号': name,
                         'IC': st.get('IC(Spearman)', np.nan),
                         't(NW)': st.get('t值(Newey-West)', np.nan),
                         '分段IC_IR': st.get('分段IC_IR', np.nan),
                         '策略年化%': bt.loc['策略(择时)', '年化收益%'] if not bt.empty else np.nan,
                         '基准年化%': bt.loc['基准(买入持有)', '年化收益%'] if not bt.empty else np.nan,
                         '策略夏普': bt.loc['策略(择时)', '夏普'] if not bt.empty else np.nan,
                         '基准夏普': bt.loc['基准(买入持有)', '夏普'] if not bt.empty else np.nan,
                         '策略回撤%': bt.loc['策略(择时)', '最大回撤%'] if not bt.empty else np.nan,
                         '基准回撤%': bt.loc['基准(买入持有)', '最大回撤%'] if not bt.empty else np.nan,
                         '在场占比%': bt.loc['策略(择时)', '在场占比%'] if not bt.empty else np.nan})
        except Exception as e:
            print('  %s 失败: %s' % (name, e))
    df = pd.DataFrame(rows)
    print('=' * 84)
    print('【多信号对比】%s | %s ~ %s | th=%s | 决策=%s | 成本=%.4f'
          % (code, start, end, th, decision_freq, cost))
    print(df.round(4).to_string(index=False))
    print('候选标准: 策略夏普>基准夏普 且 策略回撤<基准回撤 且 |t(NW)|>=2')
    return df

def ts_ic_decay(signal_name, code, start, end,
                periods=(5, 10, 20, 60, 120), params=None, strict=False):
    """时序 IC 衰减扫描：同一信号、多个前瞻期，用于选持有期。"""
    sig, px = build_ts_signal(signal_name, code, start, end, params=params)
    close = px['close']
    rows = []
    for p in periods:
        fwd = ts_forward_return(close, p, strict)
        stats, _ = ts_ic_stats(sig, fwd, overlap_hint=p)
        if stats.empty:
            continue
        rows.append({'前瞻期(日)': p, '样本数': int(stats['样本数']),
                     'IC(Spearman)': stats['IC(Spearman)'],
                     't(OLS)': stats['t值(OLS)'],
                     't(NW)': stats['t值(Newey-West)'],
                     'NW滞后': int(stats['NW滞后']),
                     '分段IC均值': stats['分段IC均值'],
                     '分段IC_IR': stats['分段IC_IR']})
    df = pd.DataFrame(rows).round(4)
    print('【时序IC衰减】%s | %s | params=%s' % (signal_name, code, params))
    print(df.to_string(index=False))
    print('判读: t(NW) 仍 >=2 且 分段IC_IR 最高的前瞻期，就是该配的持有期')
    return df

# ============================================================
# Part D 回测衔接（研究环境不要调用）
# ============================================================
def filter_tradable(codes):
    """回测/模拟专用：剔除停牌与 ST（ETF 的 is_st 恒为 False，可通用）。"""
    current = get_current_data()
    out = []
    for s in codes:
        try:
            if not current[s].paused and not current[s].is_st:
                out.append(s)
        except Exception:
            out.append(s)
    return out
