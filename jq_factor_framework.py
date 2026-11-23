# -*- coding: utf-8 -*-
# Copyright 2026 Neeko-ovo
# Apache License 2.0 -- 见项目根目录 LICENSE / NOTICE（再分发请保留署名）
"""
聚宽因子研究框架 · 导入版（单文件）
================================================

把原来 notebook 里逐格 exec 的五个模块合并成一个可以直接 import 的模块：

    import jq_factor_framework as jf

    jf.preview_pool('HS300')
    r = jf.run_factor_test('px_pct_750', 'HS300', '2019-01-01', '2024-12-31', period=20)
    jf.plot_single_all(r)

内容（按加载顺序）：
    Part A~D  池配置 / 数据读取 / 三通道因子 / 截面检验 / 时序检验 / 回测衔接
    附录 A    截面可视化
    附录 B    时序可视化
    附录 C    时序有效性补丁
    Part F    历史分位打分因子（时序 pct_* / 截面 px_pct_* / 估值分位 run_pct_value_test）
    Part G    IC 视角（screen_ic / ic_tables / ic_plots / ic_report）
    Part H    仓位管理（ATR / 分位 / 底部打分 / 状态机 / 连续仓位回测）
    Part I    RSI 体系（计算 / 75·25 分区 / 反转事件与状态机 / 分区收益 / 一键报告）

运行环境：
    聚宽投资研究（需要 get_price 等平台函数）。加载时执行 from jqdata import *
    本地 mock 测试见 selftest_import_mock.py。

注意：
    1) 本文件是 build_module.py 自动生成的，改逻辑请改五个源文件后重新生成。
    2) 各模块函数名已做过冲突排查（两两交集为空），可以安全共处一个命名空间。
"""

__framework_version__ = '4.5.0'


# ============================================================
# Part A~D 通用框架  池配置 / 数据读取 / 三通道因子 / 截面检验 / 时序检验 / 回测衔接
# 源文件：jq_unified_factor_framework.py
# ============================================================
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


# ============================================================
# 附录 A 截面可视化  jq_factor_viz（只读框架返回值，不改框架逻辑）
# 源文件：jq_factor_viz.py
# ============================================================
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


# ============================================================
# 附录 B 时序可视化  jq_ts_viz（函数名统一 ts_ / _tsv_ 前缀，避免覆盖）
# 源文件：jq_ts_viz.py
# ============================================================
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


# ============================================================
# 附录 C 时序有效性补丁 jq_tsa_patch（滚动IC / 状态 / 换仓 / 年度表）
# 源文件：jq_tsa_patch.py
# ============================================================
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


# ============================================================
# Part F 历史分位打分因子  pct_rank / pct_to_score / px_pct_* / run_pct_value_test
# 源文件：jq_pct_factor.py
# ============================================================
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


# ============================================================
# Part G IC 视角        screen_ic / ic_tables / ic_plots / ic_report / ts_ic_table
# 源文件：jq_ic_report.py
# ============================================================
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


# ============================================================
# Part H 仓位管理      atr_pct_20 / atr_rank_20 / atr_squeeze / vol_ratio / bw_rank / rsi_14 / mom_60 / bottom_score_panel / weight_state / run_weight_test / weight_report
# 源文件：jq_position.py
# ============================================================
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


# ============================================================
# Part I RSI 体系      calc_rsi / rsi_panel / rsi_zone / rsi_reversal_events / rsi_reversal_state / rsi_zone_forward / rsi_report / plot_rsi / rsi / rsi_rev / rsi_rev_hold
# 源文件：jq_rsi.py
# ============================================================
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


# ============================================================
# 导出与自检
# ============================================================
__version__ = __framework_version__

_FRAMEWORK_MAP = {
    'Part A': '池配置 / 数据读取 / 三通道因子（CUSTOM_FACTORS）',
    'Part B': '截面研究：Rank IC / 分层 / 换手 / 分档 / 批量 / 组合 / 衰减',
    'Part C': '时序研究：择时信号（TS_SIGNALS）/ NW修正IC / 状态分析 / 择时回测',
    'Part D': '回测衔接：filter_tradable',
    'Part F': '历史分位打分因子：pct_rank / pct_to_score / px_pct_* / run_pct_value_test',
    'Part G': 'IC 视角：screen_ic（初筛）/ ic_tables（各表）/ ic_report（一页纸）',
    'Part H': '仓位管理：bottom_score_panel（底部打分）/ weight_state（状态机）/ run_weight_test（连续仓位回测）/ weight_report',
    'Part I': 'RSI 体系：calc_rsi（计算）/ rsi_zone（75·25 分区）/ rsi_reversal_state（反转状态机）/ rsi_zone_forward（分区收益）/ rsi_report（一键）/ plot_rsi',
    'Viz':    '截面可视化：plot_single_all / plot_batch / plot_combo / save_all',
    'TsViz':  '时序可视化：plot_all / ts_save_all / plot_batch_compare',
    'Tsa':    '时序有效性：tsa_build / tsa_report / tsa_yearly_table',
}


def usage():
    """打印最短用法（不知道从哪下手时先调它）。"""
    lines = [
        '=' * 66,
        '聚宽因子研究框架 v%s' % __version__,
        '=' * 66,
        '命名空间：import jq_factor_framework as jf',
        '',
        '—— 截面（一篮子股票，问"谁更好"）——',
        "  jf.preview_pool('HS300')",
        "  r = jf.run_factor_test('px_pct_750', 'HS300', '2019-01-01', '2024-12-31', period=20)",
        '  jf.plot_single_all(r)                 # 13 张图',
        "  jf.scan_ic_decay('px_pct_750', 'HS300', S, E)   # 先定持有期（align=True：调仓间隔=持仓期）",
        "  df, _d, panels, _f = jf.run_batch(['px_pct_750', 'rev_20'], 'HS300', S, E)",
        '  jf.screen_ic(df)                      # 初筛：按 IC/IR/t/覆盖率 过一遍',
        '  jf.ic_report(r, save=True)            # 单因子：6 张 IC 表 + 6 张 IC 图',
        '  jf.plot_batch(df); jf.plot_corr(panels)',
        '',
        '—— 时序（单个标的，问"什么时候持有"）——',
        "  t = jf.run_ts_test('pct_score', '000300.XSHG', '2014-01-01', '2024-12-31',",
        "                     period=20, th=60, decision_freq='W')",
        "  jf.plot_all(t, code='000300.XSHG', th=60)",
        "  rep = jf.tsa_report('pct_score', ['000300.XSHG'], '2014-01-01', '2024-12-31',",
        "                      period=20, th=60, decision_freq='W')",
        '  rep["table"]',
        '',
        '—— RSI（超买>75 / 超卖<25）——',
        "  r = jf.rsi_report('000300.XSHG', '2019-01-01', '2024-12-31', plot=True)",
        "  t = jf.run_ts_test('rsi_rev_hold', '000300.XSHG', '2019-01-01', '2024-12-31', th=0.5)",
        '',
        '—— 加自己的因子 ——',
        "  @jf.register_factor('my_rev_5')        # 截面",
        "  @jf.register_ts_signal('my_signal')    # 时序",
        '',
        '—— 自检 ——',
        '  jf.health_check()                      # 数据/池/信号 一次过一遍',
        '=' * 66,
    ]
    print('\n'.join(lines))
    return None


def health_check(pool='HS300', start='2019-01-01', end='2024-12-31', index_code=None, verbose=True):
    """一键自检：交易日 / 调仓日 / 股票池 / 行情 / 指数，跑通了再开始研究。

    返回 {'ok': bool, 'messages': [str, ...]}，只读不写，不改任何状态。
    """
    ok = True
    msgs = []
    msgs.append('框架版本 %s' % __version__)
    msgs.append('时序信号 %d 个，自定义截面因子 %d 个' % (len(TS_SIGNALS), len(CUSTOM_FACTORS)))

    td = None
    try:
        td = get_trade_days(start_date=start, end_date=end)
        msgs.append('[OK] 交易日 %d 天（%s ~ %s）'
                    % (len(td), str(td[0].date()), str(td[-1].date())))
    except Exception as e:
        ok = False
        msgs.append('[X] get_trade_days 失败：%r' % (e,))

    rebal = None
    if td is not None and len(td) >= 2:
        try:
            rebal = get_rebalance_dates(start, end, 'M')
            msgs.append('[OK] 月频调仓日 %d 个' % len(rebal))
            if len(rebal) < 4:
                ok = False
                msgs.append('[X] 调仓日不足 4 个，把区间拉长或改 freq')
        except Exception as e:
            ok = False
            msgs.append('[X] get_rebalance_dates 失败：%r' % (e,))

    if rebal:
        try:
            probe = [rebal[0], rebal[-1]] if len(rebal) > 1 else list(rebal)
            codes, _per = resolve_universe(pool, probe, verbose=False)
            msgs.append('[OK] 池 %s：首末两期并集 %d 只' % (pool, len(codes)))
            if codes:
                px = get_price_history(codes[:3], rebal[-1], 3, ['close']).get('close')
                if px is None or px.dropna(how='all').empty:
                    ok = False
                    msgs.append('[X] 行情取不到（get_price_history 返回空）')
                else:
                    msgs.append('[OK] 行情可用：%s' % '、'.join(list(px.columns)[:3]))
        except Exception as e:
            ok = False
            msgs.append('[X] 池/行情失败：%r' % (e,))

    code = index_code or (INDEX_POOL.get(pool) if isinstance(pool, str) else None)
    if code:
        try:
            s = get_price_range([code], start, end, ['close']).get('close')
            if s is None or s.dropna(how='all').empty:
                ok = False
                msgs.append('[X] 指数 %s 行情为空' % code)
            else:
                col = s.iloc[:, 0].dropna()
                msgs.append('[OK] 指数 %s：%d 根日线（%s ~ %s）'
                            % (code, len(col), str(col.index[0].date()), str(col.index[-1].date())))
        except Exception as e:
            ok = False
            msgs.append('[X] 指数 %s 读取失败：%r' % (code, e))

    if verbose:
        print('\n'.join(msgs))
        print('— 自检%s —' % ('通过' if ok else '有问题'))
    return {'ok': ok, 'messages': msgs}


# 所有公开名字都导出（含 matplotlib 的 np/pd/plt，方便直接画图）
__all__ = sorted([_n for _n in list(globals()) if not _n.startswith('_')])

print('[jq_factor_framework] v%s 已加载：时序信号 %d 个，自定义截面因子 %d 个'
      % (__version__, len(TS_SIGNALS), len(CUSTOM_FACTORS)))
