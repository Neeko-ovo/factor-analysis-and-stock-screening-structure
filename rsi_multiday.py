# -*- coding: utf-8 -*-
# Copyright 2026 Neeko-ovo
# Apache License 2.0 -- 见项目根目录 LICENSE / NOTICE（再分发请保留署名）
"""
RSI 多日图 —— 独立小工具（聚宽研究环境直接跑，不依赖主框架）

功能
----
1. 自定义池子：直接给代码列表，或者给池名（HS300 / ZZ500 / ZZ800 / ZZ1000 /
   SZ50 / CYB / KC50，也可以写中文「沪深300」）。
   ETF 池也能直接写池名：'ETF'（全池 64 只）、'宽基' / '防御' / '行业' / '科技' /
   '新能源' / '港股' / '海外' / '商品'，细的还有 '医药' / '金融' / '周期'，
   也可以 '宽基ETF' 这样带后缀。
2. 自定义周期：日线 D / 五日线 5D / 周线 W / 月线 M。
3. 自定义取多少根 K 线（BARS），也支持直接给时间区间（PLOT_START / PLOT_END）。
4. RSI 天数可调（默认 14），超买 / 超卖线可调（默认 70 / 30）。
5. 上方 K 线图（红涨绿跌）+ 均线，下方 RSI 曲线 + 超买超卖区。
6. 可选成交量副图、反转事件标记、批量出图、导出 PNG、RSI 快照表。
7. 筛选：`screen_rsi(pool=..., signal='hit')` 在池子里挑出超买 / 超卖 /
   刚上穿超卖（买入信号）/ 刚下穿超买（卖出信号）的标的；池子可以是指数成分股，
   也可以是 ETF 池。筛完 `plot_hits(hits)` 直接看图。

口径
----
- 日线取前复权（fq='pre'），周 / 月 / 五日线都是**本地由日线合成**，保证 K 线与
  RSI 用的是同一份收盘价，不会出现「周线用聚宽周线、均线用日线」的口径错位。
- RSI 用 Wilder 平滑（ewm(alpha=1/n, adjust=False)），与 jq_rsi.calc_rsi 一致；
  前 n 根为 NaN，图中不画。
- 五日线：按交易日每 5 根合成一根（不是自然周）。
- 周线：按 ISO 周（周一到周日）合并，一根 K 线 = 该周的开高低收。
- 月线：按自然月合并。
"""

import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:  # 交互图引擎，没装就自动退回 matplotlib
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:  # 老环境 / 没装
    plotly = None
    go = None
    make_subplots = None
    HAS_PLOTLY = False

try:  # 聚宽环境
    from jqdata import *  # noqa: F401,F403
except Exception:  # 本地 / mock 环境
    pass


# ============================================================
# 默认参数
# ============================================================
DEFAULTS = dict(
    POOL=['000300.XSHG'],   # 标的池：代码列表，或池名列表
    FREQ='D',               # 'D' 日 / '5D' 五日 / 'W' 周 / 'M' 月
    BARS=300,               # 取多少根 K 线（数据窗口）
    PLOT_BARS=0,            # 图上只显示最后多少根（0 = 用 BARS）；数据窗口不受影响
    PLOT_START=None,        # 图左边界，如 '2020-01-01'
    PLOT_END=None,          # 图右边界，如 '2024-12-31'
    RSI_N=14,               # RSI 天数
    OB=70.0,                # 超买线（想改就改这一个数，图上的红虚线和分档都跟着走）
    OS=30.0,                # 超卖线
    MA_LIST=(20, 60),       # 均线周期，可空元组
    SHOW_VOLUME=True,       # 成交量子图
    MARK_EVENTS=True,       # 标反转（上穿超卖 / 下穿超买）
    MAX_CHARTS=6,           # 一次最多画几只（0 = 不限）
    SAVE_DIR=None,          # 填目录就导出 png
    DPI=130,
    FIGSIZE=(14.0, 8.2),
    CJK_FONT=True,          # 关掉就用默认字体（没有中文字体时）
    DATA_SOURCE=None,       # None/'auto' 自动；'jq' 聚宽；'bao' baostock；'ak' akshare

    # ---- FVG（公允价值缺口）+ 斐波那契回踩 ----
    FVG_ON=True,            # 图上画不画缺口
    FVG_RATIO=0.618,        # 回踩线在缺口里的深度：0=上沿（刚碰到），1=下沿（完全回补）
    FVG_REF='gap',          # 'gap' 按缺口高度算 / 'leg' 按形成缺口那几根的脉冲腿算
    FVG_DIR='bull',         # 筛股方向：'bull' 上涨缺口（回调买）/ 'bear' 下跌缺口（反弹）/ 'both' 两个都看
    FVG_MIN_PCT=0.3,        # 缺口高度至少占价格的百分之几，滤掉针尖大的缺口（跨度就调这个）
    FVG_LOOKBACK=120,       # 只找最近多少根 K 线内形成的缺口
    FVG_CONFIRM='close',    # 'touch' 碰到就算 / 'close' 碰到还收在线上 / 'bull' 还要求阳线
    FVG_WITHIN=2,           # 最近几根内回踩算「刚刚」
    FVG_SHOW=3,             # 图上画最近几个缺口
    FVG_BEAR=True,          # 图上画不画空头缺口

    # ---- FVG 的「像不像话」门槛：把针尖缺口、小阳小阴造出来的假缺口筛掉 ----
    FVG_MIN_BODY=0.5,       # 造缺口那根 K 线的实体至少百分之几（0.5 = 小阳小阴蹭出来的不要）
    FVG_MIN_RANGE=0.0,      # 造缺口那根 K 线的振幅至少百分之几（0 = 不看）
    FVG_MAX_FILL=100.0,     # 缺口最多被吃掉多少 % 还算有效（80 = 吃掉八成就不认）
    FVG_BIG_PCT=2.0,        # 「大缺口」档位的门槛：高度 >= 2%
    FVG_FRESH_DAYS=3,       # 「新缺口」档位的门槛：形成不超过几根
    FVG_NEAR_DIST=2.0,      # 现价离回踩线多少 % 以内算「就快碰到了」

    # ---- 综合筛选的默认参数 ----
    SCREEN_WITHIN=1,        # RSI / FVG 「刚刚」的默认窗口
    TREND_FAST=20,          # 趋势确认的短均线
    TREND_SLOW=60,          # 趋势确认的长均线
    VOL_N=5,                # 缩量比较用的均量天数
    VOL_SHRINK=0.8,         # 量 < 均量 * 这个数 算缩量
    VOL_NEAR=1.0,           # 量比 < 这个数 算「接近缩量」
    RSI_NEAR=5.0,           # RSI 落在 [超卖线, 超卖线+这个数] 里 算「接近超卖」
    TREND_NEAR=0.02,        # 趋势差一点的容差（0.02 = 2%）

    # ---- ATR（平均真实波幅）：波动大小 + 日/周/月各自一套档位 ----
    ATR_ON=True,            # 算不算 ATR（关掉表格里那几列就是空的）
    ATR_N=14,               # ATR 周期（Wilder 平滑，跟 RSI 一样用天数）
    ATR_MA_N=20,            # 判断扩张/收缩：ATR 跟它自己的多少根均值比
    ATR_EXP=0.10,           # 比自己均线高/低这个比例以上，才算扩张/收缩
    ATR_LEVELS={'D': (1.2, 3.5)},  # 各周期档位：(低波动上限, 高波动下限)，单位 ATR 占价格 %
    ATR_AUTO=True,          # 没单独写 W / M 的阈值时，按 sqrt(5) / sqrt(21) 从日线放大
    ATR_PCTL_N=250,         # 算「ATR 分位」回看多少根
    ATR_SIDE='low',         # atr 这个指标怎么算命中：'low' 低波动 / 'high' 高波动 / 'both' 两头都要
    ATR_NEAR=1.15,          # 离档位边界多少倍以内算「接近」
    SHOW_ATR=True,          # K 线图下面再挂一个 ATR% 副图（画在 RSI 下面一格）
    ATR_PANE_H=1.0,         # ATR 副图的高度占比（K线 3.0 / 量 1.05 / RSI 1.85 作参照）
    ATR_SPIKE=1.8,          # 当根 TR 超过 ATR 的几倍算「异动」（跳空/爆发那一根）
    ATR_SPIKE_LOOK=5,       # 最近多少根里出现过异动，就算现在「有异动」
    ATR_SQ_PCTL=20.0,       # ATR% 落在近 N 根的低多少分位以下，算「极低波动」
    ATR_QUIET_N=10,         # 判断「安静」看最近多少根
    ATR_QUIET_RATIO=0.7,    # 这几根里 TR 低于 ATR 的比例达到多少算安静（配合上一项）
    ATR_SIG='both',         # atr_sig 指标认哪种信号：'spike' 只认异动 / 'squeeze' 只认收缩 / 'both'
    SCREEN_TOP=0,           # 筛完只留分数最高的几个：10=前 10 只，0.1=前 10%，0=全留
    SCREEN_CLOSEST=0,       # 筛完只留离回踩线最近的几个：同上写法，0=不算

    # ---- 画图引擎 ----
    PLOT_ENGINE='mpl',      # 'mpl' 出静态图（快、能存 png）/ 'plotly' 出可缩放悬停的交互图
    PLOTLY_HEIGHT=840,      # plotly 图的高度像素
    PLOTLY_FORMAT='png',    # plotly + SAVE_DIR 时导出成什么：'png'（要 kaleido）/ 'html'
)

C_UP = '#c0392b'            # 涨（红）
C_DN = '#27ae60'            # 跌（绿）
C_RSI = '#6c5ce7'           # RSI 线
C_MA = ['#2c7fb8', '#e67e22', '#8e44ad', '#16a085', '#d35400']

# ------------------------------------------------------------
# ETF 池（1~20 只沿用主框架 jq_unified_factor_framework.py 的 ETF_POOL，其后为扩充）
# 池名直接写 'ETF'（全池）或 '宽基' / '防御' / '行业' / '科技' / '新能源' / '港股' / '海外' / '商品'
# 细的还有 '医药' / '金融' / '周期'
# ------------------------------------------------------------
ETF_POOL = {
    '宽基': [('510300.XSHG', '沪深300ETF'), ('510500.XSHG', '中证500ETF'),
             ('512100.XSHG', '中证1000ETF'), ('563300.XSHG', '中证2000ETF'),
             ('510050.XSHG', '上证50ETF'), ('159915.XSHE', '创业板ETF'),
             ('159949.XSHE', '创业板50ETF'), ('588000.XSHG', '科创50ETF'),
             ('159601.XSHE', 'A50ETF')],
    '防御': [('512890.XSHG', '红利低波ETF'), ('510880.XSHG', '红利ETF'),
             ('511360.XSHG', '短融ETF'), ('511010.XSHG', '国债ETF'),
             ('511090.XSHG', '30年国债ETF')],
    '行业': [('512010.XSHG', '医药ETF'), ('512170.XSHG', '医疗ETF'),
             ('159992.XSHE', '创新药ETF'), ('159928.XSHE', '消费ETF'),
             ('512690.XSHG', '酒ETF'), ('159996.XSHE', '家电ETF'),
             ('516810.XSHG', '农业ETF'), ('159865.XSHE', '养殖ETF'),
             ('159766.XSHE', '旅游ETF'), ('512660.XSHG', '军工ETF'),
             ('512880.XSHG', '证券ETF'), ('512800.XSHG', '银行ETF'),
             ('515630.XSHG', '证券保险ETF'), ('512400.XSHG', '有色金属ETF'),
             ('515220.XSHG', '煤炭ETF'), ('159870.XSHE', '化工ETF'),
             ('516150.XSHG', '稀土ETF'), ('512200.XSHG', '房地产ETF'),
             ('159611.XSHE', '电力ETF'), ('512580.XSHG', '环保ETF'),
             ('512980.XSHG', '传媒ETF'), ('159869.XSHE', '游戏ETF')],
    '科技': [('512480.XSHG', '半导体ETF'), ('512760.XSHG', '芯片ETF'),
             ('515880.XSHG', '通信ETF'), ('159998.XSHE', '计算机ETF'),
             ('159852.XSHE', '软件ETF'), ('159819.XSHE', '人工智能ETF'),
             ('159530.XSHE', '机器人ETF'), ('516510.XSHG', '云计算ETF'),
             ('515400.XSHG', '大数据ETF')],
    '新能源': [('515030.XSHG', '新能源车ETF'), ('159755.XSHE', '电池ETF'),
               ('515790.XSHG', '光伏ETF'), ('159566.XSHE', '储能电池ETF'),
               ('516160.XSHG', '新能源ETF'), ('159790.XSHE', '碳中和ETF')],
    '港股': [('513180.XSHG', '恒生科技ETF'), ('513330.XSHG', '恒生互联网ETF'),
             ('513050.XSHG', '中概互联网ETF'), ('513120.XSHG', '港股创新药ETF')],
    '海外': [('513100.XSHG', '纳指ETF'), ('513500.XSHG', '标普500ETF'),
             ('513000.XSHG', '日经225ETF'), ('513030.XSHG', '德国ETF'),
             ('513080.XSHG', '法国ETF'), ('513730.XSHG', '东南亚科技ETF')],
    '商品': [('518880.XSHG', '黄金ETF'), ('159985.XSHE', '豆粕ETF'),
             ('159980.XSHE', '有色期货ETF')],
}

# 代码 -> 中文名（画图 / 快照表用）
ETF_NAME = {}
for _v in ETF_POOL.values():
    for _c, _n in _v:
        ETF_NAME[_c] = _n

# 池名 -> 固定代码列表
_ETF_ALIAS = {}
_ETF_ALL = []
for _k, _v in ETF_POOL.items():
    _codes = [c for c, _ in _v]
    _ETF_ALIAS[_k] = _codes
    _ETF_ALIAS[_k + 'ETF'] = _codes
    _ETF_ALL.extend(_codes)
_ETF_ALIAS['ETF'] = _ETF_ALL
_ETF_ALIAS['ETF池'] = _ETF_ALL
_ETF_ALIAS['全部ETF'] = _ETF_ALL

# 跨类的小组合，方便只盯一条线
_ETF_SUB = {
    '医药': ['512010.XSHG', '512170.XSHG', '159992.XSHE'],
    '金融': ['512880.XSHG', '512800.XSHG', '515630.XSHG'],
    '周期': ['512400.XSHG', '515220.XSHG', '159870.XSHE', '516150.XSHG'],
}
_ETF_ALIAS.update(_ETF_SUB)

# 池名 -> 指数代码（字符串）或固定代码列表（list）
POOL_ALIAS = {
    'HS300': '000300.XSHG', '沪深300': '000300.XSHG',
    'ZZ500': '000905.XSHG', '中证500': '000905.XSHG',
    'ZZ1000': '000852.XSHG', '中证1000': '000852.XSHG',
    'ZZ800': '000906.XSHG', '中证800': '000906.XSHG',
    'SZ50': '000016.XSHG', '上证50': '000016.XSHG',
    'CYB': '399006.XSHE', '创业板': '399006.XSHE',
    'KC50': '000688.XSHG', '科创50': '000688.XSHG',
    'SZ': '399001.XSHE', '深证成指': '399001.XSHE',
    'SH': '000001.XSHG', '上证指数': '000001.XSHG',
}
POOL_ALIAS.update(_ETF_ALIAS)

FREQ_ALIAS = {
    'D': 'D', 'DAY': 'D', 'DAILY': 'D', '日': 'D', '日线': 'D', '1D': 'D',
    '5D': '5D', '5': '5D', 'W5': '5D', '五日': '5D', '5日': '5D', '周5': '5D',
    'W': 'W', 'WEEK': 'W', 'WEEKLY': 'W', '周': 'W', '周线': 'W', '1W': 'W',
    'M': 'M', 'MONTH': 'M', 'MONTHLY': 'M', '月': 'M', '月线': 'M', '1M': 'M',
}

FREQ_LABEL = {'D': '日线', '5D': '5日线', 'W': '周线', 'M': '月线'}

# 合成一根 K 线大约要几根日线
_FREQ_MULT = {'D': 1.0, '5D': 5.0, 'W': 5.0, 'M': 21.0}


# ============================================================
# 小工具
# ============================================================
def _norm_freq(freq):
    key = str(freq).strip().upper()
    if key not in FREQ_ALIAS:
        raise ValueError('FREQ 只能是 D / 5D / W / M，收到 %r' % (freq,))
    return FREQ_ALIAS[key]


def _merge_cfg(kw=None):
    cfg = dict(DEFAULTS)
    if kw:
        for k, v in kw.items():
            if k in cfg:
                cfg[k] = v
    return cfg


def _setup_style(cjk=True):
    """中文字体 + 负号 + 网格风格。找不到中文字体也不报错。"""
    try:
        if cjk:
            matplotlib.rcParams['font.sans-serif'] = [
                'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Hiragino Sans GB',
                'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC',
                'Arial Unicode MS', 'DejaVu Sans',
            ]
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['figure.autolayout'] = False
    except Exception:
        pass


def _to_ts(x):
    if x is None:
        return None
    return pd.Timestamp(x).normalize()


def _today():
    try:
        return pd.Timestamp.today().normalize()
    except Exception:
        return pd.Timestamp('2024-12-31')


_JQ_API_CACHE = {}


def _jq_api(name):
    """在聚宽环境里找 API：当前命名空间 → builtins → jqdata 模块。

    只缓存找到的结果，找不到不缓存（先跑依赖单元、后导入 jqdata 的情况也能认到）。
    """
    fn = _JQ_API_CACHE.get(name)
    if fn is not None:
        return fn
    try:
        cand = eval(name)  # noqa: F821  —— 走完整名字解析，builtins 里的也能拿到
        if callable(cand):
            fn = cand
    except Exception:
        fn = None
    if fn is None:
        try:
            import jqdata as _jq
            cand = getattr(_jq, name, None)
            if callable(cand):
                fn = cand
        except Exception:
            pass
    if fn is not None:
        _JQ_API_CACHE[name] = fn
    return fn


def _has_jq():
    return _jq_api('get_price') is not None


def _call_get_price(**kw):
    """兼容不同聚宽版本：panel 参数可能不存在。"""
    fn = _jq_api('get_price')
    if fn is None:
        raise RuntimeError(
            '找不到 get_price()。如果在聚宽研究环境里，先单独跑一次 from jqdata import *；'
            '本地跑就先 pip install akshare（会自动切到 akshare 数据源）'
        )
    try:
        return fn(**kw)
    except TypeError:
        kw.pop('panel', None)
        return fn(**kw)


# ---------- akshare（本地跑用） ----------
_AK_COLS = {
    '日期': 'date', '时间': 'date',
    '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
    '成交量': 'volume', '成交额': 'amount',
}

# 沪市指数代码（其余 000xxx 在深市是股票）
_SH_INDEX = set([
    '000001', '000016', '000300', '000688', '000852', '000905', '000906',
    '000985', '000903', '000922', '000015', '000009',
])


def _code6(code):
    s = str(code).strip().upper().split('.')[0]
    digits = ''.join(ch for ch in s if ch.isdigit())
    return digits[:6] if digits else s


def _code_suffix(code):
    s = str(code).strip().upper()
    return s.split('.')[-1] if '.' in s else ''


def _guess_kind(code):
    """猜是哪类标的：'index' / 'etf' / 'stock'。"""
    c6 = _code6(code)
    sfx = _code_suffix(code)
    if c6.startswith('399'):
        return 'index'
    if c6[:2] in ('51', '56', '58', '15', '16', '18', '50'):
        return 'etf'
    if c6.startswith('000'):
        if sfx == 'XSHE':
            return 'stock'
        if sfx == 'XSHG':
            return 'index'
        return 'index' if c6 in _SH_INDEX else 'stock'
    return 'stock'


def _ak_symbol(code):
    """6 位代码 -> akshare 指数接口要的 'sh000300' / 'sz399006'。"""
    c6 = _code6(code)
    sfx = _code_suffix(code)
    if sfx == 'XSHE':
        return 'sz' + c6
    if sfx == 'XSHG':
        return 'sh' + c6
    return ('sh' if c6.startswith(('6', '5', '9', '000')) else 'sz') + c6


def _ak_daily(code, start, end, adjust='qfq'):
    """用 akshare 取日线，返回 index=日期、columns=OHLCV 的标准表。"""
    try:
        import akshare as ak
    except Exception:
        raise RuntimeError('本地没装 akshare：pip install akshare')
    c6 = _code6(code)
    sym = _ak_symbol(code)
    s = pd.Timestamp(start).strftime('%Y%m%d')
    e = pd.Timestamp(end).strftime('%Y%m%d')
    adj = 'qfq' if str(adjust).lower() in ('pre', 'qfq') else None

    def _idx():
        return ak.stock_zh_index_daily_em(symbol=sym)

    def _etf():
        return ak.fund_etf_hist_em(symbol=c6, period='daily', start_date=s,
                                   end_date=e, adjust=adj or '')

    def _stk():
        return ak.stock_zh_a_hist(symbol=c6, period='daily', start_date=s,
                                  end_date=e, adjust=adj or '')

    order = {'index': [_idx, _etf, _stk],
             'etf': [_etf, _idx, _stk],
             'stock': [_stk, _etf, _idx]}
    errs = []
    for fn in order.get(_guess_kind(code), [_stk, _etf, _idx]):
        try:
            df = fn()
        except Exception as ex:
            errs.append(str(ex)[:60])
            continue
        if df is None or len(df) == 0:
            errs.append('返回空表')
            continue
        df = df.rename(columns=_AK_COLS)
        if 'date' not in df.columns or 'close' not in df.columns:
            errs.append('列名不认识：%s' % list(df.columns)[:4])
            continue
        out = df.copy()
        out.index = pd.to_datetime(out['date'])
        for col, fallback in (('open', 'close'), ('high', 'close'),
                              ('low', 'close'), ('volume', None)):
            if col not in out.columns:
                out[col] = out[fallback] if fallback else np.nan
        out = out[['open', 'high', 'low', 'close', 'volume']].astype(float)
        out = out.sort_index()
        out = out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
        out = out[~out.index.duplicated(keep='last')].dropna(subset=['close'])
        if len(out) == 0:
            errs.append('区间内没数据')
            continue
        return out
    raise RuntimeError('akshare 取不到 %s：%s' % (code, ' / '.join(errs)))


def _ak_cons(index_code):
    """akshare 取指数成分股，转成聚宽代码格式。"""
    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol=_code6(index_code))
    col = '成分券代码' if '成分券代码' in df.columns else df.columns[4]
    out = []
    for c in df[col].astype(str):
        c = ''.join(ch for ch in c if ch.isdigit())
        if len(c) != 6:
            continue
        out.append(c + ('.XSHG' if c.startswith(('6', '5', '9')) else '.XSHE'))
    return out


# ---------- baostock（本地免费，推荐） ----------
# 有 jq_io.py 就走它的缓存（同一只股票在多个 notebook 之间共用）；
# 没有就直连 baostock，逻辑一样：一次查询直接要前复权价。
_BAO_ADJ = {'pre': '2', 'qfq': '2', 'post': '1', 'hfq': '1', None: '3', 'none': '3', 'raw': '3', '': '3'}
_BAO_IDX = {'000300.XSHG': 'hs300', '000905.XSHG': 'zz500', '000016.XSHG': 'sz50',
            '000906.XSHG': 'zz800', '399006.XSHE': 'cyb'}
_BAO_FIELDS = 'date,open,high,low,close,volume,amount,tradestatus'


def _has_bao():
    try:
        import baostock  # noqa: F401
        return True
    except Exception:
        return False


def _bao_code(code):
    c6 = _code6(code)
    sfx = _code_suffix(code)
    if sfx == 'XSHE':
        return 'sz.' + c6
    if sfx == 'XSHG':
        return 'sh.' + c6
    return ('sh.' if c6.startswith(('6', '5', '9')) else 'sz.') + c6


def _bao_daily(code, start, end, adjust='pre'):
    """baostock 取日线，返回 index=日期、columns=OHLCV 的标准表。"""
    start = pd.Timestamp(start).strftime('%Y-%m-%d')
    end = pd.Timestamp(end).strftime('%Y-%m-%d')
    flag = _BAO_ADJ.get(str(adjust).lower() if adjust is not None else None, '2')

    try:                                  # 优先用 jq_io 的本地缓存
        import jq_io as _io
        px = _io.get_price([code], start_date=start, end_date=end,
                           fields=['open', 'high', 'low', 'close', 'volume'],
                           fq=('pre' if flag == '2' else ('post' if flag == '1' else None)),
                           panel=False)
        out = _normalize_price(px, code, ['open', 'high', 'low', 'close', 'volume'])
        if len(out):
            return out
    except Exception:
        pass

    try:
        import baostock as bs
    except Exception:
        raise RuntimeError('本地没装 baostock：pip install baostock')

    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError('baostock 登录失败：%s %s' % (lg.error_code, lg.error_msg))
    try:
        rs = bs.query_history_k_data_plus(_bao_code(code), _BAO_FIELDS, start_date=start,
                                          end_date=end, frequency='d', adjustflag=flag)
        if rs.error_code != '0':
            raise RuntimeError('%s %s' % (rs.error_code, rs.error_msg))
        rows = []
        while (rs.error_code == '0') and rs.next():
            rows.append(rs.get_row_data())
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    if not rows:
        raise RuntimeError('区间内没数据')
    df = pd.DataFrame(rows, columns=rs.fields)
    out = pd.DataFrame({
        'open': pd.to_numeric(df['open'], errors='coerce').values,
        'high': pd.to_numeric(df['high'], errors='coerce').values,
        'low': pd.to_numeric(df['low'], errors='coerce').values,
        'close': pd.to_numeric(df['close'], errors='coerce').values,
        'volume': pd.to_numeric(df['volume'], errors='coerce').values,
    }, index=pd.to_datetime(df['date']))
    out = out.sort_index()
    out = out[~out.index.duplicated(keep='last')].dropna(subset=['close'])
    if not len(out):
        raise RuntimeError('区间内没数据')
    return out


def _bao_cons(index_code):
    """baostock 取指数成分股（支持 hs300 / zz500 / sz50 / zz800），转聚宽格式。"""
    import baostock as bs
    key = _BAO_IDX.get(str(index_code).upper())
    if key is None:
        raise RuntimeError('baostock 没有 %s 的成分股' % index_code)
    names = ['hs300', 'zz500'] if key == 'zz800' else [key]
    if key == 'cyb':
        raise RuntimeError('baostock 没有创业板指成分股')
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError('baostock 登录失败：%s' % lg.error_msg)
    out = []
    try:
        for nm in names:
            fn = getattr(bs, 'query_%s_stocks' % nm, None)
            if fn is None:
                continue
            rs = fn()
            while (rs.error_code == '0') and rs.next():
                row = rs.get_row_data()
                c6 = row[1].split('.')[-1]
                out.append(c6 + ('.XSHG' if row[1].startswith('sh') else '.XSHE'))
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    seen = set()
    res = []
    for c in out:
        if c not in seen:
            seen.add(c)
            res.append(c)
    return res


def _normalize_price(raw, code, fields):
    """把 get_price 的返回统一成 index=日期、columns=fields 的表。

    聚宽单标的 panel=False 返回宽表；多标的返回 time/code 长表；
    本地 mock 一律返回长表。两种都接住。
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=fields)
    df = raw.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if 'time' in df.columns:
        if 'code' in df.columns:
            df = df[df['code'].astype(str) == str(code)]
        df = df.set_index('time')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    keep = [f for f in fields if f in df.columns]
    if 'close' not in keep:
        return pd.DataFrame(columns=fields)
    out = df[keep].astype(float)
    out = out[~out.index.duplicated(keep='last')]
    out = out.dropna(subset=['close'])
    # 缺列用 close 兜底，别让画图因为没 volume 就崩
    for f in fields:
        if f not in out.columns:
            out[f] = np.nan
    return out[fields]


# ============================================================
# 池子 / 取数 / 合成
# ============================================================
def resolve_pool(pool, date=None):
    """代码列表原样返回；池名展开成成分股。

    列表里可以直接塞 ('510300.XSHG', '沪深300ETF') 这种带名字的元组，
    也可以把 ETF_POOL['宽基'] 整个传进来，这里会自动取第一个元素当代码。
    """
    if pool is None:
        return []
    if isinstance(pool, str):
        pool = [pool]
    out = []
    for p in pool:
        if isinstance(p, (list, tuple)):        # (代码, 名称) 元组：只要代码
            p = p[0] if len(p) else ''
        elif isinstance(p, dict):               # {'code':..,'name':..} 这种也认
            p = p.get('code') or p.get('代码') or ''
        p = str(p).strip()
        if not p:
            continue
        alias = POOL_ALIAS.get(p.upper()) or POOL_ALIAS.get(p)
        if alias is None:
            out.append(p)
            continue
        if isinstance(alias, (list, tuple)):
            out.extend(list(alias))     # 固定列表池（ETF 池 / 宽基 / 行业 …）
            continue
        fn = _jq_api('get_index_stocks')
        if fn is None:
            for _try, _tag in ((_bao_cons, 'baostock'), (_ak_cons, 'akshare')):
                try:  # 本地模式：先 baostock，再 akshare
                    codes = _try(alias)
                    if codes:
                        out.extend(list(codes))
                        break
                except Exception:
                    continue
            else:
                print('[提示] %s 需要 get_index_stocks() / baostock / akshare，这里直接画指数本身' % p)
                out.append(alias)
            continue
        try:
            codes = fn(alias, date=date) if date is not None else fn(alias)
            out.extend(list(codes))
        except Exception as e:
            print('[提示] 展开 %s 失败（%s），改成直接画指数' % (p, e))
            out.append(alias)
    seen = set()
    res = []
    for c in out:
        if c not in seen:
            seen.add(c)
            res.append(c)
    return res


def fetch_daily(code, end_date=None, count=500, fq='pre', start_date=None, source=None):
    """取日线（前复权），返回 index=日期 的 OHLCV 表。

    数据源顺序：聚宽 get_price → baostock（免费，推荐本地用）→ akshare。
    DATA_SOURCE 可写 'auto' / 'jq' / 'bao' / 'ak' 强制指定。
    """
    fields = ['open', 'high', 'low', 'close', 'volume']
    src = str(source or 'auto').strip().lower()

    if src in ('auto', 'jq', 'joinquant') and _has_jq():
        raw = _call_get_price(
            security=code, count=int(count), end_date=end_date,
            frequency='daily', fields=fields, skip_paused=False, fq=fq, panel=False,
        )
        return _normalize_price(raw, code, fields)

    if src == 'jq':
        raise RuntimeError('DATA_SOURCE=\'jq\' 但当前环境没有 get_price()，换个地方跑或改成 auto')

    end = _to_ts(end_date) or _today()
    start = _to_ts(start_date)
    if start is None:
        start = end - pd.Timedelta(days=int(int(count) * 1.55) + 15)

    if src in ('auto', 'bao', 'baostock'):
        try:
            return _bao_daily(code, start, end, adjust=fq)
        except Exception as e:
            if src in ('bao', 'baostock'):
                raise
            print('[提示] baostock 取不到 %s（%s），改用 akshare' % (code, str(e)[:70]))
    return _ak_daily(code, start, end, adjust=fq)


def to_bars(df, freq):
    """日线 -> 5日 / 周 / 月 K 线。按交易日切，不依赖 pandas 的 resample 别名。"""
    f = _norm_freq(freq)
    if f == 'D' or df is None or len(df) == 0:
        return df.copy()
    n = len(df)
    if f == '5D':
        g = list(np.arange(n) // 5)
    elif f == 'W':
        # 不用 DatetimeIndex.isocalendar()（pandas 1.1 才有），也不用 tuple 分组键
        # （老版本 pandas 赋值 list of tuple 会退化成二维数组）。用 'YYYY-Www' 字符串。
        g = ['%04d-W%02d' % (ts.isocalendar()[0], ts.isocalendar()[1]) for ts in df.index]
    else:  # 'M'
        g = ['%04d-%02d' % (d.year, d.month) for d in df.index]

    tmp = df.copy()
    tmp['_g'] = g
    tmp['_d'] = df.index.values
    agg = tmp.groupby('_g', sort=True).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum', '_d': 'last',
    })
    agg = agg.set_index('_d')
    agg.index = pd.to_datetime(agg.index)
    agg.index.name = df.index.name
    return agg[['open', 'high', 'low', 'close', 'volume']]


def load_bars(code, cfg=None):
    """按配置取够长、合成到目标周期，并砍掉最后的空窗。"""
    cfg = _merge_cfg(cfg)
    f = _norm_freq(cfg['FREQ'])
    mult = _FREQ_MULT[f]
    warm = max([int(cfg['RSI_N'])] + [int(m) for m in (cfg['MA_LIST'] or ())]) + 5

    end = _to_ts(cfg['PLOT_END']) or _today()
    start = _to_ts(cfg['PLOT_START'])
    if start is not None:
        days = max((end - start).days, 40)
        # 一年约 250 个交易日 = 0.685
        count = int(days * 0.72) + int(warm * mult) + 40
    else:
        count = int((int(cfg['BARS']) + warm) * mult * 1.5) + 30

    daily = fetch_daily(code, end_date=end, count=count, start_date=start,
                        source=cfg.get('DATA_SOURCE'))
    if daily is None or len(daily) == 0:
        return daily
    bars = to_bars(daily, f)
    return bars


# ============================================================
# RSI
# ============================================================
def calc_rsi(close, n=14):
    """Wilder RSI，前 n 根是 NaN。口径与 jq_rsi.calc_rsi 一致。"""
    s = pd.Series(close).astype(float)
    n = int(n)
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(ad > 0, 100.0)
    rsi = rsi.where(~((ad <= 0) & (au <= 0)), 50.0)
    rsi = rsi.where(au.notna() & ad.notna(), np.nan)
    return rsi.clip(0.0, 100.0)


def rsi_zone(v, ob=70.0, os_=30.0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '无'
    if v > ob:
        return '超买'
    if v < os_:
        return '超卖'
    return '中性'


# ============================================================
# ATR（平均真实波幅）+ 日 / 周 / 月三套档位
# ============================================================
# TR  = max(当根振幅, |最高-昨收|, |最低-昨收|) —— 把跳空也算进波动
# ATR = TR 的 Wilder 平滑（和 RSI 同一种平滑，alpha = 1/N）
#
# 只看 ATR 绝对值没意义：15 块的票 ATR 0.3 和 1500 块的票 ATR 30 是一回事。
# 所以对外一律用 ATR% = ATR / 收盘 × 100，跨标的、跨周期都能比。
#
# 档位 = 用 ATR% 卡两道门：低于下限算「低波动」（安静、蓄势），
# 高于上限算「高波动」（躁动、真突破），中间是「正常」。
#
# 日 / 周 / 月各一套阈值：周线的 ATR% 天然比日线大 sqrt(5) 倍、月线约 sqrt(21) 倍。
# 默认只写日线 ATR_LEVELS={'D': (1.2, 3.5)}，W / M 自动按上面这个比例放大（ATR_AUTO=True）；
# 想给某个周期单独设，就把它写进 ATR_LEVELS，比如 {'D':(1.2,3.5), 'W':(4.0,10.0)}。
# ============================================================

_ATR_SCALE = {'D': 1.0, '5D': 5.0 ** 0.5, 'W': 5.0 ** 0.5, 'M': 21.0 ** 0.5}
_ATR_COLS = ['ATR', 'ATR%', 'ATR档', 'ATR扩张', 'ATR分位']


def calc_tr(df):
    """真实波幅 TR（价格单位、不平滑）：max(高-低, |高-昨收|, |低-昨收|)。

    和 ATR 的关系：TR 是「今天这一根到底晃了多少」，ATR 是它的 Wilder 平滑。
    TR 上下乱窜、ATR 慢慢跟，两条线在图上会互相穿过、缠在一起。
    """
    h = pd.Series(np.asarray(df['high'], dtype=float), index=df.index)
    lo = pd.Series(np.asarray(df['low'], dtype=float), index=df.index)
    c = pd.Series(np.asarray(df['close'], dtype=float), index=df.index)
    pc = c.shift(1)
    return pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)


def calc_atr(df, n=14):
    """Wilder ATR（价格单位），前 n 根是 NaN。df 需要 high / low / close。"""
    n = max(int(n), 1)
    return calc_tr(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def atr_levels(cfg=None, freq=None):
    """当前周期用的档位阈值 (低波动上限, 高波动下限)，单位是 ATR 占价格的 %。"""
    cfg = _merge_cfg(cfg)
    f = _norm_freq(freq if freq is not None else cfg.get('FREQ'))
    lv = cfg.get('ATR_LEVELS') or {}
    if f in lv:                      # 这个周期手填了，就照手填的来
        lo, hi = lv[f]
        return float(lo), float(hi)
    base = lv.get('D') or (1.2, 3.5)  # 没填就按日线基准缩放
    k = _ATR_SCALE.get(f, 1.0) if bool(cfg.get('ATR_AUTO', True)) else 1.0
    return float(base[0]) * k, float(base[1]) * k


def atr_levels_table(cfg=None):
    """把日 / 周 / 月三套档位打印出来 —— 改参数之前先看一眼现在到底是多少。"""
    cfg = _merge_cfg(cfg)
    lv = cfg.get('ATR_LEVELS') or {}
    rows = []
    for f, cn in (('D', '日线'), ('W', '周线'), ('M', '月线')):
        lo, hi = atr_levels(cfg, f)
        if f in lv:
            src = '手填'
        elif f == 'D':
            src = '基准'
        else:
            src = '自动 x%.2f' % _ATR_SCALE.get(f, 1.0) if bool(cfg.get('ATR_AUTO', True)) else '未缩放'
        rows.append({'周期': cn, '低波动 <': round(lo, 2), '正常': '%.2f ~ %.2f' % (lo, hi),
                     '高波动 >': round(hi, 2), '阈值来源': src})
    df = pd.DataFrame(rows, columns=['周期', '低波动 <', '正常', '高波动 >', '阈值来源'])
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.to_string(index=False))
    return df


def atr_state(df, cfg=None):
    """一只票的 ATR 状态（按 cfg['FREQ'] 那个周期的 K 线算）。

    返回 dict：
      ATR      当前 ATR（价格单位，跟收盘同量级）
      ATR%     ATR / 收盘 × 100 —— 跨标的可比的那个数
      ATR档    '低' / '正常' / '高'
      低阈值 / 高阈值   这一档实际用的两个边界（ATR%）
      ATR扩张  当前 ATR / 它自己前 N 根均值（>1 波动在放大）
      ATR趋势  '扩张' / '收缩' / '平稳'
      ATR分位  当前 ATR% 在过去 ATR_PCTL_N 根里的百分位（0~100）
    """
    cfg = _merge_cfg(cfg)
    empty = {'ATR': np.nan, 'ATR%': np.nan, 'ATR档': '', 'ATR扩张': np.nan,
             'ATR趋势': '', 'ATR分位': np.nan, '低阈值': np.nan, '高阈值': np.nan,
             '有数据': False}
    if not bool(cfg.get('ATR_ON', True)) or df is None or len(df) < 3:
        return empty
    for col in ('high', 'low', 'close'):
        if col not in df.columns:
            return empty
    n = int(cfg['ATR_N'])
    atr = calc_atr(df, n)
    if len(atr.dropna()) < 1:
        return empty
    c = pd.Series(np.asarray(df['close'], dtype=float), index=df.index)
    pct = (atr / c.replace(0.0, np.nan)) * 100.0
    a = float(atr.iloc[-1])
    p = float(pct.iloc[-1])
    if not np.isfinite(a) or not np.isfinite(p):
        return empty
    m = max(int(cfg['ATR_MA_N']), 2)
    ma = float(atr.iloc[-m:].mean()) if len(atr) >= m else float(atr.mean())
    ratio = (a / ma) if ma else np.nan
    lo, hi = atr_levels(cfg)
    band = '低' if p < lo else ('高' if p > hi else '正常')
    eps = float(cfg['ATR_EXP'])
    if not np.isfinite(ratio):
        trend = ''
    elif ratio >= 1.0 + eps:
        trend = '扩张'
    elif ratio <= 1.0 - eps:
        trend = '收缩'
    else:
        trend = '平稳'
    k = max(int(cfg['ATR_PCTL_N']), 20)
    hist = pct.iloc[-k:].dropna()
    pctl = float((hist <= p).mean() * 100.0) if len(hist) else np.nan
    sig = atr_signal(df, cfg)
    return {'ATR': a, 'ATR%': p, 'ATR档': band, 'ATR扩张': ratio, 'ATR趋势': trend,
            'ATR分位': pctl, '低阈值': lo, '高阈值': hi, '有数据': True,
            'ATR信号': sig['信号'], '异动倍数': sig['异动倍数'], '异动日': sig['异动日'],
            '距异动': sig['距异动'], '安静比': sig['安静比']}


def _atr_ratio(df, cfg):
    """逐根的 TR / ATR 倍数（价格单位相除，和 ATR% 无关）。"""
    atr = calc_atr(df, int(cfg['ATR_N']))
    tr = calc_tr(df)
    return tr / atr.replace(0.0, np.nan)


def atr_signal(df, cfg=None):
    """ATR 的两个信号：异动 / 收缩。

    异动 = 当根 TR 明显超过 ATR（默认 1.8 倍）。
        一根就晃出常态，最常见的成因就是跳空缺口、消息爆点、放量突破。
        单独看 ATR 是看不出来的（平滑会把它抹平），必须看 TR 对 ATR 的偏离。

    收缩 = ATR% 落在近 ATR_PCTL_N 根的低分位（默认 <=20%），
        并且最近 ATR_QUIET_N 根里大部分（默认 70%）TR 都贴着甚至低于 ATR。
        波动被压到极低、来回幅度越来越小 —— 蓄势，之后往哪边破不一定，
        但「安静到不正常」本身就是个信号。

    返回 dict：信号 / 异动 / 收缩 / 异动倍数 / 异动日 / 距异动 / 安静比 / ATR分位 / 有数据
    """
    cfg = _merge_cfg(cfg)
    out = {'信号': '', '异动': False, '收缩': False, '异动倍数': np.nan, '异动日': None,
           '距异动': np.nan, '安静比': np.nan, 'ATR分位': np.nan, '有数据': False}
    if not bool(cfg.get('ATR_ON', True)) or df is None or len(df) < 5:
        return out
    for col in ('high', 'low', 'close'):
        if col not in df.columns:
            return out
    ratio = _atr_ratio(df, cfg)
    if not len(ratio.dropna()):
        return out
    r = float(ratio.iloc[-1]) if np.isfinite(ratio.iloc[-1]) else np.nan
    spike_k = float(cfg.get('ATR_SPIKE', 1.8))
    look = max(int(cfg.get('ATR_SPIKE_LOOK', 5)), 1)
    win = ratio.iloc[-look:].dropna()
    spike = bool(len(win) and (win >= spike_k).any())
    spike_i = None
    spike_r = r
    if spike:
        hits = np.where(win.values >= spike_k)[0]
        if len(hits):
            spike_i = len(ratio) - len(win) + int(hits[-1])
            spike_r = float(win.values[hits[-1]])   # 报那根自己的倍数，不是当根的
    # 收缩：ATR% 的低分位 + 最近几根 TR 普遍不放大
    n = max(int(cfg['ATR_N']), 1)
    atr = calc_atr(df, n)
    c = pd.Series(np.asarray(df['close'], dtype=float), index=df.index)
    pct = (atr / c.replace(0.0, np.nan)) * 100.0
    k = max(int(cfg['ATR_PCTL_N']), 20)
    hist = pct.iloc[-k:].dropna()
    p = float(pct.iloc[-1]) if np.isfinite(pct.iloc[-1]) else np.nan
    pctl = float((hist <= p).mean() * 100.0) if len(hist) else np.nan
    qn = max(int(cfg.get('ATR_QUIET_N', 10)), 2)
    quiet = (ratio.iloc[-qn:] < 1.0).dropna()
    quiet_ratio = float(quiet.mean()) if len(quiet) else np.nan
    sq = bool(np.isfinite(pctl) and pctl <= float(cfg.get('ATR_SQ_PCTL', 20.0))
              and np.isfinite(quiet_ratio) and quiet_ratio >= float(cfg.get('ATR_QUIET_RATIO', 0.7)))
    out.update({'异动': spike, '收缩': sq, '异动倍数': spike_r if spike else r,
                '安静比': quiet_ratio,
                'ATR分位': pctl, '有数据': True,
                '异动日': (df.index[spike_i] if spike_i is not None else None),
                '距异动': (len(df) - 1 - spike_i) if spike_i is not None else np.nan})
    out['信号'] = '异动' if spike else ('收缩' if sq else '')
    return out


# ============================================================
# FVG（公允价值缺口）+ 斐波那契回踩
# ============================================================
# 多头缺口：第 3 根的最低价 > 第 1 根的最高价。中间那根往上冲，留下没人成交的空档。
#   下沿 = 第 1 根最高，上沿 = 第 3 根最低。价格回头跌进这个空档 = 支撑。
# 空头缺口：第 3 根的最高价 < 第 1 根的最低价。下沿 = 第 3 根最高，上沿 = 第 1 根最低。
#   价格反弹进这个空档 = 压力。
#
# 回踩线 = 缺口里按斐波那契取的一个深度，默认 61.8%：
#   FVG_REF='gap'：0% = 缺口上沿（价格刚碰到缺口），100% = 缺口下沿（完全回补）
#   FVG_REF='leg'：把形成缺口那 3 根的最高最低当成一段脉冲腿，按腿的回撤算
# 「回踩到位」= 价格探到这条线、但缺口还没被完全回补掉。
# ============================================================

_FVG_COLS = ['形成日', '方向', '下沿', '上沿', '高度', '高度%',
             '实体%', '振幅%', '已回补', '回补日', '最深%', '第几根']


def _fvg_empty():
    return pd.DataFrame(columns=_FVG_COLS)


def find_fvg(df, direction='both', min_pct=0.0, lookback=None,
             min_body=0.0, min_range=0.0, max_fill=None):
    """扫出所有 FVG，返回每行一个缺口。

    direction : 'bull' 多头缺口 / 'bear' 空头缺口 / 'both' 两种都要
    min_pct   : 缺口高度占价格的百分比下限（0.3 表示 0.3%）
    lookback  : 只认最近多少根 K 线内形成的缺口（None = 全部）
    min_body  : 中间那根（造缺口那根）的实体 |close-open|/open 至少百分之几，
                用来滤掉「小阳小阴蹭出来的缺口」。1.0 = 实体不到 1% 不要
    min_range : 同一根 K 线的振幅 (high-low)/open 至少百分之几（0 = 不看）
    max_fill  : 缺口最多被吃掉多少 % 还算数（80 = 吃掉八成的不列出来）

    返回列：形成日 / 方向 / 下沿 / 上沿 / 高度 / 高度% /
            实体% / 振幅%（造缺口那根 K 线的）/
            已回补（后来有没有被完全填掉）/ 回补日 /
            最深%（形成后最深探到缺口的百分之几，100 = 刚好填满）/ 第几根（在 df 里的位置）
    """
    if df is None or len(df) < 3:
        return _fvg_empty()
    d = str(direction or 'both').strip().lower()
    if d in ('bull', 'b', 'long', '多', '多头'):
        dirs = ['bull']
    elif d in ('bear', 's', 'short', '空', '空头'):
        dirs = ['bear']
    else:
        dirs = ['bull', 'bear']

    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    o = df['open'].values.astype(float) if 'open' in df.columns else c
    idx = df.index
    n = len(df)
    first = 2
    if lookback is not None and int(lookback) > 0:
        first = max(first, n - int(lookback) - 1)

    # 后缀最低 / 最高，O(1) 取「后面还有没有跌破 / 涨破」
    suf_lo = np.minimum.accumulate(l[::-1])[::-1]
    suf_hi = np.maximum.accumulate(h[::-1])[::-1]

    rows = []
    for i in range(first, n):
        # 中间那根 K 线（i-1）够不够「大阳 / 大阴」
        mid_o = float(o[i - 1]) if np.isfinite(o[i - 1]) and o[i - 1] > 0 else 0.0
        if mid_o > 0:
            body_pct = abs(float(c[i - 1]) - float(o[i - 1])) / mid_o * 100.0
            range_pct = (float(h[i - 1]) - float(l[i - 1])) / mid_o * 100.0
        else:
            body_pct = 0.0
            range_pct = 0.0
        if min_body and body_pct < float(min_body):
            continue
        if min_range and range_pct < float(min_range):
            continue

        for dd in dirs:
            if dd == 'bull':
                if not (l[i] > h[i - 2]):
                    continue
                lo, hi = float(h[i - 2]), float(l[i])
            else:
                if not (h[i] < l[i - 2]):
                    continue
                lo, hi = float(h[i]), float(l[i - 2])
            size = hi - lo
            if size <= 0:
                continue
            if min_pct and size / c[i] * 100.0 < float(min_pct):
                continue

            deep = 0.0
            filled = False
            fill_i = None
            if i + 1 < n:
                if dd == 'bull':
                    deep = (hi - suf_lo[i + 1]) / size
                    if suf_lo[i + 1] <= lo:
                        filled = True
                        hit = np.where(l[i + 1:] <= lo)[0]
                        fill_i = i + 1 + int(hit[0]) if len(hit) else i + 1
                else:
                    deep = (suf_hi[i + 1] - lo) / size
                    if suf_hi[i + 1] >= hi:
                        filled = True
                        hit = np.where(h[i + 1:] >= hi)[0]
                        fill_i = i + 1 + int(hit[0]) if len(hit) else i + 1
            deep = min(max(deep, 0.0), 1.0)

            # 吃得太深的缺口（快补完了）不算有效支撑 / 压力
            if not filled and max_fill is not None and float(max_fill) < 100.0:
                if deep * 100.0 > float(max_fill):
                    filled = True
                    if fill_i is None:
                        if dd == 'bull':
                            probe = np.where(l[i + 1:] <= hi - float(max_fill) / 100.0 * size)[0]
                        else:
                            probe = np.where(h[i + 1:] >= lo + float(max_fill) / 100.0 * size)[0]
                        fill_i = i + 1 + int(probe[0]) if len(probe) else None

            rows.append({
                '形成日': idx[i],
                '方向': '多头' if dd == 'bull' else '空头',
                '下沿': lo,
                '上沿': hi,
                '高度': size,
                '高度%': round(size / c[i] * 100.0, 3),
                '实体%': round(body_pct, 2),
                '振幅%': round(range_pct, 2),
                '已回补': bool(filled),
                '回补日': idx[fill_i] if fill_i is not None else None,
                '最深%': round(float(deep) * 100.0, 2),
                '第几根': int(i),
            })
    if not rows:
        return _fvg_empty()
    return pd.DataFrame(rows, columns=_FVG_COLS)


def fvg_state(df, ratio=0.618, ref='gap', direction='bull', min_pct=0.3,
              lookback=120, confirm='close', within=0, zones=None,
              min_body=0.0, min_range=0.0, max_fill=None, near_dist=2.0):
    """看最新一根 K 线：FVG 是什么状态、回踩线在哪儿、有没有回踩信号。

    返回一个 dict（没缺口也返回，字段是 None / False，方便直接填表）：
      方向 / 有效 / 缺口数 / 缺口日 / 下沿 / 上沿 / 高度 / 高度% / 实体% / 振幅%
      回踩线 / 已触及 / 触及日 / 距触及（几根之前碰到的）/ 距线%（现价离线多远）
      天数（缺口形成至今几根）/ 信号（'' 或 '回踩'）/ 信号日 / 最深 / 回补%
      接近（还没碰到线、但离得不远）
    """
    bull = str(direction).strip().lower() not in ('bear', 's', 'short', '空', '空头')
    ratio = float(min(max(float(ratio), 0.0), 1.0))
    res = {'方向': '多头' if bull else '空头', 'zones': None,
           '有缺口': False, '有效': False, '缺口数': 0,
           '缺口日': None, '下沿': None, '上沿': None, '高度': None, '高度%': None,
           '实体%': None, '振幅%': None,
           '回踩线': None, '已触及': False, '触及日': None, '距触及': None,
           '距线': None, '天数': 0, '信号': '', '信号日': None, '最深': None,
           '回补%': None, '接近': False}
    if df is None or len(df) < 4:
        return res

    if zones is None:
        zones = find_fvg(df, direction='bull' if bull else 'bear',
                         min_pct=min_pct, lookback=lookback,
                         min_body=min_body, min_range=min_range, max_fill=max_fill)
    res['zones'] = zones
    if zones is None or len(zones) == 0:
        return res
    res['有缺口'] = True

    z = zones.copy()
    z['_fill'] = z['已回补'].astype(bool)
    valid = z[~z['_fill']]
    res['缺口数'] = int(len(valid))
    # 主缺口 = 最近一个还没回补的；全回补了就取最近那个（用来看「补在哪儿」）
    cur = valid.iloc[-1] if len(valid) else z.iloc[-1]
    res['有效'] = bool(not cur['_fill'])

    lo, hi = float(cur['下沿']), float(cur['上沿'])
    size = hi - lo
    i = int(cur['第几根'])
    if str(ref).lower() == 'leg':
        leg_lo = float(np.nanmin(df['low'].values[max(0, i - 2):i + 1]))
        leg_hi = float(np.nanmax(df['high'].values[max(0, i - 2):i + 1]))
        span = leg_hi - leg_lo
        line = (leg_hi - ratio * span) if bull else (leg_lo + ratio * span)
    else:
        line = (hi - ratio * size) if bull else (lo + ratio * size)

    res['缺口日'] = cur['形成日']
    res['下沿'] = lo
    res['上沿'] = hi
    res['高度'] = size
    res['高度%'] = cur['高度%']
    res['实体%'] = cur['实体%']
    res['振幅%'] = cur['振幅%']
    res['回踩线'] = line
    res['最深'] = cur['最深%']
    res['回补%'] = cur['最深%']
    res['天数'] = int(len(df) - 1 - i)

    o = df['open'].values.astype(float)
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    n = len(df)

    # 形成后有没有碰到过回踩线
    if i + 1 < n:
        hit_arr = (l[i + 1:] <= line) if bull else (h[i + 1:] >= line)
        if bool(hit_arr.any()):
            pos = i + 1 + int(np.argmax(hit_arr))
            res['已触及'] = True
            res['触及日'] = df.index[pos]
            res['距触及'] = int(n - 1 - pos)

    # 最近 within 根里，有没有「碰到线 + 收回来」的确认
    st = max(i + 1, n - 1 - int(within))
    if st < n and res['有效']:
        touch = (l[st:] <= line) if bull else (h[st:] >= line)
        mk = str(confirm or 'close').strip().lower()
        ok = touch.copy()
        if mk in ('close', 'bull'):
            ok = ok & ((c[st:] > line) if bull else (c[st:] < line))
        if mk == 'bull':
            ok = ok & ((c[st:] > o[st:]) if bull else (c[st:] < o[st:]))
        if bool(ok.any()):
            pos = st + int(np.where(ok)[0][0])
            res['信号'] = '回踩'
            res['信号日'] = df.index[pos]

    last = float(c[-1])
    res['距线'] = round((last - line) / line * 100.0 * (1.0 if bull else -1.0), 2)
    # 还没碰到、但离得很近：距线为正且不超过 near_dist
    res['接近'] = bool(res['有效'] and not res['已触及']
                     and res['距线'] is not None and 0 <= res['距线'] <= float(near_dist))
    return res


def fvg_pick(df, cfg):
    """按 FVG_DIR 取状态：'bull' / 'bear' / 'both'（两个都算，挑信号更强那个）。"""
    d = str(cfg.get('FVG_DIR', 'bull') or 'bull').strip().lower()
    kw = dict(ratio=cfg['FVG_RATIO'], ref=cfg['FVG_REF'],
              min_pct=cfg['FVG_MIN_PCT'], lookback=cfg['FVG_LOOKBACK'],
              confirm=cfg['FVG_CONFIRM'], within=cfg['SCREEN_WITHIN'],
              min_body=cfg.get('FVG_MIN_BODY', 0.0),
              min_range=cfg.get('FVG_MIN_RANGE', 0.0),
              max_fill=cfg.get('FVG_MAX_FILL', 100.0),
              near_dist=cfg.get('FVG_NEAR_DIST', 2.0))
    if d in ('bear', 's', 'short', '空', '空头'):
        return fvg_state(df, direction='bear', **kw)
    if d in ('bull', 'b', 'long', '多', '多头'):
        return fvg_state(df, direction='bull', **kw)
    a = fvg_state(df, direction='bull', **kw)
    b = fvg_state(df, direction='bear', **kw)
    return _fvg_better(a, b)


def _fvg_rank(st):
    """状态好坏排序：有信号 > 快碰到 > 有效缺口 > 已回补 > 没缺口。"""
    if st.get('信号'):
        return 4
    if st.get('接近'):
        return 3
    if st.get('有效'):
        return 2
    if st.get('有缺口'):
        return 1
    return 0


def _fvg_better(a, b):
    ka, kb = _fvg_rank(a), _fvg_rank(b)
    if kb != ka:
        return b if kb > ka else a
    da = a.get('距线')
    db = b.get('距线')
    if da is None:
        return b
    if db is None:
        return a
    return b if abs(db) < abs(da) else a


# ============================================================
# 画图
# ============================================================
def _tick_labels(idx, max_ticks=12):
    """优先按「月份变化」放刻度，月份太多时抽稀；短区间按天等距。"""
    n = len(idx)
    if n == 0:
        return [], []
    span = (idx[-1] - idx[0]).days

    if span <= 150:
        step = max(1, int(np.ceil(n / float(min(max_ticks, 8)))))
        pos = list(range(0, n, step))
        if pos[-1] != n - 1:
            pos.append(n - 1)
        return pos, [idx[i].strftime('%m-%d') for i in pos]

    keys = [(d.year, d.month) for d in idx]
    mpos = [0] + [i for i in range(1, n) if keys[i] != keys[i - 1]]
    if len(mpos) > max_ticks:
        step = int(np.ceil(len(mpos) / float(max_ticks)))
        mpos = mpos[::step]
    years = set(d.year for d in idx)
    fmt = '%Y-%m' if len(years) > 1 else '%m月'
    return mpos, [idx[i].strftime(fmt) for i in mpos]


def _xpos(idx, ts):
    """日期 -> 显示区间的横坐标位置。区间外的夹到两端。"""
    n = len(idx)
    if n == 0:
        return 0
    if ts is None or pd.isna(ts):
        return n - 1
    try:
        p = int(idx.searchsorted(pd.Timestamp(ts)))
    except Exception:
        return n - 1
    return max(0, min(p, n - 1))


def _pxf(v):
    """价格显示：大数给少点小数，小数给多位，别一会儿 9.9 一会儿 9.8876。"""
    try:
        a = abs(float(v))
    except Exception:
        return str(v)
    if a >= 1000:
        return '%.1f' % float(v)
    if a >= 100:
        return '%.2f' % float(v)
    return '%.3f' % float(v)


def _draw_fvg(ax, d, zones, cfg):
    """在 K 线图上画缺口方框 + 斐波那契回踩线。

    每个缺口右侧留白里挂一个两行标签：缺口多大（价格差 + 百分比）、回踩线具体多少钱。
    只画最近 FVG_SHOW 个。返回画了几个。
    """
    if zones is None or len(zones) == 0:
        return 0
    ratio = float(cfg['FVG_RATIO'])
    z = zones
    if not bool(cfg.get('FVG_BEAR', True)):
        z = z[z['方向'] == '多头']
    if len(z) == 0:
        return 0
    n = len(d)
    tail = int(cfg['FVG_SHOW'])
    if tail > 0:
        z = z.tail(tail)
    idx = d.index
    # 纵向尺度：文字错位要靠它，别硬编码
    try:
        vspan = float(np.nanmax(d['high'].values) - np.nanmin(d['low'].values))
    except Exception:
        vspan = 0.0
    unit = vspan * 0.045 if vspan > 0 else 1.0
    taken = []
    drawn = 0
    for _, row in z.iterrows():
        bull = (row['方向'] == '多头')
        x0 = _xpos(idx, row['形成日'])
        x1 = n - 1
        if bool(row['已回补']) and pd.notna(row['回补日']):
            x1 = _xpos(idx, row['回补日'])
        if x1 < x0:
            x1 = x0
        lo, hi = float(row['下沿']), float(row['上沿'])
        size = hi - lo
        edge = C_UP if bull else C_DN
        ax.add_patch(Rectangle((x0 - 0.4, lo), (x1 - x0) + 0.8, size,
                               facecolor=edge, alpha=0.09, edgecolor=edge,
                               linewidth=0.6, zorder=1))
        # 上下沿细虚线，一眼看出缺口从哪儿到哪儿
        for yy in (hi, lo):
            ax.plot([x0 - 0.4, x1 + 0.4], [yy, yy], color=edge, ls=':', lw=0.55,
                    alpha=0.5, zorder=1)
        line = (hi - ratio * size) if bull else (lo + ratio * size)
        ax.plot([x0, x1], [line, line], color='#b8860b', ls='--', lw=1.05,
                alpha=0.95, zorder=4)
        # 标签挂在缺口框左侧：缺口多大（差价 + 百分比）+ 回踩线具体价位
        ty = line
        for t in taken:
            if abs(ty - t) < unit * 2.0:
                ty = t + unit * 2.2
        taken.append(ty)
        tip = 'FVG↑' if bull else 'FVG↓'
        # 缺口靠近右端时文字往左挂，否则会被画布裁掉
        if x0 > n - 14:
            tx, ha = x0 - 0.6, 'right'
        else:
            tx, ha = x0 + 0.6, 'left'
        ax.text(tx, ty,
                '%s %s (%+.1f%%)\n%.1f%%  %s' % (tip, _pxf(size), float(row['高度%']),
                                                 ratio * 100.0, _pxf(line)),
                fontsize=6.4, color='#7d6208', va='center', ha=ha,
                zorder=6, clip_on=True,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.78))
        drawn += 1
    return drawn


def _footer_text(cfg, last_close, last_rsi, fvg_st, atr_st=None):
    """页脚：收盘 / RSI / 分区 / ATR 档位 / FVG 缺口区间与回踩线。两个引擎共用。"""
    foot = '收盘 %.3f      RSI(%.0f) %.2f      %s' % (
        float(last_close), cfg['RSI_N'], float(last_rsi),
        rsi_zone(float(last_rsi), cfg['OB'], cfg['OS']))
    if atr_st is not None and atr_st.get('有数据'):
        foot += '      ATR %s (%.2f%%) %s档' % (
            _pxf(atr_st['ATR']), float(atr_st['ATR%']), atr_st['ATR档'])
        sg = atr_st.get('ATR信号') or ''
        if sg == '异动':
            foot += ' · 异动 %.2fx' % float(atr_st.get('异动倍数') or 0.0)
        elif sg == '收缩':
            foot += ' · 收缩 %d%%分位' % int(round(float(atr_st.get('ATR分位') or 0.0)))
    if fvg_st is not None and fvg_st['回踩线'] is not None:
        tag = '' if fvg_st['方向'] == '多头' else '空'
        gd = fvg_st['缺口日'].strftime('%m-%d') if fvg_st['缺口日'] is not None else ''
        if fvg_st.get('下沿') is not None:
            foot += '      FVG%s %s %s~%s (差%s/%+.1f%%) %.1f%%线 %s' % (
                tag, gd, _pxf(fvg_st['下沿']), _pxf(fvg_st['上沿']),
                _pxf(fvg_st['高度']), float(fvg_st['高度%']),
                float(cfg['FVG_RATIO']) * 100.0, _pxf(fvg_st['回踩线']))
        else:
            foot += '      FVG%s %s %.1f%%线 %s' % (
                tag, gd, float(cfg['FVG_RATIO']) * 100.0, _pxf(fvg_st['回踩线']))
    return foot


# ============================================================
# plotly 交互图（PLOT_ENGINE='plotly'）
# ============================================================
def _norm_engine(v):
    """把各种写法归一成 'mpl' / 'plotly'；没装 plotly 就自动退回 mpl。"""
    e = str(v or 'mpl').strip().lower()
    if e in ('plotly', 'ply', 'interactive', '交互', 'i'):
        if not HAS_PLOTLY:
            print('[提示] 这个环境没装 plotly，先按 matplotlib 出图（pip install plotly 之后再试）')
            return 'mpl'
        return 'plotly'
    return 'mpl'


def _rgba(hex_color, alpha):
    h = str(hex_color).lstrip('#')
    return 'rgba(%d,%d,%d,%.3f)' % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), float(alpha))


def _ply_yref(row):
    return 'y' if int(row) == 1 else 'y%d' % int(row)


def _ply_xdom(row):
    return 'x domain' if int(row) == 1 else 'x%d domain' % int(row)


def _ply_hline(fig, y, row, color='#888888', dash='dash', width=1.0):
    """在指定子图画一条横贯的线（老版本 plotly 没有 add_hline，这里手搓）。"""
    fig.add_shape(type='line', xref=_ply_xdom(row), yref=_ply_yref(row),
                  x0=0, x1=1, y0=float(y), y1=float(y),
                  line=dict(color=color, width=width, dash=dash), layer='below')


def _ply_band(fig, y0, y1, row, color, alpha=0.07):
    fig.add_shape(type='rect', xref=_ply_xdom(row), yref=_ply_yref(row),
                  x0=0, x1=1, y0=float(y0), y1=float(y1),
                  fillcolor=_rgba(color, alpha), line=dict(width=0), layer='below')


def _ply_rangebreaks(freq):
    """日线把周末的空档抹掉，不然图上每七天就有两条空白。"""
    if _norm_freq(freq) != 'D':
        return None
    return [dict(bounds=['sat', 'mon'])]


def _plot_kline_plotly(df, code, cfg, name=''):
    """plotly 版 K 线 + 成交量 + RSI，可缩放、悬停看每根的开高低收。"""
    full = df
    rsi_full = calc_rsi(full['close'], cfg['RSI_N'])
    atr_full = calc_atr(full, int(cfg.get('ATR_N', 14)))
    tr_full = calc_tr(full)
    ma_list = [int(m) for m in (cfg['MA_LIST'] or ())]
    ma_full = dict((m, full['close'].rolling(m, min_periods=m).mean()) for m in ma_list)

    fvg_on = bool(cfg.get('FVG_ON', True))
    fvg_zones = None
    fvg_st = None
    if fvg_on:
        fvg_zones = find_fvg(full, direction='both', min_pct=cfg['FVG_MIN_PCT'],
                             lookback=cfg['FVG_LOOKBACK'],
                             min_body=cfg.get('FVG_MIN_BODY', 0.0),
                             min_range=cfg.get('FVG_MIN_RANGE', 0.0),
                             max_fill=cfg.get('FVG_MAX_FILL', 100.0))
        fvg_st = fvg_pick(full, cfg)

    start = _to_ts(cfg['PLOT_START'])
    end = _to_ts(cfg['PLOT_END'])
    d = full
    if start is not None:
        d = d[d.index >= start]
    if end is not None:
        d = d[d.index <= end]
    if start is None and end is None:
        nb = int(cfg.get('PLOT_BARS') or 0)
        d = d.iloc[-nb:] if nb > 0 else d.iloc[-int(cfg['BARS']):]
    if len(d) < 2:
        print('[跳过] %s 显示区间里 K 线不足 2 根' % code)
        return None

    idx = d.index
    rsi = rsi_full.reindex(idx)
    o = d['open'].values.astype(float)
    h = d['high'].values.astype(float)
    lo_ = d['low'].values.astype(float)
    c = d['close'].values.astype(float)
    v = d['volume'].values.astype(float) if 'volume' in d.columns else np.zeros(len(d))
    up = c >= o
    colors = np.where(up, C_UP, C_DN)

    show_vol = bool(cfg['SHOW_VOLUME']) and np.nansum(v) > 0
    show_atr = bool(cfg.get('SHOW_ATR', True)) and bool(cfg.get('ATR_ON', True))
    heights = [3.0, 1.05, 1.85] if show_vol else [3.0, 1.85]
    if show_atr:
        heights = heights + [float(cfg.get('ATR_PANE_H', 1.0))]
    nrow = len(heights)
    row_rsi = nrow - 1 if show_atr else nrow
    row_atr = nrow if show_atr else None
    fig = make_subplots(rows=nrow, cols=1, shared_xaxes=True,
                        vertical_spacing=0.035, row_heights=heights)

    # --- K 线 ---
    fig.add_trace(go.Candlestick(
        x=idx, open=o, high=h, low=lo_, close=c, name='K线',
        increasing_line_color=C_UP, increasing_fillcolor=C_UP,
        decreasing_line_color=C_DN, decreasing_fillcolor=C_DN,
        line=dict(width=1), whiskerwidth=0.3), row=1, col=1)

    # --- 均线 ---
    for i, m in enumerate(ma_list):
        fig.add_trace(go.Scatter(
            x=idx, y=ma_full[m].reindex(idx).values, mode='lines', name='MA%d' % m,
            line=dict(color=C_MA[i % len(C_MA)], width=1.2),
            hovertemplate='MA%d %%.2f<extra></extra>' % m), row=1, col=1)

    # --- FVG 方框 / 上下沿 / 回踩线 / 标签 ---
    fvg_label = None
    if fvg_zones is not None and len(fvg_zones):
        z = fvg_zones
        if not bool(cfg.get('FVG_BEAR', True)):
            z = z[z['方向'] == '多头']
        tail = int(cfg['FVG_SHOW'])
        if tail > 0 and len(z):
            z = z.tail(tail)
        ratio = float(cfg['FVG_RATIO'])
        try:
            vspan = float(np.nanmax(h) - np.nanmin(lo_))
        except Exception:
            vspan = 0.0
        unit = vspan * 0.045 if vspan > 0 else 1.0
        taken = []
        for _, row in z.iterrows():
            bull = (row['方向'] == '多头')
            x0 = idx[_xpos(idx, row['形成日'])]
            x1 = idx[-1]
            if bool(row['已回补']) and pd.notna(row['回补日']):
                x1 = idx[_xpos(idx, row['回补日'])]
            if x1 < x0:
                x1 = x0
            zlo, zhi = float(row['下沿']), float(row['上沿'])
            size = zhi - zlo
            edge = C_UP if bull else C_DN
            fig.add_shape(type='rect', xref='x', yref='y', x0=x0, x1=x1, y0=zlo, y1=zhi,
                          fillcolor=_rgba(edge, 0.09),
                          line=dict(color=_rgba(edge, 0.55), width=0.6), layer='below')
            for yy in (zhi, zlo):
                fig.add_shape(type='line', xref='x', yref='y', x0=x0, x1=x1, y0=yy, y1=yy,
                              line=dict(color=_rgba(edge, 0.5), width=0.6, dash='dot'),
                              layer='below')
            line = (zhi - ratio * size) if bull else (zlo + ratio * size)
            fig.add_shape(type='line', xref='x', yref='y', x0=x0, x1=x1, y0=line, y1=line,
                          line=dict(color='#b8860b', width=1.1, dash='dash'))
            ty = line
            for t in taken:
                if abs(ty - t) < unit * 2.0:
                    ty = t + unit * 2.2
            taken.append(ty)
            fig.add_annotation(
                x=x0, y=ty, xref='x', yref='y', showarrow=False, xanchor='left',
                text='<b>%s %s (%+.1f%%)</b><br>%.1f%%  %s' % (
                    'FVG↑' if bull else 'FVG↓', _pxf(size), float(row['高度%']),
                    ratio * 100.0, _pxf(line)),
                font=dict(size=9.5, color='#7d6208'), align='left',
                bgcolor='rgba(255,255,255,0.82)', bordercolor=_rgba('#b8860b', 0.45),
                borderwidth=0.8, borderpad=2)
            fvg_label = 'FVG %.1f%%' % (ratio * 100.0)

    # --- 成交量 ---
    if show_vol:
        fig.add_trace(go.Bar(x=idx, y=v, marker_color=colors, opacity=0.62,
                             name='成交量', showlegend=False,
                             hovertemplate='量 %{y:,.0f}<extra></extra>'), row=2, col=1)
        fig.update_yaxes(title_text='成交量', row=2, col=1)

    # --- RSI ---
    ob, os_ = float(cfg['OB']), float(cfg['OS'])
    _ply_band(fig, ob, 100, row_rsi, C_UP, 0.07)
    _ply_band(fig, 0, os_, row_rsi, C_DN, 0.07)
    _ply_hline(fig, 50, row_rsi, '#9aa0a6', 'dot', 0.9)
    _ply_hline(fig, ob, row_rsi, C_UP, 'dash', 1.1)
    _ply_hline(fig, os_, row_rsi, C_DN, 'dash', 1.1)
    fig.add_trace(go.Scatter(x=idx, y=rsi.values, mode='lines', name='RSI(%d)' % cfg['RSI_N'],
                             line=dict(color=C_RSI, width=1.3),
                             hovertemplate='RSI %{y:.2f}<extra></extra>'), row=row_rsi, col=1)

    if bool(cfg['MARK_EVENTS']):
        prev = rsi.shift(1)
        cu = ((rsi > os_) & (prev <= os_) & rsi.notnull() & prev.notnull()).values
        cd = ((rsi < ob) & (prev >= ob) & rsi.notnull() & prev.notnull()).values
        if cu.any():
            fig.add_trace(go.Scatter(x=idx[cu], y=rsi.values[cu], mode='markers', name='上穿超卖',
                                     marker=dict(symbol='triangle-up', size=11, color=C_DN)),
                          row=row_rsi, col=1)
        if cd.any():
            fig.add_trace(go.Scatter(x=idx[cd], y=rsi.values[cd], mode='markers', name='下穿超买',
                                     marker=dict(symbol='triangle-down', size=11, color=C_UP)),
                          row=row_rsi, col=1)

    # --- ATR% / TR% 副图（TR 细线窜、ATR 粗线追，两条互相穿过）---
    if show_atr:
        a_lo, a_hi = atr_levels(cfg)
        cser = pd.Series(c, index=idx).replace(0.0, np.nan)
        pct = (atr_full.reindex(idx) / cser) * 100.0
        trp = (tr_full.reindex(idx) / cser) * 100.0
        top = float(np.nanmax(np.append(np.concatenate([pct.values, trp.values]),
                                        [a_hi * 1.5 if np.isfinite(a_hi) else 1.0])))
        _ply_band(fig, 0, a_lo, row_atr, C_DN, 0.07)
        _ply_band(fig, a_hi, top, row_atr, C_UP, 0.07)
        _ply_hline(fig, a_lo, row_atr, C_DN, 'dash', 1.0)
        _ply_hline(fig, a_hi, row_atr, C_UP, 'dash', 1.0)
        fig.add_trace(go.Scatter(
            x=idx, y=trp.values, mode='lines', name='TR%',
            line=dict(color='#8d9aa5', width=0.7), opacity=0.75,
            hovertemplate='TR% %{y:.2f}<extra></extra>'), row=row_atr, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=pct.values, mode='lines', name='ATR%%(%d)' % cfg['ATR_N'],
            line=dict(color='#b8860b', width=1.5),
            hovertemplate='ATR%% %{y:.2f}<extra></extra>'), row=row_atr, col=1)
        # 异动 / 收缩 标记：用「全历史」算完再截到显示窗口，
        # 和 atr_state / 表格 / 页脚的口径保持一致（否则分位只看窗口里那几十根，圈会多出来）
        _cfull = pd.Series(np.asarray(df['close'], dtype=float), index=df.index).replace(0.0, np.nan)
        _pfull = (atr_full / _cfull) * 100.0
        _rfull = tr_full / atr_full.replace(0.0, np.nan)
        _spk_full = _rfull >= float(cfg.get('ATR_SPIKE', 1.8))
        _k = max(int(cfg['ATR_PCTL_N']), 20)
        _qfull = _pfull.rolling(_k, min_periods=max(20, _k // 4)).quantile(
            float(cfg.get('ATR_SQ_PCTL', 20.0)) / 100.0)
        _qn = max(int(cfg.get('ATR_QUIET_N', 10)), 2)
        _qfull2 = _rfull.lt(1.0).rolling(_qn, min_periods=max(2, _qn // 2)).mean()
        _sq_full = (_pfull <= _qfull) & (_qfull2 >= float(cfg.get('ATR_QUIET_RATIO', 0.7)))
        _spk = _spk_full.reindex(idx).fillna(False).values.astype(bool)
        _sq = _sq_full.reindex(idx).fillna(False).values.astype(bool)
        if _spk.any():
            fig.add_trace(go.Scatter(
                x=idx[_spk], y=trp[_spk], mode='markers',
                name='异动 %gx' % float(cfg.get('ATR_SPIKE', 1.8)),
                marker=dict(symbol='triangle-up', size=9, color='#ff7f0e'),
                hovertemplate='异动 TR%% %{y:.2f}<extra></extra>'), row=row_atr, col=1)
        if _sq.any():
            fig.add_trace(go.Scatter(
                x=idx[_sq], y=pct[_sq], mode='markers', name='收缩',
                marker=dict(symbol='circle-open', size=7, color='#00a0a0',
                            line=dict(width=1.2)),
                hovertemplate='收缩 ATR%% %{y:.2f}<extra></extra>'), row=row_atr, col=1)
        fig.update_yaxes(range=[0, top if top > 0 else 1.0], title_text='ATR% / TR%',
                         row=row_atr, col=1)

    # --- 价格轴 / 版本无关的布局 ---
    vals = [np.nanmin(lo_), np.nanmax(h)]
    for m in ma_list:
        arr = ma_full[m].reindex(idx).values.astype(float)
        if np.isfinite(arr).any():
            vals += [np.nanmin(arr), np.nanmax(arr)]
    y0, y1 = np.nanmin(vals), np.nanmax(vals)
    pad = (y1 - y0) * 0.06 if y1 > y0 else 1.0
    fig.update_yaxes(range=[y0 - pad, y1 + pad], row=1, col=1)
    fig.update_yaxes(range=[0, 100], dtick=10, title_text='RSI(%d)' % cfg['RSI_N'],
                     row=row_rsi, col=1)
    fig.update_yaxes(gridcolor='#dddddd', gridwidth=0.8, zeroline=False)

    label = name or code
    if not name:
        _nice = ETF_NAME.get(str(code).upper())
        if _nice:
            label = '%s  %s' % (label, _nice)
    title_main = '%s  %s  K线 %d 根' % (label, FREQ_LABEL[_norm_freq(cfg['FREQ'])], len(d))
    last_rsi = rsi.dropna()
    foot = ''
    if len(last_rsi) > 0:
        foot = _footer_text(cfg, c[-1], float(last_rsi.iloc[-1]), fvg_st,
                            atr_state(df, cfg) if cfg.get('ATR_ON', True) else None)
    fig.update_layout(
        title=dict(text='<b>%s</b><br><span style="font-size:11px;color:#555">%s</span>'
                        % (title_main, foot), x=0.01, xanchor='left', font=dict(size=13)),
        height=int(cfg.get('PLOTLY_HEIGHT', 840)),
        hovermode='x unified',
        dragmode='pan',
        template='plotly_white',
        margin=dict(l=64, r=20, t=76, b=36),
        legend=dict(orientation='h', yanchor='bottom', y=1.005, x=0, font=dict(size=10)),
        bargap=0.15)
    fig.update_xaxes(showgrid=False)
    try:
        rb = _ply_rangebreaks(cfg['FREQ'])
        if rb:
            fig.update_xaxes(rangebreaks=rb)
    except Exception:  # 老版本 plotly 不认 rangebreaks 就跳过
        pass
    try:
        fig.update_xaxes(rangeslider_visible=False)
    except Exception:
        pass
    return fig


def plot_kline_rsi(df, code, cfg=None, name=''):
    """上 K 线 + 均线，下 RSI。df 必须是目标周期（已合成）的 OHLCV。

    cfg['PLOT_ENGINE'] = 'mpl'（默认，静态图）/ 'plotly'（可缩放悬停）
    """
    cfg = _merge_cfg(cfg)
    if _norm_engine(cfg.get('PLOT_ENGINE')) == 'plotly':
        return _plot_kline_plotly(df, code, cfg, name=name)
    _setup_style(cfg['CJK_FONT'])
    if df is None or len(df) == 0:
        print('[跳过] %s 没有数据' % code)
        return None

    full = df
    rsi_full = calc_rsi(full['close'], cfg['RSI_N'])
    atr_full = calc_atr(full, int(cfg.get('ATR_N', 14)))
    tr_full = calc_tr(full)
    ma_list = [int(m) for m in (cfg['MA_LIST'] or ())]
    ma_full = dict((m, full['close'].rolling(m, min_periods=m).mean()) for m in ma_list)

    # FVG 用完整历史算（显示区间裁掉的那些缺口也可能正在起作用）
    fvg_on = bool(cfg.get('FVG_ON', True))
    fvg_zones = None
    fvg_st = None
    if fvg_on:
        fvg_zones = find_fvg(full, direction='both', min_pct=cfg['FVG_MIN_PCT'],
                             lookback=cfg['FVG_LOOKBACK'],
                             min_body=cfg.get('FVG_MIN_BODY', 0.0),
                             min_range=cfg.get('FVG_MIN_RANGE', 0.0),
                             max_fill=cfg.get('FVG_MAX_FILL', 100.0))
        fvg_st = fvg_pick(full, cfg)

    # --- 显示区间 ---
    start = _to_ts(cfg['PLOT_START'])
    end = _to_ts(cfg['PLOT_END'])
    d = full
    if start is not None:
        d = d[d.index >= start]
    if end is not None:
        d = d[d.index <= end]
    if start is None and end is None:
        nb = int(cfg.get('PLOT_BARS') or 0)
        d = d.iloc[-nb:] if nb > 0 else d.iloc[-int(cfg['BARS']):]
    if len(d) < 2:
        print('[跳过] %s 显示区间里 K 线不足 2 根' % code)
        return None

    idx = d.index
    rsi = rsi_full.reindex(idx)
    mas = dict((m, ma_full[m].reindex(idx)) for m in ma_list)

    x = np.arange(len(d))
    o = d['open'].values.astype(float)
    h = d['high'].values.astype(float)
    lo_ = d['low'].values.astype(float)
    c = d['close'].values.astype(float)
    v = d['volume'].values.astype(float) if 'volume' in d.columns else np.zeros(len(d))
    up = c >= o
    colors = np.where(up, C_UP, C_DN)

    show_vol = bool(cfg['SHOW_VOLUME']) and np.nansum(v) > 0
    show_atr = bool(cfg.get('SHOW_ATR', True)) and bool(cfg.get('ATR_ON', True))
    heights = [3.0, 1.05, 1.85] if show_vol else [3.0, 1.85]
    if show_atr:
        heights = heights + [float(cfg.get('ATR_PANE_H', 1.0))]
    fig = plt.figure(figsize=cfg['FIGSIZE'], facecolor='white')
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.07,
                          left=0.055, right=0.985, top=0.905, bottom=0.075)
    ax_p = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_p) if show_vol else None
    ax_r = fig.add_subplot(gs[-2] if show_atr else gs[-1], sharex=ax_p)
    ax_a = fig.add_subplot(gs[-1], sharex=ax_p) if show_atr else None

    # --- K 线 ---
    ax_p.vlines(x, lo_, h, color=colors, linewidth=0.9, alpha=0.85, zorder=2)
    span0 = np.nanmax(h) - np.nanmin(lo_)
    min_body = span0 * 0.0008 if span0 > 0 else 0.0
    body = np.maximum(np.abs(c - o), min_body)
    ax_p.bar(x, body, bottom=np.minimum(o, c), width=0.62, color=colors,
             edgecolor=colors, linewidth=0.45, zorder=3)
    flat = h == lo_
    if flat.any():
        ax_p.scatter(x[flat], c[flat], s=7, color=colors[flat], zorder=4)

    # --- FVG 方框 + 回踩线 ---
    fvg_label = None
    if fvg_on and fvg_zones is not None and len(fvg_zones):
        if _draw_fvg(ax_p, d, fvg_zones, cfg):
            fvg_label = 'FVG %.1f%%' % (float(cfg['FVG_RATIO']) * 100.0)

    # --- 均线 ---
    for i, m in enumerate(ma_list):
        ax_p.plot(x, mas[m].values, color=C_MA[i % len(C_MA)],
                  lw=1.15, label='MA%d' % m, alpha=0.95, zorder=5)

    # --- 价格轴范围（把均线也算进去，不然均线会被切） ---
    vals = [np.nanmin(lo_), np.nanmax(h)]
    for m in ma_list:
        arr = mas[m].values.astype(float)
        if np.isfinite(arr).any():
            vals += [np.nanmin(arr), np.nanmax(arr)]
    y0, y1 = np.nanmin(vals), np.nanmax(vals)
    pad = (y1 - y0) * 0.06 if y1 > y0 else 1.0
    ax_p.set_ylim(y0 - pad, y1 + pad)
    ax_p.grid(True, ls=':', lw=0.7, color='#bbbbbb', alpha=0.55)
    ax_p.tick_params(labelbottom=False, labelsize=9)
    handles, labels_ = ax_p.get_legend_handles_labels()
    if fvg_label is not None:
        handles.append(Rectangle((0, 0), 1, 1, facecolor='#b8860b', edgecolor='#b8860b',
                                 alpha=0.28, linewidth=0.6))
        labels_.append(fvg_label)
    if handles:
        ax_p.legend(handles, labels_, loc='upper left', ncol=min(4, len(handles)),
                    fontsize=9, frameon=False, handlelength=1.6, columnspacing=1.2)

    # 最新一根的信息放到标题下面，避免和图例打架
    last_rsi = rsi.dropna()

    # --- 成交量 ---
    if show_vol:
        ax_v.bar(x, v, width=0.62, color=colors, alpha=0.6, linewidth=0)
        ax_v.set_ylim(0, np.nanmax(v) * 1.12)
        ax_v.set_ylabel('成交量', fontsize=9)
        ax_v.grid(True, axis='y', ls=':', lw=0.7, color='#bbbbbb', alpha=0.55)
        ax_v.tick_params(labelbottom=False, labelsize=8)
        ax_v.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax_v.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_vol_fmt))

    # --- RSI ---
    ob, os_ = float(cfg['OB']), float(cfg['OS'])
    ax_r.plot(x, rsi.values, color=C_RSI, lw=1.25, zorder=4, label='RSI(%d)' % cfg['RSI_N'])
    ax_r.axhline(ob, color=C_UP, ls='--', lw=1.0, alpha=0.9)
    ax_r.axhline(os_, color=C_DN, ls='--', lw=1.0, alpha=0.9)
    ax_r.axhline(50, color='#9aa0a6', ls=':', lw=0.85, alpha=0.8)
    ax_r.fill_between(x, ob, 100, color=C_UP, alpha=0.07, linewidth=0)
    ax_r.fill_between(x, 0, os_, color=C_DN, alpha=0.07, linewidth=0)
    ax_r.set_ylim(0, 100)
    ticks = sorted(set([0, int(round(os_)), 50, int(round(ob)), 100]))
    ax_r.set_yticks(ticks)
    ax_r.set_ylabel('RSI(%d)' % cfg['RSI_N'], fontsize=9.5)
    ax_r.grid(True, ls=':', lw=0.7, color='#bbbbbb', alpha=0.55)
    ax_r.tick_params(labelsize=9)

    # 反转标记：上穿超卖（绿三角）/ 下穿超买（红倒三角）
    if bool(cfg['MARK_EVENTS']):
        prev = rsi.shift(1)
        cu = ((rsi > os_) & (prev <= os_)).values
        cd = ((rsi < ob) & (prev >= ob)).values
        if cu.any():
            ax_r.scatter(x[cu], rsi.values[cu], marker='^', s=32, color=C_DN,
                         zorder=5, label='上穿超卖')
        if cd.any():
            ax_r.scatter(x[cd], rsi.values[cd], marker='v', s=32, color=C_UP,
                         zorder=5, label='下穿超买')
        if cu.any() or cd.any():
            ax_r.legend(loc='upper right', ncol=2, fontsize=8.5, frameon=False,
                        handletextpad=0.3, columnspacing=1.0)

    # --- ATR% / TR% 副图（TR 抖、ATR 慢，两条会互相穿过、缠在一起）---
    if show_atr:
        a_lo, a_hi = atr_levels(cfg)
        cser = pd.Series(c, index=idx).replace(0.0, np.nan)
        pct = ((atr_full.reindex(idx) / cser) * 100.0).values.astype(float)
        trp = ((tr_full.reindex(idx) / cser) * 100.0).values.astype(float)
        both = np.concatenate([pct, trp])
        top = float(np.nanmax(both)) if np.isfinite(np.nanmax(both)) else 1.0
        top = max(top, a_hi * 1.5)
        ax_a.axhspan(0, a_lo, color=C_DN, alpha=0.07, linewidth=0)
        ax_a.axhspan(a_hi, top, color=C_UP, alpha=0.07, linewidth=0)
        ax_a.axhline(a_lo, color=C_DN, ls='--', lw=0.95, alpha=0.85)
        ax_a.axhline(a_hi, color=C_UP, ls='--', lw=0.95, alpha=0.85)
        ax_a.plot(x, trp, color='#8d9aa5', lw=0.7, alpha=0.75, zorder=2, label='TR%')
        ax_a.plot(x, pct, color='#b8860b', lw=1.5, zorder=3,
                  label='ATR%%(%d)' % cfg['ATR_N'])
        # 异动（TR 冲出 ATR）：橙色三角；收缩（波幅压死）：青色空心圈
        # 注意：这两个判定都用「全历史」算完再截到显示窗口，
        # 否则分位数只看窗口里那几十根，图上圈的和表格里写的不一致。
        _cfull = pd.Series(np.asarray(df['close'], dtype=float), index=df.index).replace(0.0, np.nan)
        _pfull = (atr_full / _cfull) * 100.0
        _rfull = tr_full / atr_full.replace(0.0, np.nan)
        _spk_full = _rfull >= float(cfg.get('ATR_SPIKE', 1.8))
        _k = max(int(cfg['ATR_PCTL_N']), 20)
        _qfull = _pfull.rolling(_k, min_periods=max(20, _k // 4)).quantile(
            float(cfg.get('ATR_SQ_PCTL', 20.0)) / 100.0)
        _qn = max(int(cfg.get('ATR_QUIET_N', 10)), 2)
        _qfull2 = _rfull.lt(1.0).rolling(_qn, min_periods=max(2, _qn // 2)).mean()
        _sq_full = (_pfull <= _qfull) & (_qfull2 >= float(cfg.get('ATR_QUIET_RATIO', 0.7)))
        _spk = _spk_full.reindex(idx).fillna(False).values.astype(bool)
        _sq = _sq_full.reindex(idx).fillna(False).values.astype(bool)
        if _spk.any():
            ax_a.scatter(x[_spk], trp[_spk], marker='^', s=26, color='#ff7f0e',
                         zorder=5, label='异动 %gx' % float(cfg.get('ATR_SPIKE', 1.8)))
        if _sq.any():
            ax_a.scatter(x[_sq], pct[_sq], marker='o', s=15, facecolors='none',
                         edgecolors='#00a0a0', linewidths=0.9, zorder=4, label='收缩')
        ax_a.set_ylim(0, top * 1.02)
        ax_a.set_ylabel('ATR% / TR%', fontsize=9)
        ax_a.grid(True, ls=':', lw=0.7, color='#bbbbbb', alpha=0.55)
        ax_a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax_a.legend(loc='upper left', ncol=2, fontsize=8, frameon=False,
                    handletextpad=0.4, columnspacing=1.0)
        if a_lo <= float(np.nanmean(pct)) <= a_hi:
            ax_a.text(0.995, 0.90, '正常档', transform=ax_a.transAxes,
                      ha='right', va='top', fontsize=8, color='#666666')

    pos, labels = _tick_labels(idx)
    ax_last = ax_a if show_atr else ax_r
    ax_last.set_xticks(pos)
    ax_last.set_xticklabels(labels, fontsize=9)
    ax_last.set_xlim(-1, len(d))
    if show_atr:
        ax_r.tick_params(labelbottom=False, labelsize=9)

    label = name or code
    if not name:
        _nice = ETF_NAME.get(str(code).upper())
        if _nice:
            label = '%s  %s' % (label, _nice)
    title = '%s  %s  K线 %d 根' % (label, FREQ_LABEL[_norm_freq(cfg['FREQ'])], len(d))
    fig.suptitle(title, fontsize=12.5, y=0.975)
    if len(last_rsi) > 0:
        lv = float(last_rsi.iloc[-1])
        fig.text(0.5, 0.938, _footer_text(cfg, c[-1], lv, fvg_st,
                                         atr_state(df, cfg) if cfg.get('ATR_ON', True) else None),
                 ha='center', va='top', fontsize=10, color='#444444')
    return fig


def _vol_fmt(v, pos=None):
    v = float(v)
    if abs(v) >= 1e8:
        return '%.1f亿' % (v / 1e8)
    if abs(v) >= 1e4:
        return '%.0f万' % (v / 1e4)
    return '%.0f' % v


# ============================================================
# 对外入口
# ============================================================
def _is_plotly_fig(fig):
    return HAS_PLOTLY and isinstance(fig, go.Figure)


def _show_fig(fig):
    """plotly 图直接 show()，matplotlib 图走 plt.show()。"""
    if fig is None:
        return None
    if _is_plotly_fig(fig):
        try:
            fig.show()
            return None
        except Exception as e:
            print('  [提示] plotly 显示失败（%s），改走 HTML 兜底' % e)
        try:  # notebook 里 show() 不灵时，直接把 html 塞进输出
            from IPython.display import display, HTML
            display(HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)))
            return None
        except Exception:
            return None
    return plt.show()


def plot_one(code, **kw):
    """单只画一张：plot_one('000300.XSHG', FREQ='W', BARS=120, PLOT_ENGINE='plotly')"""
    cfg = _merge_cfg(kw)
    if _norm_engine(cfg.get('PLOT_ENGINE')) != 'plotly':
        _setup_style(cfg['CJK_FONT'])
    df = load_bars(code, cfg)
    fig = plot_kline_rsi(df, code, cfg)
    if fig is not None:
        _maybe_save(fig, code, cfg)
        _show_fig(fig)
    return fig


def _maybe_save(fig, code, cfg):
    """mpl 存 png；plotly 存 html（PLOTLY_FORMAT='png' 且有 kaleido 时存 png）。"""
    if not cfg['SAVE_DIR'] or fig is None:
        return None
    try:
        if not os.path.isdir(cfg['SAVE_DIR']):
            os.makedirs(cfg['SAVE_DIR'])
        stem = 'RSI_%s_%s' % (str(code).replace('.', '_'), cfg['FREQ'])
        if _is_plotly_fig(fig):
            fmt = str(cfg.get('PLOTLY_FORMAT') or 'png').strip().lower()
            if fmt == 'png':
                path = os.path.join(cfg['SAVE_DIR'], stem + '.png')
                try:
                    fig.write_image(path, height=int(cfg.get('PLOTLY_HEIGHT', 840)))
                    print('  存图: %s' % path)
                    return path
                except Exception as e:
                    print('  [提示] plotly 导 png 失败（缺 kaleido？pip install kaleido）：%s' % e)
                    fmt = 'html'
            if fmt == 'html':
                path = os.path.join(cfg['SAVE_DIR'], stem + '.html')
                fig.write_html(path, include_plotlyjs='cdn')
                print('  存网页: %s' % path)
                return path
            return None
        path = os.path.join(cfg['SAVE_DIR'], stem + '.png')
        fig.savefig(path, dpi=cfg['DPI'], bbox_inches='tight', facecolor='white')
        print('  存图: %s' % path)
        return path
    except Exception as e:
        print('  [提示] 存图失败: %s' % e)
        return None


def run_all(**kw):
    """按配置把池子里的标的逐只画出来，返回 fig 列表。"""
    cfg = _merge_cfg(kw)
    if _norm_engine(cfg.get('PLOT_ENGINE')) != 'plotly':
        _setup_style(cfg['CJK_FONT'])
    if _has_jq():
        print('[数据源] 聚宽 get_price')
    else:
        print('[数据源] akshare')
    codes = resolve_pool(cfg['POOL'], date=cfg['PLOT_END'])
    if not codes:
        print('池子是空的，先设 POOL')
        return []
    n_show = len(codes) if int(cfg['MAX_CHARTS']) <= 0 else min(len(codes), int(cfg['MAX_CHARTS']))
    figs = []
    for code in codes[:n_show]:
        try:
            df = load_bars(code, cfg)
            if df is None or len(df) < 10:
                print('[跳过] %s 数据不足（%d 根）' % (code, 0 if df is None else len(df)))
                continue
            fig = plot_kline_rsi(df, code, cfg)
            if fig is None:
                continue
            figs.append(fig)
            _maybe_save(fig, code, cfg)
            _show_fig(fig)
        except Exception as e:
            print('[失败] %s: %s' % (code, e))
    if len(codes) > n_show:
        print('池子里还有 %d 只没画（MAX_CHARTS=%d，调大就行）' % (len(codes) - n_show, n_show))
    return figs


def rsi_snapshot(pool=None, **kw):
    """池子里每只的「最新 RSI + 状态」一张表，用来快速扫。"""
    cfg = _merge_cfg(kw)
    codes = resolve_pool(pool if pool is not None else cfg['POOL'], date=cfg['PLOT_END'])
    rows = []
    for code in codes:
        try:
            df = load_bars(code, cfg)
            if df is None or len(df) < int(cfg['RSI_N']) + 2:
                continue
            r = calc_rsi(df['close'], cfg['RSI_N']).dropna()
            if len(r) == 0:
                continue
            lv = float(r.iloc[-1])
            rows.append({
                '代码': code,
                '名称': ETF_NAME.get(str(code).upper(), ''),
                '日期': df.index[-1].strftime('%Y-%m-%d'),
                '收盘': round(float(df['close'].iloc[-1]), 3),
                'RSI': round(lv, 2),
                '状态': rsi_zone(lv, cfg['OB'], cfg['OS']),
            })
        except Exception as e:
            print('[失败] %s: %s' % (code, e))
    if not rows:
        return pd.DataFrame(columns=['代码', '名称', '日期', '收盘', 'RSI', '状态'])
    return pd.DataFrame(rows)


# ============================================================
# 筛选：一堆标的里挑出「超买 / 超卖 / 刚上穿超卖 / 刚下穿超买」
# ============================================================
SCREEN_MODES = {
    'hit': '超买超卖 + 刚交叉（默认）',
    'buy': '刚上穿超卖（买入信号）',
    'sell': '刚下穿超买（卖出信号）',
    'cross': '只要有交叉，不管方向',
    'overbought': '当前超买',
    'oversold': '当前超卖',
    'all': '不过滤，全列出来',
}

_SCREEN_COLS = ['代码', '名称', '日期', '收盘', 'RSI', '状态', '信号', '信号日期']


def _screen_pri(has_buy, zone, has_sell):
    """排序优先级：买入信号 > 超卖 > 卖出信号 > 超买 > 其它。"""
    if has_buy:
        return 0
    if zone == '超卖':
        return 1
    if has_sell:
        return 2
    if zone == '超买':
        return 3
    return 9


def screen_rsi(pool=None, signal='hit', freq='D', rsi_n=14, ob=None, os_=None,
               within=0, bars=400, limit=0, sort=True, verbose=True, **kw):
    """在池子里按 RSI 状态 + 交叉信号筛标的。

    参数
    ----
    pool   : 池子，写法跟 POOL 一样（代码列表 / 'HS300' / 'ETF' / '行业' …）
    signal : 'hit'（默认）超买、超卖、刚上穿超卖、刚下穿超买 任一命中
             'buy' 刚上穿超卖 / 'sell' 刚下穿超买 / 'cross' 只要有交叉
             'overbought' 当前超买 / 'oversold' 当前超卖 / 'all' 全列出来
    freq   : 用哪个周期筛（'D' 最灵敏，'W' 信号更少更稳）
    within : 最近几根内发生交叉算「刚刚」，0 = 只看最后一根
    bars   : 每只取多少根（要够 RSI 预热，默认 400）
    limit  : 只扫前 N 只（调试用，0 = 全扫）

    返回 DataFrame：代码 / 名称 / 日期 / 收盘 / RSI / 状态 / 信号 / 信号日期
    """
    mode = str(signal).strip().lower()
    if mode not in SCREEN_MODES:
        raise ValueError('signal 只能是 %s，收到 %r' % ('/'.join(sorted(SCREEN_MODES)), signal))

    cfg = _merge_cfg(kw)
    cfg['FREQ'] = _norm_freq(freq)
    cfg['BARS'] = int(bars)
    cfg['RSI_N'] = int(rsi_n)
    cfg['OB'] = float(cfg['OB'] if ob is None else ob)
    cfg['OS'] = float(cfg['OS'] if os_ is None else os_)
    cfg['PLOT_START'] = None

    codes = resolve_pool(pool if pool is not None else cfg['POOL'], date=cfg['PLOT_END'])
    if limit and int(limit) > 0:
        codes = codes[:int(limit)]
    if not codes:
        print('[筛选] 池子是空的，先设 pool')
        return pd.DataFrame(columns=_SCREEN_COLS)

    ob_f, os_f = float(cfg['OB']), float(cfg['OS'])
    win = max(int(within), 0) + 1
    need = int(rsi_n) + 3

    rows = []
    total = len(codes)
    for i, code in enumerate(codes, 1):
        try:
            df = load_bars(code, cfg)
            if df is None or len(df) < need:
                continue
            rsi = calc_rsi(df['close'], cfg['RSI_N'])
            if rsi.notna().sum() < 2 or pd.isna(rsi.iloc[-1]):
                continue

            last = float(rsi.iloc[-1])
            zone = rsi_zone(last, ob_f, os_f)

            up = (rsi > os_f) & (rsi.shift(1) <= os_f)
            dn = (rsi < ob_f) & (rsi.shift(1) >= ob_f)
            sub_up = up.iloc[-win:].fillna(False).astype(bool)
            sub_dn = dn.iloc[-win:].fillna(False).astype(bool)
            has_buy = bool(sub_up.any())
            has_sell = bool(sub_dn.any())

            marks = []
            if has_buy:
                marks.append(('买入', sub_up[sub_up].index[-1]))
            if has_sell:
                marks.append(('卖出', sub_dn[sub_dn].index[-1]))
            marks.sort(key=lambda t: t[1])
            sig_name = '、'.join(m[0] for m in marks)
            sig_date = marks[-1][1].strftime('%Y-%m-%d') if marks else ''

            rows.append({
                '代码': code,
                '名称': ETF_NAME.get(str(code).upper(), ''),
                '日期': df.index[-1].strftime('%Y-%m-%d'),
                '收盘': round(float(df['close'].iloc[-1]), 3),
                'RSI': round(last, 2),
                '状态': zone,
                '信号': sig_name,
                '信号日期': sig_date,
                '_pri': _screen_pri(has_buy, zone, has_sell),
            })
        except Exception as e:
            print('[失败] %s: %s' % (code, e))
        if verbose and total > 40 and i % 50 == 0:
            print('  扫描 %d/%d …' % (i, total))

    if not rows:
        if verbose:
            print('[筛选] %d 只里没有可用的数据' % total)
        return pd.DataFrame(columns=_SCREEN_COLS)

    full = pd.DataFrame(rows)
    n_buy = int((full['信号'].str.contains('买入')).sum())
    n_sell = int((full['信号'].str.contains('卖出')).sum())
    n_os = int((full['状态'] == '超卖').sum())
    n_ob = int((full['状态'] == '超买').sum())

    if mode == 'buy':
        hit = full[full['信号'].str.contains('买入')]
    elif mode == 'sell':
        hit = full[full['信号'].str.contains('卖出')]
    elif mode == 'cross':
        hit = full[full['信号'] != '']
    elif mode == 'overbought':
        hit = full[full['状态'] == '超买']
    elif mode == 'oversold':
        hit = full[full['状态'] == '超卖']
    elif mode == 'hit':
        hit = full[(full['信号'] != '') | (full['状态'].isin(['超买', '超卖']))]
    else:
        hit = full

    hit = hit.copy()
    if sort and len(hit):
        hit = hit.sort_values(['_pri', 'RSI'], ascending=[True, True])
    hit = hit.drop(columns=['_pri']).reset_index(drop=True)

    if verbose:
        print('[筛选] %d 只 → 命中 %d 只 | 买入 %d · 卖出 %d · 超卖区 %d · 超买区 %d'
              % (total, len(hit), n_buy, n_sell, n_os, n_ob))
        if len(hit):
            print('       口径：%s 线 · RSI(%d) · 阈值 %.0f/%.0f · 最近 %d 根内算「刚刚」'
                  % (cfg['FREQ'], cfg['RSI_N'], ob_f, os_f, win))
    return hit[_SCREEN_COLS]


def plot_hits(hits, **kw):
    """把筛选结果直接画出来（参数同 run_all）。"""
    if hits is None or len(hits) == 0:
        print('没有命中的标的，先跑 screen_rsi')
        return []
    codes = [str(c) for c in list(hits['代码'])]
    cfg = dict(kw)
    cfg['POOL'] = codes
    cfg.setdefault('MAX_CHARTS', 0)
    return run_all(**cfg)


# ============================================================
# FVG 筛选 / 综合筛选（档位 + 指标插件）
# ============================================================
FVG_MODES = {
    'hit': '有有效缺口就列（回踩到位的排前面）',
    'touch': '刚回踩到斐波那契线',
    'fresh': '缺口还没回补、也还没回踩（等回踩）',
    'filled': '缺口已经被回补',
    'all': '有缺口就列（含已回补）',
}

_FVG_SCREEN_COLS = ['代码', '名称', '日期', '收盘', '方向', '信号', '状态', '缺口日',
                    '下沿', '上沿', '高度%', '实体%', '回踩线', '距线%', '回补%', '缺口天数']


def _fmt_date(v):
    if v is None or pd.isna(v):
        return ''
    try:
        return pd.Timestamp(v).strftime('%Y-%m-%d')
    except Exception:
        return str(v)


def screen_fvg(pool=None, mode='hit', ratio=0.618, ref='gap', min_pct=0.3,
               lookback=120, confirm='close', within=0, direction=None,
               min_body=None, min_range=None, max_fill=None,
               freq='D', bars=400, limit=0, sort=True, verbose=True, **kw):
    """在池子里按 FVG 状态 + 斐波那契回踩筛标的。

    参数
    ----
    pool      : 池子，写法和 POOL 一样
    mode      : 'hit' 有效缺口 / 'touch' 刚回踩到位 / 'fresh' 等回踩 /
                'filled' 已回补 / 'all' 有缺口就列
    ratio     : 回踩线深度，默认 0.618
    ref       : 'gap' 按缺口高度 / 'leg' 按脉冲腿
    min_pct   : 缺口高度占价格的最小百分比（跨度门槛）
    lookback  : 只认最近多少根内形成的缺口
    confirm   : 'touch' / 'close' / 'bull'
    within    : 最近几根内回踩算「刚刚」
    direction : 'bull' 上涨缺口（回调买）/ 'bear' 下跌缺口（反弹）/ 'both' 两个都看
                （不给就取 FVG_DIR）
    min_body  : 造缺口那根 K 线的实体%下限（滤小阳小阴）
    min_range : 造缺口那根 K 线的振幅%下限
    max_fill  : 缺口最多被吃掉多少 % 还算有效

    返回：代码 / 名称 / 日期 / 收盘 / 方向 / 信号 / 状态 / 缺口日 /
          下沿 / 上沿 / 高度% / 实体% / 回踩线 / 距线% / 回补% / 缺口天数
    """
    mode = str(mode).strip().lower()
    if mode not in FVG_MODES:
        raise ValueError('mode 只能是 %s，收到 %r' % ('/'.join(sorted(FVG_MODES)), mode))

    cfg = _merge_cfg(kw)
    cfg['FREQ'] = _norm_freq(freq)
    cfg['BARS'] = int(bars)
    cfg['PLOT_START'] = None
    cfg['FVG_RATIO'] = float(ratio)
    cfg['FVG_REF'] = ref
    cfg['FVG_MIN_PCT'] = float(min_pct)
    cfg['FVG_LOOKBACK'] = int(lookback)
    cfg['FVG_CONFIRM'] = confirm
    cfg['SCREEN_WITHIN'] = int(within)
    cfg['FVG_DIR'] = str(direction if direction is not None else cfg['FVG_DIR'])
    if min_body is not None:
        cfg['FVG_MIN_BODY'] = float(min_body)
    if min_range is not None:
        cfg['FVG_MIN_RANGE'] = float(min_range)
    if max_fill is not None:
        cfg['FVG_MAX_FILL'] = float(max_fill)

    codes = resolve_pool(pool if pool is not None else cfg['POOL'], date=cfg['PLOT_END'])
    if limit and int(limit) > 0:
        codes = codes[:int(limit)]
    if not codes:
        print('[FVG筛选] 池子是空的，先设 pool')
        return pd.DataFrame(columns=_FVG_SCREEN_COLS)

    rows = []
    total = len(codes)
    for i, code in enumerate(codes, 1):
        try:
            df = load_bars(code, cfg)
            if df is None or len(df) < 10:
                continue
            st = fvg_pick(df, cfg)
            if not st['有缺口']:
                continue
            if st['信号']:
                status = '回踩到位'
            elif st['有效']:
                status = '等回踩'
            else:
                status = '已回补'
            rows.append({
                '代码': code,
                '名称': ETF_NAME.get(str(code).upper(), ''),
                '日期': df.index[-1].strftime('%Y-%m-%d'),
                '收盘': round(float(df['close'].iloc[-1]), 3),
                '方向': st['方向'],
                '信号': st['信号'],
                '状态': status,
                '缺口日': _fmt_date(st['缺口日']),
                '下沿': round(float(st['下沿']), 3),
                '上沿': round(float(st['上沿']), 3),
                '高度%': st['高度%'],
                '实体%': st['实体%'],
                '回踩线': round(float(st['回踩线']), 3),
                '距线%': st['距线'],
                '回补%': st['回补%'],
                '缺口天数': st['天数'],
                '_pri': 0 if status == '回踩到位' else (1 if status == '等回踩' else 2),
            })
        except Exception as e:
            print('[失败] %s: %s' % (code, e))
        if verbose and total > 40 and i % 50 == 0:
            print('  扫描 %d/%d …' % (i, total))

    if not rows:
        if verbose:
            print('[FVG筛选] %d 只里没找到缺口' % total)
        return pd.DataFrame(columns=_FVG_SCREEN_COLS)

    full = pd.DataFrame(rows)
    n_touch = int((full['状态'] == '回踩到位').sum())
    n_wait = int((full['状态'] == '等回踩').sum())
    n_fill = int((full['状态'] == '已回补').sum())

    if mode == 'touch':
        hit = full[full['状态'] == '回踩到位']
    elif mode == 'fresh':
        hit = full[full['状态'] == '等回踩']
    elif mode == 'filled':
        hit = full[full['状态'] == '已回补']
    elif mode == 'hit':
        hit = full[full['状态'] != '已回补']
    else:
        hit = full

    hit = hit.copy()
    if sort and len(hit):
        hit = hit.sort_values(['_pri', '距线%'], ascending=[True, True])
    hit = hit.drop(columns=['_pri']).reset_index(drop=True)

    if verbose:
        print('[FVG筛选] %d 只 → 有缺口 %d｜回踩到位 %d · 等回踩 %d · 已回补 %d'
              % (total, len(full), n_touch, n_wait, n_fill))
        if len(hit):
            print('       口径：%s 线 · %s · 缺口高度≥%.2f%% · 实体≥%.1f%% · 回踩 %.1f%%（%s）· 确认 %s'
                  % (cfg['FREQ'], cfg['FVG_DIR'], float(cfg['FVG_MIN_PCT']),
                     float(cfg['FVG_MIN_BODY']), float(ratio) * 100.0, ref, confirm))
    return hit[_FVG_SCREEN_COLS]


# ---------- 指标插件：想加 MACD / KDJ / 量价，照下面这个模板写就行 ----------
SCREEN_INDICATORS = {}
SCREEN_ORDER = []
_IND_CN = {'fvg': 'FVG回踩', 'rsi': 'RSI超卖', 'trend': '趋势多头', 'vol': '缩量',
           'atr': 'ATR波动', 'atr_sig': 'ATR信号'}
# 这几项不管档位用不用，都算出来填进表格（否则 RSI / 趋势 / 量比 那几列是空的）
_DISPLAY_INDS = ['fvg', 'rsi', 'trend', 'vol', 'atr', 'atr_sig']


def register_indicator(name, desc):
    """把一个指标注册进综合筛选。函数签名 fn(df, cfg) -> dict(hit=, score=, near=, info=)。
    注册时顺手把中文名写进 _IND_CN：档位条件文字和表格「命中」列会显示中文，而不是变量名。"""
    def _deco(fn):
        SCREEN_INDICATORS[name] = {'fn': fn, 'desc': desc}
        if name not in SCREEN_ORDER:
            SCREEN_ORDER.append(name)
        _IND_CN.setdefault(name, desc)
        return fn
    return _deco


@register_indicator('fvg', '公允价值缺口 + 斐波那契回踩')
def _ind_fvg(df, cfg):
    st = fvg_pick(df, cfg)
    info = {'FVG': '', '缺口高度%': '', '实体%': '', '回踩线': '', '距线%': '',
            '缺口天数': '', '回补%': '', 'FVG方向': '', '信号日': ''}
    if st['有缺口']:
        info.update({'缺口高度%': st['高度%'], '实体%': st['实体%'],
                     '回踩线': round(float(st['回踩线']), 3) if st['回踩线'] is not None else '',
                     '距线%': st['距线'], '缺口天数': st['天数'],
                     '回补%': st['回补%'], 'FVG方向': st['方向']})
    if st['信号']:
        score, tag, near = 2.0, '回踩', False
        info['信号日'] = _fmt_date(st['信号日'])
    elif st['接近']:
        score, tag, near = 1.5, '就快碰到', True
    elif st['有效']:
        score, tag, near = 1.0, '待回踩', False
    else:
        score, tag, near = 0.0, ('已回补' if st['有缺口'] else '无'), False
    info['FVG'] = tag
    return {'hit': bool(st['信号']), 'score': score, 'near': near, 'info': info}


@register_indicator('rsi', 'RSI 超卖 / 刚上穿超卖 / 差一点到超卖')
def _ind_rsi(df, cfg):
    r = calc_rsi(df['close'], cfg['RSI_N']).dropna()
    if len(r) < 2:
        return {'hit': False, 'score': 0.0, 'near': False, 'info': {}}
    last = float(r.iloc[-1])
    zone = rsi_zone(last, cfg['OB'], cfg['OS'])
    win = max(int(cfg['SCREEN_WITHIN']), 0) + 1
    os_ = float(cfg['OS'])
    ob_ = float(cfg['OB'])
    up = (r > os_) & (r.shift(1) <= os_)
    cross = bool(up.iloc[-win:].fillna(False).astype(bool).any())
    hit = bool(cross or zone == '超卖')
    near = False
    if hit:
        score = 2.0 if cross else 1.5
        tag = '上穿' if cross else '超卖'
    elif os_ < last <= os_ + float(cfg['RSI_NEAR']):
        score, near, tag = 1.0, True, '差一点'
    elif last >= ob_:
        score, tag = 0.0, '超买'
    else:
        score, tag = 0.0, '中性'
    return {'hit': hit, 'score': score, 'near': near,
            'info': {'RSI': round(last, 2), 'RSI状态': tag}}


@register_indicator('trend', '均线多头（收盘 > 短均 > 长均）/ 差一点金叉')
def _ind_trend(df, cfg):
    c = df['close'].astype(float)
    f, s = int(cfg['TREND_FAST']), int(cfg['TREND_SLOW'])
    if len(c) < s + 1:
        return {'hit': False, 'score': 0.0, 'near': False, 'info': {}}
    mf = float(c.rolling(f, min_periods=f).mean().iloc[-1])
    ms = float(c.rolling(s, min_periods=s).mean().iloc[-1])
    last = float(c.iloc[-1])
    if not (np.isfinite(mf) and np.isfinite(ms) and ms > 0):
        return {'hit': False, 'score': 0.0, 'near': False, 'info': {}}
    hit = bool(last > mf > ms)
    near = False
    if hit:
        score, tag = 1.0, '多头'
    else:
        tol = float(cfg['TREND_NEAR'])
        # 差一点金叉：价格已站上短均、短均离长均只有一点点
        gap = abs(mf - ms) / ms
        # 或者多头排列已成、价格刚回踩到短均附近
        near = bool((last > mf and mf <= ms and gap <= tol)
                    or (mf > ms and abs(last - mf) / mf <= tol))
        score = 0.5 if near else 0.0
        if near:
            tag = '差一点'
        elif last < mf < ms:
            tag = '空头'
        else:
            tag = '纠缠'
    return {'hit': hit, 'score': score, 'near': near,
            'info': {'趋势': tag, 'MA%d' % f: round(mf, 3), 'MA%d' % s: round(ms, 3)}}


@register_indicator('vol', '回踩缩量（量 < N 日均量 × 系数）/ 量比不高')
def _ind_vol(df, cfg):
    if 'volume' not in df.columns or len(df) < int(cfg['VOL_N']) + 2:
        return {'hit': False, 'score': 0.0, 'near': False, 'info': {}}
    v = df['volume'].astype(float)
    n = int(cfg['VOL_N'])
    base = float(v.iloc[-n - 1:-1].mean())
    last = float(v.iloc[-1])
    if not np.isfinite(base) or base <= 0:
        return {'hit': False, 'score': 0.0, 'near': False, 'info': {}}
    ratio = last / base
    hit = bool(ratio < float(cfg['VOL_SHRINK']))
    near = bool((not hit) and ratio < float(cfg['VOL_NEAR']))
    score = 1.0 if hit else (0.5 if near else 0.0)
    return {'hit': hit, 'score': score, 'near': near,
            'info': {'量比': round(ratio, 2)}}


@register_indicator('atr', 'ATR 波动档位（日 / 周 / 月各一套阈值）')
def _ind_atr(df, cfg):
    """ATR 档位。命中方向看 ATR_SIDE：'low' 低波动算命中 / 'high' 高波动 / 'both' 两头都算。"""
    blank = dict((c, '') for c in _ATR_COLS)
    st = atr_state(df, cfg)
    if not st['有数据']:
        return {'hit': False, 'score': 0.0, 'near': False, 'info': blank}
    p, band = float(st['ATR%']), st['ATR档']
    lo, hi = float(st['低阈值']), float(st['高阈值'])
    side = str(cfg.get('ATR_SIDE') or 'low').strip().lower()
    if side in ('high', 'h', '高', '高波动'):
        want = ('高',)
    elif side in ('both', 'all', '双', '两头'):
        want = ('低', '高')
    else:
        want = ('低',)
    hit = band in want
    near = False
    if not hit:   # 没到档，但已经贴到边界附近
        k = max(float(cfg['ATR_NEAR']), 1.0)
        if '低' in want and lo <= p < lo * k:
            near = True
        if '高' in want and hi / k < p <= hi:
            near = True
    score = 1.0 if hit else (0.5 if near else 0.0)
    info = {'ATR': round(st['ATR'], 4), 'ATR%': round(p, 2), 'ATR档': band,
            'ATR扩张': round(st['ATR扩张'], 2) if np.isfinite(st['ATR扩张']) else '',
            'ATR分位': round(st['ATR分位'], 1) if np.isfinite(st['ATR分位']) else ''}
    return {'hit': hit, 'score': score, 'near': near, 'info': info}


@register_indicator('atr_sig', 'ATR 信号（TR 突刺=跳空/爆发，波幅压死=蓄势）')
def _ind_atr_sig(df, cfg):
    """ATR 的两个事件信号，和「档位」是两回事：

    异动 = TR > ATR × ATR_SPIKE（默认 1.8）。一根就够了，多数是跳空 / 消息 / 放量突破。
    收缩 = ATR% 落在低分位 + 最近几根 TR 普遍低于 ATR。波动压死，蓄势。

    认哪种看 ATR_SIG：'spike' 只认异动 / 'squeeze' 只认收缩 / 'both' 两种都算。
    """
    st = atr_signal(df, cfg)
    mode = str(cfg.get('ATR_SIG') or 'both').strip().lower()
    want_spike = mode in ('spike', 'sp', '异动', 'both', 'all', 'both_', '两种', '都')
    want_sq = mode in ('squeeze', 'sq', '收缩', 'both', 'all', 'both_', '两种', '都')
    sig = st['信号']
    hit = (sig == '异动' and want_spike) or (sig == '收缩' and want_sq)
    near = False
    if not hit:
        r = st['异动倍数']
        k = float(cfg.get('ATR_SPIKE', 1.8))
        if want_spike and np.isfinite(r) and k * 0.85 <= r < k:
            near = True          # 快到异动线了
        pc = st['ATR分位']
        sq = float(cfg.get('ATR_SQ_PCTL', 20.0))
        if want_sq and np.isfinite(pc) and sq < pc <= sq + 10.0 \
                and np.isfinite(st['安静比']) and st['安静比'] >= float(cfg.get('ATR_QUIET_RATIO', 0.7)) * 0.9:
            near = True          # 快压到收缩区了
    score = 1.5 if hit else (0.75 if near else 0.0)
    info = {'ATR信号': sig,
            '异动倍数': round(st['异动倍数'], 2) if np.isfinite(st['异动倍数']) else '',
            '异动日': _fmt_date(st['异动日']),
            'ATR分位': round(st['ATR分位'], 1) if np.isfinite(st['ATR分位']) else '',
            '安静比': round(st['安静比'] * 100.0, 0) if np.isfinite(st['安静比']) else ''}
    return {'hit': hit, 'score': score, 'near': near, 'info': info}


# ============================================================
# 档位 / 特色：每个档位就是一套「用哪几个指标 + 必须命中谁 + 差一点也算」
# 想加自己的档位，照着写一条塞进 SETUPS 就行（key 可以是数字也可以是名字）
# ============================================================
def _big_gap(info, cfg):
    return (info.get('缺口高度%') or 0) >= float(cfg['FVG_BIG_PCT'])


def _fresh_gap(info, cfg):
    d = info.get('缺口天数')
    return d is not None and d != '' and float(d) <= float(cfg['FVG_FRESH_DAYS'])


def _is_pullback(info, cfg):
    return info.get('FVG') == '回踩'


def _atr_is_low(info, cfg):
    return info.get('ATR档') == '低'


def _atr_is_high(info, cfg):
    return info.get('ATR档') == '高'


def _atr_sig_is_spike(info, cfg):
    return info.get('ATR信号') == '异动'


def _atr_sig_is_squeeze(info, cfg):
    return info.get('ATR信号') == '收缩'


SETUPS = {
    0: {'name': '0 档 · 自定义',
        'desc': '自己给 use / need / near / min_score',
        'use': ['fvg', 'rsi', 'trend', 'vol'], 'need': [], 'min_score': 0},
    1: {'name': '1 档 · 只看 FVG 回踩形态',
        'desc': '有有效缺口、没被补掉，回踩到位的排前面',
        'use': ['fvg'], 'need': ['fvg'], 'min_score': 0},
    2: {'name': '2 档 · FVG + RSI 共振',
        'desc': '缺口还在 + RSI 超卖（或刚上穿）',
        'use': ['fvg', 'rsi'], 'need': ['fvg', 'rsi'], 'min_score': 0},
    3: {'name': '3 档 · FVG + RSI + 趋势',
        'desc': '再加均线多头，回踩买在上升途中',
        'use': ['fvg', 'rsi', 'trend'], 'need': ['fvg', 'rsi', 'trend'], 'min_score': 0},
    4: {'name': '4 档 · 全条件',
        'desc': 'FVG + RSI + 趋势 + 缩量，最严',
        'use': ['fvg', 'rsi', 'trend', 'vol'],
        'need': ['fvg', 'rsi', 'trend', 'vol'], 'min_score': 0},
    5: {'name': '5 档 · FVG + RSI + 低波动',
        'desc': '共振之外再要求 ATR 落在低波动档（安静时候的回踩更干净）',
        'use': ['fvg', 'rsi', 'atr'], 'need': ['fvg', 'rsi', 'atr'], 'min_score': 0},
    'big_gap': {'name': '特色 · 大缺口',
                'desc': '缺口高度 ≥ FVG_BIG_PCT（默认 2%），而且没被吃掉',
                'use': ['fvg'], 'need': ['fvg'], 'where': _big_gap, 'min_score': 0},
    'atr_quiet': {'name': '特色 · 波动安静',
                  'desc': 'FVG 有效，ATR% 掉进低波动档（缩量窄幅，蓄势的样子）',
                  'use': ['fvg', 'atr'], 'need': ['fvg'], 'where': _atr_is_low, 'min_score': 0},
    'atr_loud': {'name': '特色 · 波动放大',
                 'desc': '趋势多头，ATR% 冲进高波动档（真突破而不是磨上来）',
                 'use': ['fvg', 'trend', 'atr'], 'need': ['fvg', 'trend'],
                 'where': _atr_is_high, 'min_score': 0},
    'fresh': {'name': '特色 · 新缺口',
              'desc': '缺口刚形成不超过 FVG_FRESH_DAYS 根（默认 3 根）',
              'use': ['fvg'], 'need': ['fvg'], 'where': _fresh_gap, 'min_score': 0},
    'pullback_vol': {'name': '特色 · 回踩到位 + 缩量',
                     'desc': '已经碰到 61.8% 线，而且量缩下来了',
                     'use': ['fvg', 'vol'], 'need': ['fvg', 'vol'],
                     'where': _is_pullback, 'min_score': 0},
    'near_rsi': {'name': '特色 · RSI 差一点',
                 'desc': 'FVG 有效，RSI 就快进超卖（还差一点没到）',
                 'use': ['fvg', 'rsi'], 'need': ['fvg'], 'near': ['rsi'], 'min_score': 0},
    'near_trend': {'name': '特色 · 趋势差一点',
                   'desc': 'FVG 有效，均线就快多头排列（还差一点）',
                   'use': ['fvg', 'trend'], 'need': ['fvg'], 'near': ['trend'],
                   'min_score': 0},
    'weak': {'name': '特色 · 差一点（哪个都行）',
             'desc': 'FVG 有效，RSI 接近超卖 或 趋势接近多头，沾一个就算',
             'use': ['fvg', 'rsi', 'trend'], 'need': ['fvg'],
             'any_near': ['rsi', 'trend'], 'min_score': 0},
    'gap_burst': {'name': '特色 · 缺口异动',
                  'desc': 'FVG 有效，且最近几根 TR 猛地冲出 ATR（那根多半就是造缺口的那根）',
                  'use': ['fvg', 'atr_sig'], 'need': ['fvg', 'atr_sig'],
                  'where': _atr_sig_is_spike, 'min_score': 0},
    'squeeze_wait': {'name': '特色 · 波幅压死蓄势',
                     'desc': 'ATR% 掉进低分位、TR 连续贴着 ATR —— 安静到不正常，等方向',
                     'use': ['fvg', 'atr_sig', 'atr'], 'need': ['fvg', 'atr_sig'],
                     'where': _atr_sig_is_squeeze, 'min_score': 0},
}

# 兼容老写法：TIERS 就是 SETUPS 的别名
TIERS = SETUPS

_STOCK_COLS = ['代码', '名称', '日期', '收盘', '分数', '命中', 'FVG', 'FVG方向',
               '缺口高度%', '实体%', '缺口天数', '回补%', '回踩线', '距线%',
               'RSI', 'RSI状态', '趋势', '量比',
               'ATR', 'ATR%', 'ATR档', 'ATR扩张', 'ATR分位', 'ATR信号', '特征', '信号日']


def _top_n(v, n):
    """「只留前几个」的写法：10 → 前 10 只；0.1 → 前 10%；<=0 → 全留。"""
    try:
        v = float(v or 0)
    except Exception:
        return 0
    if v <= 0 or n <= 0:
        return 0
    if v < 1:
        return max(1, int(round(n * v)))
    return min(int(v), n)


def _dist_abs(v):
    """距线% 取绝对值（空的排最后）。"""
    try:
        x = float(v)
    except Exception:
        return 9e9
    return abs(x) if np.isfinite(x) else 9e9


def _setup_cfg(tier):
    if isinstance(tier, dict):
        _check_setup(tier, '自定义')
        return dict(tier)
    key = tier
    if isinstance(key, str):
        s = key.strip()
        key = int(s) if s.isdigit() else s.lower()
    preset = SETUPS.get(key)
    if preset is None:
        raise ValueError('tier 只能是 %s 或一个 dict，收到 %r'
                         % (sorted([str(k) for k in SETUPS.keys()]), tier))
    _check_setup(preset, key)
    return dict(preset)


def _check_setup(st, key):
    """档位要是「一个条件都没写」，会放过整池 —— 这种静默失效必须在跑之前喊出来。"""
    if key == 0:
        return
    use = st.get('use') or []
    if not use:
        return
    cond = (st.get('need') or []) + (st.get('near') or []) + (st.get('any_near') or [])
    if not cond and not st.get('where') and not float(st.get('min_score') or 0):
        raise ValueError(
            '档位 %r（%s）一个过滤条件都没设：need / near / any_near / min_score / where '
            '全是空的，这样会把整个池子原样放回来。给它加个 need（比如 need=["fvg"]）再跑。'
            % (key, st.get('name', '')))
    for nm in cond:
        if nm not in SCREEN_INDICATORS:
            raise ValueError('档位 %r 用到了没注册的指标 %r，可选 %s'
                             % (key, nm, ' / '.join(SCREEN_ORDER)))


# 老名字，别的地方可能还在用
_tier_cfg = _setup_cfg


def setup_table():
    """把现有档位列成一张表，方便挑。"""
    rows = []
    for k in sorted([x for x in SETUPS.keys() if isinstance(x, int)]) + \
            [x for x in SETUPS.keys() if not isinstance(x, int)]:
        st = SETUPS[k]
        cond = []
        for nm in (st.get('need') or []):
            cond.append('%s 必中' % _IND_CN.get(nm, nm))
        for nm in (st.get('near') or []):
            cond.append('%s 差一点' % _IND_CN.get(nm, nm))
        if st.get('any_near'):
            cond.append(' / '.join(_IND_CN.get(n, n) for n in st['any_near']) + ' 差一点（沾一个）')
        if st.get('where'):
            cond.append('附加条件')
        rows.append({'档位': k, '名字': st['name'], '说明': st['desc'],
                     '条件': '｜'.join(cond) or '不打分只列'},
                     )
    return pd.DataFrame(rows, columns=['档位', '名字', '说明', '条件'])


def _pass_setup(row, st, cfg):
    """一只票能不能过这个档位。"""
    hit, near, score = row['hit'], row['near'], row['score']
    if not all(hit.get(nm) for nm in (st.get('need') or [])):
        return False
    if not all(near.get(nm) for nm in (st.get('near') or [])):
        return False
    an = st.get('any_near') or []
    if an and not any(near.get(nm) for nm in an):
        return False
    total = 0.0
    for nm in (st.get('use') or []):
        total += float(score.get(nm, 0.0))
    if total < float(st.get('min_score', 0) or 0):
        return False
    where = st.get('where')
    if where is not None:
        try:
            if not where(row['info'], cfg):
                return False
        except Exception:
            return False
    return True


def _scan_pool(pool, cfg, use, limit=0, verbose=True):
    """把池子扫一遍，每只算好指定的指标。返回 (rows, 总数)。

    row = {'代码','名称','日期','收盘','info':{...},'hit':{...},'near':{...},'score':{...}}
    """
    codes = resolve_pool(pool if pool is not None else cfg['POOL'], date=cfg['PLOT_END'])
    if limit and int(limit) > 0:
        codes = codes[:int(limit)]
    if not codes:
        return [], 0
    # 打分只认 use，但表格里的列（RSI / 趋势 / 量比）不管哪一档都填满，别让人看空格
    names = list(use)
    for nm in _DISPLAY_INDS:
        if nm not in names and nm in SCREEN_INDICATORS:
            names.append(nm)
    rows = []
    total = len(codes)
    for i, code in enumerate(codes, 1):
        try:
            df = load_bars(code, cfg)
            if df is None or len(df) < 30:
                continue
            info, hit, near, score = {}, {}, {}, {}
            for nm in names:
                try:
                    r = SCREEN_INDICATORS[nm]['fn'](df, cfg)
                except Exception:
                    r = {'hit': False, 'score': 0.0, 'near': False, 'info': {}}
                hit[nm] = bool(r.get('hit'))
                near[nm] = bool(r.get('near'))
                score[nm] = float(r.get('score', 0.0))
                info.update(r.get('info') or {})
            rows.append({'代码': code,
                         '名称': ETF_NAME.get(str(code).upper(), ''),
                         '日期': df.index[-1].strftime('%Y-%m-%d'),
                         '收盘': round(float(df['close'].iloc[-1]), 3),
                         'info': info, 'hit': hit, 'near': near, 'score': score})
        except Exception as e:
            print('[失败] %s: %s' % (code, e))
        if verbose and total > 40 and i % 50 == 0:
            print('  扫描 %d/%d …' % (i, total))
    return rows, total


def _out_row(row, st):
    hits = [nm for nm in (st.get('use') or []) if row['hit'].get(nm)]
    weak = [nm for nm in (st.get('use') or []) if row['near'].get(nm) and not row['hit'].get(nm)]
    score = 0.0
    for nm in (st.get('use') or []):
        score += float(row['score'].get(nm, 0.0))
    out = {'代码': row['代码'], '名称': row['名称'], '日期': row['日期'],
           '收盘': row['收盘'], '分数': round(score, 2),
           '命中': '+'.join(_IND_CN.get(n, n) for n in hits),
           '特征': '+'.join('%s差一点' % _IND_CN.get(n, n) for n in weak),
           '_d': row['info'].get('距线%')}
    for col in _STOCK_COLS[6:]:
        if col in ('命中', '特征'):
            continue
        v = row['info'].get(col, '')
        if isinstance(v, float) and not np.isfinite(v):
            v = ''
        out[col] = v
    return out


def setup_overview(pool=None, setups=None, within=None, freq='D', bars=400,
                   limit=0, verbose=True, **kw):
    """一次扫描，把每个档位各命中几只列成一张表 —— 先看哪档有货，再决定筛哪档。"""
    cfg = _merge_cfg(kw)
    cfg['FREQ'] = _norm_freq(freq)
    cfg['BARS'] = int(bars)
    cfg['PLOT_START'] = None
    cfg['SCREEN_WITHIN'] = int(within if within is not None else cfg['SCREEN_WITHIN'])

    # 0 档是「自定义、啥都不过滤」，放进来会显示成全池命中，没意义，默认不列
    keys = list(setups) if setups is not None else [k for k in SETUPS.keys() if k != 0]
    # 档位可以写数字 / 名字 / 直接给个 dict，dict 没法当 key，所以这里转成 (标签, 配置) 列表
    presets = []
    for k in keys:
        st = _setup_cfg(k)
        label = k if isinstance(k, (int, float, str)) else st.get('name', '?')
        presets.append((label, st))
    use = []
    for _lab, st in presets:
        for nm in (st.get('use') or []):
            if nm not in use:
                use.append(nm)
    for nm in use:
        if nm not in SCREEN_INDICATORS:
            raise ValueError('没有这个指标：%r，可选 %s' % (nm, ' / '.join(SCREEN_ORDER)))

    rows, total = _scan_pool(pool, cfg, use, limit=limit, verbose=verbose)
    if not total:
        print('[档位概览] 池子是空的，先设 pool')
        return pd.DataFrame(columns=['档位', '名字', '说明', '命中', '通过率'])

    scanned = len(rows)
    out = []
    for k, st in presets:
        n = 0
        for r in rows:
            if _pass_setup(r, st, cfg):
                n += 1
        pct = (100.0 * n / scanned) if scanned else 0.0
        out.append({'档位': k, '名字': st['name'], '说明': st['desc'],
                    '命中': n, '通过率': '%.1f%%' % pct})
    df = pd.DataFrame(out, columns=['档位', '名字', '说明', '命中', '通过率'])
    if verbose:
        print('[档位概览] 扫了 %d 只（池子 %d 只），各档命中数：' % (scanned, total))
        for _, r in df.iterrows():
            print('   %-14s %-32s %3d 只 / %d（%s）'
                  % (str(r['档位']), r['名字'], int(r['命中']), scanned, r['通过率']))
        # 单独看每个指标命中多少 —— 哪一档「满」，先看是哪个指标放宽了
        head = []
        for nm in use:
            h = sum(1 for r in rows if r['hit'].get(nm))
            ne = sum(1 for r in rows if r['near'].get(nm))
            head.append('%s %d/%d' % (_IND_CN.get(nm, nm), h, scanned))
            if ne:
                head[-1] += '（接近 %d）' % ne
        print('   [指标单独命中] ' + ' · '.join(head))
        # 某档把整池原样放回来，基本就是它的条件没生效 —— 直接喊出来
        for _, r in df.iterrows():
            k = r['档位']
            if int(r['命中']) == scanned and scanned > 0 and k != 0:
                st = presets[k]
                cond = (st.get('need') or []) + (st.get('near') or []) + \
                       (st.get('any_near') or [])
                if cond or st.get('where'):
                    print('   [!] 档位 %r 把 %d 只全放回来了（条件 %s）——'
                          '内核里可能还是旧函数：Restart & Run All 再跑一次'
                          % (k, scanned, cond or 'where'))
        if 0 not in keys:
            print('   （0 档是自定义档，啥都不过滤，不列在这）')
    return df


def screen_stocks(pool=None, tier=2, use=None, need=None, near=None, any_near=None,
                  min_score=None, within=None, freq='D', bars=400, limit=0,
                  sort=True, top=None, closest=None, verbose=True, **kw):
    """综合筛选：多个指标一起过，按档位出结果。

    参数
    ----
    pool      : 池子（同 POOL）
    tier      : 档位，数字 0~4 或名字：'big_gap' / 'near_rsi' / 'near_trend' /
                'pullback_vol' / 'fresh' / 'weak'；也可以直接给 dict
    use       : 参与打分的指标（默认取档位里的）
    need      : 必须命中的指标（默认取档位里的）
    near      : 必须「差一点」的指标（给了才启用）
    any_near  : 这几个里只要有一个「差一点」就行
    min_score : 分数下限
    within    : 「刚刚」的窗口，同时喂给 RSI 和 FVG
    top       : 筛完只留分数最高的几个（10 = 前 10 只，0.1 = 前 10%，0 = 全留）
    closest   : 筛完只留离回踩线最近的几个（同上写法，优先于 top）

    FVG 相关的门槛（FVG_DIR / FVG_MIN_PCT / FVG_MIN_BODY / FVG_MAX_FILL ...）走 kw。

    返回：代码 / 名称 / 日期 / 收盘 / 分数 / 命中 / FVG / FVG方向 / 缺口高度% /
          实体% / 缺口天数 / 回补% / 回踩线 / 距线% / RSI / RSI状态 / 趋势 / 量比 / 特征 / 信号日
    """
    cfg = _merge_cfg(kw)
    cfg['FREQ'] = _norm_freq(freq)
    cfg['BARS'] = int(bars)
    cfg['PLOT_START'] = None
    cfg['SCREEN_WITHIN'] = int(within if within is not None else cfg['SCREEN_WITHIN'])
    top = cfg['SCREEN_TOP'] if top is None else top
    closest = cfg['SCREEN_CLOSEST'] if closest is None else closest

    st = _setup_cfg(tier)
    if use is not None:
        st['use'] = list(use)
    if need is not None:
        st['need'] = list(need)
    if near is not None:
        st['near'] = list(near)
    if any_near is not None:
        st['any_near'] = list(any_near)
    if min_score is not None:
        st['min_score'] = float(min_score)

    names = list(st.get('use') or []) + list(st.get('need') or []) + \
        list(st.get('near') or []) + list(st.get('any_near') or [])
    for nm in names:
        if nm not in SCREEN_INDICATORS:
            raise ValueError('没有这个指标：%r，可选 %s' % (nm, ' / '.join(SCREEN_ORDER)))

    rows, total = _scan_pool(pool, cfg, st.get('use') or [], limit=limit, verbose=verbose)
    if not total:
        print('[综合筛选] 池子是空的，先设 pool')
        return pd.DataFrame(columns=_STOCK_COLS)

    out = []
    for r in rows:
        if _pass_setup(r, st, cfg):
            out.append(_out_row(r, st))

    if not out:
        if verbose:
            print('[综合筛选] %d 只 · %s → 0 只' % (total, st.get('name', tier)))
            print('       条件：%s' % _cond_text(st))
        return pd.DataFrame(columns=_STOCK_COLS)

    df = pd.DataFrame(out)
    n_pass = len(df)
    if sort:
        df['_d'] = df['_d'].apply(lambda x: 9e9 if x is None or x == '' else x)
        df = df.sort_values(['分数', '_d'], ascending=[False, True])
    n_take = 0
    if closest:
        df['_ad'] = df['距线%'].apply(_dist_abs)
        df = df.sort_values(['_ad', '分数'], ascending=[True, False])
        n_take = _top_n(closest, len(df))
        if n_take:
            df = df.iloc[:n_take]
        df = df.drop(columns=['_ad'])
    elif top:
        n_take = _top_n(top, len(df))
        if n_take:
            df = df.iloc[:n_take]
    if '_d' in df.columns:
        df = df.drop(columns=['_d'])
    df = df.reset_index(drop=True)

    if verbose:
        pct = (100.0 * n_pass / total) if total else 0.0
        print('[综合筛选] 池子 %d 只 · %s → 命中 %d 只（%.1f%%）'
              % (total, st.get('name', tier), n_pass, pct))
        if n_take and n_take < n_pass:
            print('       再按「%s」留了前 %d 只' % ('离回踩线最近' if closest else '分数最高', n_take))
        print('       条件：%s' % _cond_text(st))
    return df[_STOCK_COLS]


def _cond_text(st):
    parts = []
    for nm in (st.get('need') or []):
        parts.append('%s 必中' % _IND_CN.get(nm, nm))
    for nm in (st.get('near') or []):
        parts.append('%s 差一点' % _IND_CN.get(nm, nm))
    if st.get('any_near'):
        parts.append('%s 差一点（沾一个）' % '/'.join(_IND_CN.get(n, n) for n in st['any_near']))
    if st.get('min_score'):
        parts.append('分数 ≥ %g' % float(st['min_score']))
    return '；'.join(parts) or '只打分不筛'


def plot_setup(tier=2, max_charts=6, show_table=True, verbose=True, **kw):
    """按档位筛完直接把图也画出来（每个档位的「特点」都会画在图上）。

    tier      : 同 screen_stocks，数字或名字（'big_gap' / 'near_rsi' ...）
    max_charts: 最多画几只（0 = 全画）
    show_table: 画之前先把命中表完整打出来（不缩略、不省略列）
    """
    cfg = _merge_cfg(kw)
    hits = screen_stocks(tier=tier, verbose=verbose, **kw)
    if hits is None or len(hits) == 0:
        print('没有命中的，换个档位或者放宽门槛')
        return []
    if show_table:
        print('\n' + '=' * 78)
        print('命中 %d 只（完整表格）' % len(hits))
        print(hits.to_string(index=False))
        print('=' * 78 + '\n')

    pass_kw = dict((k, cfg[k]) for k in (
        'FVG_ON', 'FVG_RATIO', 'FVG_REF', 'FVG_DIR', 'FVG_MIN_PCT', 'FVG_MIN_BODY',
        'FVG_MIN_RANGE', 'FVG_MAX_FILL', 'FVG_LOOKBACK', 'FVG_CONFIRM', 'FVG_WITHIN',
        'FVG_SHOW', 'FVG_BEAR', 'FVG_NEAR_DIST', 'RSI_N', 'OB', 'OS', 'MA_LIST',
        'SHOW_VOLUME', 'MARK_EVENTS', 'CJK_FONT', 'DPI', 'FIGSIZE', 'SAVE_DIR',
        'TREND_FAST', 'TREND_SLOW', 'VOL_N', 'VOL_SHRINK', 'RSI_NEAR', 'TREND_NEAR'))
    pass_kw['MAX_CHARTS'] = int(max_charts)
    pass_kw['PLOT_BARS'] = int(cfg['PLOT_BARS'])
    pass_kw['PLOT_START'] = None
    pass_kw['PLOT_END'] = cfg['PLOT_END']
    pass_kw['BARS'] = int(cfg['BARS'])
    return plot_hits(hits, **pass_kw)

