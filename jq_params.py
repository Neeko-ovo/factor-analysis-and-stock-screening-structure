# -*- coding: utf-8 -*-
"""参数设置层（全框架共用一套口径）。

改法：直接改下面的赋值，然后重跑这一格；或者 `apply_params(TH=60, PERIOD=60)` 临时改。
跑任何检验都用 `cs_run()` / `ts_run()` / `cs_batch_run()` / `ts_batch_run()`，
它们会自动把这些参数填进去，不用每次手写日期和 period。
"""

# ====================== 1. 标的与区间 ======================
CODE = '000300.XSHG'          # 时序研究的标的（指数或股票/ETF 代码）
POOL = 'HS300'                # 截面研究的池子：HS300 / ZZ500 / ZZ800 / 全A / 代码列表
START = '2019-01-01'          # 回测/检验起点
END = '2025-06-30'            # 终点（超过今天会自动截断）

# ====================== 2. 检验口径 ======================
PERIOD = 20                   # 前瞻期（交易日）；IC 与状态分析用，不影响回测
FREQ = 'M'                    # 截面调仓频率：D / W / M / Q / Y
DEC_FREQ = 'W'                # 时序决策频率：D / W / M（W＝每周看一次信号）
TH = 0.0                      # 信号阈值：signal > TH 才持有（分位因子常用 60）
COST = 0.0005                 # 每次仓位切换成本（单边）
DIRECTION = 1                 # 1 正向；-1 翻转（波动率、换手率这类负向信号）
STRICT = False                # True = 前瞻收益从 t+1 起算（更保守）

# ====================== 3. 截面专用 ======================
Q = 5                         # 分层数（5 = 五档）
RULE = 'first'                # 每期取第一个交易日：first / last
EVERY = 1                     # 间隔倍数（freq='M', every=2 → 双月调仓）
NON_OVERLAP = False           # True = 强制非重叠（持仓期多长就等多久再调）

# ====================== 4. 批量清单 ======================
FACTORS = ['px_pct_750', 'px_pct_dev250', 'px_pct_score']   # 截面批量要跑的因子
SIGNALS = ['ma200', 'mom_120', 'pct_score']                 # 时序批量要跑的信号
CS_PERIODS = (5, 10, 20, 63, 126, 250)   # 截面 IC 衰减扫描的持仓期
TS_PERIODS = (5, 10, 20, 60, 120)        # 时序 IC 衰减扫描的前瞻期

# ====================== 5. 输出 ======================
PLOT_ENGINE = 'mpl'           # 'mpl' 静态图 / 'plotly' 可交互
OUT_DIR = 'jq_out'            # 图片落盘目录
DPI = 130                     # 存图分辨率

# ====================== 6. Barra 风格中性化（可选） ======================
# 'off'     : 不启用，跟以前完全一样（默认）
# 'neutral' : 先把因子对风格做横截面回归取残差，再用残差检验 -> 适合「非风格因子」
# 'attr'    : 因子不动，把收益对「因子+风格」回归，看控制风格后还有没有 alpha
# 'both'    : 两个都做（慢一点，但一次看全）
# 改完记得跑一次 barra_switch() 让它同步进框架。
BARRA_MODE = 'off'
BARRA_STYLES = ('beta', 'size', 'momentum', 'resvol', 'nlsize',
                'liquidity', 'volat', 'reversal')   # 要保留哪些风格
BARRA_WINSOR = 0.01           # 两端各砍 1% 极值（防单只妖股带偏回归）

# ====================== 校验与查看 ======================
PARAM_DOC = [
    ('CODE', '时序标的', 'str'),
    ('POOL', '截面池子', 'str_or_list'),
    ('START', '起点', 'date'),
    ('END', '终点', 'date'),
    ('PERIOD', '前瞻期', 'pos_int'),
    ('FREQ', '截面调仓频率', 'freq'),
    ('DEC_FREQ', '时序决策频率', 'dec'),
    ('TH', '信号阈值', 'num'),
    ('COST', '单边成本', 'nonneg'),
    ('DIRECTION', '方向', 'sign'),
    ('STRICT', '保守口径', 'bool'),
    ('Q', '分层数', 'q'),
    ('RULE', '取首末', 'rule'),
    ('EVERY', '间隔倍数', 'pos_int'),
    ('NON_OVERLAP', '非重叠', 'bool'),
    ('FACTORS', '截面清单', 'list'),
    ('SIGNALS', '时序清单', 'list'),
    ('CS_PERIODS', '截面衰减期', 'periods'),
    ('TS_PERIODS', '时序衰减期', 'periods'),
    ('PLOT_ENGINE', '画图引擎', 'engine'),
    ('OUT_DIR', '输出目录', 'str'),
    ('DPI', '存图分辨率', 'pos_int'),
    ('BARRA_MODE', 'Barra 模式', 'barra'),
    ('BARRA_STYLES', '启用风格', 'list'),
    ('BARRA_WINSOR', '缩尾比例', 'frac'),
]


def _param_names():
    return [x[0] for x in PARAM_DOC]


def check_params(**over):
    """按 PARAM_DOC 的规则校验；over 里的键优先（用于改之前的预检）。"""
    g = dict((k, globals().get(k)) for k in _param_names())
    g.update(over)
    for name, _, kind in PARAM_DOC:
        v = g.get(name)
        if kind == 'date' and not (isinstance(v, str) and len(v) == 10):
            raise ValueError('%s 要写成 "YYYY-MM-DD"，现在是 %r' % (name, v))
        if kind == 'pos_int' and not (isinstance(v, int) and v > 0):
            raise ValueError('%s 要是正整数，现在是 %r' % (name, v))
        if kind == 'nonneg' and not (isinstance(v, (int, float)) and v >= 0):
            raise ValueError('%s 不能为负，现在是 %r' % (name, v))
        if kind == 'num' and not isinstance(v, (int, float)):
            raise ValueError('%s 要是数字，现在是 %r' % (name, v))
        if kind == 'freq' and v not in ('D', 'W', 'M', 'Q', 'Y'):
            raise ValueError('%s 只认 D/W/M/Q/Y，现在是 %r' % (name, v))
        if kind == 'dec' and v not in ('D', 'W', 'M'):
            raise ValueError('%s 只认 D/W/M，现在是 %r' % (name, v))
        if kind == 'sign' and v not in (1, -1):
            raise ValueError('%s 只认 1 或 -1，现在是 %r' % (name, v))
        if kind == 'rule' and v not in ('first', 'last'):
            raise ValueError('%s 只认 first/last，现在是 %r' % (name, v))
        if kind == 'q' and not (isinstance(v, int) and 2 <= v <= 20):
            raise ValueError('%s 建议 2~20，现在是 %r' % (name, v))
        if kind == 'engine' and v not in ('mpl', 'plotly'):
            raise ValueError('%s 只认 mpl/plotly，现在是 %r' % (name, v))
        if kind == 'barra' and str(v).strip().lower() not in (
                'off', 'none', 'neutral', '中性', '中性化', 'attr', '归因', 'both', '都做'):
            raise ValueError('%s 只认 off/neutral/attr/both，现在是 %r' % (name, v))
        if kind == 'frac' and not (isinstance(v, (int, float)) and 0 <= v < 0.5):
            raise ValueError('%s 要在 [0, 0.5) 之间，现在是 %r' % (name, v))
        if kind == 'list' and not (isinstance(v, (list, tuple)) and len(v) > 0):
            raise ValueError('%s 不能是空列表' % name)
        if kind == 'periods' and not (isinstance(v, (list, tuple)) and len(v) > 0):
            raise ValueError('%s 不能是空元组' % name)
    if str(g['START']) >= str(g['END']):
        raise ValueError('START(%s) 必须早于 END(%s)' % (g['START'], g['END']))
    return True


def params_table():
    """当前参数一览（DataFrame）。"""
    import pandas as pd
    rows = []
    for name, desc, kind in PARAM_DOC:
        rows.append({'参数': name, '当前值': globals().get(name),
                     '说明': desc, '规则': kind})
    return pd.DataFrame(rows, columns=['参数', '当前值', '说明', '规则'])


def apply_params(**over):
    """改参数并校验：apply_params(TH=60, PERIOD=60)。键写错会直接报出来。"""
    unknown = [k for k in over if k not in _param_names()]
    if unknown:
        raise KeyError('没有这些参数：%s（可用：%s）' % (unknown, ', '.join(_param_names())))
    check_params(**over)
    for k, v in over.items():
        globals()[k] = v
    print('已更新：' + ', '.join('%s=%r' % (k, over[k]) for k in sorted(over)))
    return params_table()


def no_lookahead():
    """把口径一次性切到最保守：前瞻收益从 t+1 起算 + 强制非重叠。"""
    return apply_params(STRICT=True, NON_OVERLAP=True)


# ====================== 统一入口（自动套参数） ======================
def cs_run(factor=None, pool=None, start=None, end=None, **kw):
    """截面单因子检验：cs_run() / cs_run('px_pct_750') / cs_run(period=60)。"""
    args = dict(period=PERIOD, freq=FREQ, rule=RULE, every=EVERY,
                strict=STRICT, q=Q, non_overlap=NON_OVERLAP, barra=BARRA_MODE)
    args.update(kw)
    f = factor if factor is not None else FACTORS[0]
    return run_factor_test(f, pool if pool is not None else POOL,
                           start if start is not None else START,
                           end if end is not None else END, **args)


def cs_batch_run(factors=None, pool=None, start=None, end=None, **kw):
    """截面批量：默认跑 FACTORS 里那几只。"""
    args = dict(period=PERIOD, freq=FREQ, rule=RULE, every=EVERY,
                strict=STRICT, q=Q, non_overlap=NON_OVERLAP, barra=BARRA_MODE)
    args.update(kw)
    return run_batch(list(factors if factors is not None else FACTORS),
                     pool if pool is not None else POOL,
                     start if start is not None else START,
                     end if end is not None else END, **args)


def cs_decay_run(factor=None, pool=None, start=None, end=None, **kw):
    """截面 IC 衰减扫描：默认扫 CS_PERIODS。"""
    args = dict(periods=CS_PERIODS, freq=FREQ, rule=RULE, every=EVERY,
                strict=STRICT, non_overlap=NON_OVERLAP)
    args.update(kw)
    f = factor if factor is not None else FACTORS[0]
    return scan_ic_decay(f, pool if pool is not None else POOL,
                         start if start is not None else START,
                         end if end is not None else END, **args)


def ts_run(signal=None, code=None, start=None, end=None, **kw):
    """时序单信号检验：ts_run() / ts_run('pct_score', th=60)。"""
    args = dict(period=PERIOD, th=TH, decision_freq=DEC_FREQ, cost=COST,
                direction=DIRECTION, strict=STRICT)
    args.update(kw)
    s = signal if signal is not None else SIGNALS[0]
    return run_ts_test(s, code if code is not None else CODE,
                       start if start is not None else START,
                       end if end is not None else END, **args)


def ts_batch_run(signals=None, code=None, start=None, end=None, **kw):
    """时序批量：默认跑 SIGNALS 里那几只。"""
    args = dict(period=PERIOD, th=TH, decision_freq=DEC_FREQ, cost=COST,
                direction=DIRECTION)
    args.update(kw)
    return run_ts_batch(list(signals if signals is not None else SIGNALS),
                        code if code is not None else CODE,
                        start if start is not None else START,
                        end if end is not None else END, **args)


def ts_decay_run(signal=None, code=None, start=None, end=None, **kw):
    """时序 IC 衰减扫描：默认扫 TS_PERIODS。"""
    args = dict(periods=TS_PERIODS, strict=STRICT)
    args.update(kw)
    s = signal if signal is not None else SIGNALS[0]
    return ts_ic_decay(s, code if code is not None else CODE,
                       start if start is not None else START,
                       end if end is not None else END, **args)


def plot_engine(v=None):
    """把引擎同步到两个画图模块（顺带把参数区的 PLOT_ENGINE 也改成同一个值）。"""
    global PLOT_ENGINE
    e = PLOT_ENGINE if v is None else v
    if e not in ('mpl', 'plotly'):
        raise ValueError("PLOT_ENGINE 只认 mpl/plotly，现在是 %r" % (e,))
    PLOT_ENGINE = e
    done = []
    if 'set_viz_engine' in globals():
        set_viz_engine(e)
        done.append('截面')
    if 'ts_set_engine' in globals():
        ts_set_engine(e)
        done.append('时序')
    if not done:
        print('画图模块还没加载（先跑附录 A/B 那几格）')
    else:
        print('画图引擎 -> %s（%s）' % (e, ' / '.join(done)))
    return e


def barra_switch(mode=None, styles=None):
    """把参数区的 Barra 设置同步进框架（改完 BARRA_MODE 就重跑这一格）。

    例：barra_switch('neutral') —— 立刻切到中性化模式；
        barra_switch()           —— 只把当前参数区的值推过去。
    """
    global BARRA_MODE, BARRA_STYLES
    if mode is not None:
        check_params(BARRA_MODE=mode)
        BARRA_MODE = mode
    if styles is not None:
        check_params(BARRA_STYLES=styles)
        BARRA_STYLES = styles
    check_params(BARRA_WINSOR=BARRA_WINSOR)
    if 'set_barra_mode' in globals():
        set_barra_mode(BARRA_MODE, styles=BARRA_STYLES, winsor=BARRA_WINSOR)
        print('提示：run_factor_test 里 barra=None 时也会读框架的 BARRA_MODE，两边一致。')
    else:
        print('框架还没加载（先跑 Part A~B），现在只改了参数区')
    return BARRA_MODE


def params_brief():
    """一句话摘要（参数区跑完自动打印一次）。"""
    txt = ('标的 %s ｜ 池子 %s ｜ %s ~ %s ｜ 前瞻 %d 日 ｜ 截面 %s ｜ 时序 %s ｜ TH %s ｜ 引擎 %s'
           % (CODE, POOL, START, END, PERIOD, FREQ, DEC_FREQ, TH, PLOT_ENGINE))
    if str(BARRA_MODE).strip().lower() not in ('off', 'none'):
        txt += ' ｜ Barra %s' % BARRA_MODE
    return txt


check_params()
print(params_brief())
print('改参数：直接改上面赋值 / apply_params(TH=60)；看全部：params_table()')
