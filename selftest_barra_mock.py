# -*- coding: utf-8 -*-
# ============================================================
# 本地 mock 自测：Barra 风格中性化 / 归因（Part B-3b）
#
# 验证思路（不连聚宽，全部合成数据）：
#   1) 暴露算得对不对 —— 我事先知道每只股票真实的「流通股本 / beta / 动量」，
#      算出来的暴露排序必须和真值一致；
#   2) 中性化洗不洗得干净 —— 故意拿「Size」本身当因子，中性化后 IC 应该掉到 0 附近；
#   3) 归因能不能把 alpha 和风格分开 —— 故意让收益 = 0.02×因子 + 0.05×Size，
#      回归出来的 λf 要接近 0.02、Size 系数要接近 0.05；
#   4) 挂到 run_factor_test 上跑一遍，开关各种模式都不许炸。
#
# 运行：python selftest_barra_mock.py
# ============================================================
import io
import os
import sys
import types
import contextlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

import numpy as np
import pandas as pd

FRAMEWORK = os.path.join(HERE, 'jq_unified_factor_framework.py')

FAILED = []


def check(name, cond, extra=''):
    if cond:
        print('  OK   %s' % name)
    else:
        print('  FAIL %s %s' % (name, extra))
        FAILED.append(name)


# ------------------------------------------------------------
# 一、造数据：真实的股本 / beta / 动量都是已知的
# ------------------------------------------------------------
TRADE = pd.bdate_range('2018-01-02', '2024-12-31')
N = 40
CODES = ['B%02d.XSHG' % i for i in range(N)]
IDX = '000300.XSHG'
_POS = {d: i for i, d in enumerate(TRADE)}
_T = len(TRADE)

# 真值：流通股本 1e8 -> 1e10（差 100 倍，Size 排序一眼可看）
S_TRUE = np.logspace(8.0, 10.0, N)
B_TRUE = np.linspace(0.4, 1.8, N)          # 真实 beta
MOM_TRUE = np.linspace(-0.5, 1.5, N)       # 真实 12 个月动量
MU = 0.00005 * np.linspace(-1.0, 1.0, N)

_rng = np.random.default_rng(20261117)
_RM = 0.0002 + 0.011 * _rng.standard_normal(_T)          # 市场日收益
_EPS = 0.010 * _rng.standard_normal((_T, N))             # 个股噪声
_RET = MU + np.outer(_RM, B_TRUE) + _EPS
_DRIFT = np.zeros((_T, N))
_DRIFT[-252:] = MOM_TRUE / 252.0                          # 动量是靠漂移做出来的
_RET = _RET + _DRIFT

_CLOSE = 10.0 * np.exp(np.cumsum(np.log1p(_RET), axis=0))
_VOL = 1e6 * np.exp(0.20 * _rng.standard_normal((_T, N)))  # 成交量（股）
_MONEY = _CLOSE * _VOL
# 让 barra 反推出来的流通股本正好等于 S_TRUE：turn = 成交量 / 流通股本 × 100
_TURN = _VOL * 100.0 / S_TRUE
_IDXC = 1000.0 * np.exp(np.cumsum(np.log1p(_RM)))

DATA = {
    'close': pd.DataFrame(_CLOSE, index=TRADE, columns=CODES),
    'money': pd.DataFrame(_MONEY, index=TRADE, columns=CODES),
    'turn': pd.DataFrame(_TURN, index=TRADE, columns=CODES),
    'index': pd.Series(_IDXC, index=TRADE),
    'has_turn': True,
}
DATA_NOTURN = {'close': DATA['close'], 'money': DATA['money'], 'index': DATA['index'],
               'turn': None, 'has_turn': False}


# ------------------------------------------------------------
# 二、mock 聚宽 API（统一走 get_price，框架内部自己拼宽表）
# ------------------------------------------------------------
def _ts(x):
    return pd.Timestamp(x).normalize()


def get_trade_days(start_date=None, end_date=None, count=None):
    s = TRADE
    if start_date is not None:
        s = s[s >= _ts(start_date)]
    if end_date is not None:
        s = s[s <= _ts(end_date)]
    if count is not None:
        s = s[:int(count)] if start_date is not None else s[-int(count):]
    return pd.DatetimeIndex(s)


def get_price(security, start_date=None, end_date=None, count=None, frequency='daily',
              fields=None, skip_paused=False, fq='pre', panel=False):
    fields = list(fields or ['close'])
    codes = [security] if isinstance(security, str) else list(security)
    if start_date is not None and end_date is not None:
        sub = TRADE[(TRADE >= _ts(start_date)) & (TRADE <= _ts(end_date))]
    elif end_date is not None and count:
        sub = TRADE[TRADE <= _ts(end_date)][-int(count):]
    else:
        sub = TRADE
    if len(sub) == 0:
        return pd.DataFrame(columns=['time', 'code'] + fields)
    pos = [_POS[d] for d in sub]
    series = []
    for f in fields:
        cols = {}
        for c in codes:
            if c == IDX:
                cols[c] = _IDXC[pos] if f == 'close' else np.full(len(pos), np.nan)
            else:
                cols[c] = DATA[f].iloc[pos][c].values
        df = pd.DataFrame(cols, index=pd.DatetimeIndex(sub, name='time'))
        series.append(df.stack().rename(f))
    out = pd.concat(series, axis=1).reset_index()
    out.columns = ['time', 'code'] + fields
    return out


def get_index_stocks(index_code, date=None):
    return list(CODES)


class _SecInfo(object):
    def __init__(self, code):
        self.code = code
        self.display_name = code.split('.')[0]


def get_security_info(code):
    return _SecInfo(code)


_fake_jq = types.ModuleType('jqdata')
for _n in ('get_price', 'get_trade_days', 'get_index_stocks', 'get_security_info'):
    setattr(_fake_jq, _n, globals()[_n])
sys.modules['jqdata'] = _fake_jq


# ------------------------------------------------------------
# 三、装载框架
# ------------------------------------------------------------
NS = {'__name__': 'jq_ns', '__file__': FRAMEWORK}
with open(FRAMEWORK, encoding='utf-8') as fh:
    exec(compile(fh.read(), FRAMEWORK, 'exec'), NS)

F = NS['barra_exposures']
BUILD_PANEL = NS['build_exposure_panel']
NEUT = NS['neutralize_panel']
LOAD = NS['factor_style_loadings']
REG = NS['barra_regress']
ANALYZE = NS['barra_analyze']
MODE_OF = NS['barra_mode_of']
SET_MODE = NS['set_barra_mode']
DATA_NOTE = NS['barra_data_note']
IC_STATS = NS['factor_ic_stats']
RUN1 = NS['run_factor_test']
REGISTER = NS['register_factor']

print('框架装载完成：%s' % FRAMEWORK)


def _sp(a, b):
    """Spearman：只看排序，正好符合「暴露排序对不对」这种检查。"""
    return pd.Series(np.asarray(a, dtype=float)).corr(
        pd.Series(np.asarray(b, dtype=float)), method='spearman')


# ------------------------------------------------------------
# 四、暴露层
# ------------------------------------------------------------
print('\n[1] 风格暴露算得对不对')
e_last = F(DATA, TRADE[-1])
check('暴露表非空', not e_last.empty)
check('暴露列齐 8 个', set(['beta', 'size', 'momentum', 'resvol', 'nlsize',
                             'liquidity', 'volat', 'reversal']).issubset(set(e_last.columns)),
      str(list(e_last.columns)))
check('size 排序 = 真实股本排序', _sp(e_last['size'], np.log(S_TRUE)) > 0.9,
      '%.3f' % _sp(e_last['size'], np.log(S_TRUE)))
check('beta 排序 = 真实 beta 排序', _sp(e_last['beta'], B_TRUE) > 0.9,
      '%.3f' % _sp(e_last['beta'], B_TRUE))
check('momentum 排序 = 真实动量排序', _sp(e_last['momentum'], MOM_TRUE) > 0.85,
      '%.3f' % _sp(e_last['momentum'], MOM_TRUE))
check('nlsize = size 的三次方', abs(_sp(e_last['nlsize'], e_last['size']) - 1.0) < 1e-6)
check('暴露已标准化（均值≈0）', abs(float(e_last['size'].mean())) < 1e-8,
      '%.2e' % float(e_last['size'].mean()))

print('\n[2] 拿不到换手率时要优雅降级')
e_nt = F(DATA_NOTURN, TRADE[-1])
check('没有 turn 就不出 size', 'size' not in e_nt.columns, str(list(e_nt.columns)))
check('其它风格照常算', ('beta' in e_nt.columns) and not e_nt.empty)
with contextlib.redirect_stdout(io.StringIO()) as _buf:
    DATA_NOTE()
check('提示里告诉用户怎么补', 'upgrade_cache' in _buf.getvalue(), _buf.getvalue()[:80])


# ------------------------------------------------------------
# 五、中性化层
# ------------------------------------------------------------
print('\n[3] 中性化：把「Size 本身」当因子，IC 应该被洗到 0')
DATES = list(TRADE[-560::40][:10])
EXPO = BUILD_PANEL(DATA, DATES, verbose=False)
check('暴露面板期数 >= 5', len(EXPO) >= 5, str(len(EXPO)))

_good = [d for d in DATES if d in EXPO]
PANEL_SIZE = pd.DataFrame({d: EXPO[d]['size'] for d in _good}).T
_r2 = np.random.default_rng(11)
FWD_SIZE = pd.DataFrame(
    {d: 0.04 * EXPO[d]['size'] + 0.004 * _r2.standard_normal(len(EXPO[d])) for d in _good}
).T

ic_raw = IC_STATS(PANEL_SIZE, FWD_SIZE)[1]['IC均值']
NPNL, NINFO = NEUT(PANEL_SIZE, EXPO, verbose=False)
ic_neu = IC_STATS(NPNL, FWD_SIZE)[1]['IC均值']
print('    IC 原 %.3f / 中性化后 %.3f' % (ic_raw, ic_neu))
check('原因子（Size）确实有 IC', ic_raw > 0.4, '%.3f' % ic_raw)
check('中性化后 IC 掉到 0 附近', abs(ic_neu) < 0.15, '%.3f' % ic_neu)
check('风格解释度 R² 很高（因为因子就是 Size）', NINFO['风格R2'] > 0.8,
      '%.3f' % NINFO['风格R2'])
check('中性化期数 = 输入期数', NINFO['期数'] == len(_good), str(NINFO['期数']))

LD = LOAD(PANEL_SIZE, EXPO)
check('载荷表非空', not LD.empty)
check('因子=Size 时 size 载荷 > 0.8', LD.loc['size', '载荷'] > 0.8,
      '%.3f' % LD.loc['size', '载荷'])


# ------------------------------------------------------------
# 六、归因层：λf 和风格系数要能还原设定值
# ------------------------------------------------------------
print('\n[4] 归因：收益 = 0.02×因子 + 0.05×Size，看回归能不能还原')
_r3 = np.random.default_rng(23)
PANEL_MIX, FWD_MIX = {}, {}
for d in _good:
    zs = EXPO[d]['size'].values
    zs = (zs - zs.mean()) / zs.std(ddof=1)
    zf = 0.6 * zs + np.sqrt(1 - 0.36) * _r3.standard_normal(len(zs))
    zf = (zf - zf.mean()) / zf.std(ddof=1)
    PANEL_MIX[d] = pd.Series(zf, index=EXPO[d].index)
    FWD_MIX[d] = pd.Series(0.02 * zf + 0.05 * zs + 0.004 * _r3.standard_normal(len(zs)),
                           index=EXPO[d].index)
PANEL_MIX = pd.DataFrame(PANEL_MIX).T
FWD_MIX = pd.DataFrame(FWD_MIX).T

REG_RES = REG(PANEL_MIX, FWD_MIX, EXPO, verbose=False)
check('归因回归有结果', bool(REG_RES))
check('λf ≈ 0.02', abs(REG_RES['纯因子收益λf'] - 0.02) < 0.008,
      '%.4f' % REG_RES['纯因子收益λf'])
check('Size 平均系数 ≈ 0.05', abs(REG_RES['风格平均系数']['size'] - 0.05) < 0.02,
      '%.4f' % REG_RES['风格平均系数']['size'])
check('λf 显著（|t| >= 2）', abs(REG_RES['t值']) >= 2.0, '%.2f' % REG_RES['t值'])
check('风格 R² 介于 0~1', 0.0 <= REG_RES['风格R2'] <= 1.0, '%.3f' % REG_RES['风格R2'])


# ------------------------------------------------------------
# 七、模式开关
# ------------------------------------------------------------
print('\n[5] 模式别名与兜底')
check("mode_of('') 默认跟随全局", MODE_OF('') in ('off', 'neutral', 'attr', 'both'))
check("mode_of('中性') -> neutral", MODE_OF('中性') == 'neutral')
check("mode_of('归因') -> attr", MODE_OF('归因') == 'attr')
check("mode_of('都做') -> both", MODE_OF('都做') == 'both')
check("mode_of('关') -> off", MODE_OF('关') == 'off')
try:
    MODE_OF('瞎写')
    check('未知模式要报错', False)
except ValueError as ex:
    check('未知模式要报错', 'BARRA_MODE' in str(ex), str(ex))

with contextlib.redirect_stdout(io.StringIO()):
    SET_MODE('neutral')
check('set_barra_mode 改了全局', NS['BARRA_MODE'] == 'neutral', NS['BARRA_MODE'])
with contextlib.redirect_stdout(io.StringIO()):
    SET_MODE('off')
check('能改回 off', NS['BARRA_MODE'] == 'off')

with contextlib.redirect_stdout(io.StringIO()) as _buf2:
    NS['explain_barra']()
check('explain_barra 打得出说明', '中性化' in _buf2.getvalue() and len(_buf2.getvalue()) > 200)


# ------------------------------------------------------------
# 八、端到端：挂到 run_factor_test 上
# ------------------------------------------------------------
print('\n[6] 端到端 run_factor_test(barra=...)')


@REGISTER('barra_ln_size')
def _f_ln_size(stock_list, date):
    """故意做一个 Size 因子：只要 barra 生效，它的 IC 就会被洗掉一大截。"""
    return pd.Series(np.log(S_TRUE), index=CODES).reindex(stock_list)


with contextlib.redirect_stdout(io.StringIO()):
    R_OFF = RUN1('barra_ln_size', CODES, '2021-01-01', '2022-12-31', period=20,
                 barra='off', verbose=False)
check('barra=off 不出 barra 结果', R_OFF.get('barra') is None)
check('barra=off 有 IC 统计', not R_OFF['ic_stats'].empty)
check('barra=off 保留原始面板', R_OFF['factor_panel_raw'] is not None)

with contextlib.redirect_stdout(io.StringIO()):
    R_BOTH = RUN1('barra_ln_size', CODES, '2021-01-01', '2022-12-31', period=20,
                  barra='both', verbose=False)
_b = R_BOTH.get('barra') or {}
check('barra=both 有结果', bool(_b))
check('mode 记成 both', _b.get('mode') == 'both', str(_b.get('mode')))
check('有暴露面板', bool(_b.get('expo_panel')))
check('有中性化因子面板', _b.get('neutral_panel') is not None and not _b['neutral_panel'].empty)
check('有归因结果', bool(_b.get('reg')))
_ic_off = R_OFF['ic_stats']['IC均值']
_ic_neu = R_BOTH['ic_stats']['IC均值']
print('    Size 因子 IC：off %.3f -> both(中性化后) %.3f' % (_ic_off, _ic_neu))
check('尺寸因子的 IC 被中性化明显压下去', abs(_ic_neu) < max(0.02, abs(_ic_off) * 0.6),
      'off=%.3f neu=%.3f' % (_ic_off, _ic_neu))
check('原始面板仍留着（便于对比）', R_BOTH['factor_panel_raw'] is not None)

with contextlib.redirect_stdout(io.StringIO()):
    R_NEU = RUN1('barra_ln_size', CODES, '2021-01-01', '2022-12-31', period=20,
                 barra='neutral', verbose=False)
check('barra=neutral 不跑归因', not (R_NEU.get('barra') or {}).get('reg'))
check('barra=neutral 有中性化面板', bool((R_NEU.get('barra') or {}).get('neutral_panel') is not None))

with contextlib.redirect_stdout(io.StringIO()):
    R_ATTR = RUN1('barra_ln_size', CODES, '2021-01-01', '2022-12-31', period=20,
                  barra='attr', verbose=False)
check('barra=attr 只归因不中性化', (R_ATTR.get('barra') or {}).get('neutral_panel') is None)
check('barra=attr 有回归', bool((R_ATTR.get('barra') or {}).get('reg')))

print('\n%d 条断言失败：%s' % (len(FAILED), FAILED if FAILED else '无'))
sys.exit(1 if FAILED else 0)
