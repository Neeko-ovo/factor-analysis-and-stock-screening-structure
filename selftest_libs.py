# -*- coding: utf-8 -*-
"""合并后的两个库能不能 import、关键入口齐不齐（不联网，只做静态自检）。"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '_mplcache'))
import matplotlib
matplotlib.use('Agg')

import factor_lib as F
import rsi_multiday as R

fail = []


def ck(cond, label):
    print(('  [OK] ' if cond else '  [!!] ') + label)
    if not cond:
        fail.append(label)


print('== factor_lib ==')
print('  版本 %s' % getattr(F, '__version__', '?'))
ck(callable(getattr(F, 'run_factor_test', None)), 'run_factor_test 存在')
ck(callable(getattr(F, 'run_ts_test', None)), 'run_ts_test 存在')
ck(callable(getattr(F, 'plot_single_all', None)), 'plot_single_all 存在')
ck(callable(getattr(F, 'tsa_build', None)), 'tsa_build 存在')
ck(callable(getattr(F, 'screen_ic', None)), 'screen_ic 存在')
ck(callable(getattr(F, 'weight_state', None)), 'weight_state 存在')
ck(callable(getattr(F, 'rsi_report', None)), 'rsi_report 存在')
ck(callable(getattr(F, 'cs_run', None)), 'cs_run 存在')
ck(callable(getattr(F, 'get_price', None)), 'get_price 存在（jq_io 已并进来）')
ck(getattr(F, 'LOCAL_MODE', None) is True, 'LOCAL_MODE=True')
ck(isinstance(getattr(F, 'PARAM_DOC', None), list) and len(F.PARAM_DOC) >= 25,
   'PARAM_DOC %d 条' % len(getattr(F, 'PARAM_DOC', [])))
ck(getattr(F, 'BARRA_MODE', None) == 'off', '参数区的 BARRA_MODE 没被框架默认值盖掉')
ck(len(getattr(F, 'TS_SIGNALS', {})) >= 6, 'TS_SIGNALS %d 个' % len(F.TS_SIGNALS))
ck(len(getattr(F, 'CUSTOM_FACTORS', {})) >= 3, 'CUSTOM_FACTORS %d 个' % len(F.CUSTOM_FACTORS))

print('== rsi_multiday ==')
ck(callable(getattr(R, 'screen_stocks', None)), 'screen_stocks 存在')
ck(callable(getattr(R, 'plot_kline_rsi', None)), 'plot_kline_rsi 存在')
ck(callable(getattr(R, 'atr_signal', None)), 'atr_signal 存在')
ck(len(getattr(R, 'SETUPS', {})) >= 12, 'SETUPS %d 个档位' % len(R.SETUPS))
ck('atr_sig' in R.SCREEN_INDICATORS, 'atr_sig 指标已注册')

print('== 互不干扰 ==')
ck(F.pd is R.pd, '两边用的是同一个 pandas')
ck(F.np is not None, 'numpy 可见')

print('\n[%s] %d 项失败' % ('FAIL' if fail else 'PASS', len(fail)))
sys.exit(1 if fail else 0)
