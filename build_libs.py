# -*- coding: utf-8 -*-
# ============================================================
# 把「因子研究框架」的十个源文件合并成一个可以直接 import 的单文件：
#   python build_libs.py   ->   factor_lib.py
#
# 为什么合并：
#   notebook 里逐格 exec 5800 行代码，改一个参数要滚半天、每次重开内核还得全跑一遍。
#   合并成模块后，notebook 只剩「import + 改参数 + 调用」，
#   改代码去改源文件、重跑 build_libs.py 即可，notebook 不用动。
#
# 加载顺序（有依赖关系，别改）：
#   数据层 -> 框架 -> 可视化 -> 补丁/扩展 -> 参数区（放最后，参数值不会被框架默认值覆盖）
# ============================================================
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'factor_lib.py')

PARTS = [
    ('jq_io.py', '数据层   baostock 取数 / 本地 pkl 缓存 / 复权因子 / 交易日历 / 成分股'),
    ('jq_unified_factor_framework.py', 'Part A~D 通用框架  池配置 / 数据读取 / 三通道因子 / 截面检验 / 时序检验 / 回测衔接'),
    ('jq_factor_viz.py', '附录 A 截面可视化  plot_ic_series / plot_quantile_returns / plot_single_all ...'),
    ('jq_ts_viz.py', '附录 B 时序可视化  plot_price_signal / plot_backtest / ts_plot_ic_decay ...'),
    ('jq_tsa_patch.py', '附录 C 时序有效性补丁  tsa_build / tsa_report / tsa_yearly_table'),
    ('jq_pct_factor.py', 'Part F 历史分位打分因子  pct_rank / pct_to_score / px_pct_*'),
    ('jq_ic_report.py', 'Part G IC 视角  screen_ic / ic_tables / ic_plots / ic_report'),
    ('jq_position.py', 'Part H 仓位管理  atr_squeeze / vol_ratio / bottom_score_panel / weight_state'),
    ('jq_rsi.py', 'Part I RSI 体系  calc_rsi / rsi_zone / rsi_reversal_state / rsi_report'),
    ('jq_params.py', '参数区 + 统一入口  PARAM_DOC / cs_run / ts_run / batch_run / apply_params'),
]

HEADER = '''# -*- coding: utf-8 -*-
# Copyright 2026 Neeko-ovo
# Apache License 2.0 -- 见项目根目录 LICENSE / NOTICE（再分发请保留署名）
"""
因子研究框架 · 导入版（单文件）
================================================================

    import factor_lib as F

    F.params_brief()                       # 看一眼当前口径
    F.preview_pool('HS300')                # 池子里有什么
    res = F.cs_run('px_pct_750')           # 截面检验（自动填参数）
    F.plot_single_all(res)                 # 13 张图
    t = F.ts_run('pct_score', th=60)       # 时序择时
    F.plot_all(t, code='000300.XSHG', th=60)

分层（按加载顺序）：
    1  数据层      jq_io（baostock + 本地缓存）—— 与聚宽同名的 get_price /
                   get_index_stocks / get_trade_days / get_security_info
    2  框架        Part A~D 池配置 / 取数 / 因子 / 截面检验 / 时序检验
    3  可视化      附录 A 截面 / 附录 B 时序（mpl 与 plotly 双引擎）
    4  扩展        Part F 分位因子 / Part G IC 视图 / Part H 仓位 / Part I RSI
    5  参数区      放在最后，这样 BARRA_MODE 这类同名参数不会被框架默认值覆盖

约定：
    - 数据源自动判断：聚宽环境用平台 get_price，本地用 jq_io（baostock）。
      想强制指定就设环境变量 JQ_DATA_SOURCE = jq | local | sdk
    - 本文件是 build_libs.py 自动生成的，改逻辑请改十个源文件后重新生成。
    - 运行时端在 量化工作台.ipynb，那个 notebook 只 import、不内联代码。
"""

__version__ = '6.0.0'
'''

TAIL = '''

# ============================================================
# 导入版附加：一行看清「这个模块里有什么」
# ============================================================
_LAYERS = {
    '数据层': 'get_price / get_price_history / get_price_range / get_index_stocks'
              ' / get_trade_days / get_security_info / warm_cache / upgrade_cache',
    '框架': 'resolve_universe / run_factor_test / run_batch / run_combo / scan_ic_decay'
            ' / run_ts_test / run_ts_batch / ts_ic_decay / register_factor / register_ts_signal',
    '可视化': 'plot_single_all / plot_batch / plot_ic_series / plot_quantile_returns'
              ' / plot_price_signal / plot_backtest / ts_plot_ic_decay / set_viz_engine / ts_set_engine',
    '扩展': 'pct_rank / px_pct_750 / screen_ic / ic_report / weight_state'
            ' / atr_squeeze / rsi_report / rsi_reversal_state',
    '参数区': 'PARAM_DOC / params_table / params_brief / apply_params / cs_run / ts_run'
              ' / cs_batch_run / ts_batch_run / barra_switch / plot_engine',
    'Barra': 'barra_analyze / neutralize_panel / factor_style_loadings / barra_report / explain_barra',
}


def usage(verbose=True):
    """打印最短用法（不知道从哪下手先调它）。"""
    lines = ['=' * 68, '因子研究框架（导入版） v%s' % __version__, '=' * 68]
    for k in ('数据层', '框架', '可视化', '扩展', '参数区', 'Barra'):
        lines.append('%-6s %s' % (k, _LAYERS[k]))
    lines += ['', '— 最短路径 —',
              "  F.params_brief()",
              "  F.cs_run('px_pct_750')                 # 截面：自动用参数区的池子/区间",
              "  F.ts_run('pct_score', th=60)           # 时序",
              "  F.cs_batch_run(); F.ts_batch_run()     # 批量对比",
              "  F.set_barra_mode('neutral')            # 换成风格中性化口径",
              '=' * 68]
    text = '\\n'.join(lines)
    if verbose:
        print(text)
    return text


def describe(name):
    """查某个名字是哪一层、干什么的（记不住函数名时用）。"""
    obj = globals().get(name)
    if obj is None:
        print('%r 不在这个模块里' % (name,))
        return None
    doc = (getattr(obj, '__doc__', '') or '').strip().splitlines()
    print('%-28s %s' % (name, doc[0] if doc else '(无说明)'))
    return obj
'''

SKIP_MSG_OLD = "local(notebook 内联 jq_io)"
SKIP_MSG_NEW = "local(jq_io · baostock)"


def load(name):
    with io.open(os.path.join(HERE, name), 'r', encoding='utf-8') as fh:
        text = fh.read()
    lines = text.splitlines()
    if lines and lines[0].startswith('# -*- coding'):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    body = '\n'.join(lines).rstrip() + '\n'
    # 合并后 jq_io 就在同一个模块里，这行提示别再说「notebook 内联」
    return body.replace(SKIP_MSG_OLD, SKIP_MSG_NEW)


def main():
    chunks = [HEADER]
    for name, desc in PARTS:
        banner = ('\n\n# ' + '=' * 62 + '\n'
                  '# ' + desc + '\n'
                  '# 源文件：' + name + '\n'
                  '# ' + '=' * 62 + '\n')
        chunks.append(banner)
        chunks.append(load(name))
    chunks.append(TAIL)
    text = ''.join(chunks)
    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)

    compile(text, OUT, 'exec')          # 语法自检
    print('已生成 %s' % OUT)
    print('  行数 %d，大小 %.1f KB'
          % (text.count('\n') + 1, len(text.encode('utf-8')) / 1024.0))


if __name__ == '__main__':
    main()
