# -*- coding: utf-8 -*-
# Copyright 2026 Neeko-ovo
# Apache License 2.0 -- 见项目根目录 LICENSE / NOTICE（再分发请保留署名）
# ============================================================
# 把五个模块合并成一个可以直接 import 的单文件：jq_factor_framework.py
#   用法：python build_module.py
#   产物：jq_factor_framework.py（粘贴/上传到聚宽研究根目录即可 import）
#
# 为什么合并而不是做包：
#   聚宽研究环境里上传一个 .py 比上传一个包目录省事；
#   五个源文件本来就要求"按顺序 exec 进同一命名空间"，
#   合并后的单文件语义与 notebook 里逐格运行完全一致（已用 mock 验证）。
# ============================================================
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'jq_factor_framework.py')

PARTS = [
    ('jq_unified_factor_framework.py',
     'Part A~D 通用框架  池配置 / 数据读取 / 三通道因子 / 截面检验 / 时序检验 / 回测衔接'),
    ('jq_factor_viz.py',
     '附录 A 截面可视化  jq_factor_viz（只读框架返回值，不改框架逻辑）'),
    ('jq_ts_viz.py',
     '附录 B 时序可视化  jq_ts_viz（函数名统一 ts_ / _tsv_ 前缀，避免覆盖）'),
    ('jq_tsa_patch.py',
     '附录 C 时序有效性补丁 jq_tsa_patch（滚动IC / 状态 / 换仓 / 年度表）'),
    ('jq_pct_factor.py',
     'Part F 历史分位打分因子  pct_rank / pct_to_score / px_pct_* / run_pct_value_test'),
    ('jq_ic_report.py',
     'Part G IC 视角        screen_ic / ic_tables / ic_plots / ic_report / ts_ic_table'),
    ('jq_position.py',
     'Part H 仓位管理      atr_pct_20 / atr_rank_20 / atr_squeeze / vol_ratio / bw_rank / rsi_14 / mom_60'
     ' / bottom_score_panel / weight_state / run_weight_test / weight_report'),
    ('jq_rsi.py',
     'Part I RSI 体系      calc_rsi / rsi_panel / rsi_zone / rsi_reversal_events / rsi_reversal_state'
     ' / rsi_zone_forward / rsi_report / plot_rsi / rsi / rsi_rev / rsi_rev_hold'),
]

HEADER = '''# -*- coding: utf-8 -*-
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
'''

TAIL = '''

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
    print('\\n'.join(lines))
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
        print('\\n'.join(msgs))
        print('— 自检%s —' % ('通过' if ok else '有问题'))
    return {'ok': ok, 'messages': msgs}


# 所有公开名字都导出（含 matplotlib 的 np/pd/plt，方便直接画图）
__all__ = sorted([_n for _n in list(globals()) if not _n.startswith('_')])

print('[jq_factor_framework] v%s 已加载：时序信号 %d 个，自定义截面因子 %d 个'
      % (__version__, len(TS_SIGNALS), len(CUSTOM_FACTORS)))
'''


def load(name):
    with io.open(os.path.join(HERE, name), 'r', encoding='utf-8') as fh:
        text = fh.read()
    lines = text.splitlines()
    # 去掉文件自己的 coding 声明（生成文件已有）
    if lines and lines[0].startswith('# -*- coding'):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return '\n'.join(lines).rstrip() + '\n'


def main():
    chunks = [HEADER]
    for name, desc in PARTS:
        banner = ('\n\n# ' + '=' * 60 + '\n'
                  '# ' + desc + '\n'
                  '# 源文件：' + name + '\n'
                  '# ' + '=' * 60 + '\n')
        chunks.append(banner)
        chunks.append(load(name))
    chunks.append(TAIL)
    text = ''.join(chunks)
    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)

    # 语法自检
    compile(text, OUT, 'exec')
    print('已生成 %s' % OUT)
    print('  行数 %d，大小 %.1f KB' % (text.count('\n') + 1, len(text.encode('utf-8')) / 1024.0))


if __name__ == '__main__':
    main()



