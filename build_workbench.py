# -*- coding: utf-8 -*-
# ============================================================
# 生成「运行端」notebook：量化工作台.ipynb
#
# 这个 notebook 只做三件事：import、改参数、调用。
# 一行框架代码都不内联 —— 代码在 factor_lib.py / rsi_multiday.py 里，
# 想改逻辑去改那两个 .py（改完 factor_lib 要重跑 build_libs.py），
# notebook 只要 Restart & Run All 就能看到新结果。
# ============================================================
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '量化工作台.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.strip('\n').splitlines(True)}


TITLE = '''
# 量化工作台

这一页是**运行端**，只做三件事：`import` → 改参数 → 调用。

| 文件 | 角色 | 什么时候动它 |
|---|---|---|
| `量化工作台.ipynb`（本页） | 运行端 | 天天用；改参数、看结果、出图 |
| `factor_lib.py` | 代码端 · 因子框架（8150 行） | 想加因子/改算法才动，改完重跑 `build_libs.py` |
| `rsi_multiday.py` | 代码端 · RSI + FVG + ATR 筛股 | 想改指标/档位才动，改完重跑 `build_rsi_notebook.py` |
| `jq_io.py` | 数据层 · baostock 取数与缓存 | 想换数据源、补字段才动 |
| `build_libs.py` / `build_rsi_notebook.py` | 装配脚本 | 只在上面两个 .py 改完后跑一次 |

好处：改参数不用滚几千行代码，重开内核也不用等它一行行跑；
`.py` 还能被别的脚本 `import`，写批量回测、写定时任务都能直接复用。

**第一次用**：Kernel → Restart & Run All，一路往下跑就行。
'''

C_IMPORT = '''
%matplotlib inline
import factor_lib as F
import rsi_multiday as R

print('factor_lib %s 已载入  |  rsi_multiday 已载入' % F.__version__)
print('数据源：%s' % ('本地 jq_io / baostock' if F.LOCAL_MODE else '聚宽平台'))
'''

C_USAGE = '''
# 忘了有什么函数就调这一行（打印六层各有什么）
F.usage()
'''

MD_PARAM = '''
## 一、参数区

所有口径集中在 `jq_params` 里，改一处全框架跟着走。
想知道每个参数能填什么：`F.PARAM_DOC` 是规则表（有中文说明和取值范围）。
'''

C_PARAM = '''
F.params_table()          # 当前全部参数
print(F.params_brief())   # 一行摘要
'''

C_PARAM_SET = '''
# 改参数两种写法，效果一样：
#   ① 直接改变量：F.PERIOD = 20
#   ② 用 apply_params（会顺手校验，填错立刻报错并告诉你能填什么）
F.apply_params(PERIOD=20, TH=60.0, FREQ='M', DEC_FREQ='W')

# 切 Brava 风格中性化口径（off / neutral 中性 / attr 归因 / both）
# F.barra_switch('neutral')
# 切出图引擎（mpl 静态 / plotly 可放大拖动）
# F.plot_engine('plotly')
'''

MD_POOL = '''
## 二、池子

`HS300` / `ZZ500` / `ZZ800` / `SZ50` / `CYB`，也可以直接给代码列表。
成分股按日期取（不是拿今天的名单回测历史），避免生存偏差。
'''

C_POOL = '''
F.preview_pool(F.POOL)       # 池子默认取参数区的 POOL
'''

MD_CS = '''
## 三、截面因子检验

一个因子在横截面上有没有区分度：IC、分层收益、换手、分档。
结果里 `ic` 是逐期 IC，`quantile_returns` 是分层收益，
`barra` 是风格暴露分析（`BARRA_MODE='off'` 时为 None）。
'''

C_CS = '''
res = F.cs_run('px_pct_500')       # 池子/区间/前瞻全部来自参数区
print('因子 %s，调仓 %d 期' % (res['factor'], len(res['ic'])))
'''

C_CS_PLOT = '''
figs = F.plot_single_all(res)      # 一次出全套图
print('共 %d 张，空图 %d 张' % (len(figs), len([1 for v in figs.values() if v is None])))
'''

MD_TS = '''
## 四、时序择时检验

信号在单只标的上的择时价值。判读只认两条：**夏普提升 + 回撤收窄**，
只看收益变高不算数。
'''

C_TS = '''
t = F.ts_run('pct_score', th=60)   # 信号名看 F.TS_SIGNALS
print(t['summary'] if 'summary' in t else 'ok')
'''

C_TS_PLOT = '''
F.plot_all(t, code=F.CODE, th=60)  # 价格+信号 / 净值+回撤+仓位
'''

MD_BARRA = '''
## 五、Barra 模式（给「不是风格因子」的因子用）

风格因子（size、value 这种）本身就该被风格解释，做中性化反而把自己的 alpha 洗掉。
**不是**风格因子的（事件、财务、量价形态）才该切过来，看剔除风格后还剩多少。

- `off`：不处理，跟以前一样
- `neutral`：因子对 8 个风格暴露做横截面回归，残差再跑 IC/分层 —— 回答「剔掉风格还剩多少」
- `attr`：收益对「因子 + 风格」回归 —— 回答「这个因子的收益有多少是风格给的」
- `both`：两个都做（慢一点，但一次看全）

本地只有价量，所以风格是 8 个**价量代理**（beta / size / momentum / resvol /
nlsize / liquidity / volat / reversal），没有 B/P、E/P、成长、杠杆这些财务风格。
'''

C_BARRA = '''
F.explain_barra()                  # 每个数字怎么看，一句话一条
# F.barra_switch('both'); res = F.cs_run('px_pct_500')
'''

MD_RSI = '''
## 六、RSI + FVG + ATR 筛股

三个指标各自打分，档位决定「谁必须命中」：

- 数字档 1~5：FVG → +RSI → +趋势 → +缩量 → +低波动（层层收紧）
- 特色档：`big_gap` 大缺口 / `fresh` 新缺口 / `pullback_vol` 回踩缩量 /
  `near_rsi` 差一点超卖 / `near_trend` 差一点多头 / `weak` 勉强的 /
  `gap_burst` 跳空异动 / `squeeze_wait` 波幅压死等爆发

表格 26 列不缩略；`top=` / `closest=` 只在你想截断时才用。
'''

C_RSI_ATR = '''
R.atr_levels_table()               # 日/周/月三套 ATR 阈值
df = R.load_bars('000300.XSHG', R._merge_cfg({'BARS': 250}))
s = R.atr_signal(df)
print('信号 %s  分位 %.1f%%  安静比 %.0f%%' % (s['信号'], s['ATR分位'], s['安静比'] * 100))
'''

C_RSI_SCAN = '''
# 先看每个档位各命中几只（一眼看出哪个档位对你现在的池子太严/太松）
ov = R.setup_overview(pool=['000300.XSHG', '510300.XSHG', '512480.XSHG'],
                      within=3, freq='D', bars=250)

# 再按某一个档位取明细（表格 26 列，不缩略）
picks = R.screen_stocks(pool=['000300.XSHG', '510300.XSHG', '512480.XSHG'],
                        tier=1, freq='D', bars=250)
print(picks.head(10).to_string(index=False) if len(picks) else '没命中，看上面那张表挑个松一点的档位')
'''

C_RSI_PLOT = '''
R.plot_setup(pool=['000300.XSHG', '510300.XSHG'], tier=1, max_charts=2)
# 单只看：R.plot_one('000300.XSHG')
'''

MD_TAIL = '''
## 七、速查

| 想干的事 | 怎么写 |
|---|---|
| 看现在什么口径 | `F.params_brief()` |
| 改前瞻/换手频率 | `F.apply_params(PERIOD=20, FREQ='M')` |
| 批量测一堆因子 | `F.cs_batch_run()` |
| 批量测一堆信号 | `F.ts_batch_run()` |
| 只测一段区间 | `F.apply_params(START='2020-01-01', END='2024-12-31')` |
| 换池子 | `F.apply_params(POOL=['HS300', 'ZZ500'])` |
| 出图存盘 | `F.apply_params(OUT_DIR='./pics')`，`plot_*` 会自动落盘 |
| 看某个函数干什么 | `F.describe('run_ts_test')` |
| 改代码 | 改 `.py` → 重跑 `build_libs.py` → 本页 Restart & Run All |

**两个库的分工**：`F` 管因子/时序/回测衔接，`R` 管 RSI/FVG/ATR 逐票筛股。
两边共用一份数据层（`jq_io`），所以 K 线只下载一次，缓存落在 `jq_data/`。
'''


def main():
    cells = [
        md(TITLE),
        code(C_IMPORT),
        code(C_USAGE),
        md(MD_PARAM), code(C_PARAM), code(C_PARAM_SET),
        md(MD_POOL), code(C_POOL),
        md(MD_CS), code(C_CS), code(C_CS_PLOT),
        md(MD_TS), code(C_TS), code(C_TS_PLOT),
        md(MD_BARRA), code(C_BARRA),
        md(MD_RSI), code(C_RSI_ATR), code(C_RSI_SCAN), code(C_RSI_PLOT),
        md(MD_TAIL),
    ]
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.13.8'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write('\n')
    n_code = len([c for c in cells if c['cell_type'] == 'code'])
    size = os.path.getsize(OUT) / 1024.0
    print('已生成 %s' % OUT)
    print('  %d 单元 / %d code cell / %.1f KB' % (len(cells), n_code, size))


if __name__ == '__main__':
    main()
