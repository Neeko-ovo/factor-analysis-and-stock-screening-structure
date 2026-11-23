# 量化研究框架 · Quant Workbench

一套 A 股因子研究框架，**两个跑法**：聚宽投资研究（推荐）与本地 Jupyter Lab。截面选股、时序择时、以及 RSI / FVG / ATR 形态筛股，三条线共用同一套检验口径。

- **首选环境：聚宽投资研究** —— 平台内取数最快，**三通道因子全开**（自定义因子 / jqfactor 因子库 / 财务表），不用自己缓存
- **离线备选：本地 Jupyter Lab + baostock** —— 免费、免注册，接口与 `jqdata` 同名同参，代码不用重写；只有自定义因子通道
- **检验**：Rank IC + Newey-West 修正 t + 分层单调性 + 换手 + 分档诊断，一条链路跑完
- **代码结构**：两个 `.py` 当库，一个 `.ipynb` 只负责 import、改参数、调用

> **作者**：Neeko-ovo · **首次公开发布**：2026-11-22
> 许可证 Apache-2.0（见 [LICENSE](LICENSE)）。再分发请保留 [NOTICE](NOTICE) 中的署名。

---

## 目录

- [能做什么](#能做什么)
- [特色](#特色)
- [长什么样](#长什么样)
- [项目结构](#项目结构)
- [安装（只在本地跑才需要）](#安装只在本地跑才需要)
- [快速开始](#快速开始)
- [一、数据源](#一数据源)
- [二、数据处理方法](#二数据处理方法)
- [三、运行函数用法](#三运行函数用法)
- [附一、注册自己的因子（三个扩展点）](#附一注册自己的因子三个扩展点)
- [四、Barra 风格中性化](#四barra-风格中性化)
- [五、扩展模块](#五扩展模块)
- [六、RSI · FVG · ATR 筛股](#六rsi--fvg--atr-筛股)
- [七、口径与坑](#七口径与坑)

---

## 能做什么

| 模块 | 入口 | 一句话 |
|---|---|---|
| 截面因子检验 | `run_factor_test` / `run_batch` / `run_combo` | 单因子、多因子、多方案组合，出 IC / 分层 / 换手 / 分档 |
| 持仓期扫描 | `scan_ic_decay` / `scan_freq_period` | 5~250 日逐档看 IC 衰减，定「该拿多久、多久调一次」 |
| 时序择时检验 | `run_ts_test` / `run_ts_batch` / `ts_ic_decay` | 单信号对一只标的，出 IC、状态分组、择时回测 |
| 风格中性化 | `set_barra_mode` / `barra_analyze` | 把因子里的 beta / size / 动量剔除，看剩下的是不是真 alpha |
| 分位打分因子 | `pct_rank` / `pct_to_score` / `px_pct_*` | 价格在历史区间的位置 → 0~100 分（越高越便宜） |
| IC 专项 | `screen_ic` / `ic_tables` / `ic_report` | 只看 IC 那一面的一页纸报告 |
| 仓位管理 | `atr_squeeze` / `bottom_score_panel` / `weight_state` | 底部打分 + 仓位状态机 + 连续仓位回测 |
| RSI 筛股 | `screen_rsi` / `screen_stocks` | 超买超卖 + FVG 缺口 + ATR 波动，16 个档位 |
| 出图 | `plot_single_all` / `plot_*` / `plot_setup` | 静态 matplotlib 与可交互 plotly 双引擎 |

---

## 特色

**1. 检验链路是一条完整的流水线，不是零散工具**
拿到一个因子，从 IC 到能不能用，中间那串问题一次答完：IC 有多强、显著不显著、分层单不单调、换手吃掉多少、该拿多久、多久调一次仓、方向是动量还是反转。这些步骤共用同一份面板和同一个前瞻收益，口径不会前后打架。

**2. 重叠窗口的 t 值修正是默认开启的**
持仓 20 日、每月换仓，评估窗口重叠 20/21 倍，原始 t 值虚高 4~5 倍——这是因子研究里最常见的假显著来源。框架直接输出 `t值_修正`，时序那条线用 Newey-West（Bartlett 权重、滞后阶数随样本自适应）。判显著只认修正后的。

**3. 能用价量自建 Barra 风格，判断「真 alpha 还是风格的替身」**
八个价量代理风格（beta / size / momentum / resvol / nlsize / liquidity / volat / reversal），`neutral` 事前中性化、`attr` 事后归因、`both` 都做。实测 `mom_20` 的 Reversal 载荷 1.004，一眼看出它是反转的另一种写法。

**4. 复权口径从源头就做对**
缓存里存的是**不复权价 + 复权因子**，不存前复权价。新除权不会污染历史，去年跑的因子结果今年还能对上。这一点在长时间的因子研究里是刚需，但很多框架图省事直接存 qfq。

**5. 档位 + 指标插件：筛选逻辑是可组合的数据结构，不是硬编码**
16 个档位就是 16 个 dict（`use / need / near / any_near / where / min_score`），指标用 `@register_indicator` 注册。想加「缺口 + 缩量 + 低波动」这种组合，五行代码，不用改框架。`near` 那一层专门捞「差一点」的票——趋势快多头、RSI 快进超卖，这类常常比「已经到位」的更有意思。

**6. 形态那一条线（FVG + 斐波那契 + ATR）是别人少见的**
FVG 缺口框 + 61.8% 回踩线，框上直接标「缺口多大、线在哪个价」；ATR 日/周/月三套档位（W/M 按 √5、√21 自动放大）；异动（TR 猛冲 ATR）与收缩（波幅压死）两层状态信号，TR% 与 ATR% 两条线叠画，一眼看出哪根是爆发。图上标记和表格数字在**全历史口径**下对齐，不会出现「图上 11 个圈、表里说 1 个」。

**7. 数据层和平台解耦**
`jq_io.py` 接口与聚宽同名同参，baostock / jqdata / jqdatasdk 三个源一套代码。要搬回聚宽环境，改 `JQ_DATA_SOURCE` 就够，上层一行不动。

**8. 参数集中 + 自带说明**
25 个参数一处改，`PARAM_DOC` 逐条校验（写错直接报错，不静默用默认值）；`explain_metrics()` / `explain_barra()` 把每个数字「怎么读」打出来，不用去翻代码猜口径。

**9. 双引擎出图**
`mpl` 静态（批量出图快、论文风格）与 `plotly` 交互（框选放大、悬停看 OHLC、周末空档自动抹掉）随时切。两种引擎共用同一份计算，不是两套逻辑。

**10. 离线可验**
17 套 mock 自测 + 5 个 notebook 全 cell 空跑，不联网也能确认改动没把别处弄坏。

---

## 长什么样

一张图里有四层：K 线 + 均线、成交量、RSI（70/30 阈值与超买超卖带）、ATR 副图（TR% 与 ATR% 两条线叠画，异动标橙色三角、收缩标青色空心圈）。FVG 框上直接写缺口大小与 61.8% 回踩价，页脚汇总缺口区间和差价。

![K 线 / FVG / RSI / ATR](docs/img/kline_fvg_atr.png)

按档位筛股：完整表格在前（不缩略，26 列），符合档位的股票逐个出图在后，缺口框与回踩线都标了价。

![按档位筛股](docs/img/setup_screen.png)

---

## 项目结构

```
.
├── 量化工作台.ipynb          # ★ 运行端：只 import、改参数、调用（21 单元 / 13 code）
├── factor_lib.py            # ★ 代码端：因子框架（十源合并，8151 行 / 355 KB）
├── rsi_multiday.py          # ★ 代码端：RSI + FVG + ATR 筛股与画图
├── jq_factor_framework.py   # ★ 聚宽平台单文件版（v4.5.0，八个 Part，上传即用）
│
├── jq_io.py                 # 数据层源码（baostock + 本地缓存，接口对齐聚宽）
├── jq_params.py             # 参数区源码（PARAM_DOC 校验 + 统一入口）
├── jq_unified_factor_framework.py   # Part A~D 通用框架
├── jq_factor_viz.py         # 附录 A 截面可视化
├── jq_ts_viz.py             # 附录 B 时序可视化
├── jq_tsa_patch.py          # 附录 C 时序有效性补丁
├── jq_pct_factor.py         # Part F 历史分位打分
├── jq_ic_report.py          # Part G IC 视角
├── jq_position.py           # Part H 仓位管理
├── jq_rsi.py                # Part I RSI 体系
│
├── build_libs.py            # 十个源文件 → factor_lib.py
├── build_workbench.py       # 生成 量化工作台.ipynb
├── build_module.py          # 生成聚宽平台用的单文件 jq_factor_framework.py
│
├── selftest_*.py            # 17 套离线自测（mock 数据，不联网）
├── exec_*_mock.py           # 5 个 notebook 全 cell 空跑回归
├── check_readme_examples.py # 编译并实跑本文档里的示例代码（改文档后跑它）
├── docs/img/                # README 用的截图
└── jq_data/                 # 数据缓存（自动生成，不进仓库）
    ├── daily/<code>.pkl
    └── misc/
```

**`factor_lib.py` 是十个源文件合并出来的**（`build_libs.py`）。改功能改 `.py` 源文件，然后 `python build_libs.py` 重新合并——不要直接改 `factor_lib.py`，会被覆盖。

合并顺序有依赖关系，不要调整：`jq_io` → `jq_unified_factor_framework` → `jq_factor_viz` → `jq_ts_viz` → `jq_tsa_patch` → `jq_pct_factor` → `jq_ic_report` → `jq_position` → `jq_rsi` → `jq_params`（参数区**必须放最后**，它的赋值要覆盖框架里的默认值）。

---

## 安装（只在本地跑才需要）

聚宽投资研究里什么都不用装——平台自带 pandas / numpy / matplotlib / jqfactor。本地跑才要：

```bash
pip install pandas numpy matplotlib baostock
pip install plotly        # 可选：要交互图才装
```

- Python 3.9+（开发环境 3.13.8），pandas 2.x
- **设计时就兼容 pandas 0.25 / Python 3.6**：不用 `DatetimeIndex.isocalendar`、`DataFrame.append`、`iteritems`、`resample('ME')`，所以能直接搬到聚宽那个老环境里跑
- plotly 导 png 需要 `kaleido`；没装也能用，存图会退回 HTML

---

## 快速开始

### 方式 A：聚宽投资研究（推荐）

上传或粘贴 `jq_factor_framework.py`（单文件，v4.5.0，八个 Part 全在里面），然后：

```python
import jq_factor_framework as jf

# 1) 先看一眼池子
jf.preview_pool('HS300', n=10)

# 2) 截面：单因子检验（参数直接写在调用里）
r = jf.run_factor_test('px_pct_750', 'HS300', '2019-01-01', '2024-12-31', period=20, freq='M')
jf.plot_single_all(r)

# 3) 时序：单信号择时
tr = jf.run_ts_test('ma200', '000300.XSHG', '2019-01-01', '2024-12-31',
                    period=20, th=0.5, decision_freq='W')
jf.plot_all(tr)
```

平台里三通道都能走，直接拿财务字段或因子库因子，不用注册：

```python
jf.run_factor_test('turnover_ratio', 'HS300', '2019-01-01', '2024-12-31')  # 财务表字段
jf.run_factor_test('beta',           'HS300', '2019-01-01', '2024-12-31')  # jqfactor 因子库
```

> 聚宽版是八个 Part 的合并版，**不含参数区与 `cs_run` 这类统一入口**（那是本地 `factor_lib.py` 才有的）。平台上参数直接写在调用里，反而更直观。

### 方式 B：本地 Jupyter Lab

打开 `量化工作台.ipynb`，从上往下跑：

```python
import factor_lib as F     # 因子框架（含 baostock 数据层 + 参数区）
import rsi_multiday as R   # RSI / FVG / ATR

# 1) 先看一眼池子里有哪些票
F.preview_pool('HS300', n=10)

# 2) 截面：单因子检验
r = F.cs_run('px_pct_750')          # 参数全从参数区来
F.plot_single_all(r)

# 3) 时序：单信号择时
tr = F.ts_run('pct_score', th=60)   # 阈值 60 分以上才持有
F.plot_all(tr)

# 4) 筛股：按档位
sheets = R.setup_overview(pool='HS300', bars=300)
picks  = R.screen_stocks(pool='HS300', tier=2, top=20)
R.plot_setup(pool='HS300', tier='big_gap', max_charts=6)
```

本地第一次跑会去 baostock 拉数据，HS300 全量约 12 分钟；之后走缓存，单只 0.001 秒。

---

## 一、数据源

### 1.0 先说结论：主力用聚宽，本地当备胎

| | 聚宽投资研究（推荐） | 本地 Jupyter + baostock |
|---|---|---|
| 取数速度 | 平台内取数，全池几秒到手 | 首次 HS300 约 12 分钟，之后走缓存 0.001 秒/只 |
| 因子通道 | **三通道全开**：自定义 / jqfactor 因子库 / 财务表（valuation·balance·cash_flow·income·indicator） | 只有自定义通道 |
| 要装什么 | 什么都不用装 | pip 装 baostock 等 |
| 环境版本 | pandas 0.25 / Python 3.6（老） | pandas 2.x / Python 3.13 |
| 适合 | 正式研究、批量跑、要用财务与因子库 | 随时随地验证想法、离线、不想依赖平台 |

两条路的上层代码是同一份，差别只在数据层。所以在哪边写都行，结论可以互相对照——**推荐先在聚宽里跑，快而且因子库全**。

### 1.1 四个源的分工

| 源 | 历史成分股 | 复权因子 | 稳定性 | 成本 | 定位 |
|---|---|---|---|---|---|
| **聚宽 `jqdata`**（平台内） | 有 | 有 | 最好 | 随平台 | **主推荐**，三通道全开 |
| **baostock**（本地默认） | 有（HS300/ZZ500/SZ50 可按日期取） | 有（`adjustflag` 1后复权/2前复权/3不复权） | 好，多年没变 | 免费、免注册 | 本地研究的主力 |
| jqdatasdk（聚宽本地版） | 有 | 有 | 好 | 需数据权限 | 想在本机沿用聚宽口径时用 |
| tushare pro | 有（`index_weight`） | 有（`adj_factor`） | 好 | 积分制 | 要市值/换手/估值时补 |
| akshare | 基本没有 | 只有算好的前复权价 | 差，接口常改 | 免费 | 只当补充，不做主语料 |

`jq_io.py` 的接口与聚宽**同名同参**，所以框架代码一行不用改：

```python
from jq_io import *
get_price_range(['000300.XSHG'], '2019-01-01', '2025-06-30')
get_index_stocks('000300.XSHG', date='2019-01-02')
get_trade_days('2019-01-01', '2025-06-30')
```

切源用环境变量，框架启动时会自动挑（顺序 jqdata → jq_io → jqdatasdk）：

```bash
set JQ_DATA_SOURCE=local     # auto / jq / local / sdk
```

### 1.2 缓存长什么样

```
jq_data/
├── daily/600000.XSHG.pkl     # 一只一个文件
├── daily/000300.XSHG.pkl
├── misc/calendar.pkl         # 交易日历
├── misc/index_stocks.pkl     # 成分股快照
└── misc/security_basic.pkl   # 证券基本信息
```

单只文件的列：

| 列 | 含义 |
|---|---|
| `open high low close pre_close` | **不复权**原始价 |
| `volume` / `money` | 成交量（股）/ 成交额（元） |
| `turn` / `pct_chg` | 换手率(%) / 涨跌幅(%) |
| `paused` | 停牌标志（1=停牌） |
| `is_st` | ST 标志 |
| `adj` | 复权因子 = 后复权收盘 / 不复权收盘 |

**为什么存不复权价 + 复权因子，而不是直接存前复权价**：前复权价会随每次新除权变化，今年存的 qfq 价明年就和原来的对不上了，历史因子结果全部漂移。存原始价 + 因子，用时现算：

```
前复权价 = 不复权价 × adj / 最新 adj
后复权价 = 不复权价 × adj
```

### 1.3 baostock 能力边界

**能拿**：日/周/月 K 线、分钟线（5/15/30/60）、除权除息（`query_dividend_data`）、官方复权因子（`query_adjust_factor`）、六张财务季报表（利润/营运/成长/资产负债/现金流/杜邦）、业绩预告与快报、申万行业（`query_stock_industry`）、全市场列表（`query_all_stock`）。

**拿不到**：聚宽现成的因子库（jqfactor）、`get_fundamentals` 那种每日横截面市值/PE/PB、逐笔与龙虎榜。所以 `get_fundamentals()` 在本地会直接抛错，并提示改用 `@register_factor` 自己写。

**注意**：`query_zz1000_stocks` / `query_zz800_stocks` 不存在。ZZ800 = HS300 ∪ ZZ500 拼出来；ZZ1000 与 CYB 在本地会明确报错，不静默返回空。

### 1.4 限速

baostock 客户端是全局单例，**不能并发**（多线程会串数据且最容易触发限流）。`jq_io` 的做法：两次查询最小间隔 0.08 秒，失败重试 3 次（1/3/8 秒递增），每次失败后重连。

实测：单只 5 年全历史 2.41 秒，HS300 全量首次约 12 分钟，之后读缓存 0.001 秒。

```python
from jq_io import warm_cache, upgrade_cache
warm_cache(['000300.XSHG', '510300.XSHG'], '2020-01-01')  # 批量预热，每 50 只打进度
upgrade_cache(codes)      # 老缓存补 turn / pct_chg 两列（Barra 的 size 要用）
```

---

## 二、数据处理方法

整条链路：**价格 → 复权 → 宽表 → 面板 → 前瞻收益 → IC → 分层 → 换手 → 分档 → 衰减 → 中性化**。

### 2.1 复权与停牌

- 复权在上游做一次（`_apply_fq`），下游全部用复权后的价，不再回头算
- 停牌日按聚宽口径：**价格前值填充、成交量与成交额填 0**
- 停牌股不剔除，但成交额 0 会让换手率类因子自动变低，符合直觉

### 2.2 宽表：全框架唯一的数据形态

所有价格数据统一成 `dict{字段名: DataFrame}`，`index = 日期`、`columns = 代码`：

```python
px = get_price_range(codes, '2019-01-01', '2025-06-30', fields=('close', 'money'))
px['close']      # DataFrame，行是日期，列是代码
```

不传 `panel=False` 拿到的长表（`time`/`code` 两列那种）也能被自动识别并转成宽表——`_to_wide()` 兼容「长表 / MultiIndex / 宽表」三种结构。

### 2.3 调仓日

```python
get_rebalance_dates(start, end, freq='M', rule='first', every=1)
```

`freq` = `D/W/M/Q/Y`，`rule` = 每期取首个（`first`）或最后一个（`last`）交易日，`every` 是间隔倍数（`freq='M', every=2` → 双月）。

调仓频率与持仓期**解耦**：`freq='M'` 每月换仓，`period=20` 是每次持有 20 个交易日。这是两件事，别混。

### 2.4 前瞻收益

```python
fwd = close.shift(-period) / close - 1.0                 # 默认
fwd = close.shift(-(period + 1)) / close.shift(-1) - 1.0 # strict=True
```

`strict=True` 表示从 t+1 起算——**当天收盘看到的因子，最早也只能在次日开盘动**。做严谨结论时应当打开。

### 2.5 Rank IC 与显著性

逐期算 Spearman 秩相关（因子排名 vs 未来收益排名），再对 IC 序列做统计：

```
IC均值 / IC标准差 / IC_IR = 均值/标准差 / t = 均值 / (标准差/√n)
```

**重叠窗口会让 t 虚高**。持仓 20 日、每月换仓时，评估窗口重叠约 20/21 倍，原始 t 会把显著性放大 4~5 倍。所以 `factor_ic_stats` 输出两列：

```
t值        = mean / (std / √n)
t值_修正   = t / √重叠倍数        ← 显著性以这一列为准
```

`overlap_ratio > 1.5` 时只看 `t值_修正`。时序那条线用的是更强的 Newey-West 修正（Bartlett 权重，滞后阶数 `max(4·(n/100)^(2/9), 前瞻期)`），同时修掉自相关与重叠窗口。

### 2.6 分层与换手

```python
quantile_returns(panel, fwd, q=5)   # 每期按因子分 q 组，各组等权平均收益
quantile_turnover(panel, q=5)       # 最高分组的成分更换比例
```

- 看**单调性**：Q1→Q5 是不是逐级抬升。中间跳一下不叫单调
- 多空价差 = `Q(q) - Q1`，折年 = `价差 × 244 / period`
- 换手率是成本代理：每期换掉 80% 的票，扣完成本就没剩多少了

### 2.7 分档诊断：动量还是反转

离散因子（打分、评分、信号强度）适合按原始分值分档，看每档相对收益：

```python
score_bucket_analysis(panel, fwd)     # 已剔除大盘涨跌
diagnose_momentum_reversal(bt, ic)    # 自动判「越强越好」还是「越弱越好」
```

这一步能识破方向搞反的因子——IC 是负的，但你可能一直在按「分数越高越买」用。

### 2.8 缺失、极值与样本量

- 每期样本数 `< min_n`（默认 3）不出 IC，避免两三只票算出来的相关系数污染序列
- Barra 回归两端各砍 1% 极值（`BARRA_WINSOR=0.01`），防单只妖股带偏系数
- 因子在横截面上做标准化后再进回归，载荷 ≈ 相关系数，好读

---

## 三、运行函数用法

### 3.1 参数区：只改这一处

`factor_lib.py` 末尾（源码是 `jq_params.py`）集中了 25 个参数，改这里全框架跟着走：

| 组 | 参数 | 说明 |
|---|---|---|
| 标的与区间 | `CODE` `POOL` `START` `END` | 时序标的 / 截面池子 / 起止日期 |
| 检验口径 | `PERIOD` `FREQ` `DEC_FREQ` `TH` `COST` `DIRECTION` `STRICT` | 前瞻期、调仓频率、阈值、成本、方向 |
| 截面专用 | `Q` `RULE` `EVERY` `NON_OVERLAP` | 分层数、取首末、间隔倍数、非重叠 |
| 批量清单 | `FACTORS` `SIGNALS` `CS_PERIODS` `TS_PERIODS` | 跑哪些因子 / 信号 / 衰减期 |
| 输出 | `PLOT_ENGINE` `OUT_DIR` `DPI` | `mpl` 静态图 / `plotly` 交互图 |
| Barra | `BARRA_MODE` `BARRA_STYLES` `BARRA_WINSOR` | 见第四节 |

两种改法：

```python
# ① 直接改模块里的赋值，然后重跑这一格
F.PERIOD = 60

# ② 临时改，校验通过才生效
F.apply_params(PERIOD=60, TH=60.0)
F.params_table()    # 打一张表，看当前全部取值
F.params_brief()    # 一行摘要
F.check_params()    # 只校验不改，非法值直接报错
```

`apply_params` 按 `PARAM_DOC` 的 25 条规则校验（`freq` 只认 D/W/M/Q/Y、`frac` 必须 0≤v<0.5、`barra` 只认四种模式…），改错的东西进不来。

### 3.2 统一入口：不用手写日期

```python
F.cs_run('px_pct_750')                    # = run_factor_test + 参数区
F.cs_batch_run()                          # 跑 FACTORS 里全部
F.cs_decay_run('px_pct_750')              # IC 衰减扫描
F.ts_run('pct_score', th=60)              # 时序
F.ts_batch_run()                          # 跑 SIGNALS 里全部
F.ts_decay_run('mom_120')
F.plot_engine('plotly')                   # 切画图引擎（会写回参数区）
```

### 3.3 `run_factor_test` 全签名

```python
run_factor_test(factor, pool, start, end,
                period=20, freq='M', rule='first', every=1,
                rebal_dates=None, strict=False, q=5, non_overlap=False,
                verbose=True, barra=None)
```

| 参数 | 含义 |
|---|---|
| `factor` | 因子名（三通道：自定义 / 因子库 / 财务字段） |
| `pool` | `str` = 指数池（动态成分）；`list` = 固定池（ETF、自选股） |
| `period` | 持仓期（交易日）。财务因子 60+，量价 20，ETF 短信号 5 |
| `freq` / `rule` / `every` | 调仓频率 / 取首末 / 间隔倍数 |
| `rebal_dates` | 直接给调仓日列表，覆盖 `freq/rule/every` |
| `strict` | 前瞻收益从 t+1 起算 |
| `q` | 期望分层组数（池子小会自动降组） |
| `non_overlap` | 抽稀调仓日使评估窗口不重叠（长线建议开） |
| `barra` | `None` 跟随全局；`neutral`/`attr`/`both`/`off` 临时切 |

返回 `dict`：

```python
{'factor', 'factor_panel', 'factor_panel_raw', 'fwd_panel', 'ic', 'ic_stats',
 'quantile_returns', 'turnover', 'buckets', 'mono',
 'rebal_dates', 'interval', 'ratio', 'barra'}
```

### 3.4 其余截面函数

```python
run_batch(factors, pool, start, end, ...)      # 多因子同口径，共享面板，出汇总表
run_combo(schemes, pool, start, end, ...)      # 多方案组合（等权/加权/等分位加权），带样本外切分
scan_ic_decay(factor, pool, start, end, periods=(5,10,20,63,126,250))
scan_freq_period(factor, pool, start, end,
                 freqs=('W','M','Q'), periods=(5,10,20,40,60,120,250))
```

`scan_ic_decay` 解决「这个因子该持有多久」：每个前瞻期各跑一遍 IC，找 |IC| 最大的那档。长周期因子在第 5 日往往还没反应完，短周期因子到 60 日就衰减没了。

`scan_freq_period` 是二维的：调仓频率 × 持仓期，一次取价算所有组合，出 IC / IC_IR / 换手 / 成本后的净收益矩阵，配合 `plot_freq_period(df, metric='IC_IR')` 看热图。

### 3.5 `run_ts_test` 全签名

```python
run_ts_test(signal_name, code, start, end,
            period=20, th=0.0, decision_freq='D', cost=0.0005,
            direction=1, params=None, strict=False,
            state_thresholds=None, verbose=True)
```

| 参数 | 含义 |
|---|---|
| `signal_name` | 已注册的时序信号（`@register_ts_signal`） |
| `th` | `signal > th` 持有，否则空仓 |
| `decision_freq` | `D/W/M`，多久看一次信号 |
| `cost` | 每次仓位切换成本（单边） |
| `direction` | `1` 正向；`-1` 翻转（波动率、换手率这类） |
| `params` | 信号构造参数，如 `{'n': 200}` |
| `state_thresholds` | 状态分组用固定阈值，如 RSI 用 `[30, 70]`；`None` = 按分位切 |

返回：`signal / px / close / fwd / ic_stats / ic_segments / state_table / state_spread / backtest / detail`

内置 7 个信号：`ma200` `ma60` `mom_120` `dev_ma250` `volz_20_250` `px_pos_750` `trend_ma(params={'n':N})`。

自定义信号：

```python
@F.register_ts_signal('my_signal')
def my_signal(code, px, params=None):
    """px: index=日期，columns=close/high/low/money"""
    return px['close'] / px['close'].rolling(60).mean() - 1.0
```

返回 `Series(index=日期)` 即可，带不带 `params` 形参都兼容。

### 3.6 自定义截面因子

写法是 `fn(stock_list, date) -> Series(index=代码)`，完整示例、三通道优先级、注册后怎么组档位，都在 [附一、注册自己的因子](#附一注册自己的因子三个扩展点)。

本地环境下**只有自定义这一条通道能走**（没有 jqfactor，也没有财务横截面），写别的因子名会明确报错而不是静默返回空。

### 3.7 怎么看结果

框架自带 `explain_metrics()`，一条条讲每个数字怎么读。核心口径：

| 指标 | 门槛 | 说明 |
|---|---|---|
| Rank IC 均值 | \|IC\| > 0.05 | 低于这个基本没信息量 |
| t 值 | \|t\| ≥ 2 | 重叠窗口多时看 `t值_修正` |
| IC_IR | > 0.3 | IC 的稳定性，比 IC 均值更抗噪 |
| IC>0 占比 | 明显 > 50% | 接近 50% 就是方向随机 |
| 分层价差 | 看单调性 | 中间跳一下不算单调 |
| 择时价值 | **夏普提升 + 回撤收窄** | 只有收益提升不算数 |

---

## 附一、注册自己的因子（三个扩展点）

框架里所有可扩展的地方都是「注册一个函数」：不改源码，注册完就能按名字被调用。一共三处，分别对应**截面因子**、**时序信号**、**筛股指标**。

写之前先记住四条共同原则：

- **只返回值，不打印**。框架会在几百只 × 几十期里反复调你的函数，打印刷屏又费时
- **数据不够就返回 NaN / 空 Series**，不要抛异常，否则整期都会断
- **只用截止到 `date` 的数据**，别拿全历史切片往未来看（未来函数是回测第一大坑）
- **名字带周期后缀**（`_20` / `_60`），`run_batch` 批量对比时一眼能分清代数

### 附一.1 截面因子：`@register_factor`

签名是 `fn(stock_list, date)`，返回 `Series(index=股票代码)`（返回标量也行，会被广播）。

```python
@F.register_factor('rev_20')
def rev_20(stock_list, date):
    """20 日反转：跌得多的排前面（方向 +1 就是买跌）"""
    close = F.get_price_history(stock_list, date, 21, ['close']).get('close')
    if close is None or close.empty or len(close) < 3:
        return pd.Series(np.nan, index=list(stock_list), name='rev_20')
    s = (close.iloc[0] / close.iloc[-1] - 1.0).reindex(stock_list)   # 跌得多 → 值大
    s.name = 'rev_20'
    return s

F.run_factor_test('rev_20', 'HS300', '2019-01-01', '2025-06-30', period=20, freq='M')
```

`get_price_history(stock_list, date, count, fields)` 返回的是**宽表**（index=日期，columns=代码），所以能直接拿 `close.iloc[-1]` 做横截面运算。

**三通道优先级**（`factor_lib.py:1012` `get_factor_data`）：自定义因子 > jqfactor 因子库 > 财务表，重名时高优先级静默覆盖。所以：

| 环境 | 能用的因子名 |
|---|---|
| 聚宽平台 | 自己的 `@register_factor` + 因子库名（如 `beta` `momentum`）+ 财务字段（如 `turnover_ratio` `pb_ratio`） |
| 本地 baostock | **只有**自己注册的；写别的名字会明确 `ValueError` 而不是静默返回空 |

### 附一.2 时序信号：`@register_ts_signal`

签名是 `fn(code, px, params=None)`，返回 `Series(index=日期)`，数值是信号强度（用什么阈值去卡由 `run_ts_test` 的 `th` 决定）。

`px` 是这一只标的的全历史价格表：index=日期，columns 至少含 `close / high / low / money`。

```python
@F.register_ts_signal('boll_pos_20')
def boll_pos_20(code, px, params=None):
    """收盘价在布林带里的位置：0=下轨，1=上轨"""
    n = int((params or {}).get('n', 20))
    c = px['close']
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    return (c - (mid - 2 * sd)) / (4 * sd).replace(0.0, np.nan)

F.run_ts_test('boll_pos_20', '000300.XSHG', '2019-01-01', '2025-06-30',
              th=0.8, decision_freq='W', params={'n': 20})
```

`params` 是自己传的那本字典（`params={'n': 20}`），带不带这个形参都兼容——不带就直接忽略。

### 附一.3 筛股指标：`@register_indicator`

签名是 `fn(df, cfg)`，`df` 是合成好的日/周/月 OHLCV，返回固定四键：

```python
{'hit': True,    # 是否命中（决定能不能进档位）
 'score': 1.0,   # 命中得几分，用来在表里排序
 'near': False,  # 是否"差一点"（进「特征」列，可选档位用得到）
 'info': {...}}  # 想显示在表格里的字段，键名就是列名
```

```python
@R.register_indicator('bw_20', '布林带宽（压死=蓄势）')
def _ind_bw(df, cfg):
    n = int(cfg.get('BW_N', 20))
    c = df['close']
    bw = (c.rolling(n).std() * 4 / c.rolling(n).mean() * 100.0).dropna()
    if bw.empty:
        return {'hit': False, 'score': 0.0, 'near': False, 'info': {'带宽%': None}}
    p = float(bw.iloc[-1])
    q = float((bw <= p).mean() * 100.0)        # 当前带宽在历史里的分位 0~100
    lo = float(cfg.get('BW_LOW', 20.0))
    return {'hit': q <= lo,
            'score': 1.0 if q <= lo else 0.0,
            'near': lo < q <= lo + 10.0,
            'info': {'带宽%': round(p, 2), '带宽分位': round(q, 1)}}
```

注意 `cfg.get('BW_N', 20)` 这种写法：**新参数要同时加进 `rsi_multiday.DEFAULTS`**，否则 `_merge_cfg` 会把它静默丢掉。

注册完直接用名字组档位（`use` 打分的、`need` 必须命中的、`near` 必须差一点的）：

```python
R.SETUPS['bw_squeeze'] = {
    'name': '布林挤压 × FVG',
    'desc': '缺口回踩 + 带宽压到历史低位',
    'use':  ['fvg', 'bw_20'],
    'need': ['fvg', 'bw_20'],
}
R.screen_stocks(pool='HS300', tier='bw_squeeze', top=20)
```

`info` 里的键会自动出现在结果表里；`分数` 是 `use` 里各指标得分之和；`特征` 列收的是 `near` 为真的那些指标。

---

## 四、Barra 风格中性化

用来回答一个问题：**这个因子的 IC，是它自己的，还是从某个风格那儿借来的？**

```python
F.set_barra_mode('both')          # 开
F.run_factor_test('mom_20', 'HS300', '2021-01-01', '2022-12-31', barra='both')
F.explain_barra()                 # 阈值怎么读
```

| 模式 | 做什么 |
|---|---|
| `off` | 不启用，跟以前完全一样（默认） |
| `neutral` | 每期把因子对风格回归取残差，再用残差跑 IC / 分层 |
| `attr` | 因子不动，把收益对「因子 + 风格」回归，看控制风格后还有没有 alpha |
| `both` | 两个都做（慢一点，但一次看全） |

8 个**价量代理风格**：`beta` / `size` / `momentum` / `resvol` / `nlsize` / `liquidity` / `volat` / `reversal`。

> `size` 是用换手率反推的流通股本（`turn = 成交量/流通股本×100` → `流通股本 = money/close×100/turn`，取 21 日中位数），是近似值。

怎么读：

- **载荷**：`|载荷| < 0.3` 关系不大；`> 0.5` 基本就是这个风格换了个写法
- **中性化后 IC**：明显变小 → 原来那部分是风格给的；基本没变 → 因子干净
- **归因 λf**：`t ≥ 2` → 控制风格后还有效，alpha 是自己的
- **风格 R²**：`> 0.3` 要留意，`> 0.5` 基本可以认定它不是独立 alpha

实测例子：`mom_20` 对 HS300 前 25 只、2021–2022、`barra='both'` —— IC 从 −0.0261 变成 −0.0191（掉 27%），载荷表里 Reversal 载荷 **1.004**，报告直接标出「你的因子很大程度上就是 Reversal 的另一种写法」。`mom_20` 本来就是 20 日反转，这个结果是对的。

**已知限制**：没有财务类风格（B/P、E/P、成长、杠杆），也没有行业哑变量。这是自己搭的风格代理，不是官方 Barra 模型。

---

## 五、扩展模块

### 5.1 分位打分因子（Part F）

价格在过去 750 日区间的位置 → 0~100 分，越高越便宜：

```python
F.pct_rank(x, window=750)           # 分位 0~100
F.pct_to_score(pct, mode='linear')  # 分位 → 分数（可分段映射 edges/scores）
```

内置三个截面因子（用名字调用）：`px_pct_750`（价格 750 日分位）、`px_pct_dev250`（对 250 日均线偏离的分位）、`px_pct_score`（合成打分）。方向都是 +1：分数越高 = 越便宜。

```python
F.run_factor_test('px_pct_750', 'HS300', '2019-01-01', '2025-06-30')
```

时序版是 `pct_score`，配合 `run_ts_test(..., th=60)` 用：「便宜到 60 分以上才持有」。

> 名字容易混：`px_pct_*` 是**截面因子**（供 `run_factor_test`），`pct_*` 是**时序信号**（供 `run_ts_test`）。拿混了会看到「每期约 0 个不同取值」和一排空图。

### 5.2 IC 视角（Part G）

```python
F.screen_ic(df)                   # 按 IC 口径过一遍 run_batch 汇总表，做初筛
F.ic_tables(r)                    # 年度 IC / 滚动 IC / 分段表
F.ic_plots(r)                     # 只画 IC 那几张图
F.ic_report(r)                    # 一页纸报告
F.ts_ic_table(r)                  # 时序版（口径不同，单独给）
```

### 5.3 仓位管理（Part H）

基础算子：

```python
F.calc_tr(high, low, close)            # 真实波幅（价格单位）
F.calc_atr(close, high, low, n=20)     # ATR
F.atr_pct(close, high, low, n=20)      # ATR 占价格 %
F.atr_stop_line(close, high, n=22, k=3.0)   # ATR 追踪止损线
F.cheap_score_series(close, window=750)     # 便宜度打分（时序版）
```

现成的时序信号（直接给 `run_ts_test` 用）：`atr_pct_20`、`atr_rank_20`、`atr_squeeze`、`vol_ratio`、`bw_rank`、`rsi_14`、`mom_60`。

```python
F.run_ts_test('atr_squeeze', '000300.XSHG', '2019-01-01', '2025-06-30', th=0.5)
F.run_ts_test('vol_ratio',   '000300.XSHG', ..., direction=-1)   # 量能萎缩是好事
```

截面与状态机：

```python
F.bottom_score_panel(pool='HS300', start=..., end=...)  # 底部区域打分面板
F.bottom_candidates(pool='HS300', top=5, min_score=60)   # 直接出候选
F.weight_state(cheap, atr_rank, close, ma, stop)         # 仓位状态机 0~1
```

### 5.4 RSI 体系（Part I）

```python
F.calc_rsi(close, n=14, method='wilder')   # Wilder 平滑（也可 'ema'/'sma'）
F.rsi_zone(rsi, ob=75, os_=25)             # 超买 / 超卖 / 中性（这一支默认 75/25）
F.rsi_reversal_state(rsi, ob=75, os_=25, mode='long', confirm=0)  # 反转状态判定
F.rsi_report('000300.XSHG')                # 一只票的 RSI 报告
F.plot_rsi(rsi, ...)                       # 单看 RSI 那张图
```

> 两条线的默认阈值不一样：**因子框架这一支（Part I）是 75/25**，**`rsi_multiday` 那一支是 70/30**。两边都能改，但别指望改一处两边同步。

---

## 六、RSI · FVG · ATR 筛股

`rsi_multiday.py` 是独立的一支：K 线 + RSI + **公允价值缺口（FVG）+ 斐波那契 61.8% 回踩线** + **ATR 波动档位**，最后用「档位 + 指标插件」组合筛选。

### 6.1 一个 61.8% 回踩信号是怎么算出来的

1. **找缺口**：连续三根 K 线，`低[i] > 高[i-2]` → 多头缺口 `[高[i-2], 低[i]]`；反向为空头缺口。中间那根必须有实体（`FVG_MIN_BODY`），防止小阳小阴蹭出来的假缺口
2. **定回踩线**：`回踩线 = 上沿 - 0.618 × 缺口高度`（空头是 `下沿 + 0.618 × 高度`）
3. **判信号**：最近 `FVG_WITHIN` 根内，某根 `low <= 回踩线`，且收盘回到线上（`FVG_CONFIRM='close'`），且缺口没被完全回补
4. **画图**：缺口画半透明框，两行标签 `FVG↑ 1.800 (+15.3%)` / `61.8% 9.888` —— 缺口多大、线在哪个价，一眼看到

一根大阳线常常连出**层层叠叠**好几个缺口。`fvg_state` 取「最近一个还没回补的」当主缺口，所以回踩线偏上、信号来得更早。

### 6.2 ATR：日 / 周 / 月各一套档位

| 参数 | 默认 | 说明 |
|---|---|---|
| `ATR_N` | 14 | Wilder 平滑周期 |
| `ATR_LEVELS` | `{'D': (1.2, 3.5)}` | 低波动上限 / 高波动下限（ATR 占价格 %） |
| `ATR_AUTO` | `True` | W/M 按 `√5`、`√21` 自动放大（1.2/3.5 → 2.68/7.23 → 5.50/16.04） |
| `ATR_SPIKE` | 1.8 | 当根 TR 超过 ATR 的几倍算「异动」 |
| `ATR_SQ_PCTL` | 20 | ATR% 落在低分位多少以下算「压死」 |

**两层信号**：

- **异动**（TR/ATR ≥ 1.8）：跳空、爆发，往往就是造缺口的那一根
- **收缩**（ATR% 低分位 + TR 贴着 ATR）：安静到不正常，蓄势等方向

图上是 TR% 与 ATR% 两条线叠画——**TR 大过 ATR 的那几根就是异动**，橙色三角标出来；收缩用青色空心圈标在 ATR% 上。两个引擎（matplotlib / plotly）都在**全历史上**算标记再截显示窗口，保证图上的点和表格里的数一致。

```python
R.atr_levels_table()                     # 打 D/W/M 三套阈值
R.atr_state(df, cfg)                     # 档位 + 扩张/收缩 + 分位
R.atr_signal(df, cfg)                    # 异动 / 收缩 / 倍数 / 距异动
```

### 6.3 档位（16 个）

```python
R.setup_table()                          # 打印全部档位
R.setup_overview(pool='HS300', bars=300) # 每档命中多少只 + 通过率
```

| 档位 | 条件 |
|---|---|
| 1 | 只看 FVG 回踩形态 |
| 2 | FVG + RSI 共振 |
| 3 | + 趋势多头 |
| 4 | + 缩量（最严） |
| 5 | FVG + RSI + 低波动 |
| `big_gap` | 缺口高度 ≥ 2%（`FVG_BIG_PCT`） |
| `fresh` | 缺口形成 ≤ 3 根（`FVG_FRESH_DAYS`） |
| `pullback_vol` | 已回踩 + 缩量 |
| `atr_quiet` / `atr_loud` | ATR 低波动档 / 高波动档 |
| `gap_burst` | 缺口 + TR 异动 |
| `squeeze_wait` | 波幅压死蓄势 |
| `near_rsi` / `near_trend` / `weak` | 「差一点」档（RSI 快进超卖 / 均线快多头） |

逻辑是 `need` 全命中（必须项）+ `near` 全命中 + `any_near` 任一（沾一个就算）+ `min_score` + `where`（自定义谓词）。档位定义就是普通 dict，可以自己加：

```python
R.SETUPS['my'] = {'name': '自定义', 'desc': '缺口还在 + 量能萎缩到极致',
                  'use': ['fvg', 'vol'], 'need': ['fvg'],
                  'where': lambda info, cfg: info.get('量比', 9) < 0.5, 'min_score': 0}
```

### 6.4 指标插件

```python
@R.register_indicator('my_ind', '一句话说明')
def my_ind(df, cfg):
    """df: 单只标的 OHLCV；cfg: 已合并的配置"""
    return {'hit': True, 'score': 1.0, 'near': False, 'info': {'我的字段': 123}}
```

内置 6 个：`fvg` `rsi` `trend` `vol` `atr` `atr_sig`。`score` 是档位打分用的权重，`near=True` 表示「还差一点」，会被 `near_*` 这类档位捞出来。注册时给的那句 `desc` 会直接当显示名用（档位条件、命中列都显示它）。

完整示例（带历史分位判定、组进档位）见 [附一.3 筛股指标](#附一3-筛股指标register_indicator)。

### 6.5 筛股与出图

```python
R.screen_rsi(pool='HS300', signal='hit')      # 只跑 RSI：超买超卖 + 刚交叉
R.screen_fvg(pool='HS300', mode='hit')        # 只跑 FVG：hit/touch/fresh/filled/all
R.screen_stocks(pool='HS300', tier=2, top=20) # 综合：档位 + 分数
R.screen_stocks(pool='HS300', tier='big_gap', closest=5)  # 只看离回踩线最近的 5 只

R.plot_one('000300.XSHG')                     # 一只票的 K 线 + RSI + ATR + FVG
R.plot_setup(pool='HS300', tier=2, max_charts=6)  # 按档位出图（先把完整表格打出来）
R.run_all(pool=['510300.XSHG'])               # 批量
R.rsi_snapshot(pool='HS300')                  # 只看当下状态，不出图
```

表格 **26 列**，不做缩略：代码/名称/日期/收盘/分数/命中/FVG/FVG方向/缺口高度%/实体%/缺口天数/回补%/回踩线/距线%/RSI/RSI状态/趋势/量比/ATR/ATR%/ATR档/ATR扩张/ATR分位/ATR信号/特征/信号日。

画图双引擎：`cfg['PLOT_ENGINE'] = 'mpl'`（默认，快）或 `'plotly'`（可缩放、悬停看 OHLC、周末空档自动抹掉）。plotly 存 png 要 kaleido，没装会自动退回 html。

---

## 七、口径与坑

这些是实际踩出来的，写在这里省得重踩：

1. **重叠窗口**：持仓 20 日 + 每月换仓，t 值会虚高 4~5 倍。只看 `t值_修正`，或者干脆 `non_overlap=True`
2. **生存偏差**：用今天的 HS300 成分回测 2019 年，等于提前知道谁会被纳入，IC 会虚高一大截。必须 `get_index_stocks(code, date)` 按日期取
3. **复权价别落盘**：存原始价 + 因子。存 qfq 价的话，下次除权整个历史都变
4. **`_merge_cfg` 只认白名单**：`rsi_multiday` 的新参数必须同时加进 `DEFAULTS`，否则传进去会被静默丢弃——不会有任何报错
5. **mock 数据形态**：`high == low == close` 的假数据会让「实体%」恒为 0，任何 `FVG_MIN_BODY > 0` 都会把 FVG 全滤掉。看到「0 只命中」先怀疑数据
6. **档位「满池」先看是不是旧内核**：`setup_overview` 会在某档命中数等于池子总数时警告。多半是改了代码没 Restart & Run All
7. **矩阵相乘别用 Series**：`beta`（索引是代码）乘 `m_w`（索引是日期）会按标签乱对齐，得用 `np.outer`
8. **控制台编码**：Windows GBK 编不了 `²` `⚠` `✓`，模块里一律用 `2` / `[!]` / `[OK]`
9. **ETF 池要过滤上市日**：64 只 ETF 里一半是 2020 年后上市的，回测起点一律写 2019 会把它们当「长期没数据」
10. **不做并发取数**：baostock 客户端全局单例，多线程串数据

---

## 许可证与免责

Apache License 2.0，全文见 [LICENSE](LICENSE)。

- **版权**：Copyright 2026 Neeko-ovo
- **署名**：再分发本软件或其衍生作品时，必须保留 [NOTICE](NOTICE) 文件与其中的版权声明；改过哪个源文件，分发时注明
- **专利**：附带的专利授权条款，意味着贡献者不能反过来用专利告你侵权
- **作者信息**：项目首次公开发布于 2026-11-22；GitHub 首次提交时间即为公开时间戳

如果你在自己的工作里用到这套框架，引用方式：

```
Neeko-ovo. (2026). Quant Workbench: A-share factor research & screening toolkit.
https://github.com/Neeko-ovo
```

**免责**：个人研究与学习用途。数据来自 baostock / 聚宽等公开渠道，准确性由数据源方决定；本项目的回测与信号不构成任何投资建议。市场有风险，代码不保证赚钱。
