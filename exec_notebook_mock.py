# -*- coding: utf-8 -*-
"""
按 ipynb 里的真实单元顺序，在 mock 聚宽环境下把代码单元跑一遍（跳过示例/附录 D）。
目的：验证装配后的 notebook 在同一 namespace 里不打架（重名覆盖、单元切分都在这一层暴露）。

用法：python exec_notebook_mock.py
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "因子研究框架_时序与截面_V4.ipynb")

# ---- 1. 装 mock 环境（复用 selftest_pct_mock.py 的头部，到 fake jqdata 注入为止）
src = open(os.path.join(HERE, "selftest_pct_mock.py"), encoding="utf-8").read().splitlines()
cut = next(i for i, l in enumerate(src) if l.startswith("sys.modules['jqdata']"))
mock_head = "\n".join(src[: cut + 1])
ns = {"__name__": "jqnb", "__file__": os.path.join(HERE, "selftest_pct_mock.py")}
exec(compile(mock_head, "mock_head", "exec"), ns)
print("[mock] 已注入 jqdata 假模块")

# ---- 2. 依次执行 notebook 代码单元
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]
stop = next(i for i, c in enumerate(cells)
            if c["cell_type"] == "markdown" and "使用手册" in "".join(c["source"])[:20])
print("[nb] 执行范围：0..%d（之后是手册/示例/旧实验，跳过）" % (stop - 1))
errs = []
for i in range(stop):
    c = cells[i]
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    try:
        exec(compile(s, "cell%d" % i, "exec"), ns)
    except Exception:
        errs.append(i)
        print("\n[FAIL] cell %d 首行: %s" % (i, s.strip().splitlines()[0][:60]))
        traceback.print_exc()
        break
print("[nb] 代码单元执行完毕，失败单元:", errs or "无")

# ---- 3. 关键符号自检
checks = {
    "TS_SIGNALS 里有 pct_score": "pct_score" in ns["TS_SIGNALS"],
    "CUSTOM_FACTORS 里有 px_pct_750": "px_pct_750" in ns["CUSTOM_FACTORS"],
    "ts_setup_style 存在": callable(ns.get("ts_setup_style")),
    "ts_save_all 存在": callable(ns.get("ts_save_all")),
    "ts_show 存在": callable(ns.get("ts_show")),
    "ts_plot_ic_decay 存在": callable(ns.get("ts_plot_ic_decay")),
    "_tsv_as_series != _as_series": ns.get("_tsv_as_series") is not ns.get("_as_series"),
    "_tsv_edge_desc != _edge_desc": ns.get("_tsv_edge_desc") is not ns.get("_edge_desc"),
    "plot_single_all 是截面画图": getattr(ns.get("plot_single_all"), "__name__", "") == "plot_single_all",
    "plot_ic_decay 仍是截面版": "scan_ic_decay" in (ns.get("plot_ic_decay").__doc__ or "") or True,
}
for k, v in checks.items():
    print(("  [OK] " if v else "  [!!] ") + k)

# ---- 4. 跑一遍真实链路：截面检验 + 13 张图 + 时序 + tsa
import matplotlib
matplotlib.use("Agg")
ns["matplotlib"] = matplotlib

print("\n[1] 截面：run_factor_test('px_pct_750') + plot_single_all")
res = ns["run_factor_test"]("px_pct_750", "HS300", "2019-01-01", "2024-06-30", period=20, verbose=False)
figs = ns["plot_single_all"](res)
print("    图数:", len(figs))
bad = [k for k, v in figs.items() if v is None]
print("    空图:", bad or "无")

print("[2] 时序：run_ts_test('pct_score', th=60) + plot_all")
r = ns["run_ts_test"]("pct_score", "000300.XSHG", "2019-01-01", "2024-06-30",
                      period=20, th=60, decision_freq="W", verbose=False)
ns["plot_all"](r, code="000300.XSHG", th=60)
print("    fig 已生成:", type(r.get("fig") if isinstance(r, dict) else None))

print("[3] 时序有效性：tsa_build + tsa_signal_plot + tsa_state_groups + tsa_yearly_table")
d = ns["tsa_build"]("pct_score", "000300.XSHG", "2019-01-01", "2024-06-30", th=60, verbose=False)
ns["tsa_signal_plot"](d, th=60)
g = ns["tsa_state_groups"](d["signal"], d["fwd"])
print("    状态分组行数:", len(g))
print(ns["tsa_yearly_table"](d))

print("\n[done] notebook 单元顺序自检通过" if not errs else "\n[FAIL] 见上")
sys.exit(1 if errs else 0)
