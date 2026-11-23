# -*- coding: utf-8 -*-
# ============================================================
# 用 mock 数据把「框架导入测试_V4.ipynb」的代码单元按顺序真跑一遍
#   只验证能不能跑通、有没有 NameError/参数错；不判断数值好坏
# 运行：python exec_test_notebook.py
# ============================================================
import io
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(HERE, '_mplcache'))

import matplotlib
matplotlib.use('Agg')

# ---- mock 数据层 ----
_MOCK = os.path.join(HERE, 'selftest_pct_mock.py')
with io.open(_MOCK, 'r', encoding='utf-8') as fh:
    _src = fh.read()
_head = _src.split("NS = {'__name__': 'jq_ns'")[0]
NS = {'__name__': '__main__', '__file__': _MOCK}
exec(compile(_head, _MOCK, 'exec'), NS)

with io.open(os.path.join(HERE, '框架导入测试_V4.ipynb'), 'r', encoding='utf-8') as fh:
    nb = json.load(fh)

fails = []
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
for idx, (i, cell) in enumerate(code_cells):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    t0 = time.time()
    try:
        exec(compile(src, '<cell %d>' % i, 'exec'), NS)
        print('[OK]   cell %-3d %5.1fs' % (i, time.time() - t0))
    except Exception:
        fails.append(i)
        print('[FAIL] cell %-3d %5.1fs' % (i, time.time() - t0))
        traceback.print_exc()

print('\n代码单元 %d 个，失败 %d 个' % (len(code_cells), len(fails)))
if fails:
    print('失败单元：%s' % fails)
    raise SystemExit(1)
print('[done] notebook 全部代码单元在 mock 下跑通')
