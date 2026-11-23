# -*- coding: utf-8 -*-
# ============================================================
# jq_io.py —— 本地数据层（baostock + 本地缓存），接口对齐聚宽
#   目的：把「jqdata 上的框架」搬到自己的 Jupyter Lab，代码不重写
#   用法：from jq_io import *
#         login()                      # 建连（首次会自动建）
#         px = get_price_range(['000300.XSHG'], '2019-01-01', '2025-06-30')
#
#   已实现（与聚宽同名同参）：get_price / get_index_stocks /
#     get_trade_days / get_security_info / normalize_code / get_all_trade_days
#   未实现（本地没有数据源，调用时给明确提示）：get_fundamentals / jqfactor
#
#   三条口径（重要）
#     1) 缓存里存「不复权价 + 复权因子 adj」，不存前复权价。
#        前复权价 = 不复权价 × adj / 最新 adj，所以新除权不会污染历史。
#     2) 成分股按日期取（baostock 支持 hs300 / zz500 / sz50 的历史快照）。
#     3) 停牌日按聚宽口径处理：价格前值填充，成交量 0。
# ============================================================
import os
import datetime
import numpy as np
import pandas as pd

LOCAL_MODE = True                 # 框架据此跳过 jqfactor / 财务表通道
DATA_SOURCE = 'baostock'

# ------------------------------------------------------------
# 0 配置
# ------------------------------------------------------------
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except Exception:
    _HERE = os.getcwd()

CACHE_DIR = os.path.join(os.getcwd(), 'jq_data')   # 缓存根目录，想挪就改这里
DAILY_DIR = os.path.join(CACHE_DIR, 'daily')
MISC_DIR = os.path.join(CACHE_DIR, 'misc')
CACHE_ON = True
CACHE_START = '2010-01-01'        # 缓存最早抓到哪天
VERBOSE = True                    # 批量取数时是否打印进度

_bs = None
_logged = False
_sec_cache = {}
_idx_cache = {}
_MEM = {}
_MEM_LIMIT = 1200                 # 内存里最多缓几只（每只一块 DataFrame）


def _mem_put(key, df):
    if len(_MEM) >= _MEM_LIMIT:
        _MEM.clear()
    _MEM[key] = df


def _mem_clear():
    _MEM.clear()


def _mkdirs():
    for d in (CACHE_DIR, DAILY_DIR, MISC_DIR):
        if not os.path.isdir(d):
            try:
                os.makedirs(d)
            except OSError:
                pass


def set_cache_dir(path):
    """换缓存目录（换完自动建子目录）。"""
    global CACHE_DIR, DAILY_DIR, MISC_DIR
    CACHE_DIR = path
    DAILY_DIR = os.path.join(path, 'daily')
    MISC_DIR = os.path.join(path, 'misc')
    _mkdirs()
    return CACHE_DIR


def info(msg):
    if VERBOSE:
        print(msg)


# ------------------------------------------------------------
# 1 代码格式互转
# ------------------------------------------------------------
_BAO_PREFIX = {'XSHG': 'sh', 'XSHE': 'sz'}


def to_bao(code):
    """600000.XSHG -> sh.600000；已是 sh.600000 的原样返回。"""
    code = str(code).strip()
    if '.' in code:
        a, b = code.split('.', 1)
        if b.lower() in ('xshg', 'sh'):
            return 'sh.' + a
        if b.lower() in ('xshe', 'sz'):
            return 'sz.' + a
    if code[:3] in ('sh.', 'sz.'):
        return code
    # 纯数字：6 开头走上海，其余走深圳
    return ('sh.' if code[0] == '6' else 'sz.') + code


def to_jq(code):
    """sh.600000 -> 600000.XSHG；已是聚宽格式的原样返回。"""
    code = str(code).strip()
    if code[:3] == 'sh.':
        return code[3:] + '.XSHG'
    if code[:3] == 'sz.':
        return code[3:] + '.XSHE'
    return code


def normalize_code(code):
    """聚宽同名函数：统一成 XXXXXX.XSHG / XXXXXX.XSHE。"""
    return to_jq(code)


def _safe(code):
    return str(code).replace('.', '_')


# ------------------------------------------------------------
# 2 登录 / 底层查询
# ------------------------------------------------------------
def login(user=None, pwd=None, verbose=True):
    """建立 baostock 连接（baostock 免注册，user/pwd 不用传）。"""
    global _bs, _logged
    if _logged:
        return True
    try:
        import baostock as bs
    except Exception as e:
        raise RuntimeError('没装 baostock：pip install baostock（%s）' % e)
    _bs = bs
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError('baostock 登录失败：%s %s' % (lg.error_code, lg.error_msg))
    _logged = True
    _mkdirs()
    if verbose:
        print('baostock 已连接；缓存目录 %s' % CACHE_DIR)
    return True


def logout():
    global _logged
    if _logged and _bs is not None:
        try:
            _bs.logout()
        except Exception:
            pass
    _logged = False


def _ensure():
    if not _logged:
        login(verbose=False)


_RAW_FIELDS = 'date,open,high,low,close,preclose,volume,amount,tradestatus,isST,turn,pctChg'

# baostock 免费但不高频：串行 + 每次查询之间留一点空隙；失败自动重试几轮。
# 客户端是全局单例，多线程并发会串数据，所以这里只做串行限速，不做并发。
MIN_GAP = 0.08          # 两次查询之间至少隔多少秒
RETRY = 3               # 失败重试次数
RETRY_SLEEP = (1.0, 3.0, 8.0)
_last_q = [0.0]


def _sleep_gap():
    import time as _t
    dt = _t.time() - _last_q[0]
    if dt < MIN_GAP:
        _t.sleep(MIN_GAP - dt)
    _last_q[0] = _t.time()


def _query(code, fields, start, end, adjustflag='3'):
    """查一段日线，返回 DataFrame（全是字符串列，调用方自己转数值）。
    失败重试 RETRY 次（间隔 1/3/8 秒），还不行才抛。"""
    import time as _t
    _ensure()
    err = None
    for k in range(RETRY):
        _sleep_gap()
        try:
            rs = _bs.query_history_k_data_plus(code, fields, start_date=start, end_date=end,
                                               frequency='d', adjustflag=adjustflag)
        except Exception as e:            # 网络断了 / 连接被关
            err = e
        else:
            if rs.error_code == '0':
                rows = []
                while (rs.error_code == '0') and rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    return pd.DataFrame(columns=fields.split(','))
                return pd.DataFrame(rows, columns=rs.fields)
            err = RuntimeError('baostock 查询失败 %s: %s %s' % (code, rs.error_code, rs.error_msg))
        if k < RETRY - 1:
            _t.sleep(RETRY_SLEEP[min(k, len(RETRY_SLEEP) - 1)])
            if _logged:
                _reconnect()               # 连接断了就重连一次
    raise err if isinstance(err, Exception) else RuntimeError('baostock 查询失败：%s' % (err,))


def _reconnect():
    """查询出错后强制重连（baostock 的 socket 偶尔会被服务端掐掉）。"""
    global _logged
    try:
        _bs.logout()
    except Exception:
        pass
    _logged = False
    try:
        _ensure()
    except Exception:
        pass


def _num(s):
    return pd.to_numeric(s, errors='coerce')


# ------------------------------------------------------------
# 3 缓存：日线（不复权 + adj 因子）
# ------------------------------------------------------------
def _daily_path(code):
    return os.path.join(DAILY_DIR, _safe(to_bao(code)) + '.pkl')


def _read_daily(code):
    p = _daily_path(code)
    if not (CACHE_ON and os.path.isfile(p)):
        return None
    try:
        df = pd.read_pickle(p)
    except Exception:
        return None
    return df if df is not None and len(df) else None


def _write_daily(code, df):
    if not CACHE_ON:
        return
    _mkdirs()
    try:
        df.to_pickle(_daily_path(code))
    except Exception as e:
        info('  [缓存] 写入失败 %s: %s' % (code, e))


def _fetch_daily(code, start, end):
    """抓一段：不复权价 + 用「后复权/不复权」反推复权因子。"""
    bcode = to_bao(code)
    raw = _query(bcode, _RAW_FIELDS, start, end, '3')
    if raw.empty:
        return None
    try:
        hfq = _query(bcode, 'date,close', start, end, '1')
    except Exception:
        hfq = pd.DataFrame(columns=['date', 'close'])
    df = pd.DataFrame({
        'open': _num(raw['open']).values,
        'high': _num(raw['high']).values,
        'low': _num(raw['low']).values,
        'close': _num(raw['close']).values,
        'pre_close': _num(raw['preclose']).values,
        'volume': _num(raw['volume']).values,
        'money': _num(raw['amount']).values,
        # turn = 换手率(%)，pct_chg = 涨跌幅(%)。这两个字段以前查了却没存，
        # 现在补上：Barra 的 Size（流通市值）和 Liquidity 要用换手率反推流通股本。
        'turn': _num(raw['turn']).values if 'turn' in raw.columns else np.nan,
        'pct_chg': _num(raw['pctChg']).values if 'pctChg' in raw.columns else np.nan,
        'paused': (raw['tradestatus'].astype(str) == '0').astype(np.int32).values,
        'is_st': _num(raw['isST']).fillna(0).astype(np.int32).values,
    }, index=pd.to_datetime(raw['date']))
    df = df.sort_index()
    # 复权因子：后复权收盘 / 不复权收盘（指数、多数 ETF 恒为 1）
    adj = pd.Series(np.nan, index=df.index)
    if not hfq.empty:
        h = pd.Series(_num(hfq['close']).values, index=pd.to_datetime(hfq['date']))
        h = h[~h.index.duplicated(keep='last')]
        ratio = (h / df['close']).replace([np.inf, -np.inf], np.nan)
        adj = ratio.reindex(df.index)
    df['adj'] = adj.ffill().bfill().fillna(1.0)
    df.index.name = 'date'
    return df


def load_daily(code, start=None, end=None, force=False):
    """读本地缓存（不够就补抓），返回不复权价 + adj 的 DataFrame。"""
    start = pd.to_datetime(start).normalize() if start is not None else pd.Timestamp(CACHE_START)
    end = pd.to_datetime(end).normalize() if end is not None else pd.Timestamp(datetime.date.today())
    if start > end:
        start, end = end, start
    key = _safe(to_bao(code))
    ck = (key, start, end)
    if not force and ck in _MEM:
        return _MEM[ck]
    df = None if force else _read_daily(code)
    if df is not None:
        lo, hi = df.index.min(), df.index.max()
        if (lo <= start + _CAL_TOL) and (hi >= end - _CAL_TOL):
            out = df.loc[start:end]
            _mem_put(ck, out)
            return out
        need_lo = min(lo, start)
        need_hi = max(hi, end)
        fetch_start = need_lo.strftime('%Y-%m-%d')
        fetch_end = need_hi.strftime('%Y-%m-%d')
    else:
        fetch_start = max(pd.Timestamp(CACHE_START), start).strftime('%Y-%m-%d')
        fetch_end = end.strftime('%Y-%m-%d')
    new = _fetch_daily(code, fetch_start, fetch_end)
    if new is None or new.empty:
        out = df.loc[start:end] if df is not None else pd.DataFrame()
        _mem_put(ck, out)
        return out
    if df is not None:
        df = pd.concat([df, new])
        df = df[~df.index.duplicated(keep='last')].sort_index()
    else:
        df = new
    _write_daily(code, df)
    out = df.loc[start:end]
    _mem_put(ck, out)
    return out


def warm_cache(codes, start=None, end=None, verbose=True):
    """批量预热缓存（第一次跑因子时先跑它，之后各次都是本地读）。"""
    codes = list(codes)
    n = len(codes)
    t0 = datetime.datetime.now()
    done = 0
    for i, c in enumerate(codes):
        try:
            load_daily(c, start, end)
            done += 1
        except Exception as e:
            print('  [跳过] %s %s' % (c, e))
        if verbose and (i + 1) % 50 == 0:
            print('  缓存进度 %d/%d  用时 %.0fs' % (i + 1, n, (datetime.datetime.now() - t0).seconds))
    if verbose:
        print('缓存完成 %d/%d 只，用时 %.1fs，目录 %s'
              % (done, n, (datetime.datetime.now() - t0).total_seconds(), DAILY_DIR))
    return done


# 老缓存是「只有价量」的 pkl，没有 turn / pct_chg 两列。
# Barra 的 Size（流通市值）要用换手率反推流通股本，没这两列就算不出来。
# 不想重下全量也没关系：barra 那边检测到没 turn 会自动跳过 Size 并打提示，
# 需要的时候跑一次 upgrade_cache 就行（只补这两列，重抓同一区间）。
def has_field(code, field='turn'):
    """本地缓存里有没有某个字段（老缓存没有 turn）。"""
    df = _read_daily(code)
    return bool(df is not None and len(df.columns) and field in df.columns)


def upgrade_cache(codes, start=None, end=None, verbose=True):
    """把老缓存重抓一遍，补上 turn / pct_chg。返回补好的只数。"""
    codes = list(codes)
    t0 = datetime.datetime.now()
    need = [c for c in codes if not has_field(c, 'turn')]
    if verbose:
        print('需要补字段的 %d/%d 只' % (len(need), len(codes)))
    for i, c in enumerate(need):
        try:
            load_daily(c, start, end, force=True)
        except Exception as e:
            print('  [跳过] %s %s' % (c, e))
        if verbose and (i + 1) % 50 == 0:
            print('  补字段进度 %d/%d  用时 %.0fs'
                  % (i + 1, len(need), (datetime.datetime.now() - t0).seconds))
    if verbose:
        print('补字段完成，用时 %.1fs' % (datetime.datetime.now() - t0).total_seconds())
    return len(need)


# ------------------------------------------------------------
# 4 交易日历
# ------------------------------------------------------------
def _cal_path():
    return os.path.join(MISC_DIR, 'trade_days.pkl')


def _load_calendar():
    p = _cal_path()
    if CACHE_ON and os.path.isfile(p):
        try:
            c = pd.read_pickle(p)
            if c is not None and len(c):
                return pd.DatetimeIndex(c)
        except Exception:
            pass
    return None


def _refresh_calendar(start, end):
    _ensure()
    rs = _bs.query_trade_dates(start_date=start, end_date=end)
    if rs.error_code != '0':
        raise RuntimeError('交易日历读取失败：%s %s' % (rs.error_code, rs.error_msg))
    rows = []
    while (rs.error_code == '0') and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    days = pd.to_datetime(df.loc[df['is_trading_day'] == '1', 'calendar_date'])
    return pd.DatetimeIndex(sorted(days.unique()))


_CAL_TOL = pd.Timedelta(days=12)   # 周末/长假造成的边界差，小于这个就不补抓


def trade_calendar(start=None, end=None):
    """返回 DatetimeIndex（按需增量补抓）。"""
    start = pd.Timestamp(start) if start is not None else pd.Timestamp(CACHE_START)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp(datetime.date.today())
    cal = _load_calendar()
    if cal is None:
        cal = _refresh_calendar(CACHE_START, end.strftime('%Y-%m-%d'))
    elif (cal.min() > start + _CAL_TOL) or (cal.max() < end - _CAL_TOL):
        lo = min(cal.min(), start).strftime('%Y-%m-%d')
        hi = max(cal.max(), end).strftime('%Y-%m-%d')
        more = _refresh_calendar(lo, hi)
        cal = pd.DatetimeIndex(sorted(set(cal).union(set(more))))
        if CACHE_ON:
            _mkdirs()
            try:
                pd.Series(cal).to_pickle(_cal_path())
            except Exception:
                pass
    return cal[(cal >= start) & (cal <= end)]


def get_trade_days(start_date=None, end_date=None, count=None):
    """聚宽同名：返回交易日列表（datetime.date）。给 count 时取 start 后的 count 个。"""
    start_date = pd.Timestamp(start_date) if start_date is not None else pd.Timestamp(CACHE_START)
    if count is not None:
        end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(datetime.date.today())
        # 多抓一些再用切片，避免假期把 count 吃掉
        probe_end = min(end_date, start_date + pd.Timedelta(days=int(count) * 3 + 40))
        cal = trade_calendar(start_date, probe_end)
        if len(cal) < count:
            cal = trade_calendar(start_date, end_date)
        return [d.date() for d in cal[:int(count)]]
    end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(datetime.date.today())
    cal = trade_calendar(start_date, end_date)
    return [d.date() for d in cal]


def get_all_trade_days():
    return get_trade_days(CACHE_START, datetime.date.today())


# ------------------------------------------------------------
# 5 指数成分（历史快照）
# ------------------------------------------------------------
_IDX_MAP = {
    '000300.XSHG': ['hs300'],
    '000905.XSHG': ['zz500'],
    '000016.XSHG': ['sz50'],
    '000906.XSHG': ['hs300', 'zz500'],      # 中证 800 = 300 + 500
    '399006.XSHE': ['cyb'],                  # 创业板指：baostock 无成分，特判
}
_IDX_ALIAS = {'HS300': '000300.XSHG', 'ZZ500': '000905.XSHG', 'ZZ800': '000906.XSHG',
              'ZZ1000': '000852.XSHG', 'SZ50': '000016.XSHG', 'CYB': '399006.XSHE'}


def _idx_path():
    return os.path.join(MISC_DIR, 'index_stocks.pkl')


def _load_idx_cache():
    global _idx_cache
    if _idx_cache:
        return _idx_cache
    p = _idx_path()
    if CACHE_ON and os.path.isfile(p):
        try:
            _idx_cache = pd.read_pickle(p) or {}
        except Exception:
            _idx_cache = {}
    return _idx_cache


def _save_idx_cache():
    if not CACHE_ON:
        return
    _mkdirs()
    try:
        pd.to_pickle(_idx_cache, _idx_path())
    except Exception:
        pass


def _query_index(name, date_str):
    _ensure()
    fn = getattr(_bs, 'query_%s_stocks' % name, None)
    if fn is None:
        raise RuntimeError('baostock 不支持 %s 成分' % name)
    rs = fn(date=date_str) if date_str else fn()
    if rs.error_code != '0':
        raise RuntimeError('%s 成分查询失败：%s %s' % (name, rs.error_code, rs.error_msg))
    rows = []
    while (rs.error_code == '0') and rs.next():
        rows.append(rs.get_row_data())
    idx = rs.fields.index('code') if 'code' in rs.fields else 1
    return [to_jq(r[idx]) for r in rows]


def get_index_stocks(index_code, date=None):
    """聚宽同名：指数成分。支持 HS300 / ZZ500 / SZ50 / ZZ800（=300+500）。"""
    code = _IDX_ALIAS.get(str(index_code), str(index_code))
    names = _IDX_MAP.get(code)
    if names is None:
        raise ValueError('本地模式只支持这几只指数：HS300 / ZZ500 / SZ50 / ZZ800；'
                         '其它请直接给代码列表，如 get_price_range([...])')
    d = '' if date is None else pd.Timestamp(date).strftime('%Y-%m-%d')
    key = (code, d)
    cache = _load_idx_cache()
    if key in cache:
        return list(cache[key])
    out = []
    seen = set()
    for nm in names:
        if nm == 'cyb':
            raise ValueError('baostock 没有创业板指的成分快照；CYB 请换成 ['+"'159915.XSHE'"+'] 这类列表池')
        for c in _query_index(nm, d):
            if c not in seen:
                seen.add(c)
                out.append(c)
    cache[key] = out
    _save_idx_cache()
    return out


# ------------------------------------------------------------
# 6 证券信息
# ------------------------------------------------------------
class SecurityInfo(object):
    def __init__(self, code, name, start_date=None, end_date=None, type_=None):
        self.code = code
        self.display_name = name
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.type = type_

    def __repr__(self):
        return '<SecurityInfo %s %s>' % (self.code, self.display_name)


def get_security_info(code):
    """聚宽同名：返回带 display_name 的对象（取不到时返回 None）。"""
    code = normalize_code(code)
    if code in _sec_cache:
        return _sec_cache[code]
    _ensure()
    try:
        rs = _bs.query_stock_basic(code=to_bao(code))
        if rs.error_code != '0':
            return None
        if not rs.next():
            return None
        row = dict(zip(rs.fields, rs.get_row_data()))
        obj = SecurityInfo(code, row.get('code_name') or code,
                           row.get('ipoDate') or None, row.get('outDate') or None,
                           row.get('type'))
    except Exception:
        obj = None
    _sec_cache[code] = obj
    return obj


# ------------------------------------------------------------
# 7 get_price（聚宽口径）
# ------------------------------------------------------------
_DEFAULT_FIELDS = ['open', 'close', 'high', 'low', 'volume', 'money', 'pre_close', 'paused']


def _apply_fq(df, fq):
    """把不复权价按 adj 换成前/后复权；成交量金额不动。"""
    if fq in (None, 'none', 'raw', False):
        return df
    adj = df['adj'] if 'adj' in df.columns else pd.Series(1.0, index=df.index)
    last = float(adj.iloc[-1]) if len(adj) else 1.0
    if last == 0 or not np.isfinite(last):
        last = 1.0
    ratio = adj / last if str(fq).lower() in ('pre', 'qfq') else adj
    out = df.copy()
    for c in ('open', 'high', 'low', 'close', 'pre_close'):
        if c in out.columns:
            out[c] = out[c] * ratio
    return out


def get_price(security, start_date=None, end_date=None, count=None, frequency='daily',
              fields=None, skip_paused=False, fq='pre', panel=False):
    """聚宽同名。返回长表（columns: time / code / 字段），与 panel=False 一致。"""
    if frequency not in ('daily', 'd', '1d'):
        raise ValueError("本地模式只支持日线（frequency='daily'）")
    codes = [normalize_code(c) for c in (security if isinstance(security, (list, tuple, set)) else [security])]
    fields = list(fields) if fields else list(_DEFAULT_FIELDS)

    # 先把要用的每只读进内存（顺便预热缓存）
    frames = {}
    n = len(codes)
    for i, c in enumerate(codes):
        try:
            df = load_daily(c, start_date, end_date)
        except Exception as e:
            info('  [取数失败] %s %s' % (c, e))
            df = None
        if df is not None and len(df):
            frames[c] = df
        if VERBOSE and n > 60 and (i + 1) % 50 == 0:
            info('  取数进度 %d/%d' % (i + 1, n))

    if not frames:
        return pd.DataFrame(columns=['time', 'code'] + fields)

    # 统一时间轴：区间内全部交易日（停牌日 ffill 价格、量置 0）
    if count is not None:
        n_tail = int(count)
        tails = [df.index[-n_tail:] for df in frames.values()]
        cal = pd.DatetimeIndex(sorted(set().union(*[set(t) for t in tails])))
    else:
        lo = min(df.index.min() for df in frames.values())
        hi = max(df.index.max() for df in frames.values())
        cal = trade_calendar(lo, hi)

    rows = []
    for c, df in frames.items():
        d = _apply_fq(df, fq)
        d = d.reindex(cal)
        if not skip_paused:
            for col in ('open', 'high', 'low', 'close', 'pre_close', 'adj'):
                if col in d.columns:
                    d[col] = d[col].ffill()
            for col in ('volume', 'money'):
                if col in d.columns:
                    d[col] = d[col].fillna(0.0)
        d = d.dropna(how='all', subset=[x for x in ('close',) if x in d.columns])
        if d.empty:
            continue
        d = d.copy()
        d['time'] = d.index
        d['code'] = c
        rows.append(d)
    if count is not None and rows:
        # 每只都截最后 count 根后对齐
        rows = [r.iloc[-int(count):] for r in rows]
    out = pd.concat(rows, axis=0, sort=False)
    keep = ['time', 'code'] + [f for f in fields if f in out.columns]
    out = out[keep]
    out = out.sort_values(['time', 'code']).reset_index(drop=True)
    if panel:
        wide = {}
        for f in fields:
            if f in out.columns:
                wide[f] = out.pivot(index='time', columns='code', values=f)
        return wide
    return out


# ------------------------------------------------------------
# 8 本地没有的通道：给出明确的报错，而不是 NameError
# ------------------------------------------------------------
class _Unavailable(object):
    def __init__(self, what, hint):
        self._what = what
        self._hint = hint

    def __call__(self, *a, **k):
        raise RuntimeError('%s在本地数据源里没有。%s' % (self._what, self._hint))

    def __getattr__(self, item):
        raise RuntimeError('%s在本地数据源里没有。%s' % (self._what, self._hint))

    def __repr__(self):
        return '<%s 本地不可用>' % self._what


query = _Unavailable('聚宽 query 查询', '财务表需要 jqdata；本地请用自定义因子 register_factor')
valuation = _Unavailable('聚宽 valuation 表', '同上：换自定义因子')
balance = _Unavailable('聚宽 balance 表', '同上：换自定义因子')
cash_flow = _Unavailable('聚宽 cash_flow 表', '同上：换自定义因子')
income = _Unavailable('聚宽 income 表', '同上：换自定义因子')
indicator = _Unavailable('聚宽 indicator 表', '同上：换自定义因子')


def get_fundamentals(*a, **k):
    raise RuntimeError('get_fundamentals 需要 jqdata。本地模式请改用自定义因子：\n'
                       '  @register_factor("my_factor")\n'
                       '  def my_factor(stock_list, date): ...')


class _Logger(object):
    def info(self, *a):
        print(*a)

    def warn(self, *a):
        print('[warn]', *a)

    def error(self, *a):
        print('[error]', *a)

    def debug(self, *a):
        pass


log = _Logger()


# ------------------------------------------------------------
# 9 自检
# ------------------------------------------------------------
def selfcheck(code='000300.XSHG'):
    """跑一遍所有接口，确认本地数据层能正常工作。"""
    login(verbose=False)
    print('=== jq_io 自检 ===')
    days = get_trade_days('2024-01-01', '2024-01-31')
    print('get_trade_days  2024-01 共 %d 天，首尾 %s ~ %s' % (len(days), days[0], days[-1]))
    px = get_price([code], end_date='2024-01-31', count=5, fields=['close'], panel=False)
    print('get_price       %s 行 %d，列 %s' % (code, len(px), list(px.columns)))
    print(px.tail(2).to_string(index=False))
    codes = get_index_stocks('HS300')
    print('get_index_stocks HS300 %d 只，前 3 %s' % (len(codes), codes[:3]))
    si = get_security_info(codes[0]) if codes else None
    print('get_security_info %s' % (si,))
    df = load_daily(code, '2024-01-01', '2024-06-30')
    print('load_daily      %d 行，adj 末值 %.4f' % (len(df), df['adj'].iloc[-1]))
    return True


if __name__ == '__main__':
    selfcheck()
