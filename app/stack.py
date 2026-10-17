"""Učící se vrstva: z backtestu se naučí, jak vážit odhad modelu, formu, odpočinek a kurzy trhu.

Je to malá multinomická logistická regrese, která startuje z „modelu beze změny“ a od něj se odchyluje jen tehdy,
když to na nikdy neviděných zápasech (posledních 30 % času) zlepší log loss. Jinak se nepoužije.
"""
import json
from pathlib import Path

import numpy as np

NAMES_PLAIN = ["model: domácí vs remíza", "model: hosté vs remíza", "forma", "odpočinek"]
NAMES_ODDS = NAMES_PLAIN + ["trh: domácí vs remíza", "trh: hosté vs remíza"]
MIN_ROWS, GAIN = 400, 0.001


def _lr(P):
    P = np.clip(np.asarray(P, float), 1e-6, 1)
    return np.log(P[:, 0] / P[:, 1]), np.log(P[:, 2] / P[:, 1])


def _clean(x):
    return np.nan_to_num(np.asarray(x, float), nan=0.0)


def features(P, hform, aform, hrest, arest, M=None):
    lh, la = _lr(P)
    form = _clean(np.asarray(hform, float) - np.asarray(aform, float)) / 3.0
    rest = _clean(np.clip(np.nan_to_num(hrest, nan=7.0), 0, 14) - np.clip(np.nan_to_num(arest, nan=7.0), 0, 14)) / 7.0
    cols = [lh, la, form, rest]
    if M is not None:
        mh, ma = _lr(M)
        cols += [mh, ma]
    return np.column_stack(cols)


def _init(k, with_market):
    W0 = np.zeros((k, 2))
    if with_market:
        W0[4, 0], W0[5, 1] = 1.0, 1.0      # start = čistý trh
    else:
        W0[0, 0], W0[1, 1] = 1.0, 1.0      # start = čistý model
    return W0


def _probs(X, W, b):
    z = np.column_stack([X @ W[:, 0] + b[0], np.zeros(len(X)), X @ W[:, 1] + b[1]])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def _fit(X, y, W0, lam=0.02, iters=500, lr=0.4):
    W, b = W0.copy(), np.zeros(2)
    Y = np.eye(3)[y]
    for _ in range(iters):
        P = _probs(X, W, b)
        G = P - Y
        gz = np.column_stack([G[:, 0], G[:, 2]])
        W -= lr * (X.T @ gz / len(X) + lam * (W - W0))
        b -= lr * gz.mean(axis=0)
    return W, b


def train_one(X, y, with_market, base_P):
    """Vrací (parametry nebo None, report). Ověřuje na posledních 30 % zápasů (řazeno v čase)."""
    n, k = X.shape
    rep = {"rows": int(n), "enabled": False}
    if n < MIN_ROWS:
        rep["reason"] = f"málo zápasů na učení ({n} < {MIN_ROWS})"
        return None, rep
    cut = int(n * 0.7)
    W0 = _init(k, with_market)
    W, b = _fit(X[:cut], y[:cut], W0)
    base = _logloss(base_P[cut:], y[cut:])
    new = _logloss(_probs(X[cut:], W, b), y[cut:])
    rep.update(base_log_loss=base, learned_log_loss=new, test_rows=int(n - cut))
    if new < base - GAIN:
        W, b = _fit(X, y, W0)
        rep["enabled"] = True
        rep["coefs"] = {nm: [float(W[i, 0]), float(W[i, 1])] for i, nm in enumerate(NAMES_ODDS if with_market else NAMES_PLAIN)}
        return {"W": W.tolist(), "b": b.tolist()}, rep
    rep["reason"] = "naučené úpravy na neviděných zápasech nepomohly, používá se původní odhad"
    return None, rep


def train(bt, blend_fn):
    """bt = backtest s hform/aform/hrest/arest. Vrací (params, report) pro 'plain' (bez kurzů) a 'odds' (s kurzy)."""
    if bt.empty or "hform" not in bt:
        return {}, {}
    bt = bt.sort_values("date").reset_index(drop=True)
    P = bt[["p_h", "p_d", "p_a"]].to_numpy(float)
    y = bt["y"].to_numpy(int)
    args = [bt[c].to_numpy(float) for c in ("hform", "aform", "hrest", "arest")]
    params, report = {}, {}
    params["plain"], report["plain"] = train_one(features(P, *args), y, False, P)
    has = bt[["o_h", "o_d", "o_a"]].notna().all(axis=1).to_numpy()
    if has.sum():
        O = bt.loc[has, ["o_h", "o_d", "o_a"]].to_numpy(float)
        inv = 1 / O
        M = inv / inv.sum(axis=1, keepdims=True)
        Ph = P[has]
        Xo = features(Ph, *[a[has] for a in args], M=M)
        params["odds"], report["odds"] = train_one(Xo, y[has], True, blend_fn(Ph, O))
    return {k: v for k, v in params.items() if v}, report


def apply(params, kind, P, form4, M=None):
    """P: (1,3) odhad modelu; form4 = (hform, aform, hrest, arest); M = pravděpodobnosti trhu (1,3) nebo None."""
    p = (params or {}).get(kind)
    if not p:
        return None
    X = features(P, *[np.array([v], float) for v in form4], M=M)
    return _probs(X, np.array(p["W"]), np.array(p["b"]))[0]


def save(params, report, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"params": params, "report": report}, ensure_ascii=False), encoding="utf-8")


def load(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("params", {})
    except Exception:  # noqa: BLE001
        return {}
