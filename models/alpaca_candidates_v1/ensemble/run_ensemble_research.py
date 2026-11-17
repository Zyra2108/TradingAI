"""Leakage-controlled Alpaca ECL ensemble research.

Only train.csv and validation.csv are opened. Base-model OOF predictions are
generated with expanding chronological folds and date-boundary target purging.
Ensemble weights/meta-models/policy threshold use OOF predictions and labels
only. Validation is reserved for comparing the frozen candidate configurations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/alpaca_rebuild_v1"
MODEL_DIR = ROOT / "models/alpaca_candidates_v1"
OUT = MODEL_DIR / "ensemble"
SEED = 42
CLASSES = ["DOWN", "NEUTRAL", "UP"]
LABEL_TO_I = {c: i for i, c in enumerate(CLASSES)}
FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "ema_ratio_5",
    "ema_ratio_10", "ema_ratio_20", "rsi_14", "volatility_5",
    "volatility_10", "volatility_20", "candle_body", "candle_range",
    "upper_wick", "lower_wick", "minutes_since_open", "minutes_until_close",
    "day_of_week", "time_sin", "time_cos",
]
BASE_NAMES = ["lightgbm_conservative", "lightgbm_balanced_B_start", "multinomial_logistic"]
LGB_PARAMS = {
    "lightgbm_conservative": {
        "objective": "multiclass", "num_class": 3, "n_estimators": 350,
        "learning_rate": 0.02, "num_leaves": 15, "max_depth": -1,
        "min_child_samples": 60, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 5.0, "class_weight": None,
        "random_state": SEED, "n_jobs": -1, "verbosity": -1,
    },
    "lightgbm_balanced_B_start": {
        "objective": "multiclass", "num_class": 3, "n_estimators": 500,
        "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1,
        "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "class_weight": "balanced",
        "random_state": SEED, "n_jobs": -1, "verbosity": -1,
    },
}
LOGIT_PARAMS = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "class_weight": None, "random_state": SEED}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_split(name: str) -> pd.DataFrame:
    # Deliberately allowlist only these two split paths; there is no test reader.
    if name not in {"train", "validation"}:
        raise ValueError("Only train and validation split are allowed")
    f = pd.read_csv(DATA_DIR / f"{name}.csv")
    if "close.1" in f.columns:
        if not np.array_equal(f.close.to_numpy(), f["close.1"].to_numpy()):
            raise ValueError(f"Metadata/model close mismatch in {name}")
        f = f.drop(columns=["close.1"])
    if [c for c in f.columns if c in FEATURES] != FEATURES:
        raise ValueError(f"Feature order mismatch in {name}")
    f["datetime"] = pd.to_datetime(f.datetime, errors="raise")
    f["future_datetime"] = pd.to_datetime(f.future_datetime, errors="raise")
    for c in FEATURES:
        f[c] = pd.to_numeric(f[c], errors="raise")
    f["day_of_week"] = f.day_of_week.astype("int64")
    if f.datetime.duplicated().any() or not f.datetime.is_monotonic_increasing:
        raise ValueError(f"Bad timestamp order in {name}")
    if not (f.future_datetime == f.datetime + pd.Timedelta(minutes=15)).all():
        raise ValueError(f"Bad label horizon in {name}")
    if f[FEATURES].isna().any().any() or not np.isfinite(f[FEATURES].to_numpy(float)).all():
        raise ValueError(f"Bad feature values in {name}")
    if not set(f.target.astype(str)).issubset(CLASSES):
        raise ValueError(f"Unexpected class in {name}")
    return f


def make_preprocessor(scale: bool):
    nums = [c for c in FEATURES if c != "day_of_week"]
    nsteps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        nsteps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(nsteps), nums),
        ("weekday", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), ["day_of_week"]),
    ], remainder="drop")


def make_model(name: str):
    pre = make_preprocessor(scale=(name == "multinomial_logistic"))
    if name == "multinomial_logistic":
        return Pipeline([("preprocessor", pre), ("model", LogisticRegression(**LOGIT_PARAMS))])
    return Pipeline([("preprocessor", pre), ("model", lgb.LGBMClassifier(**LGB_PARAMS[name]))])


def align_proba(model, X):
    p0 = model.predict_proba(X)
    cls = [str(x) for x in model.classes_]
    p = np.zeros((len(X), 3), dtype=float)
    for i, c in enumerate(cls):
        p[:, LABEL_TO_I[c]] = p0[:, i]
    p /= p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-10):
        raise ValueError("Invalid class probabilities")
    return p


def date_folds(train: pd.DataFrame):
    dates = np.array(sorted(train.datetime.dt.date.unique()), dtype=object)
    # Expanding: first 40% dates seed fold 1; then predict the next contiguous
    # 20% block at each fold (roughly 60%, 80%, and 100% cumulative coverage).
    n = len(dates)
    cutpoints = [int(n * .40), int(n * .60), int(n * .80), n]
    folds = []
    for k in range(3):
        tr_dates = set(dates[:cutpoints[k]])
        va_dates = set(dates[cutpoints[k]:cutpoints[k + 1]])
        tr = train.datetime.dt.date.isin(tr_dates)
        va = train.datetime.dt.date.isin(va_dates)
        if not tr.any() or not va.any():
            continue
        first_valid = train.loc[va, "datetime"].min()
        keep = tr & (train.future_datetime < first_valid)
        folds.append((k + 1, keep, va, {
            "fold": k + 1, "train_dates": len(tr_dates), "predict_dates": len(va_dates),
            "train_date_start": str(min(tr_dates)), "train_date_end": str(max(tr_dates)),
            "predict_date_start": str(min(va_dates)), "predict_date_end": str(max(va_dates)),
            "train_rows_before_purge": int(tr.sum()), "train_rows_after_purge": int(keep.sum()),
            "purged_rows": int((tr & ~keep).sum()), "first_prediction_timestamp": str(first_valid),
        }))
    return folds


def fit_oof(train):
    folds = date_folds(train)
    oof = np.full((len(train), len(BASE_NAMES), 3), np.nan, dtype=float)
    fold_id = np.full(len(train), -1, dtype=int)
    fold_meta = []
    for k, tr_mask, va_mask, meta in folds:
        Xtr, ytr = train.loc[tr_mask, FEATURES], train.loc[tr_mask, "target"].astype(str).to_numpy()
        Xva = train.loc[va_mask, FEATURES]
        idx = np.flatnonzero(va_mask.to_numpy())
        for mi, name in enumerate(BASE_NAMES):
            model = make_model(name)
            model.fit(Xtr, ytr)
            oof[idx, mi, :] = align_proba(model, Xva)
        fold_id[idx] = k
        fold_meta.append(meta)
    good = (fold_id >= 0) & np.isfinite(oof).all(axis=(1, 2))
    return oof[good], train.loc[good, "target"].astype(str).to_numpy(), train.loc[good].reset_index(drop=True), fold_id[good], fold_meta


def fit_simplex_weights(P, y_idx, sample_mask=None, class_specific=False):
    # P [rows, models, classes]. Convex weights. All fitting uses OOF rows only.
    if sample_mask is not None:
        P, y_idx = P[sample_mask], y_idx[sample_mask]
    m = P.shape[1]
    if class_specific:
        def obj(flat):
            W = flat.reshape(3, m)
            scores = np.einsum("nmc,cm->nc", P, W)
            scores = np.clip(scores, 1e-9, None)
            scores /= scores.sum(axis=1, keepdims=True)
            return -np.log(scores[np.arange(len(y_idx)), y_idx]).mean() + .03 * np.square(W - 1/m).sum()
        x0 = np.full((3, m), 1/m)
        cons = [{"type": "eq", "fun": lambda x, c=c: np.sum(x.reshape(3, m)[c]) - 1} for c in range(3)]
        res = minimize(obj, x0.ravel(), method="SLSQP", bounds=[(0, 1)] * (3 * m), constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            raise RuntimeError(f"class-specific weighting failed: {res.message}")
        return res.x.reshape(3, m)
    def obj(w):
        scores = np.einsum("nmc,m->nc", P, w)
        scores = np.clip(scores, 1e-9, None)
        scores /= scores.sum(axis=1, keepdims=True)
        return -np.log(scores[np.arange(len(y_idx)), y_idx]).mean() + .03 * np.square(w - 1/m).sum()
    res = minimize(obj, np.full(m, 1/m), method="SLSQP", bounds=[(0, 1)] * m,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        raise RuntimeError(f"weighted soft vote failed: {res.message}")
    return res.x


def combine(P, method, weights=None, meta=None, regimes=None, regime_weights=None):
    if method == "equal_soft_vote":
        q = P.mean(axis=1)
    elif method == "weighted_soft_vote":
        q = np.einsum("nmc,m->nc", P, weights)
    elif method == "class_specific_weights":
        q = np.einsum("nmc,cm->nc", P, weights)
    elif method == "stacking_logistic":
        q = meta.predict_proba(P.reshape(len(P), -1))
        z = np.zeros((len(P), 3))
        for j, c in enumerate(meta.classes_): z[:, LABEL_TO_I[str(c)]] = q[:, j]
        q = z
    elif method == "regime_weighted":
        q = np.zeros((len(P), 3))
        for r in ["trending", "sideways", "high_volatility"]:
            mask = regimes == r
            w = regime_weights.get(r, weights)
            q[mask] = np.einsum("nmc,m->nc", P[mask], w)
    else:
        raise KeyError(method)
    q = np.clip(q, 1e-10, None)
    return q / q.sum(axis=1, keepdims=True)


def select_threshold(P, y, agreement=None, minimum_coverage=.20):
    # One fixed OOF-only procedure: maximize directional macro-F1 subject to
    # at least 20% action coverage. A small candidate set is derived from OOF
    # confidence quantiles, not outer validation predictions.
    base = np.asarray(CLASSES)[P.argmax(1)]
    conf = P.max(1)
    candidates = np.unique(np.quantile(conf, np.linspace(0, 1, 101)))
    best = None
    for t in candidates:
        action = np.isin(base, ["DOWN", "UP"]) & (conf >= t)
        if agreement is not None:
            action &= agreement
        if action.mean() < minimum_coverage:
            continue
        pred = base.copy(); pred[~action] = "NEUTRAL"
        score = f1_score(y, pred, labels=["DOWN", "UP"], average="macro", zero_division=0)
        item = (float(score), float(action.mean()), float(t))
        if best is None or item[0] > best[0] or (item[0] == best[0] and item[1] > best[1]): best = item
    return (best[2], best[1], best[0]) if best else (1.0, 0.0, 0.0)


def apply_selective(P, threshold, agreement=None):
    pred = np.asarray(CLASSES)[P.argmax(1)].copy()
    action = np.isin(pred, ["DOWN", "UP"]) & (P.max(1) >= threshold)
    if agreement is not None: action &= agreement
    pred[~action] = "NEUTRAL"
    return pred


def make_regimes(frame, vol_q, trend_q):
    v = frame.volatility_20.to_numpy(float)
    t = frame.sma_ratio_20.abs().to_numpy(float)
    out = np.full(len(frame), "sideways", dtype=object)
    out[t >= trend_q] = "trending"
    out[v >= vol_q] = "high_volatility"  # high volatility takes precedence
    return out


def metrics(name, y, pred, P):
    dlabels = ["DOWN", "UP"]
    return {
        "model": name, "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1_3class": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "directional_precision_macro": float(precision_score(y, pred, labels=dlabels, average="macro", zero_division=0)),
        "directional_recall_macro": float(recall_score(y, pred, labels=dlabels, average="macro", zero_division=0)),
        "directional_f1_macro": float(f1_score(y, pred, labels=dlabels, average="macro", zero_division=0)),
        "directional_coverage": float(np.isin(pred, dlabels).mean()),
        "log_loss": float(log_loss(y, P, labels=CLASSES)),
        "predicted_distribution": {c: int((pred == c).sum()) for c in CLASSES},
        "confusion_matrix_labels_rows_actual_cols_predicted": CLASSES,
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allowed_outputs = {
        Path(__file__).name, "__pycache__", "validation_metrics.csv", "validation_metrics_full.json",
        "validation_ensemble_predictions.csv", "validation_confusion_matrices.csv",
        "validation_per_class_metrics.csv", "training_oof_predictions.csv",
        "equal_soft_vote.joblib", "weighted_soft_vote.joblib", "class_specific_weights.joblib",
        "stacking_logistic.joblib", "regime_weighted.joblib", "selective_weighted_confidence.joblib",
        "selective_weighted_agreement.joblib", "ensemble_config.json", "ensemble_report.md",
    }
    existing = [p.name for p in OUT.iterdir() if p.name not in allowed_outputs]
    if existing: raise FileExistsError(f"Refusing to overwrite unrecognized ensemble outputs: {existing}")
    train = load_split("train")
    valid = load_split("validation")
    if train.datetime.max() >= valid.datetime.min() or train.future_datetime.max() >= valid.datetime.min():
        raise ValueError("Chronological separation or purge check failed")

    P_oof, y_oof, oof_frame, oof_fold, folds = fit_oof(train)
    yidx = np.array([LABEL_TO_I[x] for x in y_oof])
    p_corr = np.corrcoef(np.concatenate([P_oof[:, i, :] for i in range(3)], axis=1), rowvar=False)
    pred_oof = np.asarray(CLASSES)[P_oof.argmax(2)]
    agreement_oof = np.zeros(len(y_oof), dtype=bool)
    for i, pred in enumerate(pred_oof):
        direction = np.asarray(pred)
        if direction[1] in ("UP", "DOWN"):
            agreement_oof[i] = (direction == direction[1]).sum() >= 2

    # Regime thresholds are fitted only on training features; each OOF fold's
    # regime is classified using its own fold-training quantiles.
    oof_regime = np.empty(len(y_oof), dtype=object)
    for f in folds:
        k = f["fold"]
        sel = oof_fold == k
        # Recover that fold's pre-purge training rows by temporal position.
        first_pred = pd.Timestamp(f["first_prediction_timestamp"])
        fold_train = train.loc[train.datetime < first_pred]
        vq = float(fold_train.volatility_20.quantile(.75))
        tq = float(fold_train.sma_ratio_20.abs().quantile(.75))
        oof_regime[sel] = make_regimes(oof_frame.loc[sel].reset_index(drop=True), vq, tq)

    # Learn overall and class-specific voting weights from train OOF only.
    w_global = fit_simplex_weights(P_oof, yidx)
    w_class = fit_simplex_weights(P_oof, yidx, class_specific=True)
    # Regime-specific weights only for adequately represented OOF regimes.
    regime_counts = {r: int((oof_regime == r).sum()) for r in ["trending", "sideways", "high_volatility"]}
    regime_weights = {}
    for r, n in regime_counts.items():
        regime_weights[r] = fit_simplex_weights(P_oof, yidx, sample_mask=(oof_regime == r)) if n >= 300 else w_global
    # OOF-trained stacking model. The three base models' class probabilities
    # (9 fixed inputs) are the meta-features; scaling is learned on OOF only.
    stack = Pipeline([("scale", StandardScaler()), ("meta", LogisticRegression(C=.5, max_iter=1500, solver="lbfgs", random_state=SEED))])
    stack.fit(P_oof.reshape(len(P_oof), -1), y_oof)
    # Frozen selective thresholds are learned only on OOF predictions.
    oof_weighted = combine(P_oof, "weighted_soft_vote", weights=w_global)
    conf_threshold, conf_cov, conf_f1 = select_threshold(oof_weighted, y_oof, minimum_coverage=.20)
    agree_threshold, agree_cov, agree_f1 = select_threshold(oof_weighted, y_oof, agreement=agreement_oof, minimum_coverage=.20)

    # Validation inference: load the already train-fitted individual candidate
    # artifacts; validation labels are not passed into any fit/weight/policy step.
    Xv = valid[FEATURES]
    Pv = []
    for name in BASE_NAMES:
        fitted = joblib.load(MODEL_DIR / f"{name}.joblib")
        Pv.append(align_proba(fitted, Xv))
    P_valid = np.stack(Pv, axis=1)
    yv = valid.target.astype(str).to_numpy()
    # Fit final regime thresholds from all training features only.
    vq_final = float(train.volatility_20.quantile(.75))
    tq_final = float(train.sma_ratio_20.abs().quantile(.75))
    valid_regime = make_regimes(valid.reset_index(drop=True), vq_final, tq_final)
    val_base_preds = np.asarray(CLASSES)[P_valid.argmax(2)]
    val_agreement = np.array([p[1] in ("UP", "DOWN") and (p == p[1]).sum() >= 2 for p in val_base_preds])

    configs = {}
    configs["equal_soft_vote"] = combine(P_valid, "equal_soft_vote")
    configs["weighted_soft_vote"] = combine(P_valid, "weighted_soft_vote", weights=w_global)
    configs["class_specific_weights"] = combine(P_valid, "class_specific_weights", weights=w_class)
    configs["stacking_logistic"] = combine(P_valid, "stacking_logistic", meta=stack)
    configs["regime_weighted"] = combine(P_valid, "regime_weighted", weights=w_global, regimes=valid_regime, regime_weights=regime_weights)

    # Confidence/consensus selective policies are applied to the learned weighted vote.
    selective_conf_pred = apply_selective(configs["weighted_soft_vote"], conf_threshold)
    selective_agree_pred = apply_selective(configs["weighted_soft_vote"], agree_threshold, val_agreement)
    final_metrics = []
    pred_rows = []
    confusions = []
    class_rows = []
    # Reuse saved validation predictions for the individual baselines.
    individual = pd.read_csv(MODEL_DIR / "validation_predictions.csv")
    for name in BASE_NAMES:
        sub = individual.loc[individual.model == name].copy()
        if sub.empty: raise ValueError(f"No saved validation predictions for {name}")
        p = sub[["p_down", "p_neutral", "p_up"]].to_numpy(float)
        final_metrics.append(metrics(name, yv, sub.prediction.to_numpy(), p))
    for name, p in configs.items():
        pred = np.asarray(CLASSES)[p.argmax(1)]
        final_metrics.append(metrics(name, yv, pred, p))
    final_metrics.append(metrics("weighted_soft_vote_selective_confidence", yv, selective_conf_pred, configs["weighted_soft_vote"]))
    final_metrics.append(metrics("weighted_soft_vote_selective_agreement", yv, selective_agree_pred, configs["weighted_soft_vote"]))

    # Validation-only majority baseline; also report always-neutral accuracy.
    most = pd.Series(yv).mode().iloc[0]
    dummy_pred = np.full(len(yv), most, dtype=object)
    dummy_prob = np.zeros((len(yv), 3)); dummy_prob[:, LABEL_TO_I[most]] = 1
    final_metrics.insert(0, metrics("dummy_most_frequent", yv, dummy_pred, dummy_prob))

    all_pred = {name: np.asarray(CLASSES)[p.argmax(1)] for name, p in configs.items()}
    all_pred["weighted_soft_vote_selective_confidence"] = selective_conf_pred
    all_pred["weighted_soft_vote_selective_agreement"] = selective_agree_pred
    all_probs = dict(configs)
    all_probs["weighted_soft_vote_selective_confidence"] = configs["weighted_soft_vote"]
    all_probs["weighted_soft_vote_selective_agreement"] = configs["weighted_soft_vote"]
    for name, p in all_probs.items():
        df = pd.DataFrame({"datetime": valid.datetime.astype(str), "actual": yv, "prediction": all_pred[name], "regime": valid_regime})
        for i, c in enumerate(CLASSES): df[f"p_{c.lower()}"] = p[:, i]
        df.insert(0, "model", name); pred_rows.append(df)
    for m in final_metrics:
        cm = np.asarray(m["confusion_matrix"])
        confusions.extend({"model": m["model"], "actual": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j])} for i in range(3) for j in range(3))
        for c in CLASSES:
            ci = LABEL_TO_I[c]
            tp = int(cm[ci, ci]); fp = int(cm[:, ci].sum() - tp); fn = int(cm[ci, :].sum() - tp)
            pr = tp / (tp + fp) if tp + fp else 0.0
            re = tp / (tp + fn) if tp + fn else 0.0
            f1c = 2 * pr * re / (pr + re) if pr + re else 0.0
            class_rows.append({"model": m["model"], "class": c, "precision": pr, "recall": re, "f1": f1c, "support": int(cm[ci, :].sum())})

    pd.DataFrame([{k: v for k, v in m.items() if k not in {"confusion_matrix", "confusion_matrix_labels_rows_actual_cols_predicted", "predicted_distribution"}} for m in final_metrics]).to_csv(OUT / "validation_metrics.csv", index=False)
    (OUT / "validation_metrics_full.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "validation_ensemble_predictions.csv", index=False)
    pd.DataFrame(confusions).to_csv(OUT / "validation_confusion_matrices.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "validation_per_class_metrics.csv", index=False)

    # OOF diagnostic file contains only train-period rows with fold predictions.
    oof_rows = []
    for i in range(len(y_oof)):
        row = {"datetime": str(oof_frame.iloc[i].datetime), "actual": y_oof[i], "fold": int(oof_fold[i]), "regime": str(oof_regime[i])}
        for mi, model_name in enumerate(BASE_NAMES):
            for ci, cls in enumerate(CLASSES): row[f"{model_name}_p_{cls.lower()}"] = float(P_oof[i, mi, ci])
            row[f"{model_name}_prediction"] = CLASSES[int(P_oof[i, mi].argmax())]
        oof_rows.append(row)
    pd.DataFrame(oof_rows).to_csv(OUT / "training_oof_predictions.csv", index=False)

    # Save combination payloads with the independently trained base models.
    base_fitted = {n: joblib.load(MODEL_DIR / f"{n}.joblib") for n in BASE_NAMES}
    bundles = {
        "equal_soft_vote": {"method": "equal_soft_vote", "base_names": BASE_NAMES, "base_models": base_fitted},
        "weighted_soft_vote": {"method": "weighted_soft_vote", "weights": w_global, "base_names": BASE_NAMES, "base_models": base_fitted},
        "class_specific_weights": {"method": "class_specific_weights", "weights_by_class": w_class, "class_order": CLASSES, "base_names": BASE_NAMES, "base_models": base_fitted},
        "stacking_logistic": {"method": "stacking_logistic", "meta_model": stack, "base_names": BASE_NAMES, "base_models": base_fitted},
        "regime_weighted": {"method": "regime_weighted", "global_weights": w_global, "regime_weights": regime_weights, "regime_thresholds": {"volatility_20_q75": vq_final, "abs_sma_ratio_20_q75": tq_final}, "base_names": BASE_NAMES, "base_models": base_fitted},
        "selective_weighted_confidence": {"method": "weighted_soft_vote_selective_confidence", "weights": w_global, "confidence_threshold": conf_threshold, "base_names": BASE_NAMES, "base_models": base_fitted},
        "selective_weighted_agreement": {"method": "weighted_soft_vote_selective_agreement", "weights": w_global, "confidence_threshold": agree_threshold, "agreement_votes_required": 2, "base_names": BASE_NAMES, "base_models": base_fitted},
    }
    for name, bundle in bundles.items(): joblib.dump(bundle, OUT / f"{name}.joblib", compress=3)

    oof_stats = {}
    for i, name in enumerate(BASE_NAMES):
        yp = np.asarray(CLASSES)[P_oof[:, i].argmax(1)]
        oof_stats[name] = metrics(name, y_oof, yp, P_oof[:, i])
    pred_corr = pd.DataFrame(pred_oof, columns=BASE_NAMES).apply(lambda s: s.map(LABEL_TO_I)).corr()
    probability_corr = np.corrcoef(np.concatenate([P_oof[:, i, :] for i in range(3)], axis=1), rowvar=False)
    # pairwise per-model flattened probability correlations, more interpretable than 9x9 feature correlation.
    model_prob_corr = np.eye(3)
    for i in range(3):
        for j in range(3):
            if i != j:
                model_prob_corr[i, j] = np.corrcoef(P_oof[:, i, :].ravel(), P_oof[:, j, :].ravel())[0, 1]
    agree_matrix = pd.DataFrame(pred_oof, columns=BASE_NAMES).eq(pd.DataFrame(pred_oof, columns=BASE_NAMES).shift(axis="columns"))
    agreement_rates = {f"{BASE_NAMES[i]}__{BASE_NAMES[j]}": float((pred_oof[:, i] == pred_oof[:, j]).mean()) for i in range(3) for j in range(i+1, 3)}
    model_contrib = {n: {"global_weight": float(w_global[i]), "class_weights": {c: float(w_class[j, i]) for j, c in enumerate(CLASSES)}, "regime_weights": {r: float(regime_weights[r][i]) for r in regime_weights}} for i, n in enumerate(BASE_NAMES)}
    config = {
        "inputs": ["data/alpaca_rebuild_v1/train.csv", "data/alpaca_rebuild_v1/validation.csv"], "test_data_accessed": False,
        "train_sha256": sha256(DATA_DIR / "train.csv"), "validation_sha256": sha256(DATA_DIR / "validation.csv"),
        "features_exact_order": FEATURES, "class_order": CLASSES, "base_models": BASE_NAMES,
        "base_hyperparameters": {**LGB_PARAMS, "multinomial_logistic": LOGIT_PARAMS},
        "folds": folds, "oof_rows": int(len(y_oof)), "oof_date_range": [str(oof_frame.datetime.min().date()), str(oof_frame.datetime.max().date())],
        "oof_label_metrics_are": "Base-model expanding-window predictions on their OOF blocks. Ensemble weights/meta-models/policy were fit using these same OOF rows; any in-sample ensemble-fit score is not an unbiased generalization estimate.",
        "preprocessing": "Each OOF fold independently fits median numeric imputation; day_of_week most-frequent imputation + one-hot. Logistic additionally scales numeric features. Validation base models are existing train-only candidate artifacts.",
        "weight_learning": {"global": "simplex-constrained soft voting minimizes OOF multiclass log loss with mild L2 shrinkage toward equal weights", "class_specific": "per-class simplex weights, same OOF loss and shrinkage", "regime": "weights fitted separately only if regime OOF n>=300; otherwise fall back to global OOF weights"},
        "regime_method": "Causal thresholds: volatility_20 >= fold-training 75th percentile means high_volatility; else abs(sma_ratio_20) >= fold-training 75th percentile means trending; else sideways. Final validation thresholds are estimated from all train rows only; high volatility takes precedence.",
        "regime_oof_counts": regime_counts, "regime_validation_counts": {r: int((valid_regime == r).sum()) for r in ["trending", "sideways", "high_volatility"]},
        "regime_minimum_rows": 300, "global_weights_model_order": BASE_NAMES, "global_weights": w_global.tolist(),
        "class_specific_weights_rows_class_cols_model": {c: w_class[i].tolist() for i, c in enumerate(CLASSES)},
        "regime_weights_model_order": BASE_NAMES, "regime_weights": {r: np.asarray(w).tolist() for r, w in regime_weights.items()},
        "selective_policy": {"threshold_source": "training OOF only", "minimum_directional_coverage_constraint": 0.20, "confidence_threshold": conf_threshold, "oof_coverage": conf_cov, "oof_directional_macro_f1": conf_f1, "agreement_threshold": agree_threshold, "agreement_oof_coverage": agree_cov, "agreement_oof_directional_macro_f1": agree_f1, "agreement_votes_required": 2},
        "training_oof_base_metrics": oof_stats, "prediction_agreement_rates": agreement_rates,
        "flattened_probability_correlation_matrix_order": BASE_NAMES, "flattened_probability_correlation_matrix": model_prob_corr.tolist(),
        "prediction_label_correlation_matrix_order": BASE_NAMES, "prediction_label_correlation_matrix": pred_corr.to_numpy().tolist(),
        "individual_model_contributions": model_contrib,
        "selection_protocol": "All weights, meta-model and selective threshold frozen from training-only chronological OOF predictions before validation scoring. Validation compares these fixed candidates; no refitting or repeated threshold search on validation.",
        "versions": {"python": __import__("sys").version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__},
    }
    (OUT / "ensemble_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report_table = ["| Candidate | Accuracy | Balanced accuracy | Macro-F1 | Directional precision | Directional recall | Directional F1 | Coverage | Log loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in final_metrics:
        report_table.append(f"| {m['model']} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1_3class']:.3f} | {m['directional_precision_macro']:.3f} | {m['directional_recall_macro']:.3f} | {m['directional_f1_macro']:.3f} | {m['directional_coverage']:.1%} | {m['log_loss']:.3f} |")
    oof_table = ["| Base model (training OOF only) | Accuracy | Balanced accuracy | Macro-F1 | Directional F1 | Coverage | Log loss |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in oof_stats.items():
        oof_table.append(f"| {name} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1_3class']:.3f} | {m['directional_f1_macro']:.3f} | {m['directional_coverage']:.1%} | {m['log_loss']:.3f} |")
    report_lines = ["# Alpaca ECL custom ensemble research", "", f"Training-only OOF rows: {len(y_oof):,} across expanding folds; validation rows: {len(yv):,}. Test was not opened.", "", "## Validation results", "", *report_table, "", "Selective decisions return NEUTRAL when the policy abstains; the reported log loss uses the underlying probability vector before abstention. See `validation_confusion_matrices.csv` for counts. Metrics use DOWN, NEUTRAL, UP order.", "", "The equal vote is the strongest ensemble on validation, but it trails the balanced individual LightGBM on three-class macro-F1 and directional F1. No ensemble is promoted.", "", "## OOF construction and leakage controls", "", "- Base predictions use expanding chronological folds: train on earlier dates, predict the next block; the training fit is purged where `future_datetime` reaches the first predicted timestamp.", "- Each base-model preprocessing pipeline is fitted separately inside each fold. The full-training individual pipelines were reused only for validation inference.", "- Global/class-specific/regime weights, stacking logistic meta-model and selective thresholds are learned from training OOF predictions only. Validation labels are used only for final metric calculation and comparison.", "- OOF base-model scores are out-of-fold. Ensemble scores on those same rows are in-sample for the combiner and are not an unbiased estimate.", "", "## Training-only OOF base-model diagnostics", "", *oof_table, "", "## Ensemble behavior", "", f"Global weights in order `{BASE_NAMES}`: `{np.round(w_global, 4).tolist()}`.", f"OOF regime counts: `{regime_counts}`; minimum for regime-specific weights: 300. Thin regimes fall back to global weights.", f"Selective confidence threshold: {conf_threshold:.4f}; OOF coverage {conf_cov:.1%}. Agreement policy requires at least 2 of 3 base models to agree on UP or DOWN; its confidence threshold is {agree_threshold:.4f} (OOF coverage {agree_cov:.1%}).", f"OOF pairwise prediction agreement: `{agreement_rates}`.", "", "The individual balanced LightGBM from the parent candidate directory remains the fallback. Review macro-F1, directional precision/recall/F1, coverage and log loss together; complexity alone is not evidence of value.", ""]
    (OUT / "ensemble_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"oof_rows": len(y_oof), "folds": folds, "global_weights": w_global.tolist(), "regime_counts": regime_counts, "metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
