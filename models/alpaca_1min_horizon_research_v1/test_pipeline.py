from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import pipeline as p


def bars_for_day(day=date(2026, 1, 5), *, missing=(), close=100.0):
    op, cl = p._sessions(day, day)[day]
    stamps = pd.date_range(op, cl - timedelta(minutes=1), freq="1min")
    rows=[]
    for i, stamp in enumerate(stamps):
        if i in missing: continue
        px=close + i*0.001
        rows.append({"datetime":stamp,"session_date":day.isoformat(),"open":px,"high":px+.02,
                     "low":px-.02,"close":px,"volume":10.0,"trade_count":2})
    return pd.DataFrame(rows)


def slow_reference_targets(bars, horizon):
    """Small, deliberately row-oriented oracle for Definition A tests only."""
    ordered = bars.sort_values("datetime", kind="stable").reset_index(drop=True)
    by_time = {row.datetime: row for row in ordered.itertuples(index=False)}
    dates = ordered.datetime.dt.date
    sessions = p._sessions(dates.min(), dates.max()) if len(ordered) else {}
    result = []
    for row in ordered.itertuples(index=False):
        end = row.datetime + timedelta(minutes=horizon)
        bounds = sessions.get(row.datetime.date())
        in_session = bool(bounds and bounds[0] <= row.datetime
                           and end + timedelta(minutes=1) <= bounds[1])
        endpoint = by_time.get(end)
        if not in_session:
            label, ret, reason = "UNAVAILABLE", float("nan"), "session_or_endpoint_boundary"
        elif endpoint is None:
            label, ret, reason = "UNAVAILABLE", float("nan"), "missing_exact_endpoint"
        else:
            ret = (float(endpoint.close) / float(row.close) - 1.0) * 100.0
            label = "UP" if ret > .10 else "DOWN" if ret < -.10 else "NO_SIGNAL"
            reason = None
        result.append((row.datetime, end, ret, label, reason))
    return pd.DataFrame(result, columns=["datetime", "target_end", "future_return_pct", "label", "exclusion_reason"])


@pytest.mark.parametrize("horizon",[10,15])
def test_exact_endpoint_allows_missing_intervening_minutes(horizon):
    b=bars_for_day(missing={3,7})
    labels=p.endpoint_targets(b,horizon)
    # The endpoint exists, while two timestamps inside the prediction horizon
    # are absent. Definition A intentionally does not require a continuous path.
    t=p._sessions(date(2026, 1, 5),date(2026, 1, 5))[date(2026, 1, 5)][0]+timedelta(minutes=1)
    row=labels[labels.datetime==t].iloc[0]
    assert row.label in {"UP","DOWN","NO_SIGNAL"}
    assert row.target_end==t+timedelta(minutes=horizon)


@pytest.mark.parametrize("horizon",[10,15])
def test_missing_endpoint_is_unavailable(horizon):
    b=bars_for_day()
    target=b.datetime.iloc[100]+timedelta(minutes=horizon)
    b=b[b.datetime!=target]
    labels=p.endpoint_targets(b,horizon)
    decision=target-timedelta(minutes=horizon)
    assert labels.loc[labels.datetime==decision,"exclusion_reason"].iloc[0]=="missing_exact_endpoint"


def test_session_boundary_target_is_rejected():
    b=bars_for_day()
    last=b.iloc[-1].datetime
    label=p.endpoint_targets(b,10)
    row=label[label.datetime==last].iloc[0]
    assert row.label=="UNAVAILABLE" and row.exclusion_reason=="session_or_endpoint_boundary"


@pytest.mark.parametrize("ret,label",[(.10001,"UP"),(-.10001,"DOWN"),(.10,"NO_SIGNAL"),(-.10,"NO_SIGNAL")])
def test_strict_thresholds(ret,label):
    day=date(2026,1,5); op,_=p._sessions(day,day)[day]; t=op+timedelta(minutes=20); e=t+timedelta(minutes=10)
    b=pd.DataFrame([{"datetime":t,"session_date":str(day),"close":100.0},
                    {"datetime":e,"session_date":str(day),"close":100*(1+ret/100)}])
    assert p.endpoint_targets(b,10).iloc[0].label==label


def test_features_are_causal_and_reset_at_gap():
    b=bars_for_day(missing={40})
    f=p.causal_features(b)
    # Every feature at t depends only on the prefix ending at t.
    row=b[b.datetime==b.datetime.iloc[30]].iloc[0]
    prefix=b[b.datetime<=row.datetime]
    expected=p.causal_features(prefix).iloc[-1]
    actual=f[f.datetime==row.datetime].iloc[0]
    pd.testing.assert_series_equal(expected[p.FEATURES],actual[p.FEATURES],check_names=False)
    day=date(2026,1,5); session_open=p._sessions(day,day)[day][0]
    missing_time=session_open+timedelta(minutes=40)
    after=missing_time+timedelta(minutes=1)
    # Segment after a gap starts with missing rolling return/warmup values.
    after_features=f[f.datetime==after].iloc[0]
    assert pd.isna(after_features["return_1_pct"])
    # Return at the first post-gap candle is undefined; it cannot use the
    # pre-gap close as its lagged value. Subsequent values warm from this segment.
    assert pd.isna(after_features["return_3_pct"])
    second_after=f[f.datetime==after+timedelta(minutes=1)].iloc[0]
    expected_local_return=(b.loc[b.datetime==after+timedelta(minutes=1),"close"].iloc[0]
                           / b.loc[b.datetime==after,"close"].iloc[0]-1)*100
    assert second_after["return_1_pct"]==pytest.approx(expected_local_return)


def test_provenance_allowlist_rejects_other_path(tmp_path):
    with pytest.raises(ValueError,match="Only the approved"):
        p.verify_provenance(tmp_path/"other.csv")


def test_output_path_is_confined_to_new_research_directory():
    assert p.OUT.name=="alpaca_1min_horizon_research_v1"
    assert p.OUT.resolve().is_relative_to((p.ROOT/"models/alpaca_1min_horizon_research_v1").resolve())
    assert not (p.ROOT/"models/alpaca_1min_research_v1").resolve()==p.OUT.resolve()


@pytest.mark.parametrize("horizon",[10,15])
@pytest.mark.parametrize("day,missing",[
    (date(2026,1,5), {3,7,100}),  # missing path minutes plus exact endpoints
    (date(2025,11,28), {3,7,100}),  # synthetic early-close session
])
def test_vectorized_targets_match_slow_reference(horizon, day, missing):
    bars=bars_for_day(day,missing=missing)
    actual=p.endpoint_targets(bars,horizon)
    expected=slow_reference_targets(bars,horizon)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)


@pytest.mark.parametrize("horizon",[10,15])
def test_target_rows_are_deterministic_and_ordered(horizon):
    bars=bars_for_day(missing={3,7,100}).sample(frac=1,random_state=91).reset_index(drop=True)
    first=p.endpoint_targets(bars,horizon)
    second=p.endpoint_targets(bars,horizon)
    pd.testing.assert_frame_equal(first,second)
    assert first.datetime.is_monotonic_increasing
    assert first.datetime.is_unique
