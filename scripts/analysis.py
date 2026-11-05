"""Descriptive analysis for any agricultural CSV: quality checks, group comparisons,
trends and a plain-language summary.

Everything here is pandas/scipy only, so the dashboard and the test suite share one
implementation. Nothing in this module trains a model; it answers the questions an analyst
asks of a new dataset before modelling it: is the data fit to use, how do groups differ,
what is changing over time, and how does one place compare with its peers.

Statistical caveat that applies throughout: agricultural panels repeat the same unit
(a country, a farm, a plot) across seasons, so rows are not independent. Confidence
intervals and tests here describe the rows as given; they are a first look, not a
substitute for a model that accounts for repeated measures.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# IITA's mandate crops, matched against free-text crop labels. FAOSTAT spells these
# "Cassava, fresh", "Yams", "Cow peas, dry", "Soya beans", so matching is by token.
IITA_MANDATE_CROPS: dict[str, tuple[str, ...]] = {
    "Cassava": ("cassava",),
    "Yam": ("yam", "yams"),
    "Maize": ("maize", "corn"),
    "Cowpea": ("cowpea", "cowpeas", "cow pea", "cow peas"),
    "Soybean": ("soybean", "soybeans", "soya", "soy"),
    "Banana & plantain": ("banana", "bananas", "plantain", "plantains"),
}

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
SEVERITY_PENALTY = {"high": 12, "medium": 5, "low": 1}

# Column-name tokens for quantities that cannot be negative.
NON_NEGATIVE_TOKENS = (
    "yield", "area", "production", "size", "ha", "kg", "ton", "tons", "tonnes", "price",
    "count", "age", "distance", "km", "rainfall", "precipitation", "pct", "percent",
    "visits", "income", "cost", "quantity", "qty", "weight",
)
PERCENT_TOKENS = ("pct", "percent", "percentage", "share", "rate_pct")
# Columns that identify rows or periods; outlier checks on them are meaningless.
NON_MEASURE_TOKENS = ("id", "year", "month", "day", "lat", "lon", "latitude", "longitude", "code", "index")

UNIT_SUFFIXES: list[tuple[str, str]] = [
    ("_kg_per_ha", "kg/ha"),
    ("_t_per_ha", "t/ha"),
    ("_tons_per_ha", "t/ha"),
    ("_tonnes", "t"),
    ("_tons", "t"),
    ("_kg", "kg"),
    ("_ha", "ha"),
    ("_pct", "%"),
    ("_percent", "%"),
    ("_mm", "mm"),
    ("_km", "km"),
    ("_usd", "US$"),
    ("_ppm", "ppm"),
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _tokens(name: object) -> set[str]:
    return set(t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t)


def _has_token(name: object, tokens: tuple[str, ...]) -> bool:
    return bool(_tokens(name) & set(tokens))


def measure_unit(column: object) -> str:
    """Best-effort unit from a column name, e.g. ``yield_kg_per_ha`` -> ``kg/ha``."""
    name = str(column or "").lower()
    for suffix, unit in UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    return ""


def measure_label(column: object) -> str:
    """Readable label without the unit suffix: ``yield_kg_per_ha`` -> ``Yield``."""
    name = str(column or "")
    for suffix, _ in UNIT_SUFFIXES:
        if name.lower().endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    clean = re.sub(r"_+", " ", name).strip()
    return clean[:1].upper() + clean[1:] if clean else "Value"


def fmt_value(value: object, unit: str = "") -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    magnitude = abs(number)
    if magnitude >= 100:
        text = f"{number:,.0f}"
    elif magnitude >= 10:
        text = f"{number:,.1f}"
    else:
        text = f"{number:,.2f}"
    if not unit:
        return text
    return f"{text}{unit}" if unit == "%" else f"{text} {unit}"


def iita_crop_group(label: object) -> str | None:
    """Map a crop label to its IITA mandate-crop group, or None."""
    words = _tokens(label)
    text = " ".join(re.split(r"[^a-z0-9]+", str(label).lower()))
    for group, keys in IITA_MANDATE_CROPS.items():
        for key in keys:
            if (" " in key and key in text) or key in words:
                return group
    return None


def period_values(series: pd.Series) -> pd.Series:
    """Turn a year column or a date column into integer years (NaN where unparseable)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.5 and numeric.dropna().between(1800, 2200).all():
        return numeric.round()
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.dt.year.astype("float64")


def _normalise_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


# -----------------------------------------------------------------------------
# Data quality
# -----------------------------------------------------------------------------
def _issue(
    check: str,
    severity: str,
    column: str,
    count: int,
    detail: str,
    examples: list[object] | None = None,
) -> dict[str, Any]:
    shown = ", ".join(str(e) for e in (examples or [])[:5])
    return {
        "check": check,
        "severity": severity,
        "column": column,
        "rows_affected": int(count),
        "detail": detail,
        "examples": shown,
    }


def _robust_outlier_mask(values: pd.Series, groups: pd.Series | None, threshold: float) -> pd.Series:
    """Flag values far from their group median, measured in MADs.

    Positive, skewed measures (yields, areas, prices) are compared on a log scale, so a
    cassava yield ten times a cowpea yield is not an outlier but a yield ten times the
    other cassava yields is.
    """
    clean = values.dropna()
    use_log = bool(len(clean)) and (clean >= 0).all() and clean.skew() > 1
    scaled = np.log1p(values) if use_log else values
    if groups is None:
        keys = pd.Series(0, index=values.index)
    else:
        keys = groups.astype(str).where(groups.notna(), "(missing)")
    grouped = scaled.groupby(keys)
    median = grouped.transform("median")
    mad = (scaled - median).abs().groupby(keys).transform("median")
    size = grouped.transform("count")
    # 1.4826 makes the MAD comparable to a standard deviation under normality.
    z = (scaled - median) / (1.4826 * mad.replace(0, np.nan))
    return (z.abs() > threshold) & (size >= 8)


def quality_report(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
    group_column: str | None = None,
    outlier_threshold: float = 5.0,
) -> dict[str, Any]:
    """Run the checks an analyst does before trusting a new dataset.

    Returns ``issues`` (one dict per finding, most severe first), a per-column
    ``profile`` DataFrame, and a 0-100 ``score`` that falls with each finding.
    """
    issues: list[dict[str, Any]] = []
    n_rows = int(len(df))
    profile_rows: list[dict[str, Any]] = []

    if n_rows == 0:
        return {
            "n_rows": 0,
            "n_columns": int(df.shape[1]),
            "issues": [_issue("Empty dataset", "high", "(all)", 0, "The file has no data rows.")],
            "profile": pd.DataFrame(),
            "score": 0,
        }

    current_year = date.today().year
    group_series = df[group_column] if group_column and group_column in df.columns else None

    for col in df.columns:
        series = df[col]
        name = str(col)
        n_missing = int(series.isna().sum())
        missing_ratio = n_missing / n_rows
        non_null = series.dropna()
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = pd.api.types.is_numeric_dtype(series)
        profile_rows.append(
            {
                "column": name,
                "type": "number" if is_numeric else "text",
                "missing_pct": round(missing_ratio * 100, 1),
                "unique": int(non_null.nunique()),
                "example": str(non_null.iloc[0]) if len(non_null) else "",
            }
        )

        if n_missing == n_rows:
            issues.append(_issue("Empty column", "high", name, n_rows, "Every value is missing; drop the column."))
            continue
        if missing_ratio >= 0.2:
            issues.append(
                _issue(
                    "Many missing values", "medium", name, n_missing,
                    f"{missing_ratio:.0%} of rows are blank. Check whether blanks mean zero, "
                    "not collected, or not applicable before imputing.",
                )
            )
        elif missing_ratio >= 0.05:
            issues.append(
                _issue("Some missing values", "low", name, n_missing, f"{missing_ratio:.0%} of rows are blank.")
            )

        if non_null.nunique() == 1 and n_rows > 1:
            issues.append(
                _issue("Constant column", "low", name, n_rows, f"Only one value ({non_null.iloc[0]}); it carries no information.")
            )

        if not is_numeric:
            text = non_null.astype(str)
            parsed = pd.to_numeric(text.str.strip().str.replace(",", "", regex=False), errors="coerce")
            parse_ratio = float(parsed.notna().mean()) if len(text) else 0.0
            if 0.8 <= parse_ratio < 1.0:
                bad = text[parsed.isna()]
                issues.append(
                    _issue(
                        "Numbers stored as text", "medium", name, len(bad),
                        "Most values are numbers but some are not, so the column is read as text "
                        "and cannot be averaged. Fix or blank the entries shown.",
                        bad.unique().tolist(),
                    )
                )
                numeric = parsed.reindex(series.index)
            else:
                padded = text[text != text.str.strip()]
                if len(padded):
                    issues.append(
                        _issue(
                            "Extra spaces", "low", name, len(padded),
                            "Leading or trailing spaces make identical labels count as different.",
                            [repr(v) for v in padded.unique()[:5]],
                        )
                    )
                if 1 < text.nunique() <= 1000:
                    spellings = (
                        pd.DataFrame({"raw": text.str.strip(), "key": text.map(_normalise_label)})
                        .drop_duplicates()
                        .groupby("key")["raw"]
                        .agg(list)
                    )
                    variants = spellings[spellings.map(len) > 1]
                    if len(variants):
                        affected = int(text.str.strip().isin([v for vs in variants for v in vs]).sum())
                        issues.append(
                            _issue(
                                "Inconsistent labels", "medium", name, affected,
                                f"{len(variants)} value(s) are spelled more than one way and will be "
                                "counted as separate groups.",
                                [" / ".join(vs) for vs in variants.head(5)],
                            )
                        )
                continue

        valid = numeric.dropna()
        if valid.empty:
            continue
        is_quantity = _has_token(name, NON_NEGATIVE_TOKENS)
        is_identifier = _has_token(name, NON_MEASURE_TOKENS) and not is_quantity

        if is_quantity:
            negatives = valid[valid < 0]
            if len(negatives):
                issues.append(
                    _issue(
                        "Impossible negative values", "high", name, len(negatives),
                        "This quantity cannot be below zero.", negatives.head(5).tolist(),
                    )
                )
        if _has_token(name, PERCENT_TOKENS):
            over = valid[valid > 100]
            if len(over):
                issues.append(
                    _issue(
                        "Percentage above 100", "medium", name, len(over),
                        "A percentage column holds values over 100.", over.head(5).tolist(),
                    )
                )
        if "year" in _tokens(name) and not is_quantity:
            bad_years = valid[(valid < 1900) | (valid > current_year + 1)]
            if len(bad_years):
                issues.append(
                    _issue(
                        "Implausible year", "high", name, len(bad_years),
                        f"Years should fall between 1900 and {current_year + 1}.",
                        bad_years.head(5).tolist(),
                    )
                )

        if not is_identifier and valid.nunique() > 10 and len(valid) >= 20:
            groups = group_series if group_series is not None and name != group_column else None
            mask = _robust_outlier_mask(numeric, groups, outlier_threshold).fillna(False)
            flagged = numeric[mask]
            if len(flagged):
                scope = f" within each {group_column}" if groups is not None else ""
                share = len(flagged) / len(valid)
                issues.append(
                    _issue(
                        "Extreme values", "medium" if share > 0.01 else "low", name, len(flagged),
                        f"Values far from the typical level{scope}. They may be real, but check them "
                        "for unit or data-entry errors before analysis.",
                        [
                            f"{fmt_value(v)} ({groups.loc[i]})" if groups is not None else fmt_value(v)
                            for i, v in flagged.head(5).items()
                        ],
                    )
                )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        issues.append(
            _issue(
                "Duplicate rows", "high", "(all)", duplicate_rows,
                "Identical rows are counted twice in every total and average.",
            )
        )

    keys = [k for k in (key_columns or []) if k in df.columns]
    if keys:
        dup_keys = df[df.duplicated(subset=keys, keep=False)]
        if len(dup_keys):
            examples = (
                dup_keys[keys].drop_duplicates().head(5).astype(str).agg(" | ".join, axis=1).tolist()
            )
            issues.append(
                _issue(
                    "Repeated records", "high", " + ".join(keys), len(dup_keys),
                    "More than one row shares the same combination of these columns, which "
                    "should identify a record uniquely.",
                    examples,
                )
            )

    issues.sort(key=lambda item: (SEVERITY_ORDER[item["severity"]], -item["rows_affected"]))
    penalty = sum(SEVERITY_PENALTY[item["severity"]] for item in issues)
    return {
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "issues": issues,
        "profile": pd.DataFrame(profile_rows),
        "score": int(max(0, 100 - penalty)),
    }


# -----------------------------------------------------------------------------
# Group comparison
# -----------------------------------------------------------------------------
def compare_groups(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    min_n: int = 3,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Mean, median and a t-based confidence interval of ``value_col`` per group."""
    columns = ["group", "n", "mean", "median", "std", "ci_low", "ci_high"]
    if value_col not in df.columns or group_col not in df.columns:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(
        {"group": df[group_col].astype("string").str.strip(), "value": pd.to_numeric(df[value_col], errors="coerce")}
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    summary = frame.groupby("group")["value"].agg(n="count", mean="mean", median="median", std="std").reset_index()
    summary = summary[summary["n"] >= min_n].copy()
    if summary.empty:
        return pd.DataFrame(columns=columns)

    t_crit = stats.t.ppf(0.5 + confidence / 2, summary["n"] - 1)
    half_width = t_crit * summary["std"].fillna(0) / np.sqrt(summary["n"])
    summary["ci_low"] = summary["mean"] - half_width
    summary["ci_high"] = summary["mean"] + half_width
    summary["group"] = summary["group"].astype(str)
    return summary[columns].sort_values("mean", ascending=False).reset_index(drop=True)


def group_difference_test(df: pd.DataFrame, value_col: str, group_col: str, min_n: int = 3) -> dict[str, Any] | None:
    """Kruskal-Wallis test: do the groups share one distribution?

    Rank-based, so a few very large values (common in yields and prices) do not drive it.
    """
    if value_col not in df.columns or group_col not in df.columns:
        return None
    frame = pd.DataFrame(
        {"group": df[group_col].astype("string").str.strip(), "value": pd.to_numeric(df[value_col], errors="coerce")}
    ).dropna()
    samples = [g["value"].to_numpy() for _, g in frame.groupby("group") if len(g) >= min_n]
    if len(samples) < 2 or all(np.ptp(s) == 0 for s in samples):
        return None
    statistic, p_value = stats.kruskal(*samples)
    return {
        "test": "Kruskal-Wallis",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_groups": len(samples),
        "n_rows": int(sum(len(s) for s in samples)),
    }


# -----------------------------------------------------------------------------
# Trends
# -----------------------------------------------------------------------------
def period_means(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Mean of ``value_col`` per period (and group), one row each."""
    frame = pd.DataFrame(
        {
            "period": period_values(df[time_col]),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
            "group": df[group_col].astype("string").str.strip() if group_col else "All",
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["group", "period", "value", "n"])
    out = frame.groupby(["group", "period"])["value"].agg(value="mean", n="count").reset_index()
    out["period"] = out["period"].astype(int)
    out["group"] = out["group"].astype(str)
    return out


def trend_by_group(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    group_col: str | None = None,
    min_periods: int = 5,
) -> pd.DataFrame:
    """Linear trend of the per-period mean for each group.

    ``pct_per_period`` expresses the slope relative to the group's average level, so a
    cassava trend and a cowpea trend can be compared despite very different yields.
    """
    columns = [
        "group", "n_periods", "first_period", "last_period", "slope",
        "pct_per_period", "r2", "p_value", "fitted_start", "fitted_end",
    ]
    means = period_means(df, value_col, time_col, group_col)
    rows: list[dict[str, Any]] = []
    for group, part in means.groupby("group"):
        if len(part) < min_periods or part["value"].nunique() < 2:
            continue
        fit = stats.linregress(part["period"], part["value"])
        level = float(part["value"].mean())
        first, last = int(part["period"].min()), int(part["period"].max())
        rows.append(
            {
                "group": str(group),
                "n_periods": int(len(part)),
                "first_period": first,
                "last_period": last,
                "slope": float(fit.slope),
                "pct_per_period": float(fit.slope / level * 100) if level else np.nan,
                "r2": float(fit.rvalue**2),
                "p_value": float(fit.pvalue),
                "fitted_start": float(fit.intercept + fit.slope * first),
                "fitted_end": float(fit.intercept + fit.slope * last),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("pct_per_period", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Focus geography against its peers
# -----------------------------------------------------------------------------
def benchmark_against_peers(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    geo_col: str,
    focus: str,
    time_col: str | None = None,
    recent_periods: int = 5,
    min_peers: int = 3,
) -> pd.DataFrame:
    """Compare one place (e.g. Nigeria) with the other places, group by group.

    Each place is first reduced to its own mean over the most recent ``recent_periods``
    periods, so a country with many rows does not outweigh one with few. The peer figure
    is the median of those place means.
    """
    columns = ["group", "focus_value", "peer_median", "gap_pct", "rank", "n_places", "window"]
    needed = [value_col, group_col, geo_col] + ([time_col] if time_col else [])
    if any(c not in df.columns for c in needed):
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(
        {
            "group": df[group_col].astype("string").str.strip(),
            "geo": df[geo_col].astype("string").str.strip(),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    window = "all periods"
    if time_col:
        frame["period"] = period_values(df[time_col])
        frame = frame.dropna(subset=["period"])
        if not frame.empty:
            latest = frame["period"].max()
            start = latest - recent_periods + 1
            frame = frame[frame["period"] >= start]
            window = f"{int(start)}-{int(latest)}"
    frame = frame.dropna(subset=["group", "geo", "value"])

    place_means = frame.groupby(["group", "geo"])["value"].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for group, part in place_means.groupby("group"):
        mine = part[part["geo"] == focus]
        peers = part[part["geo"] != focus]
        if mine.empty or len(peers) < min_peers:
            continue
        focus_value = float(mine["value"].iloc[0])
        peer_median = float(peers["value"].median())
        rank = int((part["value"] > focus_value).sum()) + 1
        rows.append(
            {
                "group": str(group),
                "focus_value": focus_value,
                "peer_median": peer_median,
                "gap_pct": (focus_value / peer_median - 1) * 100 if peer_median else np.nan,
                "rank": rank,
                "n_places": int(len(part)),
                "window": window,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("gap_pct").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plain-language summary
# -----------------------------------------------------------------------------
def write_summary(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    time_col: str | None = None,
    geo_col: str | None = None,
    focus: str | None = None,
    quality: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Turn the analyses above into short findings, each with the numbers behind it.

    Every sentence is computed from the data; nothing is generated free-form. Findings
    that the data cannot support (too few groups, periods or peers) are left out rather
    than padded.
    """
    findings: list[dict[str, str]] = []
    unit = measure_unit(value_col)
    label = measure_label(value_col).lower()
    values = pd.to_numeric(df[value_col], errors="coerce") if value_col in df.columns else pd.Series(dtype=float)
    n_valid = int(values.notna().sum())
    if n_valid == 0:
        return [{"kicker": "Coverage", "title": "No usable values", "body": f"{value_col} has no numeric values to analyse.", "tone": "warning"}]

    scope: list[str] = [f"{n_valid:,} records with a {label} value"]
    if geo_col and geo_col in df.columns:
        scope.append(f"{df[geo_col].nunique():,} {pluralise(geo_col)}")
    if group_col and group_col in df.columns:
        scope.append(f"{df[group_col].nunique():,} {pluralise(group_col)}")
    if time_col and time_col in df.columns:
        periods = period_values(df[time_col]).dropna()
        if not periods.empty:
            scope.append(f"{int(periods.min())}-{int(periods.max())}")
    findings.append(
        {
            "kicker": "Coverage",
            "title": "What the data covers",
            "body": ", ".join(scope) + f". Overall median {label}: {fmt_value(values.median(), unit)}.",
            "tone": "neutral",
        }
    )

    if group_col and group_col in df.columns:
        table = compare_groups(df, value_col, group_col, min_n=5)
        if len(table) >= 2:
            top, bottom = table.iloc[0], table.iloc[-1]
            test = group_difference_test(df, value_col, group_col, min_n=5)
            footnote = ""
            if test is not None:
                verdict = "differ clearly" if test["p_value"] < 0.05 else "do not differ clearly"
                footnote = f"Kruskal-Wallis across {test['n_groups']} groups: p {format_p(test['p_value'])}, so the groups {verdict}."
            findings.append(
                {
                    "kicker": "Comparison",
                    "title": f"{top['group']} has the highest average {label}",
                    "body": (
                        f"{top['group']} averages {fmt_value(top['mean'], unit)} (n={int(top['n']):,}); "
                        f"{bottom['group']} is lowest at {fmt_value(bottom['mean'], unit)} (n={int(bottom['n']):,})."
                    ),
                    "tone": "accent",
                    "footnote": footnote,
                }
            )

    if time_col and time_col in df.columns:
        trends = trend_by_group(df, value_col, time_col, group_col if group_col in df.columns else None)
        if not trends.empty:
            significant = trends[trends["p_value"] < 0.05]
            rising = int((significant["pct_per_period"] > 0).sum())
            falling = int((significant["pct_per_period"] < 0).sum())
            if group_col and len(trends) > 1:
                body = (
                    f"Averaged across all records, {rising} of {len(trends)} {pluralise(group_col)} with at least "
                    f"5 years of data show a statistically clear rise in {label} and {falling} a clear fall. "
                    "The median change is "
                    f"{trends['pct_per_period'].median():+.1f}% of the average level per year."
                )
                if not significant.empty:
                    best = significant.iloc[0]
                    worst = significant.iloc[-1]
                    if best["pct_per_period"] > 0:
                        body += f" Fastest rise: {best['group']} ({best['pct_per_period']:+.1f}%/yr)."
                    if worst["pct_per_period"] < 0:
                        body += f" Steepest fall: {worst['group']} ({worst['pct_per_period']:+.1f}%/yr)."
                findings.append(
                    {"kicker": "Trend", "title": "How things are changing", "body": body,
                     "tone": "positive" if rising >= falling else "warning"}
                )
            else:
                row = trends.iloc[0]
                direction = "rising" if row["pct_per_period"] > 0 else "falling"
                clear = "a statistically clear" if row["p_value"] < 0.05 else "no clear"
                findings.append(
                    {
                        "kicker": "Trend",
                        "title": f"{label.capitalize()} is {direction}" if row["p_value"] < 0.05 else "No clear trend",
                        "body": (
                            f"Between {row['first_period']} and {row['last_period']} the trend line moves from "
                            f"{fmt_value(row['fitted_start'], unit)} to {fmt_value(row['fitted_end'], unit)} "
                            f"({row['pct_per_period']:+.1f}% per year): {clear} trend (p {format_p(row['p_value'])})."
                        ),
                        "tone": "neutral",
                    }
                )

    if focus and geo_col and group_col and geo_col in df.columns and group_col in df.columns:
        bench = benchmark_against_peers(df, value_col, group_col, geo_col, focus, time_col)
        if not bench.empty:
            below = bench[bench["gap_pct"] < 0]
            parts = []
            if not below.empty:
                worst = below.iloc[0]
                parts.append(
                    f"The largest gap is {worst['group']}: {fmt_value(worst['focus_value'], unit)} against a peer "
                    f"median of {fmt_value(worst['peer_median'], unit)} ({worst['gap_pct']:+.0f}%)."
                )
            above = bench[bench["gap_pct"] > 0]
            if not above.empty:
                best = above.iloc[-1]
                parts.append(
                    f"{focus} is furthest ahead on {best['group']} ({best['gap_pct']:+.0f}%)."
                )
            findings.append(
                {
                    "kicker": f"{focus} vs peers",
                    "title": f"{focus} is below the peer median on {len(below)} of {len(bench)} {pluralise(group_col)}",
                    "body": " ".join(parts),
                    "tone": "warning" if len(below) > len(bench) / 2 else "positive",
                    "footnote": f"Each place's mean over {bench['window'].iloc[0]}, compared with the median of the other places.",
                }
            )

    if quality is not None:
        highs = sum(1 for i in quality["issues"] if i["severity"] == "high")
        total = len(quality["issues"])
        findings.append(
            {
                "kicker": "Data quality",
                "title": f"Quality score {quality['score']}/100",
                "body": (
                    f"{total} issue(s) flagged, {highs} of them high severity. "
                    + ("Resolve those before relying on the numbers above." if highs else
                       "None block analysis; see the Data Quality page for details.")
                ),
                "tone": "warning" if highs else "positive",
            }
        )
    return findings


def pluralise(column: str) -> str:
    word = measure_label(column).lower()
    if word.endswith("y") and not word.endswith(("ay", "ey", "oy")):
        return word[:-1] + "ies"
    return word if word.endswith("s") else word + "s"


def format_p(p_value: float) -> str:
    return "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
