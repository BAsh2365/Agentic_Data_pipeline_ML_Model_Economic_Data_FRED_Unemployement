import numpy as np
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


OUTPUT_COLUMNS = [
    'medallion_layer',
    'macro_category',
    'search_term',
    'series_id',
    'series_title',
    'frequency',
    'units',
    'source',
    'category_id',
    'date',
    'observation_year',
    'observation_quarter',
    'observation_month',
    'value',
    'previous_value',
    'period_change',
    'period_change_pct',
    'year_over_year_value',
    'year_over_year_change',
    'year_over_year_change_pct',
    'latest_observation_date_for_series',
    'is_latest_observation_for_series',
    'category_series_count',
    'category_observation_count',
    'is_seed_series',
]


def yoy_lag_periods(frequency: object) -> int:
    text = str(frequency or '').lower()
    if 'daily' in text:
        return 365
    if 'weekly' in text:
        return 52
    if 'monthly' in text:
        return 12
    if 'quarter' in text:
        return 4
    if 'annual' in text or 'year' in text:
        return 1
    # Most FRED macroeconomic indicators here are monthly/quarterly.
    return 12


@transformer
def transform(*args, **kwargs) -> pd.DataFrame:
    """
    Gold layer: combine all silver category outputs into an analytics-ready long panel.

    The output keeps category-level splits while adding reusable metrics for dashboards,
    models, and downstream workflow steps.
    """
    frames = [data.copy() for data in args if isinstance(data, pd.DataFrame) and not data.empty]

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['macro_category', 'search_term', 'series_id', 'date', 'value'])
    df = df.drop_duplicates(subset=['macro_category', 'search_term', 'series_id', 'date', 'realtime_start', 'realtime_end'])
    df = df.sort_values(['macro_category', 'search_term', 'series_id', 'date']).reset_index(drop=True)

    group_keys = ['macro_category', 'search_term', 'series_id']
    df['previous_value'] = df.groupby(group_keys)['value'].shift(1)
    df['period_change'] = df['value'] - df['previous_value']
    df['period_change_pct'] = np.where(
        df['previous_value'].notna() & (df['previous_value'] != 0),
        df['period_change'] / df['previous_value'],
        np.nan,
    )

    enriched_groups = []
    for _, group in df.groupby(group_keys, dropna=False, sort=False):
        group = group.sort_values('date').copy()
        frequency = group['frequency'].dropna().iloc[0] if 'frequency' in group and group['frequency'].notna().any() else None
        lag_periods = yoy_lag_periods(frequency)
        group['year_over_year_value'] = group['value'].shift(lag_periods)
        group['year_over_year_change'] = group['value'] - group['year_over_year_value']
        group['year_over_year_change_pct'] = np.where(
            group['year_over_year_value'].notna() & (group['year_over_year_value'] != 0),
            group['year_over_year_change'] / group['year_over_year_value'],
            np.nan,
        )
        enriched_groups.append(group)

    df = pd.concat(enriched_groups, ignore_index=True)
    df['medallion_layer'] = 'gold'
    df['observation_year'] = df['date'].dt.year
    df['observation_quarter'] = df['date'].dt.to_period('Q').astype(str)
    df['observation_month'] = df['date'].dt.to_period('M').astype(str)

    df['latest_observation_date_for_series'] = df.groupby(group_keys)['date'].transform('max')
    df['is_latest_observation_for_series'] = df['date'].eq(df['latest_observation_date_for_series'])
    df['category_series_count'] = df.groupby('macro_category')['series_id'].transform('nunique')
    df['category_observation_count'] = df.groupby('macro_category')['value'].transform('count')

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    return df[OUTPUT_COLUMNS].sort_values(
        ['macro_category', 'search_term', 'series_id', 'date'],
        na_position='last',
    ).reset_index(drop=True)
