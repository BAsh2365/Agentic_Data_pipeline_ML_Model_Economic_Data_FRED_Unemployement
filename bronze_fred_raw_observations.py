from datetime import timezone

import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


TERM_TO_MACRO_CATEGORY = {
    'gdp': 'gdp',
    'real gdp': 'gdp',
    'm2': 'money_supply_m2',
    'unemployment': 'unemployment',
    'unemployment rate': 'unemployment',
    'cpi': 'prices_inflation',
    'inflation': 'prices_inflation',
    'pce': 'pce',
    'federal funds rate': 'interest_rates',
    'sofr': 'interest_rates',
}


@transformer
def transform(data, *args, **kwargs) -> pd.DataFrame:
    """Bronze layer: lightly standardize the raw FRED API observation output."""
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=[
            'medallion_layer', 'macro_category', 'search_term', 'series_id', 'source',
            'category_id', 'series_title', 'frequency', 'units', 'date', 'value',
            'realtime_start', 'realtime_end', 'ingested_at_utc'
        ])

    df = data.copy()

    for column in ['search_term', 'series_id', 'source', 'series_title', 'frequency', 'units']:
        if column not in df.columns:
            df[column] = None

    df['search_term'] = df['search_term'].astype(str).str.strip().str.lower()
    df['series_id'] = df['series_id'].astype(str).str.strip()
    df['macro_category'] = df['search_term'].map(TERM_TO_MACRO_CATEGORY).fillna('other_macro')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['realtime_start'] = pd.to_datetime(df['realtime_start'], errors='coerce')
    df['realtime_end'] = pd.to_datetime(df['realtime_end'], errors='coerce')
    df['medallion_layer'] = 'bronze'
    df['ingested_at_utc'] = pd.Timestamp.now(tz=timezone.utc)

    ordered_columns = [
        'medallion_layer',
        'macro_category',
        'search_term',
        'series_id',
        'source',
        'category_id',
        'series_title',
        'frequency',
        'units',
        'date',
        'value',
        'realtime_start',
        'realtime_end',
        'ingested_at_utc',
    ]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    return df[ordered_columns + remaining_columns].sort_values(
        ['macro_category', 'search_term', 'series_id', 'date'],
        na_position='last',
    ).reset_index(drop=True)
