import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


INCLUDED_TERMS = {'m2'}
MACRO_CATEGORY = 'money_supply_m2'


@transformer
def transform(data, *args, **kwargs) -> pd.DataFrame:
    """Silver layer: validated M2 and money-supply observations."""
    if data is None or len(data) == 0:
        return pd.DataFrame()

    df = data.copy()
    df['search_term'] = df['search_term'].astype(str).str.strip().str.lower()
    df = df[df['search_term'].isin(INCLUDED_TERMS)].copy()

    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[df['date'].between(pd.Timestamp('2000-01-01'), pd.Timestamp('2026-12-31'))]
    df = df.dropna(subset=['date', 'value', 'series_id'])
    df = df.drop_duplicates(subset=['search_term', 'series_id', 'date', 'realtime_start', 'realtime_end'])

    df['medallion_layer'] = 'silver'
    df['macro_category'] = MACRO_CATEGORY
    df['observation_year'] = df['date'].dt.year
    df['observation_quarter'] = df['date'].dt.to_period('Q').astype(str)
    df['observation_month'] = df['date'].dt.to_period('M').astype(str)
    df['is_seed_series'] = df['source'].eq('seed_series')

    return df.sort_values(['series_id', 'date']).reset_index(drop=True)
