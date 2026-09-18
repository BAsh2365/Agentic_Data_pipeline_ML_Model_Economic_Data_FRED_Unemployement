import os
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd
import requests

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


FRED_API_BASE_URL = 'https://api.stlouisfed.org/fred'
DEFAULT_API_KEY = '73857a9864c8e9a4af9913ec7fe6ce55'
DEFAULT_OBSERVATION_START = '2000-01-01'
DEFAULT_OBSERVATION_END = '2026-12-31'

# Search terms from the supplied list mapped to seed FRED series IDs.
# The pipeline now uses these seeds to discover related categories and pull more series.
DEFAULT_SEARCH_TERM_SERIES = {
    'gdp': 'GDP',
    'cpi': 'CPIAUCSL',
    'inflation': 'FPCPITOTLZGUSA',
    'unemployment rate': 'UNRATE',
    'real gdp': 'GDPC1',
    'unemployment': 'U6RATE',
    'pce': 'PCE',
    'federal funds rate': 'FEDFUNDS',
    'sofr': 'SOFR',
    'm2': 'M2SL',
}

SeriesInput = Union[
    str,
    Iterable[str],
    Mapping[str, str],
    Iterable[Tuple[str, str]],
]


def normalize_series_input(series_ids: SeriesInput) -> List[Tuple[str, str]]:
    """Return [(search_term, series_id), ...] from strings, lists, dicts, or tuple pairs."""
    if isinstance(series_ids, str):
        return [(series_ids, series_ids)]

    if isinstance(series_ids, Mapping):
        return [(str(search_term), str(series_id)) for search_term, series_id in series_ids.items()]

    normalized: List[Tuple[str, str]] = []
    for item in series_ids:
        if isinstance(item, str):
            normalized.append((item, item))
        else:
            search_term, series_id = item
            normalized.append((str(search_term), str(series_id)))
    return normalized


def fred_get(endpoint: str, params: Dict[str, object], api_key: str) -> Dict[str, object]:
    request_params = {
        'api_key': api_key,
        'file_type': 'json',
        **params,
    }
    response = requests.get(
        f'{FRED_API_BASE_URL}/{endpoint}',
        params=request_params,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if 'error_code' in payload:
        raise RuntimeError(
            f"FRED API error from {endpoint}: {payload.get('error_code')} - {payload.get('error_message')}"
        )
    return payload


def get_seed_categories(series_id: str, api_key: str, max_categories: int) -> List[int]:
    """Use fred/series/categories to find categories for a seed series."""
    payload = fred_get('series/categories', {'series_id': series_id}, api_key)
    categories = payload.get('categories', []) or []
    return [int(category['id']) for category in categories[:max_categories] if category.get('id') is not None]


def get_category_series(
    category_id: int,
    api_key: str,
    observation_start: str,
    observation_end: str,
    limit: int,
) -> List[Dict[str, object]]:
    """Use fred/category/series to get popular related series in a category."""
    payload = fred_get(
        'category/series',
        {
            'category_id': category_id,
            'limit': limit,
            'offset': 0,
            'order_by': 'popularity',
            'sort_order': 'desc',
        },
        api_key,
    )
    series = payload.get('seriess', []) or []

    filtered: List[Dict[str, object]] = []
    for item in series:
        title = str(item.get('title') or '')
        notes = str(item.get('notes') or '')
        start = str(item.get('observation_start') or '')
        end = str(item.get('observation_end') or '')

        # Keep series that overlap the requested historical window and skip obvious discontinued series.
        if 'DISCONTINUED' in title.upper() or 'DISCONTINUED' in notes.upper():
            continue
        if start and start > observation_end:
            continue
        if end and end < observation_start:
            continue
        filtered.append(item)

    return filtered


def discover_related_series(
    seed_series: List[Tuple[str, str]],
    api_key: str,
    observation_start: str,
    observation_end: str,
    include_related_category_series: bool = True,
    max_categories_per_seed: int = 2,
    category_series_limit: int = 25,
    max_related_series_per_term: int = 15,
) -> List[Dict[str, object]]:
    """
    Build an expanded series inventory.

    Uses the FRED docs pattern:
      1. fred/series/categories for the categories attached to a seed series.
      2. fred/category/series for additional series in those categories.
    """
    discovered: List[Dict[str, object]] = []
    seen_keys = set()

    for search_term, seed_series_id in seed_series:
        term_count = 0

        seed_key = (search_term, seed_series_id)
        if seed_key not in seen_keys:
            discovered.append({
                'search_term': search_term,
                'series_id': seed_series_id,
                'source': 'seed_series',
                'category_id': None,
                'series_title': None,
                'frequency': None,
                'units': None,
            })
            seen_keys.add(seed_key)
            term_count += 1

        if not include_related_category_series:
            continue

        try:
            category_ids = get_seed_categories(seed_series_id, api_key, max_categories_per_seed)
        except Exception as error:
            print(f'Could not fetch categories for {seed_series_id}: {error}')
            continue

        for category_id in category_ids:
            if term_count >= max_related_series_per_term:
                break

            try:
                related_series = get_category_series(
                    category_id=category_id,
                    api_key=api_key,
                    observation_start=observation_start,
                    observation_end=observation_end,
                    limit=category_series_limit,
                )
            except Exception as error:
                print(f'Could not fetch category series for category {category_id}: {error}')
                continue

            for item in related_series:
                if term_count >= max_related_series_per_term:
                    break

                series_id = item.get('id')
                if not series_id:
                    continue

                key = (search_term, str(series_id))
                if key in seen_keys:
                    continue

                discovered.append({
                    'search_term': search_term,
                    'series_id': str(series_id),
                    'source': 'category_series',
                    'category_id': category_id,
                    'series_title': item.get('title'),
                    'frequency': item.get('frequency'),
                    'units': item.get('units'),
                })
                seen_keys.add(key)
                term_count += 1

    return discovered


def fred_observations_ingestion(
    series_ids: SeriesInput,
    api_key: Optional[str] = None,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    frequency: Optional[str] = None,
    aggregation_method: Optional[str] = None,
    units: Optional[str] = None,
    limit: int = 100000,
    include_related_category_series: bool = True,
    max_categories_per_seed: int = 2,
    category_series_limit: int = 25,
    max_related_series_per_term: int = 15,
) -> pd.DataFrame:
    """Fetch observations for seed and category-discovered FRED series."""
    resolved_api_key = api_key or os.getenv('FRED_API_KEY') or DEFAULT_API_KEY
    resolved_observation_start = observation_start or DEFAULT_OBSERVATION_START
    resolved_observation_end = observation_end or DEFAULT_OBSERVATION_END
    seed_series = normalize_series_input(series_ids)

    if not seed_series:
        raise ValueError('At least one FRED series ID must be provided.')

    series_inventory = discover_related_series(
        seed_series=seed_series,
        api_key=resolved_api_key,
        observation_start=resolved_observation_start,
        observation_end=resolved_observation_end,
        include_related_category_series=include_related_category_series,
        max_categories_per_seed=max_categories_per_seed,
        category_series_limit=category_series_limit,
        max_related_series_per_term=max_related_series_per_term,
    )

    frames: List[pd.DataFrame] = []

    for series_meta in series_inventory:
        params: Dict[str, object] = {
            'series_id': series_meta['series_id'],
            'limit': limit,
            'sort_order': 'asc',
            'observation_start': resolved_observation_start,
            'observation_end': resolved_observation_end,
        }

        optional_params = {
            'frequency': frequency,
            'aggregation_method': aggregation_method,
            'units': units,
        }
        params.update({key: value for key, value in optional_params.items() if value})

        try:
            payload = fred_get('series/observations', params, resolved_api_key)
        except Exception as error:
            print(f"Could not fetch observations for {series_meta['series_id']}: {error}")
            continue

        observations = payload.get('observations', []) or []
        df = pd.DataFrame(observations)

        if df.empty:
            continue

        df['search_term'] = series_meta['search_term']
        df['series_id'] = series_meta['series_id']
        df['source'] = series_meta['source']
        df['category_id'] = series_meta['category_id']
        df['series_title'] = series_meta['series_title']
        df['frequency'] = series_meta['frequency']
        df['units'] = series_meta['units']
        df['value'] = pd.to_numeric(df['value'].replace('.', pd.NA), errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['realtime_start'] = pd.to_datetime(df['realtime_start'], errors='coerce')
        df['realtime_end'] = pd.to_datetime(df['realtime_end'], errors='coerce')
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=[
            'search_term', 'series_id', 'source', 'category_id', 'series_title',
            'frequency', 'units', 'date', 'value', 'realtime_start', 'realtime_end'
        ])

    result = pd.concat(frames, ignore_index=True)

    ordered_columns = [
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
    ]
    remaining_columns = [column for column in result.columns if column not in ordered_columns]
    return result[ordered_columns + remaining_columns]


@data_loader
def ingestion(*args, **kwargs) -> pd.DataFrame:
    """
    Runtime variables supported:
      - series_ids: string/list of FRED series IDs, or dict of search_term -> seed FRED series ID
      - fred_api_key: optional override for the FRED API key
      - observation_start: default 2000-01-01
      - observation_end: default 2026-12-31
      - include_related_category_series: default True
      - max_categories_per_seed: default 2
      - category_series_limit: default 25
      - max_related_series_per_term: default 15
      - frequency, aggregation_method, units: optional FRED observation transformations
    """
    series_ids = kwargs.get('series_ids') or DEFAULT_SEARCH_TERM_SERIES

    return fred_observations_ingestion(
        series_ids=series_ids,
        api_key=kwargs.get('fred_api_key'),
        observation_start=kwargs.get('observation_start') or DEFAULT_OBSERVATION_START,
        observation_end=kwargs.get('observation_end') or DEFAULT_OBSERVATION_END,
        frequency=kwargs.get('frequency'),
        aggregation_method=kwargs.get('aggregation_method'),
        units=kwargs.get('units'),
        include_related_category_series=kwargs.get('include_related_category_series', True),
        max_categories_per_seed=kwargs.get('max_categories_per_seed', 2),
        category_series_limit=kwargs.get('category_series_limit', 25),
        max_related_series_per_term=kwargs.get('max_related_series_per_term', 15),
    )
