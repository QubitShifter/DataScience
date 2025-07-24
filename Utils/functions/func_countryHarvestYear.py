from typing import Optional, Tuple
import pandas as pd
from Utils.dicts.dict_Cofee_Harvest_periods import coffee_harvest_seasons
from Utils.functions.func_normalize_country_names import normalize_country_key


def transform_dates_from_country_and_year(country: str, harvest_year: str, season_map: dict) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(country, str) or not isinstance(harvest_year, str):
        #print(f"DEBUG (invalid types): country={country}, harvest_year={harvest_year}")
        return None, None

    # Normalize
    harvest_year = harvest_year.replace('/', '-').strip()
    country_key = normalize_country_key(country)
    periods = season_map.get(country_key)

    if not periods:
        #print(f"DEBUG (no periods): country={country} (normalized={country_key}), harvest_year={harvest_year}")
        return None, None

    try:
        period = periods[0].lower().replace('–', '-').replace('—', '-')
        start_month, end_month = [m.strip().lower() for m in period.split('-')]
        start_month_num = coffee_harvest_seasons[start_month]
        end_month_num = coffee_harvest_seasons[end_month]
    except Exception as e:
        #print(f"DEBUG (month parse error): country={country}, period={periods[0]}, error={e}")
        return None, None

    try:
        years = [y.strip() for y in harvest_year.split('-')]
        start_year = int(years[0])
        end_year = int(years[1]) if len(years) == 2 else start_year

        # Adjust if the season spans two years
        if start_month_num > end_month_num and start_year == end_year:
            end_year += 1

        start_date = f"{start_year}-{start_month_num}-01"
        end_date = f"{end_year}-{end_month_num}-28"
        #print(f"DEBUG (success): country={country_key}, year={harvest_year}, start_date={start_date}, end_date={end_date}")
        return start_date, end_date

    except Exception as e:
        #print(f"DEBUG (year parse error): country={country}, year={harvest_year}, error={e}")
        return None, None
