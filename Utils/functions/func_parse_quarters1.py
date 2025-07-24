from typing import Optional, Tuple
import pandas as pd
import re
from Utils.dicts.dict_month_name_to_number import month_name_to_number  # Ensure this has lowercase keys


def transform_dates_from_country_and_year(country: str, harvest_year: str, season_map: dict) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(harvest_year, str) or not isinstance(country, str):
        return None, None

    # Translate Spanish to English
    harvest_year = translator(harvest_year).strip().lower()

    # Try parsing as a quarter first (e.g., "4T/2011")
    try:
        quarter_result = parse_quarters(harvest_year)
        if quarter_result:
            return quarter_result
    except Exception as e:
        print(f"[DEBUG] Quarter parse failed: {harvest_year} — {e}")

    # Try season mapping by country
    country_key = normalize_country_key(country)
    periods = season_map.get(country_key)

    if periods:
        try:
            # Use only the first period
            period = periods[0].lower().replace('–', '-').replace('—', '-')
            start_month, end_month = [m.strip().lower() for m in re.split(r'\s*-\s*', period)]

            start_month_num = int(month_name_to_number[start_month])
            end_month_num = int(month_name_to_number[end_month])

            # Try parsing years from something like '2013/2014'
            year_parts = re.findall(r'\d{4}', harvest_year)
            if not year_parts:
                raise ValueError("No valid year found")

            start_year = int(year_parts[0])
            end_year = int(year_parts[1]) if len(year_parts) > 1 else start_year

            # Adjust for cross-year seasons
            if start_month_num > end_month_num and start_year == end_year:
                end_year += 1

            return (
                f"{start_year}-{start_month_num:02}-01",
                f"{end_year}-{end_month_num:02}-28"
            )
        except Exception as e:
            print(f"[DEBUG] Country-based mapping failed: {country}, {harvest_year}, error: {e}")

    # Fallback: Use parse_date_range
    try:
        fallback = parse_date_range(harvest_year)
        if fallback:
            return fallback
    except Exception as e:
        print(f"[DEBUG] parse_date_range failed for '{harvest_year}': {e}")

    # If all fails
    return None, None

def parse_date_range(raw: str) -> Optional[Tuple[str, str]]:
        from dateutil import parser
        raw = raw.strip().lower()
        separators = [' to ', '-', '–', '—', '/', ' a ']
        for sep in separators:
            if sep in raw:
                parts = [p.strip() for p in raw.split(sep)]
                if len(parts) == 2:
                    try:
                        start = parser.parse(parts[0], fuzzy=True)
                        end = parser.parse(parts[1], fuzzy=True)
                        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
                    except:
                        return None
        try:
            single = parser.parse(raw, fuzzy=True)
            return single.strftime("%Y-%m-%d"), single.strftime("%Y-%m-%d")
        except:
            return None

    # Step 1: Clean raw input
    if pd.isna(raw_year) or not isinstance(raw_year, str) or raw_year.strip() == "":
        raw_year = ""

    raw_year = raw_year.strip()

    # Try parsing explicit dates first
    result = parse_quarters(raw_year)
    if result:
        return result

    result = parse_date_range(raw_year)
    if result:
        return result

    # Step 2: Try fallback using country harvest season
    if isinstance(country, str):
        country_key = country.strip().title()
        periods = season_dict.get(country_key)

        if not periods:
            # print(f"DEBUG (no season): country={country_key}, raw_year={raw_year}")
            return None, None

        season = periods[0].lower().replace("–", "-").replace("—", "-")
        if "-" in season:
            try:
                start_month, end_month = [m.strip().lower() for m in season.split("-")]
                start_month_num = month_name_to_number.get(start_month)
                end_month_num = month_name_to_number.get(end_month)

                if start_month_num is None or end_month_num is None:
                    raise ValueError(f"Invalid month: '{start_month}' or '{end_month}'")

                # Extract year or fallback
                year_match = re.search(r'\d{4}', raw_year)
                year = int(year_match.group(0)) if year_match else 2012

                # Adjust if it crosses year boundary
                end_year = year if end_month_num >= start_month_num else year + 1

                start_date = f"{year}-{start_month_num:02d}-01"
                end_date = f"{end_year}-{end_month_num:02d}-28"

                return start_date, end_date
            except Exception as e:
               # print(f"DEBUG (month parse error): country={country_key}, period={season}, error='{e}'")
                return None, None

    return None, None
