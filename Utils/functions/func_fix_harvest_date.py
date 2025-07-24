import re
import pandas as pd
from dateutil import parser
from typing import Optional, Tuple
from Utils.functions.func_spanish_to_english import translator
from Utils.dicts.dict_Seasons import season_map


def parse_quarters(raw: str) -> Optional[Tuple[str, str]]:
    if not isinstance(raw, str):
        return None

    # Allow 2-digit or 4-digit years
    match = re.match(r'(\d)[tT]/?(\d{2,4})$', raw)
    if match:
        quarter = int(match.group(1))
        year = int(match.group(2))

        # Normalize 2-digit year to 4-digit
        if year < 100:
            if year >= 50:
                year += 1900  # Assume 1950–1999
            else:
                year += 2000  # Assume 2000–2049

        quarter_months = {
            1: ("01-01", "03-31"),
            2: ("04-01", "06-30"),
            3: ("07-01", "09-30"),
            4: ("10-01", "12-31")
        }

        start, end = quarter_months.get(quarter, (None, None))
        if start and end:
            return f"{year}-{start}", f"{year}-{end}"

    return None


def parse_date_range(raw: str) -> Optional[Tuple[str, str]]:
    if not isinstance(raw, str):
        return None

    raw = translator(raw).strip().lower()

    # 1. Try parsing quarter-like strings
    quarter_result = parse_quarters(raw)
    if quarter_result:
        return quarter_result

    for season, (start_suffix, end_suffix) in season_map.items():
        if season in raw:
            year_match = re.search(r'(\d{4})', raw)
            if year_match:
                year = int(year_match.group(1))
                if season == "winter":
                    return f"{year}-12-01", f"{year+1}-02-28"
                return f"{year}-{start_suffix}", f"{year}-{end_suffix}"

    # 3. Handle single-month entries (e.g. "April 2010")
    if re.match(r'^[a-zA-Z]+\s+\d{4}$', raw):
        try:
            dt = parser.parse(raw)
            start = dt.replace(day=1)
            next_month = start.replace(day=28) + pd.Timedelta(days=4)
            end = next_month - pd.Timedelta(days=next_month.day)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        except:
            pass

    # 4. Handle date ranges using separators
    separators = [' to ', ' a ', '-', '–', '—', '/', '–']
    for sep in separators:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep)]
            if len(parts) == 2:
                try:
                    start = parser.parse(parts[0], fuzzy=True, default=parser.parse("2000-01-01"))
                    end = parser.parse(parts[1], fuzzy=True, default=parser.parse("2000-01-01"))

                    # Inherit year from other side if missing
                    if start.year == 2000 and end.year != 2000:
                        start = start.replace(year=end.year)
                    elif end.year == 2000 and start.year != 2000:
                        end = end.replace(year=start.year)

                    # Fix reversed dates
                    if start > end:
                        start, end = end, start

                    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
                except:
                    return None

    # 5. Handle a single date string
    try:
        single_date = parser.parse(raw, fuzzy=True)
        return single_date.strftime("%Y-%m-%d"), single_date.strftime("%Y-%m-%d")
    except:
        return None


def get_start_end_dates(raw: str) -> Tuple[Optional[str], Optional[str]]:

    if pd.isna(raw) or not isinstance(raw, str) or raw.strip() == "":
        return None, None

    # Check for quarter format first
    quarter_result = parse_quarters(raw)
    if quarter_result:
        return quarter_result

    # Otherwise, try general date range parse
    range_result = parse_date_range(raw)
    if range_result:
        return range_result

    return None, None


def parse_harvest_dates(raw_date: str) -> Optional[str]:
    if pd.isna(raw_date) or raw_date.strip() == "":
        return None

    raw = raw_date.lower().strip()
    raw = translator(raw)

    try:
        parsed = parser.parse(raw, fuzzy=True)
        return parsed.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Failed to parse: {raw} ({e})")
        return None




