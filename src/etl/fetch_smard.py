import os
import sys
import time
from pathlib import Path

# ==================== PROJECT ROOT SETUP ====================
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, text

# Import config
from src.config import DATABASE_URL

ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

# ==========================
# SMARD Configuration
# ==========================
BASE_URL = "https://www.smard.de/app/chart_data"

MODULE_MAP = {
    "1223": "gen_lignite", "1224": "gen_nuclear", "1225": "gen_offshore_wind",
    "1226": "gen_hydro", "1227": "gen_other_conv", "1228": "gen_other_renew",
    "4066": "gen_biomass", "4067": "gen_onshore_wind", "4068": "gen_solar",
    "4069": "gen_hard_coal", "4070": "gen_pumped_storage", "4071": "gen_natural_gas",
    "410": "cons_total_grid", "4359": "cons_residual", "4387": "cons_pumped_storage",
    "4169": "price_de_lu", "5078": "price_neighbors", "4996": "price_be",
    "4997": "price_no2", "4170": "price_at", "252": "price_dk1",
    "253": "price_dk2", "254": "price_fr", "255": "price_it_north",
    "256": "price_nl", "257": "price_pl", "259": "price_ch",
    "260": "price_si", "261": "price_cz", "262": "price_hu",
    "3791": "proj_offshore", "123": "proj_onshore", "125": "proj_solar",
    "715": "proj_other", "5097": "proj_wind_solar", "122": "proj_total"
}

HOURLY_MODULES = {5078, 4996, 4997, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262}


def get_region(mod_int):
    if mod_int in [4169, 5078]: return "DE-LU"
    if mod_int == 4170: return "AT"
    if mod_int in [252, 253]: return "DK"
    if mod_int == 254: return "FR"
    if mod_int == 255: return "IT"
    if mod_int == 256: return "NL"
    if mod_int in [257, 258]: return "PL"
    if mod_int == 259: return "CH"
    if mod_int == 260: return "SI"
    if mod_int == 261: return "CZ"
    if mod_int == 262: return "HU"
    if mod_int == 4996: return "BE"
    if mod_int == 4997: return "NO2"
    return "DE"


def process_module(module):
    col_name = MODULE_MAP[module]
    mod_int = int(module)
    region = get_region(mod_int)
    resolution = "hour" if mod_int in HOURLY_MODULES else "quarterhour"
    
    print(f"→ Processing {col_name} ({resolution})...")

    index_url = f"{BASE_URL}/{module}/{region}/index_{resolution}.json"
    try:
        index_res = requests.get(index_url, timeout=15)
        index_res.raise_for_status()
        available_timestamps = index_res.json().get('timestamps', [])
    except Exception as e:
        print(f"Failed to get index for {col_name}: {e}")
        return

    target_timestamps = [ts for ts in available_timestamps if 1588284000000 <= ts <= 1746057600000]

    if not target_timestamps:
        print(f"   No data in target range for {col_name}")
        return

    all_data = []
    for ts in target_timestamps:
        data_url = f"{BASE_URL}/{module}/{region}/{module}_{region}_{resolution}_{ts}.json"
        try:
            res = requests.get(data_url, timeout=15)
            if res.status_code == 200:
                all_data.extend(res.json().get('series', []))
        except Exception:
            continue
        time.sleep(0.12)

    if not all_data:
        print(f"   No data extracted for {col_name}")
        return

    df = pd.DataFrame(all_data, columns=['timestamp', col_name])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset=['timestamp']).dropna()

    temp_table = f"temp_{module}"
    try:
        df.to_sql(temp_table, ENGINE, if_exists='replace', index=False)

        with ENGINE.begin() as conn:
            conn.execute(text(f"ALTER TABLE smard_market_data ADD COLUMN IF NOT EXISTS {col_name} NUMERIC(14,4);"))
            
            conn.execute(text(f"""
                INSERT INTO smard_market_data (timestamp, {col_name})
                SELECT timestamp, {col_name} FROM {temp_table}
                ON CONFLICT (timestamp) DO UPDATE SET {col_name} = EXCLUDED.{col_name};
            """))
            
            conn.execute(text(f"DROP TABLE IF EXISTS {temp_table};"))

        print(f"Synced {col_name}: {len(df):,} rows")
        
    except Exception as e:
        print(f"Database error for {col_name}: {e}")


# ==========================
if __name__ == "__main__":
    print("Starting SMARD Data Fetch Pipeline...\n")

    with ENGINE.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS smard_market_data (
                timestamp TIMESTAMP WITH TIME ZONE PRIMARY KEY
            );
        """))

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_module, MODULE_MAP.keys())

    print("\n SMARD data synchronization completed!")