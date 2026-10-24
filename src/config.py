from typing import Dict
from pathlib import Path

# Ruta al fichero de datos
data_folder = "data"
clean_data_file = Path(data_folder) / "clean_data.parquet"

# Fecha inicial para descargar datos
default_start_date = "2006-01-01"

# Mapeo de series de precios
DEFAULT_PRICE_ASSETS: Dict[str, str] = {
    "GC=F": "gold",
    "CL=F": "oil",
    "SPY": "spy",
}

# Mapeo de series macro: Series ID de FRED -> Nombre descriptivo de columna
# Factores financieros / mercado (diarios o semanales en origen)
DAILY_MACRO_SERIES: Dict[str, str] = {
    "DFII10": "real_yield_10y",                 # 10-Year Real Yield (Diaria)
    "DTWEXBGS": "dxy_broad_goods",              # Trade Weighted USD (Diaria)
    "VIXCLS": "vix_index",                      # CBOE Volatility Index (Diaria)
    "STLFSI4": "st_louis_financial_stress",     # Financial Stress Index (Semanal -> cierre de mes)
}
# Pilares del ciclo económico fundamental (mensuales nativos en FRED)
MONTHLY_MACRO_SERIES: Dict[str, str] = {
    "INDPRO": "industrial_production",          # Industrial Production (Mensual)
    "UNRATE": "unemployment_rate",              # Unemployment Rate (Mensual)
    "CPIAUCSL": "cpi_headline",                 # CPI Urban Consumers (Mensual)
    "FEDFUNDS": "fed_funds_rate",               # Effective Federal Funds Rate (Mensual)
    "M2SL": "m2_money_supply",                  # M2 Money Supply (Mensual)
}

# Diccionario consolidado
DEFAULT_MACRO_SERIES: Dict[str, str] = {**DAILY_MACRO_SERIES, **MONTHLY_MACRO_SERIES}

# Listas automaticas:
daily_cols = list(DAILY_MACRO_SERIES.values())
monthly_cols = list(MONTHLY_MACRO_SERIES.values())