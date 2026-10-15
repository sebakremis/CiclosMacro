from typing import Dict
from pathlib import Path

# Ruta al fichero de datos
data_folder = "data"
clean_data_file = Path(data_folder) / "clean_data.parquet"

# Fecha inicial para descargar datos
default_start_date = "2010-01-01"

# Mapeo de series macro: Series ID de FRED -> Nombre descriptivo de columna
DEFAULT_GOLD_MACRO_SERIES: Dict[str, str] = {
    "DFII10": "real_yield_10y",                 # 10-Year Treasury Inflation-Indexed Yield
    "T10YIE": "breakeven_inflation_10y",        # 10-Year Breakeven Inflation Rate
    "DTWEXBGS": "dxy_broad_goods",              # Trade Weighted U.S. Dollar Index
    "DGS10": "treasury_10y_nominal",            # 10-Year Treasury Constant Maturity
    "VIXCLS": "vix_index",                      # CBOE Volatility Index
}

# Mapeo de series de precios
DEFAULT_PRICE_ASSETS: Dict[str, str] = {
    "GC=F": "gold",
    "CL=F": "oil",
    "SPY": "spy",
}