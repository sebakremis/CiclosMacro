"""
src/extract.py
Modulo para extraer series temporales desde la API de FRED
orientadas a la modelizacion del precio de Oro spot.
"""

from typing import Dict, Optional
import pandas as pd
from fredapi import Fred
import yfinance as yf

from src.data_sources import fred_api_key

# Mapeo de series clave: Series ID de FRED -> Nombre descriptivo de columna
DEFAULT_GOLD_MACRO_SERIES: Dict[str, str] = {
    "DFII10": "real_yield_10y",                 # 10-Year Treasury Inflation-Indexed Yield
    "T10YIE": "breakeven_inflation_10y",        # 10-Year Breakeven Inflation Rate
    "DTWEXBGS": "dxy_broad_goods",              # Trade Weighted U.S. Dollar Index
    "DGS10": "treasury_10y_nominal",            # 10-Year Treasury Constant Maturity
    "VIXCLS": "vix_index",                      # CBOE Volatility Index
}


def get_fred_client(api_key: Optional[str] = None) -> Fred:
    """Valida y crea una instancia de conexion con el cliente de FRED."""
    key = api_key or fred_api_key
    if not key:
        raise ValueError(
            "FRED_API_KEY no encontrada. Revisa tu archivo .env o la variable de entorno."
        )
    return Fred(api_key=key)


def fetch_fred_series(
    series_map: Optional[Dict[str, str]] = None,
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    client: Optional[Fred] = None,
) -> pd.DataFrame:
    """
    Descarga un conjunto de series temporales de FRED y las consolida en un DataFrame.

    Parameters
    ----------
    series_map : Dict[str, str], optional
        Diccionario {series_id_fred: nombre_columna}. Si es None usa DEFAULT_GOLD_MACRO_SERIES.
    start_date : str, default '2010-01-01'
        Fecha inicial de consulta (formato YYYY-MM-DD).
    end_date : str, optional
        Fecha final de consulta. Si es None, trae hasta el dato mas reciente disponible.
    client : Fred, optional
        Instancia de Fred ya configurada.

    Returns
    -------
    pd.DataFrame
        DataFrame con DatetimeIndex y cada serie en una columna, ordenado cronologicamente.
    """
    if series_map is None:
        series_map = DEFAULT_GOLD_MACRO_SERIES

    fred = client or get_fred_client()
    data_frames = []

    for series_id, col_name in series_map.items():
        try:
            # fred.get_series devuelve una pd.Series indexada por fecha
            s = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            s.name = col_name
            # Forzar tipo numerico (coercing nulos de dias festivos/sin cotizacion)
            s = pd.to_numeric(s, errors="coerce")
            data_frames.append(s)
        except Exception as e:
            print(f"[WARN] Error al descargar la serie {series_id} ({col_name}): {e}")

    if not data_frames:
        raise RuntimeError("No se pudo obtener ninguna serie desde FRED.")

    # Outer join sobre las fechas para alinear los calendarios de negociacion
    df = pd.concat(data_frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.sort_index(inplace=True)

    return df

def fetch_gold_spot_daily(
    ticker: str = "GC=F",
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Descarga el precio de cierre diario del oro (por defecto futuros continuos GC=F).
    Alternativa para spot: 'GLD' (SPDR Gold Shares).
    """
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )
    # Extraer precio de cierre
    if isinstance(data.columns, pd.MultiIndex):
        s = data["Close"][ticker]
    else:
        s = data["Close"]

    s.name = "gold_price_usd"
    s.index = pd.to_datetime(s.index)
    return s


def fetch_full_dataset(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Extrae factores macro de FRED y precio de Oro de Yahoo Finance."""
    # 1. Factores Macro
    df_macro = fetch_fred_series(
        series_map=DEFAULT_GOLD_MACRO_SERIES,
        start_date=start_date,
        end_date=end_date
    )

    # 2. Target Oro (GC=F continuo)
    s_gold = fetch_gold_spot_daily(
        ticker="GC=F",
        start_date=start_date,
        end_date=end_date
    )

    # 3. Concatenar y alinear indices temporales
    df_merged = df_macro.join(s_gold, how="outer")
    df_merged.index.name = "date"
    df_merged.sort_index(inplace=True)

    return df_merged


if __name__ == "__main__":
    # Prueba rapida de ejecucion
    df_raw = fetch_full_dataset(start_date="2020-01-01")
    print("\nShape:", df_raw.shape)
    print("\nPrimeras filas:")
    print(df_raw.head())
    print("\nValores nulos por serie:")
    print(df_raw.isna().sum())