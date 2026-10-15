"""
src/extract.py
Modulo para extraer series temporales desde la API de FRED
orientadas a la modelizacion del precio de Oro spot.
"""

from typing import Dict, List, Optional
import pandas as pd
from fredapi import Fred
import yfinance as yf

from src.data_sources import fred_api_key
from src.config import default_start_date, DEFAULT_GOLD_MACRO_SERIES, DEFAULT_PRICE_ASSETS


def get_fred_client(api_key: Optional[str] = None) -> Fred:
    """Valida y crea una instancia de conexion con el cliente de FRED."""
    key = api_key or fred_api_key
    if not key:
        raise ValueError(
            "FRED_API_KEY no encontrada. Revisa tu archivo .env o la variable de entorno."
        )
    return Fred(api_key=key)


def fetch_fred(
    series_map: Optional[Dict[str, str]] = None,
    start_date: str = default_start_date,
    end_date: Optional[str] = None,
    client: Optional[Fred] = None,
) -> pd.DataFrame:
    """
    Descarga un conjunto de series temporales de FRED y las consolida en un DataFrame.

    Parameters
    ----------
    series_map : Dict[str, str], optional
        Diccionario {series_id_fred: nombre_columna}. Si es None usa DEFAULT_GOLD_MACRO_SERIES.
    start_date : str
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


def fetch_prices(
    assets: Optional[Dict[str, str]] = None,
    start_date: str = default_start_date,
    end_date: Optional[str] = None,
    interval: str = "1wk",
) -> pd.DataFrame:
    """
    Descarga los precios de cierre para una coleccion de tickers desde Yahoo Finance.

    Parameters
    ----------
    assets : Dict[str, str], optional
        Mapeo {ticker_yfinance: nombre_columna}.
        Por defecto usa {'GC=F': 'gold_price_usd', 'SPY': 'spy_price_usd'}.
    start_date : str
        Fecha inicial de consulta (YYYY-MM-DD).
    end_date : str, optional
        Fecha final de consulta. Si es None, descarga hasta la fecha actual.
    interval : str, default '1wk'
        Intervalo temporal de las velas ('1d', '1wk', '1mo').

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por fecha con una columna de precio de cierre por activo.
    """
    if assets is None:
        assets = DEFAULT_PRICE_ASSETS

    tickers_list: List[str] = list(assets.keys())

    data = yf.download(
        tickers=tickers_list,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError("No se obtuvieron datos para los tickers solicitados.")

    # Si se pide mas de un ticker, yfinance genera columnas MultiIndex: (Price, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        df_close = data["Close"].copy()
    else:
        # En caso de pasar un solo ticker en el diccionario
        df_close = data[["Close"]].copy()
        df_close.columns = tickers_list

    # Renombrar columnas segun el mapeo definido
    df_close = df_close.rename(columns=assets)

    # Filtrar solo las columnas solicitadas y formatear el indice
    df_close = df_close[[name for name in assets.values() if name in df_close.columns]]
    df_close.index = pd.to_datetime(df_close.index)
    df_close.index.name = "date"
    df_close.sort_index(inplace=True)

    return df_close