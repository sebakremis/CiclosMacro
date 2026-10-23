from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import PowerTransformer


# Processing

def process_macro_to_monthly(
    df_raw: pd.DataFrame, 
    daily_columns: list[str], 
    monthly_columns: list[str]
) -> pd.DataFrame:
    """
    Remuestrea series mixtas (diarias y mensuales) a cierre de mes (ME)
    aplicando lag de publicacion a las variables mensuales nativas.
    """
    # Bloque Diario -> ffill acotado por festivos y tomar cierre de mes
    df_daily_m = (
        df_raw[daily_columns]
        .ffill(limit=5)
        .resample("ME")
        .last()
    )

    # Bloque Mensual Nativo -> descartar dias vacios y mover a fin de mes
    df_monthly_native = df_raw[monthly_columns].dropna(how="all")
    df_monthly_native.index = df_monthly_native.index + pd.offsets.MonthEnd(0)

    # Unir ambos bloques
    df_res = df_daily_m.join(df_monthly_native, how="inner")

    # Lag de 1 mes de publicacion a las variables mensuales nativas (al 31 de Enero, el ultimo dato conocido es de Diciembre)
    df_res = calculate_lag(df_res, cols= monthly_columns)

    return df_res


def drop_incomplete_month(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina la fila correspondiente al mes en curso si aun no ha finalizado."""
    now = pd.Timestamp.now()
    # Inicio del mes actual (ej. 2026-10-01 00:00:00)
    current_month_start = pd.Timestamp(year=now.year, month=now.month, day=1)

    # Conservar solo los registros fechados antes del inicio del mes en curso
    # (Como el indice es MonthEnd, el mes previo estara en el ultimo dia del mes anterior)
    df_closed = df[df.index < current_month_start].copy()
    return df_closed

# EDA

def show_missings(df:pd.DataFrame)->pd.Series:
    """
    Calcula la proporción de valores nulos por columna usando operaciones vectorizadas.
    """
    # Se calcula la proporción de nulos para todas las columnas
    missings = df.isna().mean()
    
    # Se filtran donde el valor es mayor a 0, ordenando luego de mayor a menor
    return missings[missings > 0].sort_values(ascending=False)


def histogram_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(9,8), bins = None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel 
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """
    # Definir tamaño letra
    sns.set(font_scale=font_scale)
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=data, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=data, ax=ax_hist2, bins=bins) if bins else sns.histplot(x=data, ax=ax_hist2)
    # Pintar una línea con la media
    ax_hist2.axvline(np.mean(data),color='g',linestyle='-')
    # Pintar una línea con la mediana
    ax_hist2.axvline(np.median(data),color='y',linestyle='--')
    # Asignar título y nombre de eje si tal
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title, xlabel="")
    # Mostrar gráfico
    plt.show()


def cat_chart(col):
     """
     Gráfico de barras para variables categóricas.
     """
     if col.dtypes == 'category':
        fig = px.bar(col.value_counts())
        #fig = sns.countplot(x=col)
        return(fig)


def plot_col(col):
     """
     Función general para aplicar al archivo por columnas, detectando el tipo de variable y aplicando el gráfico adecuado.
     """
     if col.dtypes != 'category':
        print('Cont')
        histogram_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        print('Cat')
        cat_chart(col).show()


def show_asymmetry(df:pd.DataFrame):
    """
    Muestra la asimetría de las columnas numéricas del DataFrame.
    """
    print(df.select_dtypes(include="number").skew().sort_values(ascending=False).to_string())


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    figsize: tuple = (16, 12),
    cmap: str = "vlag",
    annot: bool = True,
    top_n_pairs: int = 10,
    title: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula y grafica la matriz de correlación triangular para la fase EDA

    de CiclosMacro, extrayendo además los pares más correlacionados.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las 20 features macroeconómicas y de precios procesadas.
    method : str, default 'spearman'
        Método de correlación ('spearman' o 'pearson'). Spearman se recomienda
        por la presencia de shocks de volatilidad no lineales (VIX, retornos).
    figsize : tuple, default (16, 12)
        Dimensiones de la figura de Matplotlib.
    cmap : str, default 'vlag'
        Paleta divergente centrada en cero (e.g. 'vlag', 'coolwarm',
        'RdBu_r').
    annot : bool, default True
        Si se imprimen los coeficientes numéricos en cada celda.
    top_n_pairs : int, default 10
        Cantidad de pares de mayor correlación absoluta a reportar en tabla.
    title : str, optional
        Título del heatmap.

    Retorna
    -------
    corr_matrix : pd.DataFrame
        Matriz cuadrada de correlaciones completa.
    top_pairs_df : pd.DataFrame
        Ranking con los pares de variables más fuertemente correlacionados.
    """
    # Asegurar únicamente columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    # Calcular matriz de correlación
    corr_matrix = numeric_df.corr(method=method)

    # Generar máscara para el triángulo superior y diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Graficar Heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0.0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"Correlación de {method.capitalize()}"},
        annot_kws={"size": 8 if len(corr_matrix.columns) >= 20 else 10},
    )

    plot_title = (
        title
        if title
        else f"CiclosMacro — Matriz de Correlación ({method.capitalize()})"
    )
    plt.title(plot_title, fontsize=14, weight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()

    # Extraer ranking de pares con mayor correlación absoluta (sin duplicados ni diagonal)
    pairs = (corr_matrix.where(~mask).stack().reset_index(name="correlation").rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    )

    pairs["abs_correlation"] = pairs["correlation"].abs()
    top_pairs_df = (
        pairs.sort_values(by="abs_correlation", ascending=False)
        .drop(columns=["abs_correlation"])
        .head(top_n_pairs)
        .reset_index(drop=True)
    )

    return corr_matrix, top_pairs_df


# Feature Engineering

def calculate_lag(df:pd.DataFrame, cols:list,periods:int=1)->pd.DataFrame:
    """
    Calcula el valor rezagado (lag) de las columnas especificadas.
    """
    df_out = df.copy()
    for col in cols:
        df_out[col+f'_lag{periods}'] = df_out[col].shift(periods)

    df_out.drop(columns=cols, inplace=True)
    return df_out


def calculate_return(df:pd.DataFrame, cols:list, periods:int=1)->pd.DataFrame:
    """
    Calcula las variaciones porcentuales
    """
    df_out = df.copy()
    for col in cols:
        df_out[f"{col}_ret_{periods}m"] = df_out[col].pct_change(periods, fill_method=None)

    return df_out


def calculate_delta(df:pd.DataFrame, cols:list, periods=1)->pd.DataFrame:
    """
    Calcula las variaciones absolutas
    """
    df_out = df.copy()
    for col in cols:
        df_out[f"{col}_change_{periods}m"] = df_out[col].diff(periods)

    return df_out


# Transformations

def log_transform(df: pd.DataFrame, cols: list, calculation_1p: bool = False) -> pd.DataFrame:
    """
    Aplica la transformación logarítmica a las columnas especificadas del DataFrame.
    Se crea una nueva columna con el sufijo '_Log' o '_Log1p' y se elimina la columna original.
    """
    # Creamos una copia para no alterar el DataFrame original accidentalmente
    df_out = df.copy()
    
    # Se define la función y el sufijo
    if calculation_1p:
        funcion_log = np.log1p
        sufijo = '_log1p'
    else:
        funcion_log = np.log
        sufijo = '_log'
        
    # Se aplica la función y elimina la columna original
    for col in cols:
        df_out[f'{col}{sufijo}'] = funcion_log(df_out[col])
        df_out.drop(col, axis=1, inplace=True)
        
    return df_out


def yeo_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Aplica la transformación de Yeo-Johnson a las columnas especificadas del DataFrame.
    Se crea una nueva columna con el sufijo '_Yeo' y se elimina la columna original.
    """
    df_out = df.copy()

    # Inicializar el PowerTransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True)

    # Se aplica la función y elimina la columna original
    for col in cols:
        df_out[f'{col}_yeo'] = pt.fit_transform(df_out[[col]])
        df_out.drop(col, axis=1, inplace=True)
        
    return df_out