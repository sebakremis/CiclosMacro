from typing import Dict, List, Optional
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mapeo descriptivo y paleta semántica adaptada al perfil del modelo 3D
DEFAULT_REGIME_NAMES: Dict[int, str] = {
    0: "Expansión Sostenida",
    1: "Reflación Agresiva",
    2: "Recesión Profunda",
    3: "Sobrecalentamiento",
}

DEFAULT_REGIME_COLORS: Dict[int, str] = {
    0: "#107C41",
    1: "#0055D4",       
    2: "#D90429",
    3: "#D97706",  
}


def plot_regimes_timeline(
    df_prices: pd.DataFrame,
    regime_series: pd.Series,
    assets: Optional[List[str]] = None,
    regime_names: Optional[Dict[int, str]] = None,
    regime_colors: Optional[Dict[int, str]] = None,
    figsize: tuple = (16, 9),
    log_scale: bool = True,
    title: str = "CiclosMacro — Evolución Temporal de Activos y Regímenes Macroeconómicos",
) -> tuple[plt.Figure, np.ndarray]:
    """Grafica la evolución temporal de activos clave (ej.

    S&P 500 y Oro) superponiendo el sombreado de regímenes detectados por GMM.

    Parámetros
    ----------
    df_prices : pd.DataFrame
        DataFrame con precios nominales de los activos (e.g. 'spy', 'gold').
    regime_series : pd.Series
        Serie con las etiquetas de cluster (0, 1, 2, 3) indexada por fecha.
    assets : list of str, optional
        Columnas de df_prices a graficar.
    regime_names : dict, optional
        Mapeo {id_cluster: 'Nombre descriptivo'}.
    regime_colors : dict, optional
        Mapeo {id_cluster: 'color_hex'}.
    figsize : tuple, default (16, 9)
        Dimensiones de la figura.
    log_scale : bool, default True
        Si se grafican los precios en escala logarítmica (recomendado para
        retornos acumulados a largo plazo).
    title : str
        Título superior de la figura.
    """
    if assets is None:
        # Detectar columna de spy automáticamente si varía de nombre
        possible_spy = [
            c for c in df_prices.columns if "spy" in c.lower()
        ]
        assets = [possible_spy[0]]

    names = regime_names or DEFAULT_REGIME_NAMES
    colors = regime_colors or DEFAULT_REGIME_COLORS

    # Alinear fechas comunes entre precios y regímenes
    common_idx = df_prices.index.intersection(regime_series.index)
    prices = df_prices.loc[common_idx, assets].copy()
    regimes = regime_series.loc[common_idx].copy()

    fig, axes = plt.subplots(
        nrows=len(assets), ncols=1, figsize=figsize, sharex=True
    )
    if len(assets) == 1:
        axes = [axes]

    dates = regimes.index.to_numpy()
    n_dates = len(dates)

    # Identificar bloques temporales continuos del mismo régimen
    regime_blocks = []
    start_date = dates[0]
    current_regime = regimes.iloc[0]

    for i in range(1, n_dates):
        if regimes.iloc[i] != current_regime:
            regime_blocks.append((start_date, dates[i], current_regime))
            start_date = dates[i]
            current_regime = regimes.iloc[i]
    # Último bloque hasta la fecha final
    regime_blocks.append((start_date, dates[-1], current_regime))

    # Graficar cada activo
    for ax, asset in zip(axes, assets):
        # 1. Sombrear los regímenes macro en el fondo
        for start, end, r_id in regime_blocks:
            ax.axvspan(
                start,
                end,
                color=colors.get(r_id, "#CCCCCC"),
                alpha=0.28,
                edgecolor=None,
            )

        # 2. Curva de precio
        ax.plot(
            prices.index,
            prices[asset],
            color="#1F2937",
            linewidth=1.8,
            label=f"Precio {asset.upper()}",
        )

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel(
                f"{asset.upper()} (Escala Log USD)",
                fontsize=11,
                fontweight="bold",
            )
        else:
            ax.set_ylabel(
                f"{asset.upper()} (USD)", fontsize=11, fontweight="bold"
            )

        ax.grid(True, which="major", linestyle="--", alpha=0.45)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)

        # Formato de fechas
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Crear leyenda de regímenes personalizada
    legend_patches = [
        mpatches.Patch(color=colors[r_id], label=names[r_id], alpha=0.7)
        for r_id in sorted(names.keys())
    ]

    # Añadir leyenda de regímenes en el panel superior
    axes[0].legend(
        handles=legend_patches,
        loc="upper left",
        framealpha=0.9,
        fontsize=9.5,
        title="Regímenes GMM (Fondo)",
        title_fontsize=10,
    )

    fig.suptitle(title, fontsize=15, weight="bold", y=0.98)
    plt.tight_layout()
    plt.show()

    return fig, axes