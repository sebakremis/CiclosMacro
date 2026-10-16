import pandas as pd
import numpy as np

def split_last(
    df: pd.DataFrame, 
    label: str, 
    cols_excluded: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba basándose en la fecha más reciente.
    """
    
    # Identificamos cuál es la fecha máxima en todo el dataset
    ultima_fecha = df['date'].max()
    
    # Separamos mediante máscaras booleanas
    df_train = df[df['date'] < ultima_fecha]
    df_test = df[df['date'] == ultima_fecha]
    
    X_train = df_train.drop(columns=cols_excluded+[label])
    y_train = df_train[label]
    
    X_test = df_test.drop(columns=cols_excluded+[label])
    y_test = df_test[label]
    
    return X_train, X_test, y_train, y_test


def encode_cyclical_month(df: pd.DataFrame) -> np.ndarray:
    month = df.iloc[:, 0].astype(float)
    sin_comp = np.sin(2 * np.pi * month / 12.0)
    cos_comp = np.cos(2 * np.pi * month / 12.0)
    return np.column_stack([sin_comp, cos_comp])