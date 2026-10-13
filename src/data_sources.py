"""
src/data_sources.py
Modulo privado para gestionar la carga de claves API.
NO ESCRIBAS tu clave real en este archivo. La clave se guarda en el archivo .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Se obtiene la clave API desde las variables de entorno
fred_api_key = os.getenv("FRED_API_KEY")