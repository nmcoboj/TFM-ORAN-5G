import requests
import pandas as pd
from io import StringIO

# Configuración
url = "http://172.19.1.5:8086/api/v2/query?orgID=331bc19ea81b92dc"
token = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"


# Consulta Flux con rango de tiempo especificado
query = f"""
from(bucket: "srsran")
    |> range(start: 2024-12-26T21:47:30Z, stop: 2024-12-26T22:03:00Z)
    |> filter(fn: (r) => r["testbed"] == "default")
    |> filter(fn: (r) => r._measurement == "ue_info")
"""

# Realizar la petición a la API de InfluxDB
headers = {
    "Authorization": f"Token {token}",
    "Accept": "application/csv",
    "Content-type": "application/vnd.flux"
}

response = requests.post(url, headers=headers, data=query)

# Verificar que la petición fue exitosa
if response.status_code == 200:
    # Imprimir la respuesta completa para depuración
    print("Respuesta completa de la API:\n", response.text[:500])  # Muestra los primeros 500 caracteres

    # Verificar si la respuesta tiene contenido
    if not response.text.strip():
        print("La respuesta está vacía. Verifica la consulta o los datos en InfluxDB.")
    else:
        # Leer los datos obtenidos como un DataFrame
        try:
            data = StringIO(response.text)
            df = pd.read_csv(data)

            # Asegurarse de que las columnas necesarias existan
            required_columns = ["_time", "_field", "_value", "rnti"]
            if all(col in df.columns for col in required_columns):
                # Pivotar los datos para que cada combinación de `_field` y `rnti` sea una columna
                df["column_name"] = df["_field"] + "_rnti_" + df["rnti"].astype(str)
                df_pivot = df.pivot_table(
                    index="_time",  # Usar la columna `time` como índice
                    columns="column_name",  # Cada combinación de `_field` y `rnti` será una columna
                    values="_value",  # Usar los valores de `_value` como datos
                    aggfunc="first"  # En caso de duplicados, usar el primer valor
                ).reset_index()

                # Guardar los datos en un archivo CSV
                output_file = "datos_agrupados.csv"
                df_pivot.to_csv(output_file, index=False)

                print(f"Datos agrupados guardados en: {output_file}")
            else:
                print(f"Faltan columnas necesarias en los datos: {required_columns}")
        except pd.errors.EmptyDataError:
            print("Error al procesar los datos: la respuesta no contiene datos válidos.")
else:
    print(f"Error en la petición: {response.status_code}")
    print(response.text)