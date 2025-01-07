import pandas as pd


# Ruta de los archivos CSV
csv_files = ['datos_finales_1.csv', 'datos_finales_2.csv', 'datos_finales_3.csv']  # Lista con los nombres de los archivos

# Leer y concatenar los archivos
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenar todos los DataFrames en uno solo
combined_df = pd.concat(dfs, ignore_index=True)

# Lista de palabras clave que deben estar en los nombres de las columnas
keywords = ["time", "dl_brate", "dl_nof_nok", "pusch_snr_db", "latency"]

# Filtrar las columnas que contienen cualquiera de las palabras clave
filtered_columns = [col for col in combined_df.columns if any(keyword in col for keyword in keywords)]

# Crear un nuevo DataFrame con las columnas filtradas
filtered_df = combined_df[filtered_columns]

# Rellenar los valores NaN con el último valor disponible (forward fill)
filtered_df.fillna(method='ffill', inplace=True)

# Rellenar valores NaN restantes (si existen) con 0
filtered_df.fillna(0, inplace=True)

filtered_df.to_csv('datos_combinados.csv', index=False)

print("Archivos combinados guardados como 'datos_combinados.csv'")