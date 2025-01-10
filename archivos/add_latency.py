import pandas as pd
import re

# Ruta del archivo CSV principal
csv_file = 'influxdb_data.csv'

# Rutas de los archivos .log para cada UE
log_files = {
    'UE1': 'ue1.log',
    'UE2': 'ue2.log',
    'UE3': 'ue3.log'
}

# Leer el archivo CSV principal
df = pd.read_csv(csv_file)

# Renombrar columnas del archivo principal
renamed_columns = {col: re.sub(r'rnti_4603', 'UE1',
                   re.sub(r'rnti_4602', 'UE2',
                   re.sub(r'rnti_4601', 'UE3', col))) for col in df.columns}
df.rename(columns=renamed_columns, inplace=True)
df.rename(columns={'_time': 'time'}, inplace=True)

# Asegurarse de que la columna de tiempo esté en formato datetime
df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)

# Función para procesar un archivo .log
def process_log(file_path):
    log_data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'(\S+) .* time=(\d+\.?\d*) ms', line)
            if match:
                log_data.append({'time': match.group(1), 'latency': float(match.group(2))})
    log_df = pd.DataFrame(log_data)
    log_df['time'] = pd.to_datetime(log_df['time'], utc=True).dt.tz_localize(None)  # Convertir a datetime sin zona horaria
    return log_df


# Procesar cada archivo de logs y añadir la latencia correspondiente
for ue, log_file in log_files.items():
    # Leer el archivo de log
    log_df = process_log(log_file)

    # Unir los datos basados en el tiempo más cercano
    df = pd.merge_asof(
        df.sort_values('time'),
        log_df.sort_values('time'),
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta('2s')
    )
    df.rename(columns={'latency': f'latency_{ue}'}, inplace=True)

# Guardar el archivo final
df.to_csv('datos_agrupados.csv', index=False)
print("Archivo guardado como 'datos_agrupados.csv'")
