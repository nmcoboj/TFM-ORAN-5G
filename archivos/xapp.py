import argparse
import time
import numpy as np
import pandas as pd
from keras.api.models import load_model
from sklearn.preprocessing import MinMaxScaler
import subprocess
import requests
from io import StringIO
from datetime import datetime


# Función para capturar datos desde InfluxDB
def fetch_influx_data(ue_id):
    INFLUX_URL = "http://172.19.1.5:8086/api/v2/query?orgID=331bc19ea81b92dc"
    INFLUX_TOKEN = "605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b"

    HEADERS = {
        "Authorization": f"Token {INFLUX_TOKEN}",
        "Accept": "application/csv",
        "Content-type": "application/vnd.flux"
    }
    query = f"""
        from(bucket: "srsran")
          |> range(start: -30s) 
          |> filter(fn: (r) => r._measurement == "ue_info")
          |> filter(fn: (r) => r.rnti == "{ue_id}")
          |> filter(fn: (r) => r["_field"] == "dl_brate" or r["_field"] == "dl_nof_nok" or r["_field"] == "pusch_snr_db")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")    
    """
    response = requests.post(INFLUX_URL, headers=HEADERS, data=query)
    if response.status_code != 200:
        print(f"Error en la consulta a InfluxDB: {response.status_code}, {response.text}")
        return pd.DataFrame()  # Retornar DataFrame vacío en caso de error

    df = pd.read_csv(StringIO(response.text))

    # Verificar si el DataFrame está vacío
    if df.empty:
        print("No se encontraron datos en la consulta.")
        return df

    # Convertir _time a formato datetime y ordenar
    df['_time'] = pd.to_datetime(df['_time'], utc=True).dt.tz_localize(None)
    df = df[['_time', 'dl_brate', 'dl_nof_nok', 'pusch_snr_db']]
    df.rename(columns={'_time': 'time'}, inplace=True)
    return df


# Función para capturar latencia desde ping
def fetch_ping_latency(ue_name):
    results_df = pd.DataFrame(columns=["time", "latency"])
    current_second = None
    max_latency_line = None
    start_time = time.time()
    try:
        command = f"sudo ip netns exec {ue_name} ping -i 1 10.45.1.1"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Leer la salida línea por línea
        for line in iter(process.stdout.readline, ""):
            if time.time() - start_time > 30:
                break
            # Obtener timestamp en formato ISO 8601
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3]
            current_line_second = timestamp.split(".")[0]  # Extraer solo el segundo

            # Procesar las líneas con "bytes from" (respuestas de ping)
            if "bytes from" in line:
                # Extraer el tiempo de latencia (parte después de "time=")
                try:
                    latency = float(line.split("time=")[-1].split(" ")[0])
                except (IndexError, ValueError):
                    continue

                # Actualizar el dato máximo del segundo actual
                if current_second is None or current_second != current_line_second:
                    # Guardar el dato anterior si cambia de segundo
                    if max_latency_line:
                        max_timestamp, max_latency = max_latency_line
                        results_df = pd.concat([results_df, pd.DataFrame({"time": [max_timestamp], "latency": [max_latency]})], ignore_index=True)
                        print(max_latency_line)  # Mostrar en la terminal
                    # Reiniciar el máximo para el nuevo segundo
                    current_second = current_line_second
                    max_latency_line = (timestamp, latency)
                elif latency > max_latency_line[1]:
                    # Actualizar el máximo si el tiempo actual es mayor
                    max_latency_line = (timestamp, latency)

            elif "ping: " in line or "connect:" in line:
                print(f"[ERROR] {timestamp}: {line.strip()}")

    except Exception as e:
        print(f"[ERROR] Ocurrió un error: {e}")
    results_df['time'] = pd.to_datetime(results_df['time'], utc=True).dt.tz_localize(None)
    process.terminate()
    return results_df


# Función para normalizar y crear secuencias
def preprocess_data(latency_df, influx_df, scaler, feature_columns, sequence_length):
    if influx_df.empty:
        return None, None
    # Añadir latencia a los datos
    df = pd.merge_asof(
        influx_df.sort_values('time'),
        latency_df.sort_values('time'),
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta('2s')
    )

    # Seleccionar columnas relevantes
    data = df[feature_columns].to_numpy()

    # Normalizar datos
    data_normalized = scaler.fit_transform(data)

    # Crear secuencia
    if len(data_normalized) < sequence_length:
        return None, None  # No hay suficientes datos para una secuencia
    sequence = data_normalized[-sequence_length:]

    return np.expand_dims(sequence, axis=0), influx_df["time"].iloc[-1]


# Función principal para procesar en tiempo real
def process_realtime(ue_id, ue_name, model_path, scaler, feature_columns, sequence_length):
    model = load_model(model_path)
    while True:
        try:
            # Capturar latencia desde ping
            latency_df = fetch_ping_latency(ue_name)
            if latency_df is None:
                print(f"Latencia no disponible para {ue_id}.")
                continue
            # Capturar datos de InfluxDB
            influx_df = fetch_influx_data(ue_id)
            # Preprocesar datos
            X, timestamp = preprocess_data(latency_df, influx_df, scaler, feature_columns, sequence_length)
            if X is None:
                print(f"No hay suficientes datos para {ue_id}.")
                continue
            # Hacer predicción
            prediction = model.predict(X)
            decision_threshold = 0.5  # Umbral de decisión
            decision = "Transmitir" if prediction[0][0] > decision_threshold else "No Transmitir"
            print(f"[{timestamp}] {ue_name} - {ue_id}: Predicción = {prediction[0][0]:.2f}, Decisión = {decision}")
            time.sleep(1)  # Pausa de 1 segundo antes de la siguiente iteración
        except KeyboardInterrupt:
            print("Interrupción por el usuario. Finalizando...")
            break

# Configuración por UE
feature_columns = ["dl_brate", "dl_nof_nok", "pusch_snr_db", "latency"]
sequence_length = 8

# Normalizador (reemplazar por los valores del entrenamiento)
scaler = MinMaxScaler()
scaler.fit(np.random.rand(100, len(feature_columns)))  # Simular ajuste inicial del normalizador

# Ejecutar en tiempo real para cada UE
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Procesamiento para un UE específico.")
    parser.add_argument("--ue_name", type=str, required=True, help="Nombre del UE (e.g., ue1)")
    parser.add_argument("--ue_id", type=str, required=True, help="ID del UE (e.g., 4601)")
    parser.add_argument("--model_path", type=str, required=True, help="modelo_UE1.keras")
    args = parser.parse_args()

    process_realtime(ue_id=args.ue_id, ue_name=args.ue_name, model_path=args.model_path, scaler=scaler,
                     feature_columns=feature_columns, sequence_length=sequence_length)