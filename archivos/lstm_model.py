import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.api.layers import LSTM, Dense, Dropout


# Función: Cargar y separar datos por UE
def load_and_separate_data(file_path):
    df = pd.read_csv(file_path)
    columns_ue1 = [col for col in df.columns if "_UE1" in col]
    columns_ue2 = [col for col in df.columns if "_UE2" in col]
    columns_ue3 = [col for col in df.columns if "_UE3" in col]
    return (
        df[["time"] + columns_ue1],
        df[["time"] + columns_ue2],
        df[["time"] + columns_ue3]
    )


# Función: Etiquetar los datos
def add_labels(df, dl_brate_col, dl_nof_nok_col, pusch_snr_col, latency_col):
    # Agregar una etiqueta 'viable' para cada muestra
    df['viable'] = 1  # Por defecto, viable transmitir
    condition = (
        ((df[dl_brate_col] < 180000) & (df[dl_brate_col] > 10000)) &  # Tasa de bits en el rango problemático
        ((df[dl_nof_nok_col] > 95) |                          # Número de errores alto
        (df[pusch_snr_col] < 8) |                            # Relación señal a ruido baja
        (df[latency_col] > 800))                             # Latencia muy alta
    )
    df.loc[condition, 'viable'] = 0  # No viable transmitir
    return df


# Función: Normalizar los datos
def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df["time"] = pd.to_datetime(df["time"])
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    return df_normalized, scaler


# Función: Crear secuencias para LSTM
def create_sequences(data, labels, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i: i + sequence_length]
        sequences.append(seq)
        targets.append(labels[i + sequence_length - 1])  # Etiqueta del último paso de la ventana
    return np.array(sequences), np.array(targets)


# Función: Construir el modelo LSTM
def build_lstm_model(sequence_length, input_dim):
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Clasificación binaria
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model_for_ue(df, feature_columns, sequence_length, model_name, metrics):
    # Paso 1: Etiquetar los datos
    df = add_labels(
        df,
        dl_brate_col=metrics['dl_brate_col'],
        dl_nof_nok_col=metrics['dl_nof_nok_col'],
        pusch_snr_col=metrics['pusch_snr_col'],
        latency_col=metrics['latency_col']
    )

    # Paso 2: Normalizar los datos
    df_normalized, _ = normalize_data(df, feature_columns)

    # Paso 3: Crear secuencias para el LSTM
    data_array = df_normalized[feature_columns].to_numpy()
    labels = df_normalized['viable'].to_numpy()
    X, y = create_sequences(data_array, labels, sequence_length)

    # Paso 4: Dividir datos en entrenamiento y validación
    train_size = int(0.9 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Paso 5: Construir y entrenar el modelo
    model = build_lstm_model(sequence_length, X.shape[2])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

    # Guardar el modelo
    model.save(model_name)
    print(f"Modelo guardado como: {model_name}")


# Ejecutar para cada UE con métricas específicas
file_path = "datos.csv"  # Ruta al archivo
df_ue1, df_ue2, df_ue3 = load_and_separate_data(file_path)

# Métricas específicas para cada UE
metrics_ue1 = {
    'dl_brate_col': 'dl_brate_UE1',
    'dl_nof_nok_col': 'dl_nof_nok_UE1',
    'pusch_snr_col': 'pusch_snr_db_UE1',
    'latency_col': 'latency_UE1'
}

metrics_ue2 = {
    'dl_brate_col': 'dl_brate_UE2',
    'dl_nof_nok_col': 'dl_nof_nok_UE2',
    'pusch_snr_col': 'pusch_snr_db_UE2',
    'latency_col': 'latency_UE2'
}

metrics_ue3 = {
    'dl_brate_col': 'dl_brate_UE3',
    'dl_nof_nok_col': 'dl_nof_nok_UE3',
    'pusch_snr_col': 'pusch_snr_db_UE3',
    'latency_col': 'latency_UE3'
}

# Entrenar modelos para cada UE
train_model_for_ue(df_ue1, [col for col in df_ue1.columns if "_UE1" in col], 8, "modelo_UE1.keras", metrics_ue1)
train_model_for_ue(df_ue2, [col for col in df_ue2.columns if "_UE2" in col], 8, "modelo_UE2.keras", metrics_ue2)
train_model_for_ue(df_ue3, [col for col in df_ue3.columns if "_UE3" in col], 8, "modelo_UE3.keras", metrics_ue3)