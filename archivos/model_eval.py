import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from keras.api.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Función para etiquetar los datos
def add_labels(df, dl_brate_col, dl_nof_nok_col, pusch_snr_col, latency_col):
    # Crear una etiqueta 'viable' para cada muestra
    df['viable'] = 1  # Por defecto, viable transmitir
    condition = (
        (df[dl_brate_col] < 180000) & (df[dl_brate_col] > 10000) |  # Tasa de bits en el rango problemático
        (df[dl_nof_nok_col] > 95) |                          # Número de errores alto
        (df[pusch_snr_col] < 8) |                            # Relación señal a ruido baja
        (df[latency_col] > 800)                              # Latencia muy alta
    )
    df.loc[condition, 'viable'] = 0  # No viable transmitir
    return df

# Función para normalizar los datos
def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df[columns])
    return df_normalized, scaler

# Función para crear secuencias para LSTM
def create_sequences(data, labels, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i: i + sequence_length]
        sequences.append(seq)
        targets.append(labels[i + sequence_length - 1])  # Etiqueta del último paso de la ventana
    return np.array(sequences), np.array(targets)

# Función para evaluar el modelo
def evaluate_model(model_path, df, feature_columns, sequence_length, metrics, ue_name):
    # Paso 1: Etiquetar los datos nuevos
    df = add_labels(
        df,
        dl_brate_col=metrics['dl_brate_col'],
        dl_nof_nok_col=metrics['dl_nof_nok_col'],
        pusch_snr_col=metrics['pusch_snr_col'],
        latency_col=metrics['latency_col']
    )

    # Paso 2: Normalizar los datos
    df_normalized, _ = normalize_data(df, feature_columns)

    # Paso 3: Crear secuencias temporales
    data_array = df_normalized[feature_columns].to_numpy()
    labels = df_normalized['viable'].to_numpy()
    X_test, y_test = create_sequences(data_array, labels, sequence_length)

    # Paso 4: Cargar el modelo
    model = load_model(model_path)

    # Paso 5: Predicción
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Métricas
    print(f"=== Evaluación del Modelo para {ue_name} ===")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Transmitir", "Transmitir"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Transmitir", "Transmitir"], yticklabels=["No Transmitir", "Transmitir"])
    plt.title(f'Matriz de Confusión ({ue_name})')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.show()

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'Curva ROC ({ue_name})')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc="lower right")
    plt.show()


# Evaluar los modelos generados con datos nuevos
if __name__ == "__main__":
    # Ruta del archivo con datos nuevos
    data_path = "datos.csv"
    df_new = pd.read_csv(data_path)

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

    # Evaluar modelos para cada UE
    evaluate_model(
        model_path="modelo_UE1.keras",
        df=df_new,
        feature_columns=[col for col in df_new.columns if "_UE1" in col],
        sequence_length=8,
        metrics=metrics_ue1,
        ue_name="UE1"
    )

    evaluate_model(
        model_path="modelo_UE2.keras",
        df=df_new,
        feature_columns=[col for col in df_new.columns if "_UE2" in col],
        sequence_length=8,
        metrics=metrics_ue2,
        ue_name="UE2"
    )

    evaluate_model(
        model_path="modelo_UE3.keras",
        df=df_new,
        feature_columns=[col for col in df_new.columns if "_UE3" in col],
        sequence_length=8,
        metrics=metrics_ue3,
        ue_name="UE3"
    )
