import subprocess
import signal
import sys
import os
from datetime import datetime

# Configuración del namespace y dirección IP
UE_NAME = "ue1"  # Cambia esto al nombre del namespace del UE (e.g., ue2, ue3)
DEST_IP = "10.45.1.1"  # Cambia esto a la IP de destino
OUTPUT_FILE = f"{UE_NAME}_ping_output.log"

# Eliminar el archivo de salida si ya existe
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# Variable para almacenar el dato máximo por segundo
current_second = None
max_latency_line = None

# Función para capturar la señal de interrupción (CTRL+C)
def signal_handler(sig, frame):
    print("\n[INFO] Interrumpido por el usuario. Finalizando...")
    sys.exit(0)

# Configurar el manejador para CTRL+C
signal.signal(signal.SIGINT, signal_handler)

# Abrir el archivo para registrar los resultados
with open(OUTPUT_FILE, "a") as log_file:
    print(f"[INFO] Ejecutando ping desde {UE_NAME} hacia {DEST_IP}. Guardando resultados en {OUTPUT_FILE}...")
    try:
        # Comando ping para el namespace del UE
        command = f"sudo ip netns exec {UE_NAME} ping -i 1 {DEST_IP}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Leer la salida línea por línea
        for line in iter(process.stdout.readline, ""):
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
                        log_file.write(max_latency_line)
                        print(max_latency_line.strip())  # Mostrar en la terminal
                    # Reiniciar el máximo para el nuevo segundo
                    current_second = current_line_second
                    max_latency_line = f"{timestamp} {line.strip()}\n"
                elif latency > float(max_latency_line.split("time=")[-1].split(" ")[0]):
                    # Actualizar el máximo si el tiempo actual es mayor
                    max_latency_line = f"{timestamp} {line.strip()}\n"

            elif "ping: " in line or "connect:" in line:
                # Capturar errores o problemas de conectividad
                error_entry = f"{timestamp} {line.strip()}\n"
                print(error_entry.strip())
                log_file.write(error_entry)

    except Exception as e:
        print(f"[ERROR] Ocurrió un error: {e}")
    finally:
        # Guardar el último dato máximo cuando se detiene el script
        if max_latency_line:
            log_file.write(max_latency_line)
            print(max_latency_line.strip())
        # Asegurarse de limpiar el proceso si el script se detiene abruptamente
        process.terminate()
        print("[INFO] Proceso finalizado.")