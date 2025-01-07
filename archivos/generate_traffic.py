import subprocess
import random
import time
from datetime import datetime

# Función para obtener el timestamp actual
def current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Función para generar tráfico dinámico
def generate_traffic(ue_name, server_ip, total_duration, min_time, max_time, interval):
    end_time = time.time() + total_duration
    log_file = f"{ue_name}_iperf_output.log"  # Archivo para guardar la salida de iPerf

    try:
        with open(log_file, "a") as log:  # Abrir archivo en modo de agregar
            while time.time() < end_time:
                # Generar parámetros aleatorios
                duration = random.randint(min_time, max_time)  # Duración de la ejecución
                interval_time = 1
                stop_traffic = random.choice([True, False])  # Pausar tráfico

                if stop_traffic:
                    pause_duration = random.randint(3, 10)  # Duración de la pausa
                    print(f"{current_time()} [{ue_name}] Pausando tráfico por {pause_duration} segundos...")
                    for _ in range(pause_duration):
                        log.write(f"{current_time()}, {ue_name}, 0, 0, 0, 0\n")  # Registrar ceros
                        time.sleep(1)  # Esperar un segundo antes de registrar el siguiente cero
                    continue

                # Construir el comando de iPerf
                command = f"sudo ip netns exec {ue_name} iperf -c {server_ip} -i {interval_time} -t {duration}"
                print(f"{current_time()} [{ue_name}] Ejecutando: {command}")

                # Ejecutar el comando y capturar la salida
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Leer la salida línea por línea
                for line in iter(process.stdout.readline, ""):
                    timestamped_line = f"{current_time()}, {ue_name}, {line.strip()}\n"
                    print(timestamped_line.strip())  # Mostrar en tiempo real
                    log.write(timestamped_line)
                
                process.stdout.close()
                process.wait()

                # Pausa entre ejecuciones
                time.sleep(interval)

    except KeyboardInterrupt:
        print(f"{current_time()} [{ue_name}] Interrumpido por el usuario. Finalizando...")
    finally:
        print(f"{current_time()} [{ue_name}] Generación de tráfico completada.")

generate_traffic(
    ue_name= 'ue1',
    server_ip= '10.53.1.1',
    total_duration=10800, #Duración total del tráfico (en segundos)
    min_time= 30, #Duración mínima de cada ejecución de iPerf (en segundos)
    max_time= 300, #Duración máxima de cada ejecución de iPerf (en segundos)
    interval=20 #Intervalo entre ejecuciones de iPerf (en segundos)
)
