# TFM-ORAN-5G
Los archivos que se encuentran en este repositorio son los siguientes:
- Archivo de configuración del gNB: gnb_zmq.yaml 
- Archivos de configuración de los UEs: ue1_zmq.conf, ue2_zmq.conf, ue3_zmq.conf
- Archivo del escenario de GNU Radio: multi_ue_scenario.grc
- Scripts:
    - Generación de tráfico dinámico: generate_traffic.py
    - Captura de latencia: capture_latency.py
    - Extracción de métricas de InfluxDB: extract_influxdb.py
    - Agrupación de la latencia con las métricas de InfluxDB: add_latency.py
    - Filtrar métricas y concatenar archivos: filter_concat.py
    - Entrenamiento del modelo de inteligencia artificial: lstm_model.py
    - xApp: xapp.py

Las rutas de instalación o de ejecución de los archivos se encuentra detallada en la memoria de TFM.
