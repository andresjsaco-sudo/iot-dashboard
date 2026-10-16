from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import json
from datetime import datetime
import time
import numpy as np
import subprocess

print("Cargando modelo de IA...")
modelo = YOLO('yolov8n.pt')

print("Iniciando camara en alta resolucion...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 360), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)

ultimo_guardado = time.time()
intervalo_guardado = 3

fps_tiempo_anterior = time.time()
fps_contador = 0
fps_actual = 0

frame_skip = 0
personas_detectadas = 0
personas_anterior = -1

historial_detecciones = []

def enviar_curl(personas, fps):
    try:
        data = {
            "x": str(personas),
            "y": str(round(fps, 1))
        }
        
        comando = [
            'curl',
            '--location',
            'https://iotrp.sandbox.portillonino.dev/position',
            '--header', 'Content-Type: application/json',
            '--data', json.dumps(data)
        ]
        
        resultado = subprocess.run(comando, capture_output=True, text=True, timeout=5)
        print(f"[OK] Enviado a API: Personas={personas}, FPS={fps:.1f}")
        print(f"    Respuesta del servidor: {resultado.stdout.strip()}")
        
    except Exception as e:
        print(f"[ERROR] Error enviando a API: {e}")

print("\n[OK] Sistema iniciado correctamente")
print("Presiona Ctrl+C para salir")
print("[INFO] Guardando datos cada 3 segundos en: conteo_tiempo_real.json")
print("[INFO] Enviando a API cada vez que cambie el numero de personas\n")

try:
    while True:
        frame = picam2.capture_array()
        
        if frame is None:
            print("Error al capturar video")
            break
        
        frame_skip += 1
        
        if frame_skip % 2 == 0:
            resultados = modelo(frame, verbose=False, imgsz=192, conf=0.5)
            personas_detectadas = 0
            
            for resultado in resultados:
                for caja in resultado.boxes:
                    if int(caja.cls[0]) == 0:
                        personas_detectadas += 1
                        
                        x1, y1, x2, y2 = map(int, caja.xyxy[0])
                        confianza = float(caja.conf[0])
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        texto = f"P{personas_detectadas}"
                        cv2.putText(frame, texto, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            if personas_detectadas != personas_anterior:
                fecha_cambio = datetime.now().strftime("%d/%m/%Y")
                hora_cambio = datetime.now().strftime("%H:%M:%S")
                
                historial_detecciones.append({
                    "fecha": fecha_cambio,
                    "hora": hora_cambio,
                    "personas": personas_detectadas,
                    "fps": round(fps_actual, 1)
                })
                
                enviar_curl(personas_detectadas, fps_actual)
                personas_anterior = personas_detectadas
        
        fps_contador += 1
        tiempo_transcurrido = time.time() - fps_tiempo_anterior
        if tiempo_transcurrido >= 1.0:
            fps_actual = fps_contador / tiempo_transcurrido
            fps_contador = 0
            fps_tiempo_anterior = time.time()
        
        fecha_hora_actual = datetime.now()
        fecha = fecha_hora_actual.strftime("%d/%m/%Y")
        hora = fecha_hora_actual.strftime("%H:%M:%S")
        
        cv2.putText(frame, f"{fecha} {hora}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Personas: {personas_detectadas}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {fps_actual:.1f}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('Detector', frame)
        
        tiempo_actual = time.time()
        if tiempo_actual - ultimo_guardado >= intervalo_guardado:
            datos = {
                "fecha": fecha,
                "hora": hora,
                "timestamp_completo": fecha_hora_actual.strftime("%Y-%m-%d %H:%M:%S"),
                "personas_detectadas": personas_detectadas,
                "fps": round(fps_actual, 1)
            }
            
            with open('conteo_tiempo_real.json', 'w') as archivo:
                json.dump(datos, archivo, indent=4)
            
            ultimo_guardado = tiempo_actual
            print(f"[{fecha} {hora}] Personas: {personas_detectadas} | FPS: {fps_actual:.1f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[STOP] Detenido por el usuario")

total_detecciones = len(historial_detecciones)
max_personas = max([d["personas"] for d in historial_detecciones]) if historial_detecciones else 0

resumen = {
    "resumen": {
        "fecha_inicio": historial_detecciones[0]["fecha"] if historial_detecciones else "N/A",
        "hora_inicio": historial_detecciones[0]["hora"] if historial_detecciones else "N/A",
        "fecha_final": datetime.now().strftime("%d/%m/%Y"),
        "hora_final": datetime.now().strftime("%H:%M:%S"),
        "total_cambios": total_detecciones,
        "maximo_personas_detectadas": max_personas
    },
    "historial": historial_detecciones
}

with open('resumen_final.json', 'w') as archivo:
    json.dump(resumen, archivo, indent=4)

print(f"\n[OK] Deteccion finalizada")
print(f"[INFO] Total de cambios registrados: {total_detecciones}")
print(f"[INFO] Maximo de personas detectadas: {max_personas}")
print(f"[INFO] Resumen completo guardado en: resumen_final.json")

picam2.stop()
cv2.destroyAllWindows()
