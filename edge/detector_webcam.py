import cv2
from ultralytics import YOLO
import json
from datetime import datetime
import time

print("Cargando modelo de IA...")
modelo = YOLO('yolov8n.pt')

print("Iniciando cámara...")
camara = cv2.VideoCapture(0)

camara.set(cv2.CAP_PROP_FPS, 30)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ultimo_guardado = time.time()
intervalo_guardado = 1

fps_tiempo_anterior = time.time()
fps_contador = 0
fps_actual = 0

print("\n[OK] Sistema iniciado correctamente")
print("Presiona 'q' para salir")
print("[INFO] Guardando datos cada segundo en: conteo_tiempo_real.json\n")

while True:
    ret, frame = camara.read()
    
    if not ret:
        print("Error al capturar video")
        break
    
    resultados = modelo(frame, verbose=False)
    personas_detectadas = 0
    
    for resultado in resultados:
        for caja in resultado.boxes:
            if int(caja.cls[0]) == 0:
                personas_detectadas += 1
                
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                confianza = float(caja.conf[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                texto = f"persona {confianza:.2f}"
                cv2.putText(frame, texto, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    fps_contador += 1
    tiempo_transcurrido = time.time() - fps_tiempo_anterior
    if tiempo_transcurrido >= 1.0:
        fps_actual = fps_contador / tiempo_transcurrido
        fps_contador = 0
        fps_tiempo_anterior = time.time()
    
    fecha_hora_actual = datetime.now()
    fecha = fecha_hora_actual.strftime("%d/%m/%Y")
    hora = fecha_hora_actual.strftime("%H:%M:%S")
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 130), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    cv2.putText(frame, f"Fecha: {fecha}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Hora: {hora}", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Personas: {personas_detectadas}", (10, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(frame, f"FPS: {fps_actual:.1f}", (10, 115),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow('Detector de Personas', frame)
    
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

resumen = {
    "fecha_final": datetime.now().strftime("%d/%m/%Y"),
    "hora_final": datetime.now().strftime("%H:%M:%S"),
    "timestamp_completo": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('resumen_final.json', 'w') as archivo:
    json.dump(resumen, archivo, indent=4)

print(f"\n[OK] Detección finalizada")
print(f"[INFO] Resumen final guardado en: resumen_final.json")

camara.release()
cv2.destroyAllWindows()
