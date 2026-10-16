# Classroom occupancy with edge AI and a 3D point cloud

An IoT system that detects and counts people in a university classroom and shows them in near real time on a cloud dashboard and a 3D point cloud of the room. Built for the Communications course challenge at Universidad del Norte by Juan Betancourt, Darwin Diaz, Laura Guerrero and Jose Noriega.

**Live dashboard:** [sensordashboardiot.vercel.app](https://sensordashboardiot.vercel.app)

## How it works

```
Raspberry Pi 4 + 120° camera            LTE (SIMCOM A7670SA, PPP)            AWS                           Web
YOLOv8n person detection on-device  ->  cellular uplink, no Wi-Fi needed ->  IoT Core, API Gateway,   ->   Next.js dashboard
6 to 10 FPS, sends only on change                                            Lambda, DynamoDB              + Potree 3D point cloud
```

1. **Edge detection.** A Raspberry Pi 4 runs YOLOv8 Nano on frames from a wide-angle camera through Picamera2. Every second frame is processed at a 192 px inference size with a 0.5 confidence threshold.
2. **Event-driven uplink.** The Pi only transmits when the person count changes, which keeps mobile data use and latency low. It also writes a local snapshot every 3 seconds and a session summary on exit.
3. **Cellular link.** An LTE module (SIMCOM A7670SA) connects the Pi to the internet over PPP, so the system works in rooms without reliable Wi-Fi.
4. **Cloud.** AWS IoT Core, API Gateway, Lambda and DynamoDB ingest and store the detections. The infrastructure is defined as code with AWS CDK in TypeScript and deployed through a pipeline.
5. **Visualization.** This Next.js dashboard shows occupancy over time, and a Potree viewer places detections on a point cloud and mesh of the classroom.

## Performance

The first version ran at 1080p and processed every frame, reaching only 2 to 3 FPS on the Pi. Lowering the capture resolution to 640x360, shrinking the inference size, skipping alternate frames and batching disk writes brought it to a stable **6 to 10 FPS**, enough for continuous occupancy monitoring.

## Repository layout

| Path | Contents |
|---|---|
| `edge/detector_picamera.py` | Production detector for the Raspberry Pi: Picamera2 capture, YOLOv8n, event-driven upload |
| `edge/detector_webcam.py` | Desktop version used for development with a USB webcam |
| `app/`, `components/`, `hooks/` | Next.js dashboard (occupancy chart, room view, Potree viewer, `useSensorData` hook) |
| `app/api/sensor-data/route.ts` | Data endpoint for the dashboard |

## Running the detector

```bash
pip install ultralytics opencv-python picamera2
python edge/detector_picamera.py      # on the Raspberry Pi
python edge/detector_webcam.py        # on a laptop with a webcam
```

## Running the dashboard

```bash
pnpm install
pnpm dev
```

## Stack

YOLOv8 Nano, OpenCV, Picamera2, Raspberry Pi 4, SIMCOM A7670SA LTE, AWS IoT Core, API Gateway, Lambda, DynamoDB, AWS CDK, Next.js, Tailwind CSS, Potree.
