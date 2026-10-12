<div align="center">

# 🚦 ETLE Vision Engine
### Automated Electronic Traffic Law Enforcement Prototype using AI & Computer Vision

<img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" />
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg" />
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg" />

</div>

---

## 📌 About The Project

**ETLE Vision Engine** is a prototype automated traffic violation detection system designed to assist in traffic monitoring[cite: 7, 12]. Developed as part of a collaborative high school research and technology project[cite: 8, 9], this system simulates electronic law enforcement by combining object detection, object tracking, and automated evidence logging[cite: 7].

Instead of relying on heavy, expensive infrastructure, this prototype demonstrates how computer vision models can process video feeds to detect vehicles, estimate positions relative to a violation line, and record infractions in real time[cite: 13, 18, 21].

## 👥 Team Gerbang Logika / Research Team
* **Marquis Verino** (AI & Computer Vision)[cite: 19]
* **Christiano Orlando Ruslie** (OCR & Data Processing)[cite: 19]
* **Timothea Owen** (Frontend & Dashboard)[cite: 19]
* *Institution:* SMA Kristen Gloria 2 Surabaya[cite: 8]

## ⚡ Core Features

- **Real-Time YOLO Tracking:** Leverages YOLOv8 and OpenCV to detect and track vehicle classes (cars, motorcycles, buses, trucks) across video frames with unique persistence IDs[cite: 18].
- **Interactive Control:** Toggle the detection engine on/off dynamically using the spacebar without restarting the application.
- **Automated Violation Logging:** Automatically flags vehicles crossing the designated violation line, records real timestamps, estimates color heuristics, generates unique Indonesian-style license plate simulations, and saves snapshot image evidence.
- **CSV Data Persistence:** Exports all violation records directly into an organized `.csv` file (`data_pelanggaran_etle.csv`) for dashboard monitoring.

## 🛠️ Tech Stack

* **Language:** Python[cite: 17]
* **Computer Vision & Processing:** OpenCV (`cv2`)[cite: 17]
* **Object Detection & Tracking:** Ultralytics YOLOv8 (`model.track`)[cite: 17, 18]
* **Data Management:** CSV, OS, Datetime modules

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone [https://github.com/riberii/ETLE-Vision-Engine.git](https://github.com/riberii/ETLE-Vision-Engine.git)
   cd ETLE-Vision-Engine
