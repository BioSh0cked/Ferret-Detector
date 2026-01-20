import sys
import cv2
import time
import torch
import threading
from pygrabber.dshow_graph import FilterGraph
from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QListWidget,
    QListWidgetItem, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

def get_camera_devices():
    graph = FilterGraph()
    return list(enumerate(graph.get_input_devices()))

def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices


class YOLOModel:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.conf = 0.25
        self.allowed_classes = None

        torch.backends.cudnn.benchmark = True

    def load(self, path, device):
        self.device = device
        self.model = YOLO(path)

        if device.startswith("cuda"):
            self.model.to(device)

    def infer(self, frame):
        return self.model(
            frame,
            conf=self.conf,
            classes=self.allowed_classes,
            device=self.device if self.device != "tensorrt" else None,
            verbose=False
        )[0]




class CameraThread(threading.Thread):
    def __init__(self, cam_index, engine, label, record):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.engine = engine
        self.label = label
        self.record = record
        self.running = True
        self.writer = None
        self.last_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)

        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                f"camera_{self.cam_index}.mp4",
                fourcc,
                30,
                (int(cap.get(3)), int(cap.get(4)))
            )

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.engine.infer(frame)
            frame = results.plot()

            fps = 1 / (time.time() - self.last_time)
            self.last_time = time.time()

            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if self.writer:
                self.writer.write(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

            self.label.setPixmap(
                QPixmap.fromImage(img).scaled(
                    self.label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio
                )
            )

        cap.release()
        if self.writer:
            self.writer.release()




class YoloWebcamImplement(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YoloWebcamImplement")
        self.resize(1400, 900)

        self.engine = YOLOModel()
        self.active_cameras = {}

        self.init_ui()

    def init_ui(self):
        self.video_layout = QHBoxLayout()
        self.control_layout = QVBoxLayout()
        self.model_btn = QPushButton("Load Model")
        self.model_btn.clicked.connect(self.load_model)
        self.device_box = QComboBox()
        self.device_box.addItems(available_devices())
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("Confidence: 0.25")
        def update_conf(v):
            val = v / 100
            self.engine.conf = val
            self.conf_label.setText(f"Confidence: {val:.2f}")
        self.conf_slider.valueChanged.connect(update_conf)
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.camera_list = QListWidget()
        for idx, name in get_camera_devices():
            item = QListWidgetItem(f"[{idx}] {name}")
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.camera_list.addItem(item)
        self.add_camera_btn = QPushButton("Add Camera")
        self.add_camera_btn.clicked.connect(self.add_camera)
        self.remove_camera_btn = QPushButton("Remove Camera")
        self.remove_camera_btn.clicked.connect(self.remove_camera)
        self.record_box = QCheckBox("Record Output")
        self.dark_box = QCheckBox("Dark Mode")
        self.dark_box.stateChanged.connect(self.toggle_dark)


        ##Layout
        self.control_layout.addWidget(self.model_btn)
        self.control_layout.addWidget(QLabel("Device"))
        self.control_layout.addWidget(self.device_box)
        self.control_layout.addWidget(QLabel("Confidence"))
        self.control_layout.addWidget(self.conf_slider)
        self.control_layout.addWidget(self.conf_label)
        self.control_layout.addWidget(QLabel("Classes"))
        self.control_layout.addWidget(self.class_list)
        self.control_layout.addWidget(QLabel("Cameras"))
        self.control_layout.addWidget(self.camera_list)
        self.control_layout.addWidget(self.add_camera_btn)
        self.control_layout.addWidget(self.remove_camera_btn)
        self.control_layout.addWidget(self.record_box)
        self.control_layout.addWidget(self.dark_box)

        main = QHBoxLayout()
        main.addLayout(self.control_layout, 1)
        main.addLayout(self.video_layout, 3)
        self.setLayout(main)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "*.pt"
        )
        if not path:
            return

        device = self.device_box.currentText()
        self.engine.load(path, device)

        self.class_list.clear()
        for i, name in self.engine.model.names.items():
            item = QListWidgetItem(name)
            item.setSelected(True)
            self.class_list.addItem(item)

    def add_camera(self):
        item = self.camera_list.currentItem()
        if not item:
            return

        cam_index = item.data(Qt.ItemDataRole.UserRole)
        if cam_index in self.active_cameras:
            return

        self.engine.allowed_classes = [
            i for i in range(self.class_list.count())
            if self.class_list.item(i).isSelected()
        ]

        label = QLabel(item.text())
        label.setMinimumSize(400, 300)
        label.setStyleSheet("border: 1px solid gray;")
        self.video_layout.addWidget(label)

        thread = CameraThread(
            cam_index,
            self.engine,
            label,
            self.record_box.isChecked()
        )
        thread.start()

        self.active_cameras[cam_index] = (thread, label)

    def remove_camera(self):
        item = self.camera_list.currentItem()
        if not item:
            return

        cam_index = item.data(Qt.ItemDataRole.UserRole)
        if cam_index not in self.active_cameras:
            return

        thread, label = self.active_cameras.pop(cam_index)
        thread.running = False
        thread.join(timeout=1.0)

        self.video_layout.removeWidget(label)
        label.deleteLater()

    def toggle_dark(self, enabled):
        if enabled:
            self.setStyleSheet("""
                QWidget { background-color: #121212; color: #E0E0E0; }
                QPushButton { background-color: #1F1F1F; }
            """)
        else:
            self.setStyleSheet("")

    def closeEvent(self, event):
        for thread, _ in self.active_cameras.values():
            thread.running = False
            thread.join(timeout=1.0)
        event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = YoloWebcamImplement()
    win.show()
    sys.exit(app.exec())
