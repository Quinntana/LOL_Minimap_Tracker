"""Transparent GUI overlay for displaying champion data."""
from PyQt5.QtWidgets import QMainWindow, QApplication  
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor
import math
from typing import Dict  

class TransparentOverlay(QMainWindow):
    """Overlay window displaying champion positions and directions."""
    
    def __init__(self, get_data_callback, get_rect_center_callback, config: Dict):
        """Initialize the overlay."""
        super().__init__()
        self.get_data = get_data_callback
        self.get_rect_center = get_rect_center_callback
        self.config = config
        self.data = []
        self.rect_center = None
        self.show_arrows = True
        self.show_last_seen = True
        self.init_ui()

    def init_ui(self):
        """Set up the overlay window."""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        screen = QApplication.primaryScreen().size()
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(100)  # Update every 100ms

    def update_overlay(self):
        """Update overlay data and trigger repaint."""
        self.data = self.get_data()
        self.rect_center = self.get_rect_center()
        self.update()

    def toggle_arrows(self):
        """Toggle the visibility of the arrow overlay."""
        self.show_arrows = not self.show_arrows
        self.update()  

    def toggle_last_seen(self):
        self.show_last_seen = not self.show_last_seen
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(255, 0, 0, 255))
        painter.drawRect(self.config["left"], self.config["top"], 
                        self.config["width"], self.config["height"])
        center = (self.width() // 2, self.height() // 2)
        y_offset = 20
        for entry in self.data:
            name = entry["Champion"]
            pos = entry["Position"]
            is_current = entry["IsCurrent"]
            text = f"{name} - ({pos['X']}, {pos['Y']})" if pos else f"{name} - (-)"
            painter.setPen(QColor(255, 255, 255, 255))
            painter.drawText(10, y_offset, text)
            y_offset += 20

            if self.show_arrows and pos and self.rect_center:
                dx, dy = pos["X"] - self.rect_center[0], pos["Y"] - self.rect_center[1]
                end = (center[0] + dx, center[1] + dy)
                distance = math.sqrt(dx ** 2 + dy ** 2)
                color = QColor(255, 0, 0, 255) if is_current else QColor(255, 255, 0, 255)
                painter.setPen(color)
                painter.drawLine(center[0], center[1], end[0], end[1])
                painter.setPen(QColor(255, 255, 255, 255))
                painter.drawText(end[0] + 5, end[1] - 5, f"{name} ({int(distance)})")

        if self.show_last_seen:
            for entry in self.data:
                if not entry["IsCurrent"] and entry["Position"]:
                    pos = entry["Position"]
                    screen_x = self.config["left"] + pos["X"]
                    screen_y = self.config["top"] + pos["Y"]
                    painter.setPen(QColor(255, 0, 0, 255))  # Red dot
                    painter.drawEllipse(screen_x - 2, screen_y - 2, 4, 4)
                    painter.setPen(QColor(255, 255, 255, 255))  # White text
                    painter.drawText(screen_x - 10, screen_y - 5, entry["Champion"])