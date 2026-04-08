import os
import subprocess
import sys
import time
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from moviepy.editor import VideoFileClip
from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRect,
    QRegularExpression,
    QSettings,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QRegularExpressionValidator,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

VIDEO_FILTER = "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v *.webm)"
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}

QUALITY_PRESETS: List[Tuple[str, int, str]] = [
    ("Original quality (largest file)", 16, "slow"),
    ("High quality", 18, "medium"),
    ("Balanced (recommended)", 20, "medium"),
    ("Compact file", 22, "fast"),
    ("Preview draft", 26, "veryfast"),
]


def configure_local_vlc() -> Optional[str]:
    """Prefer libvlc.dll from the app folder and return plugin path if found."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(app_dir, "plugins")

    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(app_dir)
        except Exception:
            pass

    if app_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{app_dir};{os.environ.get('PATH', '')}"

    if os.path.isdir(plugin_dir):
        os.environ.setdefault("VLC_PLUGIN_PATH", plugin_dir)
        return plugin_dir
    return None


_vlc_plugin_path = configure_local_vlc()
import vlc


@dataclass
class VideoMetadata:
    duration_ms: int
    width: int
    height: int
    fps: float


@dataclass
class ExportJob:
    input_path: str
    output_path: str
    format_ext: str
    start_ms: int
    end_ms: int
    width: int
    height: int
    fps: int
    crf: int
    preset: str
    mute_audio: bool


def format_ms(ms: int) -> str:
    ms = max(0, int(ms))
    total_seconds, ms_part = divmod(ms, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}.{ms_part:03}"
    return f"{minutes:02}:{seconds:02}.{ms_part:03}"


def format_timecode_frames(ms: int, fps: float) -> str:
    fps = max(1.0, fps)
    total_seconds, ms_part = divmod(max(0, int(ms)), 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    frame = int((ms_part / 1000.0) * fps)
    max_frame = max(1, int(round(fps)))
    frame = min(frame, max_frame - 1)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}:{frame:02}"
    return f"{minutes:02}:{seconds:02}:{frame:02}"


def parse_timecode(text: str) -> int:
    value = (text or "").strip().replace(",", ".")
    if not value:
        raise ValueError("Empty time value")

    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError("Use ss(.mmm), mm:ss(.mmm), or hh:mm:ss(.mmm)")

    try:
        if len(parts) == 1:
            total_seconds = float(parts[0])
        elif len(parts) == 2:
            total_seconds = int(parts[0]) * 60 + float(parts[1])
        else:
            total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError as exc:
        raise ValueError("Time parse error") from exc

    if total_seconds < 0:
        raise ValueError("Negative time is not allowed")
    return int(round(total_seconds * 1000))


def even(value: float) -> int:
    n = max(2, int(round(value)))
    return n if n % 2 == 0 else n - 1


def fit_to_height(width: int, height: int, target_height: int) -> Tuple[int, int]:
    if target_height >= height:
        return width, height
    scale = target_height / float(height)
    return even(width * scale), even(target_height)


def is_supported_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def frame_to_pixmap(frame: np.ndarray, width: int = 208, height: int = 116) -> QPixmap:
    rgb = np.ascontiguousarray(frame[:, :, :3].astype(np.uint8))
    h, w, c = rgb.shape
    image = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )


class Card(QFrame):
    def __init__(self, title: str, subtitle: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title_label = QLabel(title, self)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle, self)
            subtitle_label.setObjectName("cardSub")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        layout.addLayout(self.body)


class DropVideoArea(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("videoShell")
        self.setMinimumSize(920, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.placeholder = QLabel("Drop video here\nor click Open", self)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setObjectName("placeholder")
        self.placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.placeholder)

    def set_placeholder_visible(self, visible: bool):
        self.placeholder.setVisible(visible)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            local = urls[0].toLocalFile()
            if local and is_supported_video(local):
                event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            local = urls[0].toLocalFile()
            if local and is_supported_video(local):
                self.file_dropped.emit(local)
                event.acceptProposedAction()


class TimelinePreviewPopup(QFrame):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("previewPopup")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(208, 116)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setObjectName("previewImage")
        layout.addWidget(self.image_label)

        self.time_label = QLabel("00:00.000", self)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setObjectName("previewTime")
        layout.addWidget(self.time_label)

        self.opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(0.0)

        self.fade_in = QPropertyAnimation(self.opacity, b"opacity", self)
        self.fade_in.setDuration(100)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.fade_out = QPropertyAnimation(self.opacity, b"opacity", self)
        self.fade_out.setDuration(100)
        self.fade_out.setStartValue(1.0)
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self.fade_out.finished.connect(self.hide)

    def update_preview(self, pixmap: Optional[QPixmap], time_ms: int):
        if pixmap is not None:
            self.image_label.setText("")
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Loading...")
        self.time_label.setText(format_ms(time_ms))

    def show_at(self, global_center_x: int, global_bottom_y: int):
        self.adjustSize()
        x = global_center_x - self.width() // 2
        y = global_bottom_y - self.height() - 8
        self.move(x, y)
        if not self.isVisible():
            self.show()
            self.fade_out.stop()
            self.fade_in.start()

    def hide_with_fade(self):
        if self.isVisible():
            self.fade_in.stop()
            self.fade_out.start()


class WaveTimeline(QWidget):
    seekRequested = pyqtSignal(int)
    trimChanged = pyqtSignal(int, int)
    hoverPreview = pyqtSignal(int, int)
    hoverLeave = pyqtSignal()
    zoomChanged = pyqtSignal(float, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setObjectName("waveTimeline")

        self.duration_ms = 0
        self.position_ms = 0
        self.start_ms = 0
        self.end_ms = 0
        self.hover_ms: Optional[int] = None
        self.wave_samples: List[float] = []
        self.fps = 30.0

        self.zoom_factor = 1.0
        self.max_zoom = 80.0
        self.view_start_ms = 0

        self.drag_mode: Optional[str] = None
        self.ruler_font = QFont("Consolas", 9)
        self.ruler_font.setStyleHint(QFont.StyleHint.TypeWriter)

    def set_range(self, duration_ms: int):
        self.duration_ms = max(0, int(duration_ms))
        self.position_ms = 0
        self.start_ms = 0
        self.end_ms = self.duration_ms
        self.zoom_factor = 1.0
        self.view_start_ms = 0
        self.drag_mode = None
        self.hover_ms = None
        self.zoomChanged.emit(self.zoom_factor, self._visible_ms() / 1000.0)
        self.update()

    def set_fps(self, fps: float):
        self.fps = max(1.0, float(fps))
        self.update()

    def set_position(self, position_ms: int, auto_scroll: bool = True):
        if self.duration_ms <= 0:
            return
        self.position_ms = max(0, min(int(position_ms), self.duration_ms))
        if auto_scroll:
            self._ensure_visible(self.position_ms)
        self.update()

    def set_markers(self, start_ms: int, end_ms: int):
        frame_ms = max(1, int(round(1000.0 / max(1.0, self.fps))))
        self.start_ms = max(0, min(int(start_ms), self.duration_ms))
        self.end_ms = max(self.start_ms + frame_ms, min(int(end_ms), self.duration_ms))
        if self.end_ms > self.duration_ms:
            self.end_ms = self.duration_ms
            self.start_ms = max(0, self.end_ms - frame_ms)
        self.update()

    def set_wave_samples(self, samples: List[float]):
        self.wave_samples = samples
        self.update()

    def _track_rect(self) -> QRect:
        r = self.rect().adjusted(8, 6, -8, -8)
        return QRect(r.left(), r.top() + 28, r.width(), max(38, r.height() - 28))

    def _ruler_rect(self) -> QRect:
        t = self._track_rect()
        return QRect(t.left(), t.top() - 24, t.width(), 22)

    def _visible_ms(self) -> int:
        if self.duration_ms <= 0:
            return 1
        return max(200, int(self.duration_ms / self.zoom_factor))

    def _max_view_start(self) -> int:
        return max(0, self.duration_ms - self._visible_ms())

    def _set_view_start(self, ms: int):
        self.view_start_ms = max(0, min(int(ms), self._max_view_start()))

    def _ms_to_x(self, ms: int) -> int:
        track = self._track_rect()
        visible = self._visible_ms()
        if visible <= 0:
            return track.left()
        ratio = (ms - self.view_start_ms) / visible
        ratio = min(max(ratio, 0.0), 1.0)
        return track.left() + int(ratio * track.width())

    def _x_to_ms(self, x: int) -> int:
        track = self._track_rect()
        visible = self._visible_ms()
        if visible <= 0:
            return 0
        ratio = (x - track.left()) / max(1, track.width())
        ratio = min(max(ratio, 0.0), 1.0)
        return int(self.view_start_ms + ratio * visible)

    def _ensure_visible(self, ms: int):
        visible = self._visible_ms()
        left = self.view_start_ms
        right = left + visible
        margin = int(visible * 0.15)
        if ms < left + margin:
            self._set_view_start(ms - margin)
        elif ms > right - margin:
            self._set_view_start(ms + margin - visible)

    def _hit_handle(self, x: int) -> Optional[str]:
        if self.duration_ms <= 0:
            return None
        start_x = self._ms_to_x(self.start_ms)
        end_x = self._ms_to_x(self.end_ms)
        if abs(x - start_x) <= 8:
            return "start"
        if abs(x - end_x) <= 8:
            return "end"
        return None

    def _set_cursor_for_x(self, x: int):
        handle = self._hit_handle(x)
        if handle:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        if self.duration_ms <= 0:
            return

        delta_steps = event.angleDelta().y() / 120.0
        if delta_steps == 0:
            return

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            shift = int(self._visible_ms() * 0.12 * (-delta_steps))
            self._set_view_start(self.view_start_ms + shift)
            self.update()
            event.accept()
            return

        old_zoom = self.zoom_factor
        factor = 1.15 ** delta_steps
        new_zoom = max(1.0, min(self.max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 0.001:
            return

        anchor_x = int(event.position().x())
        anchor_ms = self._x_to_ms(anchor_x)
        self.zoom_factor = new_zoom

        visible = self._visible_ms()
        track = self._track_rect()
        ratio = (anchor_x - track.left()) / max(1, track.width())
        self._set_view_start(int(anchor_ms - ratio * visible))

        self.zoomChanged.emit(self.zoom_factor, self._visible_ms() / 1000.0)
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.duration_ms <= 0:
            return

        x = int(event.position().x())
        handle = self._hit_handle(x)
        if handle == "start":
            self.drag_mode = "start"
        elif handle == "end":
            self.drag_mode = "end"
        else:
            self.drag_mode = "scrub"
            ms = self._x_to_ms(x)
            self.seekRequested.emit(ms)

        ms = self._x_to_ms(x)
        self.hover_ms = ms
        self.hoverPreview.emit(ms, x)
        self.update()

    def mouseMoveEvent(self, event):
        x = int(event.position().x())
        self._set_cursor_for_x(x)

        ms = self._x_to_ms(x)
        self.hover_ms = ms
        self.hoverPreview.emit(ms, x)

        if self.drag_mode == "start":
            frame_ms = int(round(1000.0 / max(1.0, self.fps)))
            new_start = max(0, min(ms, self.end_ms - max(1, frame_ms)))
            self.start_ms = new_start
            self.trimChanged.emit(self.start_ms, self.end_ms)
        elif self.drag_mode == "end":
            frame_ms = int(round(1000.0 / max(1.0, self.fps)))
            new_end = max(self.start_ms + max(1, frame_ms), min(ms, self.duration_ms))
            self.end_ms = new_end
            self.trimChanged.emit(self.start_ms, self.end_ms)
        elif self.drag_mode == "scrub":
            self.seekRequested.emit(ms)

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_mode = None

    def leaveEvent(self, event):
        self.hover_ms = None
        self.drag_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.hoverLeave.emit()
        self.update()

    def _choose_major_tick_ms(self, pixels_per_ms: float) -> float:
        frame_ms = 1000.0 / max(1.0, self.fps)
        candidates = [
            frame_ms,
            frame_ms * 2,
            frame_ms * 5,
            frame_ms * 10,
            frame_ms * 15,
            frame_ms * 30,
            1000,
            2000,
            5000,
            10000,
            15000,
            30000,
            60000,
            120000,
            300000,
        ]
        target_px = 150.0
        best = candidates[0]
        best_score = abs(candidates[0] * pixels_per_ms - target_px)
        for c in candidates[1:]:
            score = abs(c * pixels_per_ms - target_px)
            if score < best_score:
                best = c
                best_score = score
        return best

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        outer = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor("#2A2F3D"), 1))
        painter.setBrush(QColor("#141923"))
        painter.drawRect(outer)

        if self.duration_ms <= 0:
            return

        track = self._track_rect()
        ruler = self._ruler_rect()

        painter.fillRect(ruler, QColor("#111722"))
        painter.fillRect(track, QColor("#0F141F"))

        visible = self._visible_ms()
        pixels_per_ms = track.width() / max(1.0, float(visible))

        major_ms = self._choose_major_tick_ms(pixels_per_ms)
        minor_ms = major_ms / 2.0
        painter.setFont(self.ruler_font)
        metrics = QFontMetrics(self.ruler_font)
        min_label_gap_px = metrics.horizontalAdvance("00:00:00:00") + 12
        major_gap_px = max(1.0, major_ms * pixels_per_ms)
        label_every = max(1, int(math.ceil(min_label_gap_px / major_gap_px)))

        start_tick = int(self.view_start_ms // major_ms) * major_ms
        end_tick = self.view_start_ms + visible

        tick = float(start_tick)
        tick_index = 0
        while tick <= end_tick + major_ms:
            x = self._ms_to_x(int(tick))
            painter.setPen(QPen(QColor("#4F5C73"), 1))
            painter.drawLine(x, ruler.bottom() - 6, x, ruler.bottom())
            painter.drawLine(x, track.top(), x, track.top() + 8)

            if tick_index % label_every == 0:
                painter.setPen(QColor("#A8B5CC"))
                painter.drawText(x + 3, ruler.top() + 14, format_timecode_frames(int(tick), self.fps))

            sub = tick + minor_ms
            if sub < end_tick:
                sx = self._ms_to_x(int(sub))
                painter.setPen(QPen(QColor("#374359"), 1))
                painter.drawLine(sx, ruler.bottom() - 3, sx, ruler.bottom())

            tick += major_ms
            tick_index += 1

        samples = self.wave_samples or [0.25] * 200
        bar_count = max(140, track.width() // 3)
        bar_w = max(1, track.width() / max(1, bar_count))

        for i in range(bar_count):
            ms = self.view_start_ms + int((i / max(1, bar_count - 1)) * visible)
            idx = int((ms / max(1, self.duration_ms)) * (len(samples) - 1))
            amp = samples[max(0, min(idx, len(samples) - 1))]
            x = track.left() + int(i * bar_w)
            h = max(2, int(track.height() * (0.15 + 0.85 * amp)))
            y = track.center().y() - h // 2
            color = QColor("#45526A") if i % 2 == 0 else QColor("#3A455B")
            painter.fillRect(x, y, max(1, int(bar_w) - 1), h, color)

        start_x = self._ms_to_x(self.start_ms)
        end_x = self._ms_to_x(self.end_ms)
        if end_x > start_x:
            painter.fillRect(start_x, track.top(), end_x - start_x, track.height(), QColor(59, 122, 255, 58))

        self._draw_trim_handle(painter, track, start_x, QColor("#4EA0FF"), True)
        self._draw_trim_handle(painter, track, end_x, QColor("#4EA0FF"), False)

        play_x = self._ms_to_x(self.position_ms)
        painter.setPen(QPen(QColor("#FF8A3C"), 2))
        painter.drawLine(play_x, ruler.top(), play_x, track.bottom() + 1)

        if self.hover_ms is not None:
            hover_x = self._ms_to_x(self.hover_ms)
            painter.setPen(QPen(QColor("#9DBEF8"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(hover_x, ruler.bottom(), hover_x, track.bottom())

    def _draw_trim_handle(self, painter: QPainter, track: QRect, x: int, color: QColor, is_start: bool):
        painter.setPen(QPen(color, 2))
        painter.drawLine(x, track.top(), x, track.bottom())

        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        if is_start:
            points = [
                QPoint(x, track.top() - 1),
                QPoint(x + 10, track.top() + 8),
                QPoint(x, track.top() + 16),
            ]
        else:
            points = [
                QPoint(x, track.top() - 1),
                QPoint(x - 10, track.top() + 8),
                QPoint(x, track.top() + 16),
            ]
        painter.drawPolygon(points)


class TimelineAnalysisWorker(QObject):
    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, sample_count: int = 220, thumb_count: int = 42):
        super().__init__()
        self.file_path = file_path
        self.sample_count = sample_count
        self.thumb_count = thumb_count

    @pyqtSlot()
    def run(self):
        try:
            clip = VideoFileClip(self.file_path)
            try:
                duration = max(0.001, float(clip.duration or 0.001))
                times = np.linspace(0, max(0.001, duration - 0.001), self.sample_count)
                samples: List[float] = []

                # Prefer real audio amplitude for waveform; fallback to frame luma if audio unavailable.
                if clip.audio is not None:
                    for t in times:
                        try:
                            audio_frame = clip.audio.get_frame(float(t))
                            if isinstance(audio_frame, np.ndarray):
                                amp = float(np.mean(np.abs(audio_frame)))
                            else:
                                amp = abs(float(audio_frame))
                        except Exception:
                            amp = 0.0
                        samples.append(amp)
                else:
                    for t in times:
                        frame = clip.get_frame(float(t))
                        tiny = frame[::12, ::12, :3]
                        luma = (0.2126 * tiny[:, :, 0] + 0.7152 * tiny[:, :, 1] + 0.0722 * tiny[:, :, 2]).mean() / 255.0
                        samples.append(float(luma))

                max_val = max(samples) if samples else 1.0
                min_val = min(samples) if samples else 0.0
                spread = max(1e-6, max_val - min_val)
                normalized = [0.04 + ((v - min_val) / spread) * 0.96 for v in samples]

                thumbs = []
                for t in np.linspace(0, max(0.001, duration - 0.001), self.thumb_count):
                    frame = clip.get_frame(float(t))
                    rgb = np.ascontiguousarray(frame[:, :, :3].astype(np.uint8))
                    thumbs.append((int(round(t * 1000)), rgb))
            finally:
                clip.close()

            self.finished.emit(normalized, thumbs)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, job: ExportJob):
        super().__init__()
        self.job = job
        self._cancel_requested = False
        self._process: Optional[subprocess.Popen] = None

    @pyqtSlot()
    def run(self):
        try:
            ffmpeg = self._resolve_ffmpeg_binary()
            duration_ms = max(1, self.job.end_ms - self.job.start_ms)
            duration_us = duration_ms * 1000
            self.progress.emit(0, "Encoding")

            ext = self.job.format_ext.lower()
            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{self.job.start_ms / 1000.0:.3f}",
                "-i",
                self.job.input_path,
                "-t",
                f"{duration_ms / 1000.0:.3f}",
            ]

            if ext == "gif":
                cmd += [
                    "-vf",
                    f"fps={self.job.fps},scale={self.job.width}:{self.job.height}:flags=lanczos",
                    "-loop",
                    "0",
                ]
            elif ext == "webm":
                cmd += [
                    "-vf",
                    f"scale={self.job.width}:{self.job.height}:flags=lanczos",
                    "-r",
                    str(self.job.fps),
                    "-c:v",
                    "libvpx-vp9",
                    "-b:v",
                    "0",
                    "-crf",
                    str(self.job.crf),
                ]
                if self.job.mute_audio:
                    cmd += ["-an"]
                else:
                    cmd += ["-c:a", "libopus", "-b:a", "160k"]
            elif ext == "avi":
                cmd += [
                    "-vf",
                    f"scale={self.job.width}:{self.job.height}:flags=lanczos",
                    "-r",
                    str(self.job.fps),
                    "-c:v",
                    "mpeg4",
                    "-q:v",
                    "3",
                ]
                if self.job.mute_audio:
                    cmd += ["-an"]
                else:
                    cmd += ["-c:a", "mp3", "-b:a", "192k"]
            else:
                cmd += [
                    "-vf",
                    f"scale={self.job.width}:{self.job.height}:flags=lanczos",
                    "-r",
                    str(self.job.fps),
                    "-c:v",
                    "libx264",
                    "-preset",
                    self.job.preset,
                    "-crf",
                    str(self.job.crf),
                ]
                if self.job.mute_audio:
                    cmd += ["-an"]
                else:
                    cmd += ["-c:a", "aac", "-b:a", "192k"]
                if ext in {"mp4", "mov"}:
                    cmd += ["-movflags", "+faststart"]

            cmd += [
                "-progress",
                "pipe:1",
                "-nostats",
                self.job.output_path,
            ]

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            tail = []
            for raw_line in self._process.stdout:
                if self._cancel_requested:
                    self._terminate_process()
                    break

                line = raw_line.strip()
                if not line:
                    continue

                tail.append(line)
                if len(tail) > 40:
                    tail.pop(0)

                if line.startswith("out_time_ms="):
                    raw_value = line.split("=", 1)[1].strip()
                    try:
                        out_us = int(raw_value)
                    except ValueError:
                        out_us = 0
                    pct = min(99, int((out_us / duration_us) * 100))
                    self.progress.emit(pct, "Encoding")
                elif line == "progress=end":
                    self.progress.emit(100, "Done")

            return_code = self._process.wait()
            if self._cancel_requested:
                self.failed.emit("Export canceled")
                return

            if return_code != 0:
                msg = "\n".join(tail[-10:])
                raise RuntimeError(f"ffmpeg exited with code {return_code}\n{msg}")

            self.finished.emit(self.job.output_path)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._process = None

    def cancel(self):
        self._cancel_requested = True
        self._terminate_process()

    def _terminate_process(self):
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=1)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass

    @staticmethod
    def _resolve_ffmpeg_binary() -> str:
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

class VideoPlayer(QMainWindow):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("VideoCut")
        self.resize(1460, 920)
        self.setAcceptDrops(True)

        self.settings = QSettings("videocut", "studio")
        self.default_open_dir = self._read_setting("paths/open", self._default_open_path())
        self.default_save_dir = self._read_setting("paths/save", os.path.join(os.path.expanduser("~"), "Downloads"))
        self.current_volume = int(self._read_setting("player/volume", 85))
        self.export_mute_audio = bool(int(self._read_setting("export/mute_audio", 0)))

        # VLC 4+ removed --plugin-path; rely on VLC_PLUGIN_PATH env instead.
        self.vlc_instance = vlc.Instance("--quiet")
        self.media_player = self.vlc_instance.media_player_new()

        self.media_path: Optional[str] = None
        self.metadata: Optional[VideoMetadata] = None
        self.start_ms = 0
        self.end_ms = 0

        self.thumbnail_cache: Dict[int, QPixmap] = {}
        self.preview_lowres_clip: Optional[VideoFileClip] = None

        self.export_thread: Optional[QThread] = None
        self.export_worker: Optional[ExportWorker] = None
        self.analysis_thread: Optional[QThread] = None
        self.analysis_worker: Optional[TimelineAnalysisWorker] = None

        self.card_animations: List[QPropertyAnimation] = []

        self.last_vlc_ms = 0
        self.last_vlc_clock = time.perf_counter()
        self.last_preview_fetch_ts = 0.0
        self.last_preview_requested_ms = -1

        self._set_window_icon()
        self._build_ui()
        self._apply_style()
        self._disable_button_focus_outline()
        self._create_shortcuts()
        self._bind_video_surface()
        self._animate_panels_in()

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_playback_ui)
        self.ui_timer.start(16)

        if file_path:
            self.open_video(file_path)

    def _build_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        top_row = QHBoxLayout()
        self.open_button = QPushButton("Open", self)
        self.open_button.setObjectName("primaryBtn")
        self.open_button.clicked.connect(self.open_video)
        top_row.addWidget(self.open_button)

        self.file_label = QLabel("No file selected", self)
        self.file_label.setObjectName("fileInfo")
        top_row.addWidget(self.file_label, 1)
        left_col.addLayout(top_row)

        self.video_area = DropVideoArea(self)
        self.video_area.file_dropped.connect(self.open_video)
        left_col.addWidget(self.video_area, 1)

        transport = QFrame(self)
        transport.setObjectName("transport")
        t_layout = QVBoxLayout(transport)
        t_layout.setContentsMargins(10, 10, 10, 10)
        t_layout.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.play_button = QPushButton("Play", self)
        self.play_button.setObjectName("accentBtn")
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)

        self.back_button = QPushButton("<< 5s", self)
        self.back_button.clicked.connect(lambda: self.seek_relative(-5000))
        controls.addWidget(self.back_button)

        self.forward_button = QPushButton("5s >>", self)
        self.forward_button.clicked.connect(lambda: self.seek_relative(5000))
        controls.addWidget(self.forward_button)

        controls.addWidget(QLabel("Volume", self))

        self.volume = QSlider(Qt.Orientation.Horizontal, self)
        self.volume.setRange(0, 100)
        self.volume.setValue(self.current_volume)
        self.volume.setFixedWidth(140)
        self.volume.valueChanged.connect(self.change_volume)
        controls.addWidget(self.volume)

        controls.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self.zoom_label = QLabel("Zoom 1.0x | View 0.0s", self)
        self.zoom_label.setObjectName("metaLabel")
        controls.addWidget(self.zoom_label)

        self.time_label = QLabel("00:00.000 / 00:00.000", self)
        self.time_label.setObjectName("timeLabel")
        controls.addWidget(self.time_label)

        t_layout.addLayout(controls)

        self.timeline = WaveTimeline(self)
        self.timeline.seekRequested.connect(self.seek_to)
        self.timeline.trimChanged.connect(self.on_timeline_trim_changed)
        self.timeline.hoverPreview.connect(self.on_timeline_hover)
        self.timeline.hoverLeave.connect(self.hide_timeline_preview)
        self.timeline.zoomChanged.connect(self.on_timeline_zoom_changed)
        t_layout.addWidget(self.timeline)
        left_col.addWidget(transport)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        self.marker_card = Card("Trim", "Drag trim handles on timeline or edit values below")
        marker_body = QVBoxLayout()
        marker_body.setSpacing(8)
        trim_label_width = 54
        trim_button_width = 96

        start_row = QHBoxLayout()
        start_label = QLabel("Start", self)
        start_label.setFixedWidth(trim_label_width)
        start_row.addWidget(start_label)
        self.start_edit = QLineEdit("00:00.000", self)
        self.start_edit.setObjectName("timeEdit")
        self.start_edit.setFont(QFont("Consolas", 10))
        start_row.addWidget(self.start_edit, 1)
        self.set_start_btn = QPushButton("Set Start", self)
        self.set_start_btn.setFixedWidth(trim_button_width)
        self.set_start_btn.clicked.connect(self.set_start_from_playhead)
        start_row.addWidget(self.set_start_btn)
        marker_body.addLayout(start_row)

        end_row = QHBoxLayout()
        end_label = QLabel("End", self)
        end_label.setFixedWidth(trim_label_width)
        end_row.addWidget(end_label)
        self.end_edit = QLineEdit("00:00.000", self)
        self.end_edit.setObjectName("timeEdit")
        self.end_edit.setFont(QFont("Consolas", 10))
        end_row.addWidget(self.end_edit, 1)
        self.set_end_btn = QPushButton("Set End", self)
        self.set_end_btn.setFixedWidth(trim_button_width)
        self.set_end_btn.clicked.connect(self.set_end_from_playhead)
        end_row.addWidget(self.set_end_btn)
        marker_body.addLayout(end_row)

        nav_row = QHBoxLayout()
        self.goto_start_btn = QPushButton("Go Start", self)
        self.goto_start_btn.clicked.connect(lambda: self.seek_to(self.start_ms))
        nav_row.addWidget(self.goto_start_btn)
        self.goto_end_btn = QPushButton("Go End", self)
        self.goto_end_btn.clicked.connect(lambda: self.seek_to(self.end_ms))
        nav_row.addWidget(self.goto_end_btn)
        self.selection_len = QLabel("Selection: 00:00.000", self)
        self.selection_len.setObjectName("metaLabel")
        nav_row.addWidget(self.selection_len, 1, Qt.AlignmentFlag.AlignRight)
        marker_body.addLayout(nav_row)

        self.marker_card.body.addLayout(marker_body)
        right_col.addWidget(self.marker_card)

        self.export_card = Card("Export", "Presets or manual settings + output format")
        export_grid = QGridLayout()
        export_grid.setHorizontalSpacing(8)
        export_grid.setVerticalSpacing(8)
        export_label_width = 88

        format_label = QLabel("Format", self)
        format_label.setFixedWidth(export_label_width)
        export_grid.addWidget(format_label, 0, 0)
        self.format_combo = QComboBox(self)
        self.format_combo.addItem("MP4 (H.264)", "mp4")
        self.format_combo.addItem("MOV (H.264)", "mov")
        self.format_combo.addItem("MKV (H.264)", "mkv")
        self.format_combo.addItem("WEBM (VP9)", "webm")
        self.format_combo.addItem("AVI (MPEG4)", "avi")
        self.format_combo.addItem("GIF (animated)", "gif")
        export_grid.addWidget(self.format_combo, 0, 1, 1, 3)
        self._configure_combo_dropdown(self.format_combo)

        mode_label = QLabel("Mode", self)
        mode_label.setFixedWidth(export_label_width)
        export_grid.addWidget(mode_label, 1, 0)
        self.export_mode_combo = QComboBox(self)
        self.export_mode_combo.addItems(["Preset", "Custom"])
        self._configure_combo_dropdown(self.export_mode_combo)
        export_grid.addWidget(self.export_mode_combo, 1, 1, 1, 3)

        aspect_label = QLabel("Aspect", self)
        aspect_label.setFixedWidth(export_label_width)
        export_grid.addWidget(aspect_label, 2, 0)
        self.aspect_combo = QComboBox(self)
        self.aspect_combo.addItem("Original", "original")
        self.aspect_combo.addItem("1:1", "1:1")
        self.aspect_combo.addItem("16:9", "16:9")
        self.aspect_combo.addItem("4:3", "4:3")
        self.aspect_combo.addItem("3:2", "3:2")
        self.aspect_combo.addItem("9:16", "9:16")
        self.aspect_combo.addItem("Free", "free")
        self._configure_combo_dropdown(self.aspect_combo)
        export_grid.addWidget(self.aspect_combo, 2, 1, 1, 3)

        audio_label = QLabel("Audio", self)
        audio_label.setFixedWidth(export_label_width)
        export_grid.addWidget(audio_label, 3, 0)
        self.export_audio_btn = QPushButton("Keep audio", self)
        self.export_audio_btn.setCheckable(True)
        self.export_audio_btn.setChecked(self.export_mute_audio)
        export_grid.addWidget(self.export_audio_btn, 3, 1, 1, 3)

        self.export_mode_stack = QStackedWidget(self)
        export_grid.addWidget(self.export_mode_stack, 4, 0, 1, 4)

        preset_page = QWidget(self)
        preset_grid = QGridLayout(preset_page)
        preset_grid.setContentsMargins(0, 0, 0, 0)
        preset_grid.setHorizontalSpacing(8)
        preset_grid.setVerticalSpacing(8)
        preset_grid.setColumnStretch(1, 1)
        preset_grid.setColumnStretch(2, 1)

        p_res_label = QLabel("Resolution", self)
        p_res_label.setFixedWidth(export_label_width)
        preset_grid.addWidget(p_res_label, 0, 0)
        self.resolution_combo = QComboBox(self)
        self._configure_combo_dropdown(self.resolution_combo)
        preset_grid.addWidget(self.resolution_combo, 0, 1, 1, 2)

        p_fps_label = QLabel("FPS", self)
        p_fps_label.setFixedWidth(export_label_width)
        preset_grid.addWidget(p_fps_label, 1, 0)
        self.fps_preset_combo = QComboBox(self)
        self._configure_combo_dropdown(self.fps_preset_combo)
        preset_grid.addWidget(self.fps_preset_combo, 1, 1, 1, 2)

        p_quality_label = QLabel("Quality", self)
        p_quality_label.setFixedWidth(export_label_width)
        preset_grid.addWidget(p_quality_label, 2, 0)
        self.quality_combo = QComboBox(self)
        for title, crf, preset in QUALITY_PRESETS:
            self.quality_combo.addItem(title, (crf, preset))
        self.quality_combo.setCurrentIndex(2)
        self._configure_combo_dropdown(self.quality_combo)
        preset_grid.addWidget(self.quality_combo, 2, 1, 1, 2)

        custom_page = QWidget(self)
        custom_grid = QGridLayout(custom_page)
        custom_grid.setContentsMargins(0, 0, 0, 0)
        custom_grid.setHorizontalSpacing(8)
        custom_grid.setVerticalSpacing(8)
        custom_grid.setColumnStretch(1, 1)
        custom_grid.setColumnStretch(2, 1)

        c_res_label = QLabel("Resolution", self)
        c_res_label.setFixedWidth(export_label_width)
        custom_grid.addWidget(c_res_label, 0, 0)
        self.custom_width_edit = QLineEdit("1920", self)
        self.custom_width_edit.setPlaceholderText("W")
        self.custom_height_edit = QLineEdit("1080", self)
        self.custom_height_edit.setPlaceholderText("H")
        custom_grid.addWidget(self.custom_width_edit, 0, 1)
        custom_grid.addWidget(self.custom_height_edit, 0, 2)

        c_fps_label = QLabel("FPS", self)
        c_fps_label.setFixedWidth(export_label_width)
        custom_grid.addWidget(c_fps_label, 1, 0)
        self.custom_fps_edit = QLineEdit("30", self)
        self.custom_fps_edit.setPlaceholderText("FPS")
        custom_grid.addWidget(self.custom_fps_edit, 1, 1, 1, 2)

        c_quality_label = QLabel("Quality", self)
        c_quality_label.setFixedWidth(export_label_width)
        custom_grid.addWidget(c_quality_label, 2, 0)
        self.custom_crf_edit = QLineEdit("20", self)
        self.custom_crf_edit.setPlaceholderText("CRF 0..51")
        custom_grid.addWidget(self.custom_crf_edit, 2, 1)
        self.custom_preset_combo = QComboBox(self)
        self.custom_preset_combo.addItems(["slow", "medium", "fast", "veryfast"])
        self._configure_combo_dropdown(self.custom_preset_combo)
        custom_grid.addWidget(self.custom_preset_combo, 2, 2)
        self.crf_help_label = QLabel("CRF: lower = better quality/larger file, higher = smaller file (0..51).", self)
        self.crf_help_label.setObjectName("metaLabel")
        self.crf_help_label.setWordWrap(True)
        custom_grid.addWidget(self.crf_help_label, 3, 0, 1, 3)

        self.export_mode_stack.addWidget(preset_page)
        self.export_mode_stack.addWidget(custom_page)

        save_label = QLabel("Save To", self)
        save_label.setFixedWidth(export_label_width)
        export_grid.addWidget(save_label, 5, 0)
        self.save_dir_edit = QLineEdit(self.default_save_dir, self)
        export_grid.addWidget(self.save_dir_edit, 5, 1, 1, 2)
        self.browse_save_dir_btn = QPushButton("...", self)
        self.browse_save_dir_btn.setFixedWidth(40)
        self.browse_save_dir_btn.clicked.connect(self.choose_save_directory)
        export_grid.addWidget(self.browse_save_dir_btn, 5, 3)

        self.export_card.body.addLayout(export_grid)

        self.progress = QProgressBar(self)
        self.progress.hide()
        self.export_card.body.addWidget(self.progress)
        self.expected_size_label = QLabel("Expected size: --", self)
        self.expected_size_label.setObjectName("metaLabel")
        self.export_card.body.addWidget(self.expected_size_label)

        right_col.addWidget(self.export_card)

        self.hotkey_card = Card("Hotkeys")
        self._add_hotkey_row("Play / Pause", "Space")
        self._add_hotkey_row("Set Start", "I")
        self._add_hotkey_row("Set End", "O")
        self._add_hotkey_row("Seek -5 sec", "Left")
        self._add_hotkey_row("Seek +5 sec", "Right")
        self._add_hotkey_row("Open File", "Ctrl+O")
        self._add_hotkey_row("Export", "Ctrl+S")
        right_col.addWidget(self.hotkey_card)
        right_col.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        bottom_actions = QFrame(self)
        bottom_actions.setObjectName("bottomActions")
        bottom_actions_layout = QHBoxLayout(bottom_actions)
        bottom_actions_layout.setContentsMargins(0, 0, 0, 0)
        bottom_actions_layout.setSpacing(8)

        self.export_btn = QPushButton("Export Clip", self)
        self.export_btn.setObjectName("primaryBtn")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.cut_and_save)
        bottom_actions_layout.addWidget(self.export_btn, 1)

        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_export)
        bottom_actions_layout.addWidget(self.cancel_btn)

        right_col.addWidget(bottom_actions)

        left_host = QWidget(self)
        left_host.setLayout(left_col)
        right_host = QWidget(self)
        right_host.setLayout(right_col)
        right_host.setFixedWidth(390)

        root_layout.addWidget(left_host, 1)
        root_layout.addWidget(right_host)

        self.preview_popup = TimelinePreviewPopup(self)

        validator = QRegularExpressionValidator(QRegularExpression(r"[0-9:.,]*"), self)
        self.start_edit.setValidator(validator)
        self.end_edit.setValidator(validator)
        int_validator = QRegularExpressionValidator(QRegularExpression(r"[0-9]{0,5}"), self)
        float_validator = QRegularExpressionValidator(QRegularExpression(r"[0-9]{0,3}([.,][0-9]{0,3})?"), self)
        self.custom_width_edit.setValidator(int_validator)
        self.custom_height_edit.setValidator(int_validator)
        self.custom_crf_edit.setValidator(int_validator)
        self.custom_fps_edit.setValidator(float_validator)
        self.start_edit.editingFinished.connect(self.apply_start_input)
        self.end_edit.editingFinished.connect(self.apply_end_input)
        self.export_mode_combo.currentIndexChanged.connect(self._update_export_mode_controls)
        self.aspect_combo.currentIndexChanged.connect(self.on_aspect_changed)
        self.format_combo.currentIndexChanged.connect(self.on_export_setting_changed)
        self.fps_preset_combo.currentIndexChanged.connect(self.on_export_setting_changed)
        self.resolution_combo.currentIndexChanged.connect(self.on_export_setting_changed)
        self.quality_combo.currentIndexChanged.connect(self.on_export_setting_changed)
        self.custom_preset_combo.currentIndexChanged.connect(self.on_export_setting_changed)
        self.export_audio_btn.toggled.connect(self.on_export_audio_toggled)
        self.custom_width_edit.editingFinished.connect(self.on_custom_resolution_edited)
        self.custom_height_edit.editingFinished.connect(self.on_custom_resolution_edited)
        self.custom_fps_edit.editingFinished.connect(self.on_export_setting_changed)
        self.custom_crf_edit.editingFinished.connect(self.on_export_setting_changed)
        self._update_export_mode_controls()
        self.on_export_audio_toggled(self.export_audio_btn.isChecked())

        self.statusBar().showMessage("Ready")

    def _configure_combo_dropdown(self, combo: QComboBox):
        view = QListView(combo)
        view.setObjectName("comboPopup")
        combo.setView(view)
        combo.setMaxVisibleItems(14)

    def _update_export_mode_controls(self, *_):
        mode_index = self.export_mode_combo.currentIndex()
        self.export_mode_stack.setCurrentIndex(mode_index)
        self.update_expected_size_estimate()

    @staticmethod
    def _parse_ratio_text(value: str) -> Optional[Tuple[int, int]]:
        if value in {"original", "free", ""}:
            return None
        try:
            w_text, h_text = value.split(":")
            rw = int(w_text)
            rh = int(h_text)
            if rw <= 0 or rh <= 0:
                return None
            return rw, rh
        except Exception:
            return None

    def on_export_audio_toggled(self, checked: bool):
        self.export_audio_btn.setText("Mute audio" if checked else "Keep audio")
        self.export_mute_audio = bool(checked)
        self.settings.setValue("export/mute_audio", int(self.export_mute_audio))
        self.update_expected_size_estimate()

    def on_aspect_changed(self, *_):
        self.populate_resolution_and_fps(reset_custom=False)
        self.update_expected_size_estimate()

    def on_custom_resolution_edited(self):
        if not self.metadata:
            return
        ratio_value = str(self.aspect_combo.currentData())
        ratio = self._parse_ratio_text(ratio_value)
        if ratio is None:
            self.update_expected_size_estimate()
            return

        aw, ah = ratio
        src = self.sender()
        if src is self.custom_width_edit:
            try:
                current_w = int(self.custom_width_edit.text().strip())
            except Exception:
                current_w = self.metadata.width
            current_w = max(16, min(16384, current_w))
            new_h = max(16, min(16384, even((current_w * ah) / float(aw))))
            self.custom_width_edit.setText(str(current_w))
            self.custom_height_edit.setText(str(new_h))
        else:
            try:
                current_h = int(self.custom_height_edit.text().strip())
            except Exception:
                current_h = self.metadata.height
            current_h = max(16, min(16384, current_h))
            new_w = max(16, min(16384, even((current_h * aw) / float(ah))))
            self.custom_width_edit.setText(str(new_w))
            self.custom_height_edit.setText(str(current_h))
        self.update_expected_size_estimate()

    def on_export_setting_changed(self, *_):
        is_gif = str(self.format_combo.currentData()) == "gif"
        self.export_audio_btn.setEnabled(not is_gif)
        self.export_audio_btn.blockSignals(True)
        self.export_audio_btn.setChecked(True if is_gif else self.export_mute_audio)
        self.export_audio_btn.blockSignals(False)
        self.export_audio_btn.setText("Mute audio" if self.export_audio_btn.isChecked() else "Keep audio")
        self.update_expected_size_estimate()

    def _estimate_export_values(self) -> Optional[Tuple[int, int, int, int, str, bool]]:
        if not self.metadata:
            return None

        custom_mode = self.export_mode_combo.currentIndex() == 1
        source_fps = float(self.metadata.fps)
        format_ext = str(self.format_combo.currentData() or "mp4").lower()

        try:
            if custom_mode:
                w = int(self.custom_width_edit.text().strip() or "0")
                h = int(self.custom_height_edit.text().strip() or "0")
                fps = int(round(float(self.custom_fps_edit.text().strip().replace(",", ".") or "0")))
                crf = int(self.custom_crf_edit.text().strip() or "0")
            else:
                res_text = self.resolution_combo.currentText().strip()
                w_text, h_text = res_text.lower().split("x")
                w = int(w_text)
                h = int(h_text)
                fps_data = self.fps_preset_combo.currentData()
                fps = int(round(source_fps)) if fps_data == "original" else int(fps_data)
                preset_data = self.quality_combo.currentData()
                crf = int(preset_data[0]) if preset_data else 20
        except Exception:
            return None

        if w < 16 or h < 16 or fps < 1:
            return None
        if format_ext != "gif":
            w = even(w)
            h = even(h)
        crf = max(0, min(51, crf))
        mute_audio = bool(self.export_audio_btn.isChecked() or format_ext == "gif")
        return w, h, fps, crf, format_ext, mute_audio

    @staticmethod
    def _human_size(num_bytes: float) -> str:
        value = max(0.0, float(num_bytes))
        units = ["B", "KB", "MB", "GB", "TB"]
        idx = 0
        while value >= 1024.0 and idx < len(units) - 1:
            value /= 1024.0
            idx += 1
        if idx == 0:
            return f"{int(round(value))} {units[idx]}"
        return f"{value:.1f} {units[idx]}"

    def update_expected_size_estimate(self):
        if not self.metadata or not self.media_path:
            self.expected_size_label.setText("Expected size: --")
            return

        values = self._estimate_export_values()
        if values is None:
            self.expected_size_label.setText("Expected size: --")
            return

        w, h, fps, crf, format_ext, mute_audio = values
        selection_seconds = max(0.001, (self.end_ms - self.start_ms) / 1000.0)
        source_seconds = max(0.001, self.metadata.duration_ms / 1000.0)
        try:
            source_size = float(os.path.getsize(self.media_path))
        except Exception:
            self.expected_size_label.setText("Expected size: --")
            return

        source_bitrate = (source_size * 8.0) / source_seconds
        pixel_scale = (w * h) / float(max(1, self.metadata.width * self.metadata.height))
        fps_scale = fps / max(1.0, float(self.metadata.fps))
        crf_scale = 2.0 ** ((23.0 - float(crf)) / 6.0)
        video_bitrate = max(120_000.0, source_bitrate * pixel_scale * fps_scale * crf_scale * 0.55)
        if format_ext == "gif":
            video_bitrate = max(video_bitrate, w * h * fps * 0.7)

        audio_bitrate = 0.0 if mute_audio or format_ext == "gif" else 192_000.0
        estimated_bytes = selection_seconds * (video_bitrate + audio_bitrate) / 8.0
        self.expected_size_label.setText(f"Expected size: ~{self._human_size(estimated_bytes)} (estimate)")

    def _add_hotkey_row(self, action_text: str, combo_text: str):
        row = QHBoxLayout()
        action = QLabel(action_text, self)
        action.setObjectName("hotkeyAction")
        combo = QLabel(combo_text, self)
        combo.setObjectName("hotkeyCombo")
        combo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(action)
        row.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row.addWidget(combo)
        self.hotkey_card.body.addLayout(row)

    def _apply_style(self):
        self.setFont(QFont("Bahnschrift", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: #1E1E1E;
            }
            QStatusBar {
                background: #2A2A2A;
                color: #B5B5B5;
                border-top: 1px solid #3A3A3A;
            }
            QLabel {
                color: #D9D9D9;
            }
            #fileInfo {
                color: #B0B0B0;
                font-weight: 600;
            }
            #metaLabel {
                color: #9A9A9A;
                font-size: 11px;
                font-weight: 600;
                font-family: Consolas;
            }
            #timeLabel {
                color: #ECECEC;
                font-size: 16px;
                font-weight: 700;
                font-family: Consolas;
            }
            #videoShell {
                background: #121212;
                border: 1px solid #333333;
                border-radius: 0px;
            }
            #placeholder {
                color: #9C9C9C;
                font-size: 22px;
                font-weight: 700;
            }
            #transport {
                background: #252525;
                border: 1px solid #3A3A3A;
                border-radius: 0px;
            }
            #card {
                background: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 0px;
            }
            #cardTitle {
                color: #EAEAEA;
                font-size: 15px;
                font-weight: 700;
            }
            #cardSub {
                color: #A2A2A2;
                font-size: 11px;
            }
            QPushButton {
                background: #3A3A3A;
                border: 1px solid #505050;
                border-radius: 0px;
                padding: 8px 12px;
                color: #E7E7E7;
                font-weight: 600;
                outline: none;
            }
            QPushButton:focus {
                outline: none;
                border: 1px solid #505050;
            }
            QPushButton:hover {
                background: #474747;
                border-color: #676767;
            }
            QPushButton:pressed {
                background: #2F2F2F;
            }
            QPushButton::menu-indicator {
                image: none;
            }
            QPushButton:disabled {
                background: #292929;
                border-color: #3A3A3A;
                color: #707070;
            }
            #primaryBtn {
                background: #0E639C;
                border: 1px solid #1980C5;
                color: #F2F8FF;
                font-weight: 700;
            }
            #primaryBtn:hover {
                background: #1177B9;
            }
            #accentBtn {
                background: #4C4C4C;
                border: 1px solid #656565;
                color: #F2F2F2;
                font-weight: 700;
            }
            #dangerBtn {
                background: #5A2B2B;
                border: 1px solid #8B4444;
                color: #FFEDED;
            }
            #dangerBtn:hover {
                background: #6D3434;
            }
            QLineEdit, QComboBox {
                background: #1F1F1F;
                border: 1px solid #4A4A4A;
                border-radius: 0px;
                padding: 6px;
                color: #ECECEC;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #5C5C5C;
            }
            QComboBox::drop-down {
                border-left: 1px solid #505050;
                width: 24px;
                background: #2B2B2B;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #BDBDBD;
            }
            QListView#comboPopup {
                background: #1B1B1B;
                border: 1px solid #5A5A5A;
                outline: 0;
                color: #EEEEEE;
                padding: 2px;
            }
            QListView#comboPopup::item {
                height: 24px;
                padding: 2px 8px;
            }
            QListView#comboPopup::item:selected {
                background: #0E639C;
                color: #F7FBFF;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4B4B4B;
                height: 8px;
                background: #1F1F1F;
                border-radius: 0px;
            }
            QSlider::handle:horizontal {
                background: #BEBEBE;
                border: 1px solid #E2E2E2;
                width: 12px;
                margin: -3px 0;
                border-radius: 0px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 0px;
                background: #1A1A1A;
                color: #E4E4E4;
                text-align: center;
                font-weight: 700;
            }
            QProgressBar::chunk {
                background: #0E639C;
            }
            #previewPopup {
                background: #252525;
                border: 1px solid #4A4A4A;
                border-radius: 0px;
            }
            #previewImage {
                background: #101010;
                border-radius: 0px;
            }
            #previewTime {
                color: #ECECEC;
                font-weight: 700;
                font-family: Consolas;
            }
            #hotkeyAction {
                color: #F0F0F0;
                font-weight: 600;
            }
            #hotkeyCombo {
                color: #909090;
                font-weight: 700;
                font-family: Consolas;
            }
            #bottomActions {
                padding-top: 4px;
            }
            """
        )

    def _disable_button_focus_outline(self):
        for button in self.findChildren(QPushButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _create_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_video)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.cut_and_save)
        QShortcut(QKeySequence("I"), self, activated=self.set_start_from_playhead)
        QShortcut(QKeySequence("O"), self, activated=self.set_end_from_playhead)
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.seek_relative(-5000))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.seek_relative(5000))

    def _animate_panels_in(self):
        self.card_animations.clear()
        for index, widget in enumerate((self.marker_card, self.export_card, self.hotkey_card)):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            effect.setOpacity(0.0)

            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(180 + index * 70)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            QTimer.singleShot(index * 80, anim.start)
            self.card_animations.append(anim)

    def _set_window_icon(self):
        base = os.path.dirname(os.path.abspath(__file__))
        for icon_name in ("scissors.ico", "icon.ico"):
            icon_path = os.path.join(base, icon_name)
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                self.setWindowIcon(icon)
                app = QApplication.instance()
                if app is not None:
                    app.setWindowIcon(icon)
                if sys.platform.startswith("win"):
                    try:
                        import ctypes

                        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("videocut.studio.app")
                    except Exception:
                        pass
                break

    def _bind_video_surface(self):
        win_id = int(self.video_area.winId())
        if sys.platform.startswith("win"):
            self.media_player.set_hwnd(win_id)
        elif sys.platform.startswith("linux"):
            self.media_player.set_xwindow(win_id)
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(win_id)

    def _default_open_path(self) -> str:
        videos = os.path.join(os.path.expanduser("~"), "Videos")
        return videos if os.path.isdir(videos) else os.path.expanduser("~")

    def _read_setting(self, key: str, default):
        value = self.settings.value(key, default)
        if value in (None, ""):
            return default
        return value

    def open_video(self, file_path: Optional[str] = None):
        if isinstance(file_path, bool):
            file_path = None

        if self.export_thread and self.export_thread.isRunning():
            self.show_error("Wait for export to finish or cancel it")
            return

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", self.default_open_dir, VIDEO_FILTER)
            if not file_path:
                return

        if not os.path.isfile(file_path):
            self.show_error("File not found")
            return
        if not is_supported_video(file_path):
            self.show_error("Unsupported file type")
            return

        self.progress.show()
        self.progress.setRange(0, 0)
        self.progress.setFormat("Analyzing video...")
        QApplication.processEvents()

        try:
            metadata = self.probe_video(file_path)
            self.load_video(file_path, metadata)
            self.statusBar().showMessage("Video loaded")
        except Exception as exc:
            self.show_error(f"Failed to open video\n{exc}")
        finally:
            self.progress.hide()

    def probe_video(self, file_path: str) -> VideoMetadata:
        clip = VideoFileClip(file_path)
        try:
            duration = float(clip.duration or 0.0)
            fps = float(clip.fps or 30.0)
            width = int(clip.size[0])
            height = int(clip.size[1])
        finally:
            clip.close()

        if duration <= 0:
            raise ValueError("Could not detect duration")

        return VideoMetadata(
            duration_ms=max(1, int(round(duration * 1000))),
            width=max(2, width),
            height=max(2, height),
            fps=max(1.0, fps),
        )

    def load_video(self, file_path: str, metadata: VideoMetadata):
        self.stop_analysis_thread()
        self.close_preview_clip()

        self.media_player.stop()
        self.media_player.set_media(None)

        self.media_path = file_path
        self.metadata = metadata
        self.default_open_dir = os.path.dirname(file_path)
        self.settings.setValue("paths/open", self.default_open_dir)

        self.file_label.setText(file_path)
        self.video_area.set_placeholder_visible(False)

        self.timeline.set_range(metadata.duration_ms)
        self.timeline.set_fps(metadata.fps)

        self.start_ms = 0
        self.end_ms = metadata.duration_ms
        self.timeline.set_markers(self.start_ms, self.end_ms)
        self.update_marker_widgets()
        self.update_time_label(0, metadata.duration_ms)

        self.thumbnail_cache.clear()
        self.timeline.set_wave_samples([])

        self.populate_resolution_and_fps(reset_custom=True)
        self.export_btn.setEnabled(True)
        self.setup_preview_clip(file_path)

        media = self.vlc_instance.media_new(file_path)
        self.media_player.set_media(media)
        self._bind_video_surface()
        self.media_player.audio_set_volume(self.current_volume)
        self.media_player.play()
        self.last_vlc_ms = 0
        self.last_vlc_clock = time.perf_counter()
        QTimer.singleShot(140, self.pause_after_first_frame)

        self.start_timeline_analysis(file_path)

    def setup_preview_clip(self, file_path: str):
        self.close_preview_clip()
        try:
            self.preview_lowres_clip = VideoFileClip(file_path, audio=False, target_resolution=(180, -1))
        except Exception:
            try:
                self.preview_lowres_clip = VideoFileClip(file_path, audio=False)
            except Exception:
                self.preview_lowres_clip = None

    def close_preview_clip(self):
        if self.preview_lowres_clip is not None:
            try:
                self.preview_lowres_clip.close()
            except Exception:
                pass
            self.preview_lowres_clip = None

    def start_timeline_analysis(self, file_path: str):
        self.analysis_worker = TimelineAnalysisWorker(file_path)
        self.analysis_thread = QThread(self)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.finished.connect(self.on_timeline_analysis_ready)
        self.analysis_worker.failed.connect(self.on_timeline_analysis_failed)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.cleanup_analysis_thread)

        self.analysis_thread.start()

    def on_timeline_analysis_ready(self, samples_obj, thumbs_obj):
        self.timeline.set_wave_samples(list(samples_obj))

        self.thumbnail_cache.clear()
        for time_ms, frame in thumbs_obj:
            try:
                self.thumbnail_cache[int(time_ms)] = frame_to_pixmap(frame)
            except Exception:
                continue

    def on_timeline_analysis_failed(self, error_text: str):
        self.statusBar().showMessage(f"Timeline analysis warning: {error_text}")

    def cleanup_analysis_thread(self):
        if self.analysis_worker:
            self.analysis_worker.deleteLater()
        if self.analysis_thread:
            self.analysis_thread.deleteLater()
        self.analysis_worker = None
        self.analysis_thread = None

    def stop_analysis_thread(self):
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait(1200)

    def pause_after_first_frame(self):
        state = self.media_player.get_state()
        if state in (vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening):
            self.media_player.pause()
        self.play_button.setText("Play")

    def populate_resolution_and_fps(self, reset_custom: bool = False):
        if not self.metadata:
            return

        w = self.metadata.width
        h = self.metadata.height

        aspect_value = str(self.aspect_combo.currentData())
        ratio = self._parse_ratio_text(aspect_value)

        if ratio is None or aspect_value == "free":
            candidates = [
                (w, h),
                (even(w * 0.75), even(h * 0.75)),
                (even(w * 0.5), even(h * 0.5)),
                fit_to_height(w, h, 1080),
                fit_to_height(w, h, 720),
                fit_to_height(w, h, 480),
            ]
        else:
            aw, ah = ratio
            candidates = []
            for hh in [h, even(h * 0.75), even(h * 0.5), 1080, 720, 480]:
                hh = max(16, int(hh))
                ww = max(16, even((hh * aw) / float(ah)))
                candidates.append((ww, hh))

        self.resolution_combo.clear()
        seen = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            self.resolution_combo.addItem(f"{item[0]}x{item[1]}")

        preferred = f"{w}x{h}"
        if ratio is not None and aspect_value != "free":
            aw, ah = ratio
            preferred = f"{max(16, even((h * aw) / float(ah)))}x{h}"
        self.resolution_combo.setCurrentText(preferred)
        if self.resolution_combo.currentIndex() < 0 and self.resolution_combo.count() > 0:
            self.resolution_combo.setCurrentIndex(0)

        if reset_custom or not self.custom_width_edit.text().strip() or not self.custom_height_edit.text().strip():
            self.custom_width_edit.setText(str(w))
            self.custom_height_edit.setText(str(h))

        if ratio is not None and aspect_value != "free":
            try:
                base_h = int(self.custom_height_edit.text().strip())
            except Exception:
                base_h = h
            base_h = max(16, min(16384, base_h))
            self.custom_height_edit.setText(str(base_h))
            self.custom_width_edit.setText(str(max(16, even((base_h * aw) / float(ah)))))
        elif aspect_value == "original":
            self.custom_width_edit.setText(str(w))
            self.custom_height_edit.setText(str(h))

        src_fps = float(self.metadata.fps)
        display_fps = f"{src_fps:.3f}".rstrip("0").rstrip(".")
        self.fps_preset_combo.blockSignals(True)
        self.fps_preset_combo.clear()
        self.fps_preset_combo.addItem(f"{display_fps} fps (original)", "original")
        for fps_value in (60, 30, 25, 24):
            if abs(src_fps - fps_value) < 0.1:
                continue
            self.fps_preset_combo.addItem(f"{fps_value} fps", fps_value)
        self.fps_preset_combo.setCurrentIndex(0)
        self.fps_preset_combo.blockSignals(False)
        if reset_custom or not self.custom_fps_edit.text().strip():
            self.custom_fps_edit.setText(display_fps)
        self.update_expected_size_estimate()

    def toggle_play(self):
        if not self.media_path:
            return

        state = self.media_player.get_state()
        if state == vlc.State.Playing:
            self.media_player.pause()
            self.play_button.setText("Play")
            self.statusBar().showMessage("Paused")
            return

        if self.metadata:
            frame_ms = max(1, int(round(1000.0 / max(1.0, self.metadata.fps))))
            if state == vlc.State.Ended or self.current_position_ms() >= self.metadata.duration_ms - frame_ms:
                # VLC can get stuck at EOF state; restart media before playing again.
                self.media_player.stop()
                self.media_player.play()
                QTimer.singleShot(60, lambda: self.seek_to(0))
        self.media_player.play()
        self.last_vlc_clock = time.perf_counter()
        self.play_button.setText("Pause")
        self.statusBar().showMessage("Playing")

    def seek_relative(self, delta_ms: int):
        if not self.metadata:
            return
        self.seek_to(self.current_position_ms() + delta_ms)

    def seek_to(self, position_ms: int):
        if not self.metadata:
            return

        pos = max(0, min(int(position_ms), self.metadata.duration_ms))
        self.media_player.set_time(pos)
        self.last_vlc_ms = pos
        self.last_vlc_clock = time.perf_counter()

        self.timeline.set_position(pos, auto_scroll=True)
        self.update_time_label(pos, self.metadata.duration_ms)

    def on_timeline_trim_changed(self, start_ms: int, end_ms: int):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.update_marker_widgets()

    def on_timeline_zoom_changed(self, zoom: float, visible_seconds: float):
        self.zoom_label.setText(f"Zoom {zoom:.1f}x | View {visible_seconds:.1f}s")

    def on_timeline_hover(self, time_ms: int, x_local: int):
        if not self.metadata:
            return

        pixmap = self.closest_thumbnail(time_ms)
        if pixmap is None and self.media_path:
            now = time.perf_counter()
            if (
                abs(time_ms - self.last_preview_requested_ms) >= 180
                and (now - self.last_preview_fetch_ts) >= 0.08
            ):
                self.last_preview_requested_ms = time_ms
                self.last_preview_fetch_ts = now
                pixmap = self.fetch_preview_frame(time_ms)
                if pixmap is not None:
                    self.thumbnail_cache[time_ms] = pixmap

        self.preview_popup.update_preview(pixmap, time_ms)

        top_left_global = self.timeline.mapToGlobal(QPoint(0, 0))
        center_x = top_left_global.x() + x_local
        bottom_y = top_left_global.y()
        self.preview_popup.show_at(center_x, bottom_y)

    def hide_timeline_preview(self):
        self.preview_popup.hide_with_fade()

    def closest_thumbnail(self, time_ms: int) -> Optional[QPixmap]:
        if not self.thumbnail_cache:
            return None
        keys = list(self.thumbnail_cache.keys())
        nearest = min(keys, key=lambda x: abs(x - time_ms))
        if abs(nearest - time_ms) > 2000:
            return None
        return self.thumbnail_cache.get(nearest)

    def fetch_preview_frame(self, time_ms: int) -> Optional[QPixmap]:
        if not self.metadata:
            return None
        t = max(0.0, min(time_ms / 1000.0, max(0.001, self.metadata.duration_ms / 1000.0 - 0.001)))
        clip = self.preview_lowres_clip
        if clip is None:
            return None
        try:
            frame = clip.get_frame(t)
            return frame_to_pixmap(frame)
        except Exception:
            return None

    def update_playback_ui(self):
        if not self.metadata:
            return

        state = self.media_player.get_state()
        current = self.current_position_ms()

        auto_scroll = state == vlc.State.Playing
        self.timeline.set_position(current, auto_scroll=auto_scroll)
        self.update_time_label(current, self.metadata.duration_ms)

        if state == vlc.State.Ended:
            self.play_button.setText("Play")
            self.statusBar().showMessage("Reached end")

    def current_position_ms(self) -> int:
        if not self.metadata:
            return 0

        now = time.perf_counter()
        raw = self.media_player.get_time()
        state = self.media_player.get_state()

        if raw >= 0:
            if abs(raw - self.last_vlc_ms) > 8:
                self.last_vlc_ms = raw
                self.last_vlc_clock = now

        if state == vlc.State.Playing:
            est = self.last_vlc_ms + int((now - self.last_vlc_clock) * 1000)
            value = max(raw if raw >= 0 else 0, est)
        else:
            value = raw if raw >= 0 else self.last_vlc_ms

        value = max(0, min(int(value), self.metadata.duration_ms))
        return value

    def update_time_label(self, current_ms: int, total_ms: int):
        self.time_label.setText(f"{format_ms(current_ms)} / {format_ms(total_ms)}")

    def set_start_from_playhead(self):
        if not self.metadata:
            self.show_error("Open a video first")
            return
        frame_ms = max(1, int(round(1000.0 / max(1.0, self.metadata.fps))))
        self.start_ms = self.current_position_ms()
        if self.start_ms > self.end_ms - frame_ms:
            self.end_ms = min(self.metadata.duration_ms, self.start_ms + frame_ms)
        self.timeline.set_markers(self.start_ms, self.end_ms)
        self.update_marker_widgets()

    def set_end_from_playhead(self):
        if not self.metadata:
            self.show_error("Open a video first")
            return
        frame_ms = max(1, int(round(1000.0 / max(1.0, self.metadata.fps))))
        self.end_ms = self.current_position_ms()
        if self.end_ms < self.start_ms + frame_ms:
            self.start_ms = max(0, self.end_ms - frame_ms)
        self.timeline.set_markers(self.start_ms, self.end_ms)
        self.update_marker_widgets()

    def apply_start_input(self) -> bool:
        if not self.metadata:
            return False
        try:
            value = parse_timecode(self.start_edit.text())
        except ValueError as exc:
            self.show_error(f"Invalid start value\n{exc}")
            self.start_edit.setText(format_ms(self.start_ms))
            return False

        frame_ms = max(1, int(round(1000.0 / max(1.0, self.metadata.fps))))
        self.start_ms = max(0, min(value, self.metadata.duration_ms))
        if self.start_ms > self.end_ms - frame_ms:
            self.end_ms = min(self.metadata.duration_ms, self.start_ms + frame_ms)

        self.timeline.set_markers(self.start_ms, self.end_ms)
        self.update_marker_widgets()
        return True

    def apply_end_input(self) -> bool:
        if not self.metadata:
            return False
        try:
            value = parse_timecode(self.end_edit.text())
        except ValueError as exc:
            self.show_error(f"Invalid end value\n{exc}")
            self.end_edit.setText(format_ms(self.end_ms))
            return False

        frame_ms = max(1, int(round(1000.0 / max(1.0, self.metadata.fps))))
        self.end_ms = max(0, min(value, self.metadata.duration_ms))
        if self.end_ms < self.start_ms + frame_ms:
            self.start_ms = max(0, self.end_ms - frame_ms)

        self.timeline.set_markers(self.start_ms, self.end_ms)
        self.update_marker_widgets()
        return True

    def update_marker_widgets(self):
        self.start_edit.setText(format_ms(self.start_ms))
        self.end_edit.setText(format_ms(self.end_ms))
        self.selection_len.setText(f"Selection: {format_ms(max(0, self.end_ms - self.start_ms))}")
        self.update_expected_size_estimate()

    def choose_save_directory(self):
        selected = QFileDialog.getExistingDirectory(self, "Choose Save Directory", self.save_dir_edit.text().strip())
        if selected:
            self.save_dir_edit.setText(selected)

    def cut_and_save(self):
        if not self.metadata or not self.media_path:
            self.show_error("Open a video first")
            return
        if self.export_thread and self.export_thread.isRunning():
            self.show_error("Export already running")
            return

        if not self.apply_start_input() or not self.apply_end_input():
            return

        source_fps = float(self.metadata.fps)
        custom_mode = self.export_mode_combo.currentIndex() == 1

        if not custom_mode:
            fps_data = self.fps_preset_combo.currentData()
            target_fps_for_range = source_fps if fps_data == "original" else float(fps_data)
        else:
            try:
                target_fps_for_range = float(self.custom_fps_edit.text().strip().replace(",", "."))
            except Exception:
                self.show_error("Invalid FPS value")
                return

        min_frame_ms = max(1, int(round(1000.0 / max(1.0, target_fps_for_range))))
        if self.end_ms - self.start_ms < min_frame_ms:
            self.show_error(f"Minimum trim length is 1 frame ({min_frame_ms} ms)")
            return

        try:
            if not custom_mode:
                w_text, h_text = self.resolution_combo.currentText().split("x")
                target_w = int(w_text)
                target_h = int(h_text)
            else:
                target_w = int(self.custom_width_edit.text().strip())
                target_h = int(self.custom_height_edit.text().strip())
                if target_w < 16 or target_h < 16:
                    raise ValueError("Resolution is too small")
                if target_w > 16384 or target_h > 16384:
                    raise ValueError("Resolution is too large")

            if not custom_mode:
                fps_data = self.fps_preset_combo.currentData()
                target_fps = int(round(source_fps)) if fps_data == "original" else int(fps_data)
            else:
                target_fps = int(round(float(self.custom_fps_edit.text().strip().replace(",", "."))))
                if target_fps < 1 or target_fps > 240:
                    raise ValueError("FPS out of range")

            if not custom_mode:
                preset_data = self.quality_combo.currentData()
                if not preset_data:
                    raise ValueError("No quality preset")
                crf, preset = preset_data
            else:
                crf = int(self.custom_crf_edit.text().strip())
                if crf < 0 or crf > 51:
                    raise ValueError("CRF out of range")
                preset = self.custom_preset_combo.currentText().strip()

            format_ext = str(self.format_combo.currentData())
            mute_audio = bool(self.export_audio_btn.isChecked() or format_ext == "gif")
            if format_ext != "gif":
                target_w = even(target_w)
                target_h = even(target_h)
        except Exception as exc:
            self.show_error(f"Invalid export settings\n{exc}")
            return

        save_dir = self.save_dir_edit.text().strip() or self.default_save_dir
        if not os.path.isdir(save_dir):
            self.show_error("Save directory does not exist")
            return

        base_name = os.path.basename(self.media_path)
        stem, _ = os.path.splitext(base_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggestion = os.path.join(save_dir, f"{stem}_trim_{stamp}.{format_ext}")

        save_filter = self._output_filter_for_format(format_ext)
        out_file, _ = QFileDialog.getSaveFileName(self, "Export Video", suggestion, save_filter)
        if not out_file:
            return
        if not out_file.lower().endswith(f".{format_ext}"):
            out_file += f".{format_ext}"

        self.default_save_dir = os.path.dirname(out_file)
        self.settings.setValue("paths/save", self.default_save_dir)
        self.save_dir_edit.setText(self.default_save_dir)

        job = ExportJob(
            input_path=self.media_path,
            output_path=out_file,
            format_ext=format_ext,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            width=target_w,
            height=target_h,
            fps=target_fps,
            crf=int(crf),
            preset=str(preset),
            mute_audio=mute_audio,
        )
        self.start_export(job)

    @staticmethod
    def _output_filter_for_format(ext: str) -> str:
        ext = ext.lower()
        mapping = {
            "mp4": "MP4 Files (*.mp4)",
            "mov": "MOV Files (*.mov)",
            "mkv": "MKV Files (*.mkv)",
            "webm": "WEBM Files (*.webm)",
            "avi": "AVI Files (*.avi)",
            "gif": "GIF Files (*.gif)",
        }
        return mapping.get(ext, "Video Files (*.*)")

    def start_export(self, job: ExportJob):
        self.export_worker = ExportWorker(job)
        self.export_thread = QThread(self)
        self.export_worker.moveToThread(self.export_thread)

        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.progress.connect(self.on_export_progress)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.failed.connect(self.on_export_failed)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.failed.connect(self.export_thread.quit)
        self.export_thread.finished.connect(self.cleanup_export_thread)

        self.set_export_controls(running=True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Encoding")
        self.progress.show()
        self.statusBar().showMessage("Export started")
        self.export_thread.start()

    def cancel_export(self):
        if self.export_worker:
            self.export_worker.cancel()

    def on_export_progress(self, percent: int, label: str):
        self.progress.setValue(max(0, min(100, percent)))
        self.progress.setFormat(label)

    def on_export_finished(self, output_path: str):
        self.progress.hide()
        self.set_export_controls(running=False)
        self.statusBar().showMessage("Export finished")
        self.show_info(f"Export complete\n{output_path}")

    def on_export_failed(self, reason: str):
        self.progress.hide()
        self.set_export_controls(running=False)
        if reason == "Export canceled":
            self.statusBar().showMessage("Export canceled")
            self.show_info("Export canceled")
        else:
            self.statusBar().showMessage("Export failed")
            self.show_error(f"Export failed\n{reason}")

    def cleanup_export_thread(self):
        if self.export_worker:
            self.export_worker.deleteLater()
        if self.export_thread:
            self.export_thread.deleteLater()
        self.export_worker = None
        self.export_thread = None

    def set_export_controls(self, running: bool):
        self.open_button.setEnabled(not running)
        self.export_btn.setEnabled(not running and self.metadata is not None)
        self.cancel_btn.setVisible(running)
        self.cancel_btn.setEnabled(running)
        self.play_button.setEnabled(not running)

    def change_volume(self, value: int):
        self.current_volume = value
        self.settings.setValue("player/volume", value)
        self.media_player.audio_set_volume(value)

    def show_error(self, text: str):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Error")
        box.setText(text)
        box.exec()

    def show_info(self, text: str):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Info")
        box.setText(text)
        box.exec()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            local = urls[0].toLocalFile()
            if local and is_supported_video(local):
                event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            local = urls[0].toLocalFile()
            if local and is_supported_video(local):
                self.open_video(local)
                event.acceptProposedAction()

    def closeEvent(self, event):
        self.stop_analysis_thread()
        self.close_preview_clip()

        if self.export_worker:
            self.export_worker.cancel()
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.quit()
            self.export_thread.wait(2000)

        self.ui_timer.stop()
        self.media_player.stop()
        self.media_player.set_media(None)

        self.settings.setValue("paths/open", self.default_open_dir)
        self.settings.setValue("paths/save", self.default_save_dir)
        self.preview_popup.hide()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    start_file = sys.argv[1] if len(sys.argv) > 1 else None
    window = VideoPlayer(start_file)
    window.show()
    sys.exit(app.exec())
