"""
GUI-интерфейс для Jarvis: окно с историей диалога, полем ввода текста,
кнопкой микрофона и визуализацией статуса (слушает/думает/говорит).

Запуск:
    python gui.py

Требования:
    pip install PyQt5 sounddevice numpy faster-whisper torch torchaudio

Интерфейс:
    - Список сообщений (история диалога)
    - Поле ввода текста + кнопка "Отправить" (или Enter)
    - Кнопка микрофона: включение/выключение записи по клику
    - Статусная строка: "Слушает", "Думает", "Говорит", "Ожидание"
"""

from __future__ import annotations

import sys
import threading
import queue
from pathlib import Path
from typing import Optional, Callable

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from brain.ollama_client import OllamaClient
from stt.transcriber import record_until_silence, transcribe
from tts.speaker import speak

# Глобальные события для управления состоянием
stop_event = threading.Event()
interrupt_event = threading.Event()
speaking_event = threading.Event()
listening_event = threading.Event()

# Очередь для команд из GUI в основной поток
command_queue: queue.Queue[str] = queue.Queue()


class MessageBubble(QFrame):
    """Визуальный блок сообщения (пузырь)"""
    
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        
        # Стиль пузырей
        if is_user:
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #dcf8c6;
                    border-radius: 12px;
                    padding: 10px;
                    margin: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #ffffff;
                    border-radius: 12px;
                    padding: 10px;
                    margin: 4px;
                    border: 1px solid #e0e0e0;
                }
            """)
        
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = QFont("Segoe UI", 10)
        label.setFont(font)
        layout.addWidget(label)


class ChatWidget(QWidget):
    """Виджет истории чата с автопрокруткой"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setAlignment(Qt.AlignTop)
        self.container_layout.setSpacing(4)
        
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)
    
    def add_message(self, text: str, is_user: bool):
        bubble = MessageBubble(text, is_user, self.container)
        self.container_layout.addWidget(bubble)
        # Автопрокрутка вниз
        QTimer.singleShot(100, self._scroll_to_bottom)
    
    def _scroll_to_bottom(self):
        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


from PyQt5.QtCore import QTimer


class JarvisGUI(QMainWindow):
    """Главное окно приложения"""
    
    status_signal = pyqtSignal(str)
    message_signal = pyqtSignal(str, bool)  # текст, is_user
    mic_toggle_signal = pyqtSignal(bool)    # is_listening
    
    def __init__(self):
        super().__init__()
        
        self.client = OllamaClient()
        self.history = []
        self.is_mic_active = False
        self.mic_thread: Optional[threading.Thread] = None
        
        self.setWindowTitle("Jarvis Assistant")
        self.setMinimumSize(500, 700)
        
        # Центральная панель
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # История чата
        self.chat = ChatWidget()
        main_layout.addWidget(self.chat, stretch=1)
        
        # Поле ввода
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите сообщение или нажмите кнопку микрофона...")
        self.input_field.setFont(QFont("Segoe UI", 11))
        self.input_field.returnPressed.connect(self._send_text)
        input_layout.addWidget(self.input_field)
        
        self.send_btn = QPushButton("➤")
        self.send_btn.setFixedSize(50, 50)
        self.send_btn.setFont(QFont("Segoe UI", 14))
        self.send_btn.clicked.connect(self._send_text)
        input_layout.addWidget(self.send_btn)
        
        main_layout.addLayout(input_layout)
        
        # Кнопка микрофона и статус
        bottom_layout = QHBoxLayout()
        
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setFixedSize(60, 60)
        self.mic_btn.setFont(QFont("Segoe UI", 20))
        self.mic_btn.setToolTip("Включить/выключить микрофон")
        self.mic_btn.clicked.connect(self._toggle_microphone)
        bottom_layout.addWidget(self.mic_btn)
        
        self.status_label = QLabel("Ожидание")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 8px;")
        bottom_layout.addWidget(self.status_label, stretch=1)
        
        main_layout.addLayout(bottom_layout)
        
        # Подключение сигналов
        self.status_signal.connect(self._update_status)
        self.message_signal.connect(self._add_message_to_chat)
        self.mic_toggle_signal.connect(self._update_mic_button)
        
        # Запуск обработчика команд
        self.command_timer = QTimer()
        self.command_timer.timeout.connect(self._process_commands)
        self.command_timer.start(100)
        
        self.show()
    
    def _update_status(self, status: str):
        self.status_label.setText(status)
        # Цвет статуса
        colors = {
            "Ожидание": "#f0f0f0",
            "Слушает": "#d4edda",
            "Думает": "#fff3cd",
            "Говорит": "#cce5ff"
        }
        color = colors.get(status, "#f0f0f0")
        self.status_label.setStyleSheet(f"padding: 10px; background-color: {color}; border-radius: 8px;")
    
    def _add_message_to_chat(self, text: str, is_user: bool):
        self.chat.add_message(text, is_user)
    
    def _update_mic_button(self, is_listening: bool):
        if is_listening:
            self.mic_btn.setStyleSheet("background-color: #dc3545; border-radius: 30px;")
            self.mic_btn.setToolTip("Нажмите, чтобы остановить запись")
        else:
            self.mic_btn.setStyleSheet("")
            self.mic_btn.setToolTip("Включить микрофон")
    
    def _send_text(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self._handle_command(text)
    
    def _toggle_microphone(self):
        if self.is_mic_active:
            self.is_mic_active = False
            listening_event.set()  # Сигнал остановки записи
            self.mic_toggle_signal.emit(False)
            self._update_status("Ожидание")
        else:
            self.is_mic_active = True
            self.mic_toggle_signal.emit(True)
            self._update_status("Слушает")
            self.mic_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.mic_thread.start()
    
    def _record_audio(self):
        """Запись аудио до тишины"""
        try:
            audio = record_until_silence()
            if self.is_mic_active:
                self.is_mic_active = False
                self.mic_toggle_signal.emit(False)
            
            if audio is not None and len(audio) > 0:
                self._update_status("Распознаёт...")
                text = transcribe(audio, language="ru")
                if text:
                    self._handle_command(text)
                else:
                    self._add_message_to_chat("Не расслышал. Попробуйте ещё раз.", False)
            self._update_status("Ожидание")
        except Exception as e:
            self._add_message_to_chat(f"Ошибка микрофона: {e}", False)
            self.is_mic_active = False
            self.mic_toggle_signal.emit(False)
            self._update_status("Ожидание")
    
    def _handle_command(self, text: str):
        self._add_message_to_chat(text, True)
        self._update_status("Думает")
        
        # Обработка в отдельном потоке
        def process():
            try:
                response = self.client.chat(text, self.history)
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": response})
                # Trim history
                self.history = self.history[-12:]
                
                self.message_signal.emit(response, False)
                self._update_status("Говорит")
                speak(response, interrupt_event)
                self._update_status("Ожидание")
            except Exception as e:
                self._add_message_to_chat(f"Ошибка: {e}", False)
                self._update_status("Ожидание")
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _process_commands(self):
        """Обработка очереди команд (если нужно)"""
        pass
    
    def closeEvent(self, event):
        stop_event.set()
        interrupt_event.set()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Настройка стиля
    app.setStyle("Fusion")
    
    window = JarvisGUI()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
