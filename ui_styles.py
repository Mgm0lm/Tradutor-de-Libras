from config import UI_RADIUS

STYLESHEET = f"""
QMainWindow, QWidget#root {{
    background-color: #ffffff;
}}

QLabel {{
    color: #1f2430;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}

QLabel#videoLabel {{
    background-color: #101318;
    border-radius: {UI_RADIUS}px;
}}

QFrame#card {{
    background-color: #f4f6f9;
    border: 1px solid #e3e7ee;
    border-radius: {UI_RADIUS}px;
}}

QLabel#titulo {{
    font-size: 20px;
    font-weight: 600;
}}

QLabel#subtitulo {{
    font-size: 13px;
    color: #6b7280;
}}

QLabel#letraLabel {{
    font-size: 220px;
    font-weight: 800;
    color: #9ca3af;               /* cinza = nada / abaixo do threshold */
}}

QLabel#letraLabel[estado="detectando"] {{
    color: #2563eb;                /* azul = detectando, ainda estabilizando */
}}

QLabel#letraLabel[estado="confirmado"] {{
    color: #0f8a4c;                /* verde = confirmado / falado */
}}

QLabel#statusLabel {{
    font-size: 15px;
    font-weight: 600;
    color: #2563eb;
}}

QPushButton {{
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: {UI_RADIUS - 4}px;
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #1d4ed8;
}}

QPushButton:pressed {{
    background-color: #1e40af;
}}

QPushButton#thresholdBtn {{
    min-width: 36px;
    max-width: 36px;
    font-size: 18px;
    padding: 4px;
}}

QPushButton#secondaryBtn {{
    background-color: #ffffff;
    color: #374151;
    border: 1px solid #d0d5dd;
    padding: 6px 12px;
    font-size: 12px;
}}

QPushButton#secondaryBtn:hover {{
    background-color: #f4f6f9;
}}

QComboBox, QCheckBox {{
    font-size: 13px;
    padding: 4px;
}}

QComboBox {{
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 6px 8px;
    background-color: white;
}}

QProgressBar {{
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    background-color: #eef1f5;
    height: 16px;
    text-align: center;
    font-size: 10px;
}}

QProgressBar::chunk {{
    background-color: #2563eb;
    border-radius: 7px;
}}
"""
