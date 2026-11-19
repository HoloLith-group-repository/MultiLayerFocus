"""Light and dark themes (Qt style sheets + plotting colors)."""

PALETTES = {
    'light': dict(bg='#f3f5f9', surface='#ffffff', surface2='#f7f9fc', border='#e1e6ee', text='#18202e',
                  muted='#5f6c80', accent='#3366ee', accent_hover='#2453d4', accent2='#e8590c', ok='#1f9d55',
                  warn='#d97706', bad='#d6336c', plot_bg='#ffffff', plot_fg='#3a4556', grid='#d9dee7',
                  interface='#ffffff', selection='#dbe5ff'),
    'dark': dict(bg='#0d121b', surface='#151c28', surface2='#1b2331', border='#273246', text='#e7ecf4',
                 muted='#97a3b8', accent='#6b9bff', accent_hover='#88afff', accent2='#ff8a4c', ok='#3ccf7e',
                 warn='#f2b04a', bad='#ff6b98', plot_bg='#10161f', plot_fg='#c2cad8', grid='#2a3446',
                 interface='#ffffff', selection='#24345a'),
}


def stylesheet(name):
    p = PALETTES[name]
    return f"""
* {{ font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif; font-size: 9pt; color: {p['text']}; }}
QMainWindow, QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {p['bg']}; }}
QToolTip {{ background: {p['surface']}; color: {p['text']}; border: 1px solid {p['border']}; padding: 6px; }}
QFrame#card {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 12px; }}
QLabel#cardTitle {{ font-size: 10pt; font-weight: 600; }}
QLabel#cardSubtitle, QLabel#muted {{ color: {p['muted']}; font-size: 9pt; }}
QLabel#brand {{ font-size: 14pt; font-weight: 700; letter-spacing: 0.3px; }}
QLabel#brandSub {{ color: {p['muted']}; font-size: 9pt; }}
QLabel#tileValue {{ font-size: 13pt; font-weight: 600; }}
QLabel#tileCaption {{ color: {p['muted']}; font-size: 8pt; }}
QFrame#tile {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 10px; }}
QToolButton#cardHeader {{ border: none; background: transparent; font-weight: 600; text-align: left; padding: 2px; }}
QPushButton {{ background: {p['surface2']}; border: 1px solid {p['border']}; border-radius: 8px; padding: 5px 10px; }}
QPushButton:hover {{ border-color: {p['accent']}; }}
QPushButton:disabled {{ color: {p['muted']}; }}
QPushButton#primary {{ background: {p['accent']}; color: white; border: none; font-weight: 600; padding: 7px 13px; }}
QPushButton#primary:hover {{ background: {p['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {p['border']}; color: {p['muted']}; }}
QPushButton#ghost {{ background: transparent; border: none; color: {p['accent']}; padding: 4px 6px; }}
QPushButton:checked {{ background: {p['selection']}; border-color: {p['accent']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {p['surface2']}; border: 1px solid {p['border']};
    border-radius: 7px; padding: 3px 5px; min-height: 20px; selection-background-color: {p['accent']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {p['accent']}; }}
QComboBox QAbstractItemView {{ background: {p['surface']}; border: 1px solid {p['border']}; selection-background-color: {p['selection']}; selection-color: {p['text']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QSlider::groove:horizontal {{ height: 4px; background: {p['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {p['surface']}; border: 2px solid {p['accent']}; width: 12px; height: 12px; margin: -6px 0; border-radius: 7px; }}
QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 12px; background: {p['surface']}; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 7px 9px; margin-right: 2px; color: {p['muted']}; border: none; font-weight: 600; }}
QTabBar::tab:selected {{ color: {p['accent']}; border-bottom: 2px solid {p['accent']}; }}
QTabBar::tab:hover {{ color: {p['text']}; }}
QTableWidget {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 8px; gridline-color: {p['border']}; }}
QHeaderView::section {{ background: {p['surface2']}; color: {p['muted']}; border: none; border-bottom: 1px solid {p['border']}; padding: 4px; font-weight: 600; }}
QTableCornerButton::section {{ background: {p['surface2']}; border: none; border-bottom: 1px solid {p['border']}; }}
QTableWidget::item:selected {{ background: {p['selection']}; color: {p['text']}; }}
QProgressBar {{
    background: {p['surface2']};
    color: {'#ffffff' if name == 'light' else '#10161f'};
    border: 1px solid {p['border']};
    border-radius: 6px;
    min-height: 22px;
    text-align: center;
    font-weight: 700;
}}

QProgressBar::chunk {{
    background: {p['accent']};
    border-radius: 5px;
}}
QProgressBar::chunk {{ background: {p['accent']}; border-radius: 3px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background: {p['border']};
    border-radius: 4px;
    min-width: 30px;
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

QStatusBar {{ background: {p['surface']}; border-top: 1px solid {p['border']}; color: {p['muted']}; }}
QToolButton#exportButton {{
    background: {p['surface2']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 8px;
    padding: 6px 10px;
}}
QToolButton#exportButton:hover {{ border-color: {p['accent']}; }}
QMessageBox {{ background-color: {p['surface']}; }}
QMessageBox QLabel {{ color: {p['text']}; background: transparent; }}
QMessageBox QPushButton {{ color: {p['text']}; background-color: {p['surface2']}; }}
QMenu {{ background: {p['surface']}; border: 1px solid {p['border']}; padding: 4px; }}
QMenu::separator {{ height: 1px; background: {p['border']}; margin: 5px 8px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {p['selection']}; }}
QSplitter::handle {{ background: {p['bg']}; }}
QGroupBox {{ border: 1px solid {p['border']}; border-radius: 10px; margin-top: 14px; padding: 10px 8px 8px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {p['muted']}; }}
QToolButton#volumeCameraToggle {{
    background: {p['surface2']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
}}

QToolButton#volumeCameraToggle:hover {{
    border-color: {p['accent']};
}}

QToolButton#volumeCameraToggle:checked {{
    background: {p['selection']};
    border-color: {p['accent']};
}}

QToolButton#cardArrow {{
    background: transparent;
    border: none;
    font-size: 15pt;
    font-weight: 700;
    padding: 0;
}}

"""
