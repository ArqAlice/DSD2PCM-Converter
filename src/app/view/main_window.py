from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6 import QtCore, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    """メインウィンドウ（MVC の View）。"""

    def __init__(self, max_workers: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DSD (DSF) -> PCM (FLAC) Converter")

        self._max_workers = max_workers

        self._create_widgets()
        self._layout_widgets()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    def _create_widgets(self) -> None:
        self.central = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central)

        # ファイル操作ボタン
        self.add_files_button = QtWidgets.QPushButton("追加...")
        self.remove_files_button = QtWidgets.QPushButton("選択行を削除")
        self.clear_files_button = QtWidgets.QPushButton("一覧をクリア")

        # ファイル一覧テーブル
        self.table = QtWidgets.QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(
            ["入力ファイル", "出力ファイル", "DSD Fs [Hz]", "Ch", "ステータス"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # 出力ディレクトリ
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_browse_button = QtWidgets.QPushButton("参照...")

        # PCM サンプリング周波数
        self.pcm_sr_label = QtWidgets.QLabel("PCMサンプリング周波数 [Hz]:")
        self.pcm_sr_combo = QtWidgets.QComboBox()
        for fs in (44100, 88200, 176400):
            self.pcm_sr_combo.addItem(str(fs), fs)
        self.pcm_sr_combo.setCurrentIndex(1)  # デフォルト: 88.2kHz

        # FIR フィルタパラメータ
        self.stopband_label = QtWidgets.QLabel("阻止帯域開始周波数 [Hz]:")
        self.stopband_spin = QtWidgets.QDoubleSpinBox()
        self.stopband_spin.setDecimals(1)
        self.stopband_spin.setRange(1000.0, 2000000.0)
        self.stopband_spin.setSingleStep(1000.0)
        self.stopband_spin.setValue(22000.0)

        self.atten_label = QtWidgets.QLabel("阻止帯域減衰量 [dB]:")
        self.atten_spin = QtWidgets.QDoubleSpinBox()
        self.atten_spin.setDecimals(1)
        self.atten_spin.setRange(40.0, 200.0)
        self.atten_spin.setSingleStep(5.0)
        self.atten_spin.setValue(140.0)

        # 同時変換数
        self.workers_label = QtWidgets.QLabel("同時変換数:")
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, max(1, self._max_workers))
        self.workers_spin.setValue(min(1, self._max_workers))

        # Start / Stop
        self.start_button = QtWidgets.QPushButton("変換開始")
        self.stop_button = QtWidgets.QPushButton("中止")
        self.stop_button.setEnabled(False)

        # ログ表示
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)

    def _layout_widgets(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self.central)

        # ファイルボタン行
        file_btn_layout = QtWidgets.QHBoxLayout()
        file_btn_layout.addWidget(self.add_files_button)
        file_btn_layout.addWidget(self.remove_files_button)
        file_btn_layout.addWidget(self.clear_files_button)
        file_btn_layout.addStretch()

        main_layout.addLayout(file_btn_layout)
        main_layout.addWidget(self.table)

        # 出力フォルダ行
        out_layout = QtWidgets.QHBoxLayout()
        out_layout.addWidget(QtWidgets.QLabel("出力フォルダ:"))
        out_layout.addWidget(self.output_dir_edit)
        out_layout.addWidget(self.output_dir_browse_button)
        main_layout.addLayout(out_layout)

        # 各種設定
        settings_layout = QtWidgets.QGridLayout()
        settings_layout.addWidget(self.pcm_sr_label, 0, 0)
        settings_layout.addWidget(self.pcm_sr_combo, 0, 1)
        settings_layout.addWidget(self.stopband_label, 1, 0)
        settings_layout.addWidget(self.stopband_spin, 1, 1)
        settings_layout.addWidget(self.atten_label, 2, 0)
        settings_layout.addWidget(self.atten_spin, 2, 1)
        settings_layout.addWidget(self.workers_label, 0, 2)
        settings_layout.addWidget(self.workers_spin, 0, 3)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)

        main_layout.addLayout(settings_layout)

        # Start / Stop 行
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.addWidget(self.start_button)
        ctrl_layout.addWidget(self.stop_button)
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)

        main_layout.addWidget(QtWidgets.QLabel("ログ:"))
        main_layout.addWidget(self.log_edit)

    def _connect_signals(self) -> None:
        self.add_files_button.clicked.connect(self._on_add_files_clicked)
        self.remove_files_button.clicked.connect(self._on_remove_files_clicked)
        self.clear_files_button.clicked.connect(self._on_clear_files_clicked)
        self.output_dir_browse_button.clicked.connect(self._on_browse_output_dir)

    # ------------------------------------------------------------------ #
    # Controller から利用する API
    # ------------------------------------------------------------------ #
    def append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def get_max_workers_setting(self) -> int:
        return int(self.workers_spin.value())

    def get_pcm_samplerate(self) -> int:
        return int(self.pcm_sr_combo.currentData())

    def get_stopband_hz(self) -> float:
        return float(self.stopband_spin.value())

    def get_stopband_atten_db(self) -> float:
        return float(self.atten_spin.value())

    def get_output_dir(self) -> Path | None:
        text = self.output_dir_edit.text().strip()
        if not text:
            return None
        return Path(text)

    def get_file_paths(self) -> list[Path]:
        paths: list[Path] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            path_str = item.text()
            if path_str:
                paths.append(Path(path_str))
        return paths

    def get_selected_rows(self) -> List[int]:
        rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
        return rows

    def set_row_status(self, row: int, status: str) -> None:
        if 0 <= row < self.table.rowCount():
            item = self.table.item(row, 4)
            if item is None:
                item = QtWidgets.QTableWidgetItem(status)
                self.table.setItem(row, 4, item)
            else:
                item.setText(status)

    def set_row_dsd_info(self, row: int, fs: int, channels: int) -> None:
        if 0 <= row < self.table.rowCount():
            fs_item = QtWidgets.QTableWidgetItem(str(fs))
            ch_item = QtWidgets.QTableWidgetItem(str(channels))
            self.table.setItem(row, 2, fs_item)
            self.table.setItem(row, 3, ch_item)

    def set_row_output_path(self, row: int, path: Path | None) -> None:
        if 0 <= row < self.table.rowCount():
            text = str(path) if path is not None else ""
            item = QtWidgets.QTableWidgetItem(text)
            self.table.setItem(row, 1, item)

    def set_conversion_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    # ------------------------------------------------------------------ #
    # 内部 UI ハンドラ（ファイル操作系）
    # ------------------------------------------------------------------ #
    def _on_add_files_clicked(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "DSFファイルを選択",
            "",
            "DSFファイル (*.dsf);;すべてのファイル (*.*)",
        )
        if not files:
            return

        for path_str in files:
            path = Path(path_str)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(path)))
            self.set_row_status(row, "待機中")

    def _on_remove_files_clicked(self) -> None:
        rows = self.get_selected_rows()
        for row in reversed(rows):
            self.table.removeRow(row)

    def _on_clear_files_clicked(self) -> None:
        self.table.setRowCount(0)

    def _on_browse_output_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "出力フォルダを選択",
            "",
        )
        if directory:
            self.output_dir_edit.setText(directory)
