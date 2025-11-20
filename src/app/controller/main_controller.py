from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from PySide6 import QtCore

from ..model.converter import ConversionSettings, ConversionResult, convert_dsf_to_flac
from ..model.dsf_reader import DsfReader
from ..view.main_window import MainWindow

import concurrent.futures


class ConversionController(QtCore.QObject):
    """GUI と変換ロジックをつなぐ Controller。"""
    progress_changed = QtCore.Signal(int, float)  # row, progress(0.0〜1.0)

    def __init__(self, window: MainWindow, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.window = window

        self._executor: concurrent.futures.Executor | None = None
        self._futures: Dict[concurrent.futures.Future, int] = {}
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll_futures)

        # Start / Stop ボタン
        self.window.start_button.clicked.connect(self.start_conversion)
        self.window.stop_button.clicked.connect(self.stop_conversion)

        # 行追加時に DSF の Fs / Ch を埋める
        self.window.table.model().rowsInserted.connect(self._on_rows_inserted)

        # 進捗シグナルを View に接続
        self.progress_changed.connect(self._on_progress_changed)

    # ------------------------------------------------------------------ #
    def _on_rows_inserted(self, parent_index, start: int, end: int) -> None:  # noqa: ANN001
        # 追加された行について DSF ヘッダを読んで Fs / Ch を表示
        for row in range(start, end + 1):
            item = self.window.table.item(row, 0)
            if item is None:
                continue
            path = Path(item.text())
            if not path.exists():
                continue
            try:
                with DsfReader(path) as reader:
                    self.window.set_row_dsd_info(row, reader.sample_rate, reader.channels)
            except Exception as exc:  # noqa: BLE001
                self.window.append_log(f"[DSF解析エラー] {path}: {exc}")
                self.window.set_row_status(row, "DSF解析エラー")

    # ------------------------------------------------------------------ #
    def start_conversion(self) -> None:
        if self._executor is not None:
            # 既に変換中
            return

        file_paths = self.window.get_file_paths()
        if not file_paths:
            self.window.append_log("入力ファイルが指定されていません。")
            return

        output_dir = self.window.get_output_dir()
        if output_dir is None:
            self.window.append_log("出力フォルダを指定してください。")
            return

        max_workers_setting = self.window.get_max_workers_setting()
        max_workers = min(max_workers_setting, os.cpu_count() or 1)
        if max_workers <= 0:
            max_workers = 1

        fs_pcm = self.window.get_pcm_samplerate()
        stopband_hz = self.window.get_stopband_hz()
        atten_db = self.window.get_stopband_atten_db()

        settings = ConversionSettings(
            output_dir=output_dir,
            pcm_samplerate=fs_pcm,
            stopband_hz=stopband_hz,
            stopband_atten_db=atten_db,
            max_workers = max((os.cpu_count() // max_workers) , 1),
        )

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        )
        self._futures.clear()

        self.window.append_log(
            f"変換開始: {len(file_paths)} ファイル, "
            f"PCM Fs={fs_pcm} Hz, stopband={stopband_hz} Hz, attenuation={atten_db} dB, "
            f"workers={max_workers}"
        )

        for row, src_path in enumerate(file_paths):
            self.window.set_row_status(row, "変換待ち")
            self.window.set_row_progress(row, 0.0)

            # row 固有の progress コールバックを作る
            def make_progress_cb(row_index: int):
                def _cb(frac: float) -> None:
                    self.progress_changed.emit(row_index, frac)
                return _cb

            progress_cb = make_progress_cb(row)

            # 変換をスケジュール（コールバック付き）
            future = self._executor.submit(
                convert_dsf_to_flac,
                str(src_path),
                settings,
                progress_cb,
            )
            self._futures[future] = row
            self.window.set_row_status(row, "変換中 0%")

        self.window.set_conversion_running(True)
        self._poll_timer.start()

    # ------------------------------------------------------------------ #
    def stop_conversion(self) -> None:
        if self._executor is None:
            return

        self.window.append_log("変換中止要求を送信しました。")
        try:
            # cancel_futures=True は Python 3.9+
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)

        self._executor = None
        self._futures.clear()
        self._poll_timer.stop()
        self.window.set_conversion_running(False)

    # ------------------------------------------------------------------ #
    def _poll_futures(self) -> None:
        if self._executor is None:
            self._poll_timer.stop()
            return

        done_futures = [f for f in self._futures.keys() if f.done()]

        for fut in done_futures:
            row = self._futures.pop(fut)
            try:
                result: ConversionResult = fut.result()
            except Exception as exc:  # noqa: BLE001
                self.window.set_row_status(row, "エラー")
                self.window.append_log(f"[変換エラー] 行 {row}: {exc}")
                continue

            if result.success:
                self.window.set_row_status(row, "完了")
                if result.dst_path is not None:
                    self.window.set_row_output_path(row, result.dst_path)
                self.window.append_log(f"[OK] {result.src_path} -> {result.dst_path}")
                self.window.set_row_progress(row, 1.0)
                if result.message and result.message != "OK":
                    self.window.append_log(f"  メモ: {result.message}")
            else:
                self.window.set_row_status(row, "エラー")
                self.window.append_log(f"[NG] {result.src_path}: {result.message}")
                self.window.set_row_progress(row, 0.0)

        if not self._futures:
            # すべて完了
            self._poll_timer.stop()
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=False)
                except Exception:
                    pass
            self._executor = None
            self.window.set_conversion_running(False)
            self.window.append_log("すべての変換が完了しました。")
    
    @QtCore.Slot(int, float)
    def _on_progress_changed(self, row: int, frac: float) -> None:
        # 0.0〜1.0 → 0〜100%
        percent = int(frac * 100.0)
        if percent < 0:
            percent = 0
        elif percent > 100:
            percent = 100

        # ステータスを「変換中 XX%」に上書き
        self.window.set_row_status(row, f"変換中 {percent}%")

        # プログレスバーを更新
        self.window.set_row_progress(row, frac)
