from __future__ import annotations

import os
import sys

from PySide6 import QtWidgets

from .controller.main_controller import ConversionController
from .view.main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)

    max_workers = os.cpu_count() or 1
    window = MainWindow(max_workers=max_workers)
    _controller = ConversionController(window)

    window.resize(1000, 600)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
