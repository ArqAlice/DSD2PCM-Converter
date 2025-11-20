# DSD2PCM-Converter
DSD2PCM-Converter

Cythonビルド：
python setup_native_fir.py build_ext --inplace

Nuitkaビルド：
python -m nuitka --standalone --windows-disable-console --enable-plugin=pyside6 --output-dir=dist --python-flag=-m src/app