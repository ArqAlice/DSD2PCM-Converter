# app/native_fir/fir_decimator.pyx
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: language_level=3
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t  float32_t
ctypedef cnp.float64_t  float64_t


def fir_decimate_chunk_core(
    float32_t[:, :] dsd_ext,   # typed memoryview
    float32_t[:] taps,         # typed memoryview
    int decim,
    int phase_init,
    int overlap,
    int main_len,
):
    """
    dsd_ext: shape (N_ext, ch), float32
        先頭 overlap サンプルはオーバーラップ領域、
        続く main_len サンプルが今回のチャンク本体。

    taps   : shape (L,), float32
    decim  : decimation factor (例: 64)
    phase_init:
        dsd_ext[0] のグローバルインデックス mod decim
        （Python 側で global_start_ext % decim として渡している想定）
    overlap:
        オーバーラップサンプル数（通常 taps.size - 1）
    main_len:
        本体サンプル数

    戻り値:
        shape (N_out, ch) の float64 ndarray
    """

    cdef:
        Py_ssize_t N_ext = dsd_ext.shape[0]
        Py_ssize_t ch = dsd_ext.shape[1]
        Py_ssize_t L = taps.shape[0]

        Py_ssize_t main_start = overlap
        Py_ssize_t main_end = overlap + main_len  # 非包含上限

        Py_ssize_t first_i      # 最初の出力サンプル位置 (dsd_ext の index)
        Py_ssize_t n_out        # 出力サンプル数
        Py_ssize_t last_i
        Py_ssize_t span

        Py_ssize_t oi, i, k, c
        int r

        float64_t acc

    if main_len <= 0:
        # チャンク本体がない場合は空を返す
        return np.empty((0, ch), dtype=np.float64)

    if main_end > N_ext:
        raise ValueError("main_len + overlap exceeds dsd_ext length")

    # -------------------------------------------------
    # decimation の性質を利用して、最初の出力位置 first_i と
    # 出力数 n_out を O(1) で決定する
    #
    # 各サンプル i の「グローバルインデックス mod decim」は
    #   (phase_init + i) % decim
    # main_start 以降で、この値が 0 となる最初の i を求める。
    # -------------------------------------------------
    r = (phase_init + main_start) % decim

    if r == 0:
        first_i = main_start
    else:
        first_i = main_start + (decim - r)

    if first_i >= main_end:
        # このチャンク範囲内には出力位置が存在しない
        return np.empty((0, ch), dtype=np.float64)

    # 最後の出力位置は main_end - 1 以下で decim 間隔
    # first_i, first_i + decim, ..., last_i <= main_end - 1
    last_i = main_end - 1
    span = last_i - first_i        # >= 0
    n_out = span // decim + 1      # 整数除算

    # 出力配列（NumPy ndarray）と、その memoryview を用意
    cdef cnp.ndarray[float64_t, ndim=2] out_arr = np.empty((n_out, ch), dtype=np.float64)
    cdef float64_t[:, :] out = out_arr

    # -------------------------------------------------
    # ここはシングルスレッドだが GIL を解放してループを回す。
    # 後で Python 側からマルチスレッドでこの関数を呼び出せるようにする前提。
    # -------------------------------------------------
    with nogil:
        for oi in range(n_out):
            i = first_i + oi * decim

            # 各チャンネル独立
            for c in range(ch):
                acc = 0.0
                # i を中心に taps を後ろ向きにかける想定
                for k in range(L):
                    acc += <float64_t> dsd_ext[i - k, c] * <float64_t> taps[k]
                out[oi, c] = acc

    # Python 側からは ndarray として扱いたいので out_arr を返す
    return out_arr
