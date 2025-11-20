from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit, prange

@dataclass
class FIRDecimatorState:
    """ストリーミング FIR decimation 用の per-channel 状態。"""
    prev_input: np.ndarray  # shape: (channels, taps-1)
    phase: np.ndarray       # shape: (channels,), modulo decim factor

# dsp.py などのトップレベルに置く（Pickle 可能にするため）
def fir_decimate_chunk_worker(
    dsd_ext: np.ndarray,
    taps: np.ndarray,
    decim: int,
    global_start_index: int,
) -> np.ndarray:
    L = len(taps)
    overlap = L - 1
    return fir_decimate_chunk_stateless(
        dsd_ext,
        taps,
        decim,
        global_start_index,
        overlap,
    )

def create_fir_decimator_state(num_channels: int, num_taps: int, decim: int) -> FIRDecimatorState:
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    if num_taps <= 1:
        raise ValueError("num_taps must be > 1.")
    if decim <= 0:
        raise ValueError("decim must be positive.")

    prev_input = np.zeros((num_channels, num_taps - 1), dtype=np.float32)
    phase = np.zeros(num_channels, dtype=np.int64)
    return FIRDecimatorState(prev_input=prev_input, phase=phase)

def design_kaiser_lowpass(
    fs: float,
    f_stop: float,
    attenuation_db: float,
    transition_ratio: float = 0.1,
) -> np.ndarray:
    """Kaiser 窓でローパス FIR フィルタを設計する。

    Parameters
    ----------
    fs:
        入力サンプリング周波数 [Hz] (DSD 側の Fs)。
    f_stop:
        阻止帯域開始周波数 [Hz]。
    attenuation_db:
        阻止帯域減衰量 [dB]。
    transition_ratio:
        遷移帯域幅を f_stop に対する比 (0.1 なら f_stop の ±10% を遷移帯域にするイメージ)。
    """
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")
    if f_stop <= 0:
        raise ValueError("Stopband frequency must be positive.")

    # Nyquist 未満にクランプ
    nyq = fs / 2.0
    f_stop = min(f_stop, 0.99 * nyq)

    # 遷移帯域幅
    delta_f = max(f_stop * transition_ratio, fs * 1e-5)
    if delta_f <= 0:
        raise ValueError("Transition width must be positive.")

    A = max(float(attenuation_db), 0.0)

    # Kaiser β
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21.0) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    # 正規化遷移幅 Δω [rad]
    dw = 2.0 * np.pi * delta_f / fs
    # フィルタ長の近似 (Oppenheim & Schafer)
    N = int(np.ceil((A - 8.0) / (2.285 * dw)))
    if N < 5:
        N = 5
    # 奇数長にして線形位相 FIR にする
    if N % 2 == 0:
        N += 1

    # 遷移帯域の中央をカットオフに設定
    fc = max(min(f_stop - delta_f / 2.0, nyq * 0.99), 1.0)

    n = np.arange(N)
    m = n - (N - 1) / 2.0

    # 理想 LPF: 2*fc/fs * sinc(2*fc*m/fs)
    x = 2.0 * fc * m / fs
    h_ideal = 2.0 * fc / fs * np.sinc(x)

    # Kaiser 窓
    if beta == 0.0:
        window = np.ones_like(h_ideal)
    else:
        arg = beta * np.sqrt(1.0 - ((2.0 * n) / (N - 1) - 1.0) ** 2)
        window = np.i0(arg) / np.i0(beta)

    h = h_ideal * window
    # DC 利得 = 1 に正規化
    h /= np.sum(h)
    return h.astype(np.float32)

def create_fir_decimator_state(num_channels: int, num_taps: int, decim: int) -> FIRDecimatorState:
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    if num_taps <= 1:
        raise ValueError("num_taps must be > 1.")
    if decim <= 0:
        raise ValueError("decim must be positive.")

    prev_input = np.zeros((num_channels, num_taps - 1), dtype=np.float32)
    phase = np.zeros(num_channels, dtype=np.int64)
    return FIRDecimatorState(prev_input=prev_input, phase=phase)


@njit(cache=True, fastmath=True, parallel=True)
def _fir_decimate_chunk_core(
    dsd_ext_cf: np.ndarray,    # shape = (channels, N_ext), float32
    taps: np.ndarray,          # shape = (L,), float32
    decim: int,
    phase_init: int,           # dsd_ext[0] に対応する global index % decim
    overlap: int,              # 通常 L-1
    main_len: int,             # 本体サンプル数
) -> np.ndarray:
    """
    float32 前提の高速版 (channels-first)。

    dsd_ext_cf:
        shape = (channels, N_ext)。先頭 overlap サンプルは前チャンクとのオーバーラップ。
    taps:
        FIR 係数。
    decim:
        デシメーション係数。
    phase_init:
        dsd_ext_cf[:, overlap] に対応する DSD グローバル index % decim。
    overlap:
        オーバーラップサンプル数。通常 len(taps)-1。
    main_len:
        このチャンクで本体として処理する DSD サンプル数。
    """
    num_channels, num_samples = dsd_ext_cf.shape
    L = taps.shape[0]

    if num_samples != overlap + main_len:
        raise ValueError("dsd_ext length must be overlap + main_len.")

    # 本体区間
    start_main = overlap
    end_main = overlap + main_len  # 非包含

    # 出力候補の範囲 [n_min, n_max]
    n_min = start_main
    if L - 1 > n_min:
        n_min = L - 1

    n_max = end_main - 1
    if n_min > n_max:
        return np.empty((0, num_channels), dtype=np.float32)

    # 条件: (phase_init + n) % decim == 0  <=>  n ≡ (-phase_init) mod decim
    phase_mod = phase_init % decim
    n0 = (decim - phase_mod) % decim

    # n >= n_min で n ≡ n0 (mod decim) となる最初の n を求める
    if n0 >= n_min:
        n_first = n0
    else:
        delta = n_min - n0
        k0 = delta // decim
        if n0 + k0 * decim < n_min:
            k0 += 1
        n_first = n0 + k0 * decim

    if n_first > n_max:
        return np.empty((0, num_channels), dtype=np.float32)

    # 出力サンプル数
    span = n_max - n_first
    n_out = span // decim + 1

    pcm = np.empty((n_out, num_channels), dtype=np.float32)

    # ★ 出力サンプル方向に並列化
    for i in prange(n_out):
        n = n_first + i * decim
        for ch in range(num_channels):
            x = dsd_ext_cf[ch]  # shape = (N_ext,), contiguous
            acc = np.float32(0.0)
            # y[n] = Σ h[k] * x[n-k]
            for k in range(L):
                acc += taps[k] * x[n - k]
            pcm[i, ch] = acc

    return pcm


def fir_decimate_chunk_stateless(
    dsd_ext: np.ndarray,
    taps: np.ndarray,
    decim: int,
    global_start_index: int,
    overlap: int,
) -> np.ndarray:
    """stateless な 1 チャンク用 FIR+decimator（float32, channels-first版）。"""
    # 入力は (N_ext, ch) 前提。float32＋C連続に整えておく。
    dsd_ext32 = np.ascontiguousarray(dsd_ext, dtype=np.float32)

    num_samples, num_channels = dsd_ext32.shape
    main_len = num_samples - overlap
    if main_len <= 0:
        return np.empty((0, num_channels), dtype=np.float32)

    L = taps.shape[0]
    if overlap < L - 1:
        raise ValueError("overlap must be at least L-1 for seamless processing.")

    # channels-first に 1 回だけ転置して contiguous にする
    dsd_ext_cf = np.ascontiguousarray(dsd_ext32.T)  # shape = (ch, N_ext)

    taps32 = np.ascontiguousarray(taps, dtype=np.float32)

    # dsd_ext[0] に対応する global index
    ext_start_index = global_start_index - overlap
    phase_init = ext_start_index % decim

    pcm = _fir_decimate_chunk_core(
        dsd_ext_cf,
        taps32,
        int(decim),
        int(phase_init),
        int(overlap),
        int(main_len),
    )
    return pcm

def process_dsd_in_chunks_stateless(
    dsd_iter,
    taps: np.ndarray,
    decim: int,
    chunk_dsd_samples: int,
):
    """DSD サンプル列をチャンク分割して stateless FIR+decimation で処理する例。

    Parameters
    ----------
    dsd_iter:
        2D ndarray (samples, ch) を順次 yield するイテレータ。
        例: DsfReader(...).iter_blocks()
    taps:
        FIR 係数。
    decim:
        デシメーション係数。
    chunk_dsd_samples:
        1 チャンクあたりの「本体サンプル数」（DSD 側）。
        例: DSD Fs=2.8224MHz で 0.5 秒ぶんなら ≒ 1,411,200 サンプル。

    Yields
    ------
    pcm_block:
        各チャンクに対応する PCM ブロック (2D ndarray) を順に yield。
    """
    L = len(taps)
    overlap = L - 1

    # 直前までの末尾 overlap サンプル
    tail = None  # shape = (overlap, ch)
    # ファイル全体に対するグローバル DSD インデックス
    global_index = 0

    agg_blocks = []
    agg_count = 0  # 本体として貯めたサンプル数

    for blk in dsd_iter:
        if blk.size == 0:
            continue
        agg_blocks.append(blk)
        agg_count += blk.shape[0]

        while agg_count >= chunk_dsd_samples:
            # チャンク本体を取り出す
            # agg_blocks から chunk_dsd_samples 分を切り出す（実装簡略化のため concat）
            big = np.concatenate(agg_blocks, axis=0)
            main = big[:chunk_dsd_samples, :]
            rest = big[chunk_dsd_samples:, :]

            # 残りは次のチャンクの先頭として使う
            agg_blocks = [rest] if rest.shape[0] > 0 else []
            agg_count = rest.shape[0]

            # オーバーラップ付き入力を作る
            if tail is None:
                # ファイル最初のチャンク：オーバーラップは 0 埋め
                ch = main.shape[1]
                tail = np.zeros((overlap, ch), dtype=np.float32)
            # dsd_ext = [tail; main]
            dsd_ext = np.concatenate([tail, main.astype(np.float32)], axis=0)

            # このチャンク本体先頭のグローバル index
            g_start = global_index

            # FIR + decimation
            pcm_chunk = fir_decimate_chunk_stateless(
                dsd_ext,
                taps,
                decim,
                global_start_index=g_start,
                overlap=overlap,
            )

            yield pcm_chunk

            # tail を更新（今回処理した本体の末尾 overlap サンプル）
            concat_for_tail = np.concatenate([tail, main], axis=0)
            if concat_for_tail.shape[0] >= overlap:
                tail = concat_for_tail[-overlap:, :]
            else:
                # ここに来るのはほぼありえないが、保険
                pad = overlap - concat_for_tail.shape[0]
                ch = concat_for_tail.shape[1]
                new_tail = np.zeros((overlap, ch), dtype=np.float32)
                new_tail[pad:, :] = concat_for_tail
                tail = new_tail

            global_index += main.shape[0]

    # 余りがあれば最後のチャンクとして処理
    if agg_count > 0:
        big = np.concatenate(agg_blocks, axis=0)
        main = big  # 全部本体にする

        if tail is None:
            ch = main.shape[1]
            tail = np.zeros((overlap, ch), dtype=np.float32)

        dsd_ext = np.concatenate([tail, main.astype(np.float32)], axis=0)
        g_start = global_index

        pcm_chunk = fir_decimate_chunk_stateless(
            dsd_ext,
            taps,
            decim,
            global_start_index=g_start,
            overlap=overlap,
        )
        yield pcm_chunk
