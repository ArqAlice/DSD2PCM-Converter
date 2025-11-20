from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..native_fir.fir_decimator import fir_decimate_chunk_core as _fir_decimate_chunk_core

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


def fir_decimate_chunk_stateless(
    dsd_ext: np.ndarray,
    taps: np.ndarray,
    decim: int,
    global_start_index: int,  # 本体 main[0] のグローバルインデックス
    overlap: int,
) -> np.ndarray:
    """stateless な 1 チャンク用 FIR+decimation（float32版）。

    Parameters
    ----------
    dsd_ext:
        shape = (N_ext, ch)。先頭 overlap サンプルはオーバーラップ部分、
        続く main_len サンプルが今回のチャンク本体。
    taps:
        FIR 係数 (1D)。
    decim:
        decimation factor。
    global_start_index:
        本体 main[0] のグローバルインデックス（dsd 全体に対して）。
    overlap:
        オーバーラップサンプル数（通常 taps.size - 1）。
    """
    # 型とメモリレイアウトを Cython 側に合わせる
    dsd_ext32 = np.ascontiguousarray(dsd_ext, dtype=np.float32)
    taps32 = np.ascontiguousarray(taps, dtype=np.float32)

    num_samples, num_channels = dsd_ext32.shape

    if overlap < 0 or overlap > num_samples:
        raise ValueError("invalid overlap")

    # 本体サンプル数 = 全体 - overlap
    main_len = num_samples - overlap
    if main_len <= 0:
        # 本体がない場合は空配列
        return np.empty((0, num_channels), dtype=np.float64)

    # dsd_ext[0] のグローバルインデックス
    global_start_ext = global_start_index - overlap

    # Cython コアに渡す位相 (0..decim-1)
    phase_init = int(global_start_ext % decim)

    # Cython 実装を呼び出し
    pcm = _fir_decimate_chunk_core(
        dsd_ext32,
        taps32,
        int(decim),
        phase_init,
        int(overlap),
        int(main_len),
    )

    # _fir_decimate_chunk_core は float64 を返す実装にしている想定
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
    global_index = 0      # 次の main[0] のグローバルインデックス
    tail = None
    agg_blocks = []
    agg_count = 0

    for dsd_block in dsd_iter:
        agg_blocks.append(dsd_block)
        agg_count += dsd_block.shape[0]

        while agg_count >= chunk_dsd_samples:
            # チャンク本体
            big = np.concatenate(agg_blocks, axis=0)
            main = big[:chunk_dsd_samples, :]
            rest = big[chunk_dsd_samples:, :]

            agg_blocks = [rest] if rest.shape[0] > 0 else []
            agg_count = rest.shape[0]

            # オーバーラップ部分を用意
            if tail is None:
                ch = main.shape[1]
                tail = np.zeros((overlap, ch), dtype=np.float32)

            dsd_ext = np.concatenate([tail, main.astype(np.float32)], axis=0)

            # ここで global_index は main[0] のグローバルインデックス
            pcm_chunk = fir_decimate_chunk_stateless(
                dsd_ext,
                taps,
                decim,
                global_start_index=global_index,
                overlap=overlap,
            )
            yield pcm_chunk

            # 次のチャンク用に tail と global_index を更新
            tail = main[-overlap:, :].astype(np.float32)
            global_index += chunk_dsd_samples

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
