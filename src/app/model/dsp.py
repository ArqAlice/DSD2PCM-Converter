from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit, prange

@dataclass
class FIRDecimatorState:
    """ストリーミング FIR decimation 用の per-channel 状態。"""
    prev_input: np.ndarray  # shape: (channels, taps-1)
    phase: np.ndarray       # shape: (channels,), modulo decim factor

@dataclass
class CICDecimatorState:
    """CIC デシメータの状態（多チャンネル対応）。

    order:
        CIC の次数 (N)。典型値 3〜5。
    decim:
        デシメーション係数 R (例: 8, 16, 32)。
    integrator:
        shape = (channels, order), float64
    comb:
        shape = (channels, order), float64
    phase:
        shape = (1,), int64。ブロック跨ぎの位相 (0〜decim-1)
    """
    order: int
    decim: int
    integrator: np.ndarray
    comb: np.ndarray
    phase: np.ndarray  # shape=(1,)

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

def create_cic_decimator_state(num_channels: int, order: int, decim: int) -> CICDecimatorState:
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    if order <= 0:
        raise ValueError("CIC order must be positive.")
    if decim <= 1:
        raise ValueError("CIC decimation factor must be >= 2.")

    integrator = np.zeros((num_channels, order), dtype=np.float64)
    comb = np.zeros((num_channels, order), dtype=np.float64)
    phase = np.zeros(1, dtype=np.int64)  # 共通位相

    return CICDecimatorState(
        order=int(order),
        decim=int(decim),
        integrator=integrator,
        comb=comb,
        phase=phase,
    )


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


@njit(cache=True, fastmath=True)
def _process_block_cic_decimate_core(
    dsd_block: np.ndarray,      # (samples, ch), float32
    integrator: np.ndarray,     # (ch, order), float64
    comb: np.ndarray,           # (ch, order), float64
    phase_arr: np.ndarray,      # shape=(1,), int64
    decim: int,
    order: int,
) -> np.ndarray:
    """CIC デシメータ 1 ブロック処理（Numba コア）。

    dsd_block:
        入力 DSD サンプル（±1 を想定）。float32。
    integrator, comb, phase_arr:
        状態（in-place で更新）。
    decim:
        デシメーション係数 R。
    order:
        CIC 次数 N。
    """
    num_samples, num_channels = dsd_block.shape
    p0 = int(phase_arr[0])

    # 出力サンプル数の計算
    # 出力条件: サンプル n (0-origin) の処理時に p == decim-1 だったとき
    # p_n = (p0 + n) mod decim
    # => p_n == decim-1 となる最小 n を求める
    n0 = (decim - 1 - p0) % decim  # 最初に出力が出る n (>=0)
    if n0 >= num_samples:
        out_len = 0
    else:
        span = num_samples - 1 - n0
        out_len = span // decim + 1

    if out_len <= 0:
        # phase だけ進めて終わり
        phase_arr[0] = (p0 + num_samples) % decim
        return np.empty((0, num_channels), dtype=np.float32)

    y = np.empty((out_len, num_channels), dtype=np.float32)

    p = p0
    out_idx = 0

    for n in range(num_samples):
        # 各チャンネルでインテグレータを回す
        for ch in range(num_channels):
            v = float(dsd_block[n, ch])

            # N 段インテグレータ
            for s in range(order):
                integrator[ch, s] += v
                v = integrator[ch, s]

            # デシメーション境界ならコンブを回して出力
            if p == decim - 1:
                w = v
                for s in range(order):
                    diff = w - comb[ch, s]
                    comb[ch, s] = w
                    w = diff
                y[out_idx, ch] = float(w)

        # 位相更新（サンプル共通）
        if p == decim - 1:
            out_idx += 1
            p = 0
        else:
            p += 1

    phase_arr[0] = p
    return y


@njit(cache=True, fastmath=True, parallel=True)
def _fir_decimate_chunk_core(
    dsd_ext: np.ndarray,      # shape = (N_ext, ch), float32
    taps: np.ndarray,         # shape = (L,), float32
    decim: int,
    phase_init: int,          # dsd_ext[0] の global index % decim
    overlap: int,             # 通常 L-1
    main_len: int,            # 本体サンプル数
) -> np.ndarray:
    """
    float32 前提の高速版: 出力位置 n を解析的に求めてから、その位置だけ FIR 畳み込みする。
    """
    num_samples, num_channels = dsd_ext.shape
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

    # チャンネル方向を prange で並列化
    for ch in prange(num_channels):
        n = n_first
        for i in range(n_out):
            acc = np.float32(0.0)
            # y[n] = Σ h[k] * x[n-k]
            for k in range(L):
                acc += taps[k] * dsd_ext[n - k, ch]
            pcm[i, ch] = acc
            n += decim

    return pcm

def fir_decimate_chunk_stateless(
    dsd_ext: np.ndarray,
    taps: np.ndarray,
    decim: int,
    global_start_index: int,
    overlap: int,
) -> np.ndarray:
    """stateless な 1 チャンク用 FIR+decimator（float32版）。"""
    dsd_ext32 = np.ascontiguousarray(dsd_ext, dtype=np.float32)
    taps32 = np.ascontiguousarray(taps, dtype=np.float32)

    num_samples, _ = dsd_ext32.shape
    main_len = num_samples - overlap
    if main_len <= 0:
        return np.empty((0, dsd_ext32.shape[1]), dtype=np.float32)

    L = taps32.shape[0]
    if overlap < L - 1:
        raise ValueError("overlap must be at least L-1 for seamless processing.")

    # dsd_ext[0] に対応する global index
    ext_start_index = global_start_index - overlap
    phase_init = ext_start_index % decim

    pcm = _fir_decimate_chunk_core(
        dsd_ext32,
        taps32,
        int(decim),
        int(phase_init),
        int(overlap),
        int(main_len),
    )
    return pcm

def choose_cic_fir_decim_factors(total_decim: int, max_cic: int = 32) -> tuple[int, int]:
    """総デシメーション total_decim を CIC + FIR に分解する。

    Parameters
    ----------
    total_decim:
        fs_in // fs_out の値（整数）。
    max_cic:
        CIC に割り当てる最大係数（2 の累乗を想定）。デフォルト 32。

    Returns
    -------
    (D_cic, D_fir):
        D_cic * D_fir = total_decim。
        D_cic は power-of-two (<= max_cic) のうち最大のもの。
        D_cic == 1 の場合は「CIC を使わず全部 FIR」という意味。
    """
    if total_decim <= 1:
        return 1, 1

    # total_decim が 2 の累乗でない場合もとりあえず最大の 2 の累乗因子で分解
    D_cic = 1
    # 候補（大きい順）
    candidates = [32, 16, 8, 4, 2]
    for r in candidates:
        if r > max_cic:
            continue
        if total_decim % r == 0:
            D_cic = r
            break

    if D_cic == 1:
        return 1, total_decim

    D_fir = total_decim // D_cic
    return int(D_cic), int(D_fir)


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

@njit(cache=True)
def _process_block_fir_decimate_core(
    dsd_block: np.ndarray,
    taps: np.ndarray,
    prev_input: np.ndarray,
    phase: np.ndarray,
    decim: int,
) -> np.ndarray:
    """
    Numba 用のコア関数（ポリフェーズ FIR decimator）。
    state は渡さず、prev_input/phase を直接受け取って in-place 更新する。
    """
    num_samples, num_channels = dsd_block.shape
    num_taps = taps.shape[0]
    prev_len = num_taps - 1
    step = decim

    # 各 ch の出力サンプル数を見ておき、最小値で揃える
    n_out_min = num_samples
    for ch in range(num_channels):
        p = phase[ch]
        if step > 0:
            p_mod = p % step
        else:
            p_mod = p
        first_idx = (step - p_mod) % step

        if first_idx >= num_samples:
            n_out = 0
        else:
            # first_idx, first_idx+step, ... < num_samples の個数
            n_out = ((num_samples - 1 - first_idx) // step) + 1

        if n_out < n_out_min:
            n_out_min = n_out

    if n_out_min < 0:
        n_out_min = 0

    # 出力バッファ
    pcm_block = np.empty((n_out_min, num_channels), dtype=np.float64)

    # 各チャンネルごとに処理
    for ch in range(num_channels):
        prev = prev_input[ch]
        x_new = dsd_block[:, ch]

        # extended = [prev, x_new] を作る
        extended_len = prev_len + num_samples
        extended = np.empty(extended_len, dtype=np.float64)

        # 先頭に前回の末尾
        for i in range(prev_len):
            extended[i] = prev[i]
        # 後ろに今回のブロック
        for i in range(num_samples):
            extended[prev_len + i] = x_new[i]

        # decimation 位置を計算
        p = phase[ch]
        if step > 0:
            p_mod = p % step
        else:
            p_mod = p
        first_idx = (step - p_mod) % step

        out_idx = 0
        if n_out_min > 0:
            j = first_idx
            while j < num_samples and out_idx < n_out_min:
                # y_valid[j] に相当する位置
                j_base = j + prev_len

                # ここで FIR の畳み込みを直接計算（ポリフェーズ）
                acc = 0.0
                for t in range(num_taps):
                    acc += taps[t] * extended[j_base - t]

                pcm_block[out_idx, ch] = acc
                out_idx += 1
                j += step

        # 位相更新
        phase[ch] = (p + num_samples) % step

        # prev_input を更新 (拡張バッファの末尾 num_taps-1 サンプル)
        if num_samples >= prev_len:
            # ブロックが十分長い普通のケース
            start = num_samples - prev_len
            for i in range(prev_len):
                prev_input[ch, i] = x_new[start + i]
        else:
            # ブロックがフィルタ長より短いレアケース
            offset = prev_len - num_samples
            # まず前回の末尾の一部をコピー
            for i in range(offset):
                prev_input[ch, i] = prev[i + prev_len - offset]
            # 続けて今回ブロックをコピー
            for i in range(num_samples):
                prev_input[ch, offset + i] = x_new[i]

    return pcm_block

def process_block_fir_decimate(
    dsd_block: np.ndarray,
    taps: np.ndarray,
    state: FIRDecimatorState,
    decim: int,
) -> np.ndarray:
    ...
    dsd_block64 = np.ascontiguousarray(dsd_block, dtype=np.float64)
    taps64 = np.ascontiguousarray(taps, dtype=np.float64)

    pcm_block = _process_block_fir_decimate_core(
        dsd_block64,
        taps64,
        state.prev_input,
        state.phase,
        int(decim),
    )

    return pcm_block

def process_block_cic_decimate(
    dsd_block: np.ndarray,
    state: CICDecimatorState,
) -> np.ndarray:
    """CIC デシメータで DSD ブロックを decimation する。

    Parameters
    ----------
    dsd_block:
        shape = (samples, channels), float32/float64。
    state:
        CICDecimatorState。order, decim, integrator, comb, phase を含む。

    Returns
    -------
    y:
        shape = (samples/decim, channels) 程度の float32。
    """
    if dsd_block.size == 0:
        return np.empty((0, state.integrator.shape[0]), dtype=np.float32)

    if dsd_block.ndim != 2:
        raise ValueError("dsd_block must be 2D (samples, channels).")

    num_samples, num_channels = dsd_block.shape
    if num_channels != state.integrator.shape[0]:
        raise ValueError("Channel count mismatch between dsd_block and CIC state.")

    x32 = np.ascontiguousarray(dsd_block, dtype=np.float32)

    y = _process_block_cic_decimate_core(
        x32,
        state.integrator,
        state.comb,
        state.phase,
        int(state.decim),
        int(state.order),
    )
    return y
