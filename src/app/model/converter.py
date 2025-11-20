from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import concurrent.futures

import numpy as np
import soundfile as sf

from .dsf_reader import DsfReader
from .dsp import (
    design_kaiser_lowpass,
    create_fir_decimator_state,
    process_block_fir_decimate,
    create_cic_decimator_state,
    process_block_cic_decimate,
    choose_cic_fir_decim_factors,
)
from .tagging import copy_tags_dsf_to_flac


@dataclass(frozen=True)
class ConversionSettings:
    output_dir: Path
    pcm_samplerate: int
    stopband_hz: float
    stopband_atten_db: float


@dataclass(frozen=True)
class ConversionResult:
    success: bool
    src_path: Path
    dst_path: Path | None
    message: str


def fir_decimate_chunk_worker(
    dsd_ext: np.ndarray,
    taps: np.ndarray,
    decim: int,
    global_start_index: int,
) -> np.ndarray:
    """マルチプロセス用のワーカー関数。

    1 チャンク分の DSD (オーバーラップ付き) に対して FIR+デシメーションを行い、
    このチャンク本体に対応する PCM ブロックを返す。
    """
    overlap = len(taps) - 1
    return fir_decimate_chunk_stateless(
        dsd_ext,
        taps,
        decim,
        global_start_index=global_start_index,
        overlap=overlap,
    )

def convert_dsf_to_flac(
    src_path: str | Path,
    settings: ConversionSettings,
) -> ConversionResult:
    src = Path(src_path)
    try:
        if not src.exists():
            return ConversionResult(False, src, None, "Source file does not exist.")

        out_dir = settings.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (src.stem + ".flac")

        reader = DsfReader(src)
        try:
            fs_dsd = reader.sample_rate
            channels = reader.channels

            fs_pcm = int(settings.pcm_samplerate)
            if fs_pcm <= 0:
                return ConversionResult(False, src, None, "PCM sample rate must be positive.")

            if fs_dsd % fs_pcm != 0:
                msg = (
                    f"DSD sample rate {fs_dsd} is not an integer multiple of "
                    f"target PCM rate {fs_pcm}."
                )
                return ConversionResult(False, src, None, msg)

            decim = fs_dsd // fs_pcm

            # FIR 設計（float32）
            max_stop = 0.45 * fs_pcm
            stopband_hz = min(float(settings.stopband_hz), max_stop)
            if stopband_hz <= 0.0:
                stopband_hz = max_stop

            taps = design_kaiser_lowpass(
                fs=float(fs_dsd),
                f_stop=stopband_hz,
                attenuation_db=float(settings.stopband_atten_db),
            )
            L = len(taps)
            if L < 2:
                return ConversionResult(False, src, None, "FIR taps length is too short.")

            overlap = L - 1

            # -----------------------------
            # チャンクサイズを動的に決める
            #   ・基本は 0.25 秒
            #   ・ただし DSD256 のような高 Fs では、チャンクのバイト数が
            #     16MB を超えないように制限
            # -----------------------------
            target_chunk_bytes = 16 * 1024 * 1024  # 16MB くらいを目安
            bytes_per_sample = 4 * channels       # float32 × channels

            chunk_dsd_samples_time = int(fs_dsd * 0.25)  # 0.25 秒
            chunk_dsd_samples_mem = target_chunk_bytes // bytes_per_sample

            chunk_dsd_samples = min(chunk_dsd_samples_time, chunk_dsd_samples_mem)
            if chunk_dsd_samples <= overlap:
                chunk_dsd_samples = overlap * 2

            # 内部並列度（必要に応じて調整）
            cpu_cnt = os.cpu_count() or 1
            inner_workers = max(1, cpu_cnt // 2)

            # in-flight チャンク数の上限
            max_inflight_chunks = inner_workers * 2

            with sf.SoundFile(
                dst,
                mode="w",
                samplerate=fs_pcm,
                channels=channels,
                format="FLAC",
                subtype="PCM_24",
            ) as out_f, concurrent.futures.ProcessPoolExecutor(
                max_workers=inner_workers,
            ) as pool:

                # in-flight 管理用
                inflight: dict[int, concurrent.futures.Future] = {}
                next_to_write = 0
                chunk_index = 0

                def flush_completed(bound: bool) -> None:
                    """順序を保ちながら、書けるチャンクを out_f に書き出す。

                    bound=True のときは「inflight が max_inflight を切るまで」
                    を目安にループを途中で抜ける。
                    bound=False のときは inflight が空になるまで書き切る。
                    """
                    nonlocal next_to_write
                    while True:
                        if next_to_write not in inflight:
                            break
                        fut = inflight.pop(next_to_write)
                        pcm_block = fut.result()
                        if pcm_block.size != 0:
                            # 既に float32 なのでそのまま書く
                            out_f.write(pcm_block)
                        next_to_write += 1

                        if bound and len(inflight) < max_inflight_chunks:
                            break

                # 直前までの末尾 overlap サンプル（float32）
                tail = np.zeros((overlap, channels), dtype=np.float32)

                # グローバル DSD インデックス
                global_index = 0

                agg_blocks: list[np.ndarray] = []
                agg_count = 0

                # DSD ブロックを読みながらチャンク分割してワーカーに投げる
                for blk in reader.iter_blocks():
                    if blk.size == 0:
                        continue
                    if blk.dtype != np.float32:
                        blk = blk.astype(np.float32, copy=False)

                    agg_blocks.append(blk)
                    agg_count += blk.shape[0]

                    while agg_count >= chunk_dsd_samples:
                        big = np.concatenate(agg_blocks, axis=0)
                        main = big[:chunk_dsd_samples, :]
                        rest = big[chunk_dsd_samples:, :]

                        agg_blocks = [rest] if rest.size > 0 else []
                        agg_count = rest.shape[0] if rest.size > 0 else 0

                        # [tail; main] を作成
                        dsd_ext = np.concatenate([tail, main], axis=0).astype(
                            np.float32, copy=False
                        )

                        g_start = global_index  # 本体先頭の global index

                        # 新しいチャンクをサブワーカーに submit
                        fut = pool.submit(
                            fir_decimate_chunk_worker,
                            dsd_ext,
                            taps,
                            decim,
                            g_start,
                        )
                        inflight[chunk_index] = fut
                        chunk_index += 1

                        # in-flight が増えすぎないように適宜フラッシュ
                        if len(inflight) >= max_inflight_chunks:
                            flush_completed(bound=True)

                        # tail 更新
                        concat_for_tail = np.concatenate([tail, main], axis=0)
                        if concat_for_tail.shape[0] >= overlap:
                            tail = concat_for_tail[-overlap:, :].astype(
                                np.float32, copy=False
                            )
                        else:
                            pad = overlap - concat_for_tail.shape[0]
                            new_tail = np.zeros(
                                (overlap, channels), dtype=np.float32
                            )
                            new_tail[pad:, :] = concat_for_tail.astype(
                                np.float32, copy=False
                            )
                            tail = new_tail

                        global_index += main.shape[0]

                # 余りチャンクがあれば最後の 1 個を submit
                if agg_count > 0:
                    big = np.concatenate(agg_blocks, axis=0)
                    main = big.astype(np.float32, copy=False)

                    dsd_ext = np.concatenate([tail, main], axis=0)
                    g_start = global_index

                    fut = pool.submit(
                        fir_decimate_chunk_worker,
                        dsd_ext,
                        taps,
                        decim,
                        g_start,
                    )
                    inflight[chunk_index] = fut
                    chunk_index += 1

                # すべての in-flight チャンクを書き切る
                flush_completed(bound=False)

        finally:
            reader.close()

        # タグコピー
        try:
            copy_tags_dsf_to_flac(src, dst)
        except Exception as tag_exc:  # noqa: BLE001
            return ConversionResult(
                True,
                src,
                dst,
                f"Converted, but failed to copy tags: {tag_exc}",
            )

        return ConversionResult(True, src, dst, "OK")

    except Exception as exc:  # noqa: BLE001
        return ConversionResult(False, src, None, f"Error: {exc}")


def convert_dsf_to_flac_cic(
    src_path: str | Path,
    settings: ConversionSettings,
) -> ConversionResult:
    """CIC + FIR の 2 段デシメータを使った DSF -> FLAC 変換。

    - 1 ファイル内はシングルプロセス・ストリーミング処理
    - 初段: CIC (order=3, decim=D_cic)
    - 後段: FIR (Kaiser) + decim=D_fir
    """
    src = Path(src_path)
    try:
        if not src.exists():
            return ConversionResult(False, src, None, "Source file does not exist.")

        out_dir = settings.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (src.stem + ".flac")

        reader = DsfReader(src)
        try:
            fs_dsd = reader.sample_rate
            channels = reader.channels

            fs_pcm = int(settings.pcm_samplerate)
            if fs_pcm <= 0:
                return ConversionResult(False, src, None, "PCM sample rate must be positive.")

            if fs_dsd % fs_pcm != 0:
                msg = (
                    f"DSD sample rate {fs_dsd} is not an integer multiple of "
                    f"target PCM rate {fs_pcm}."
                )
                return ConversionResult(False, src, None, msg)

            total_decim = fs_dsd // fs_pcm

            # 総デシメーションを CIC + FIR に分割
            D_cic, D_fir = choose_cic_fir_decim_factors(total_decim, max_cic=8)

            # 全部 FIR に任せる fallback
            if D_cic == 1:
                # 既存の単段 FIR 版にフォールバックさせたい場合はここで呼び換える
                return convert_dsf_to_flac(src, settings)  # 単段版がある前提

            # --- 初段 CIC の設計 ---
            cic_order = 3  # 典型値。必要に応じて 4〜5 に変更可。
            cic_state = create_cic_decimator_state(
                num_channels=int(channels),
                order=cic_order,
                decim=int(D_cic),
            )

            fs_after_cic = fs_dsd // D_cic

            # --- 後段 FIR (Kaiser) の設計 ---
            # 阻止帯域は PCM Nyquist 近辺で制限
            max_stop = 0.45 * fs_pcm
            stopband_hz = min(float(settings.stopband_hz), max_stop)
            if stopband_hz <= 0.0:
                stopband_hz = max_stop

            fir_taps = design_kaiser_lowpass(
                fs=float(fs_after_cic),
                f_stop=stopband_hz,
                attenuation_db=float(settings.stopband_atten_db),
            ).astype(np.float32)

            fir_state = create_fir_decimator_state(
                num_channels=int(channels),
                num_taps=len(fir_taps),
                decim=int(D_fir),
            )

            # --- ストリーミング変換 ---
            # DSD 側で 0.25 秒ぶんをまとめて処理（メモリとのバランスで調整可）
            agg_duration_sec = 0.25
            target_dsd_samples_per_ch = int(fs_dsd * agg_duration_sec)
            if target_dsd_samples_per_ch <= 0:
                target_dsd_samples_per_ch = reader.block_size_per_channel * 8

            with sf.SoundFile(
                dst,
                mode="w",
                samplerate=fs_pcm,
                channels=channels,
                format="FLAC",
                subtype="PCM_24",
            ) as out_f:
                agg_blocks: list[np.ndarray] = []
                agg_samples = 0

                for dsd_block in reader.iter_blocks():
                    if dsd_block.size == 0:
                        continue
                    if dsd_block.dtype != np.float32:
                        dsd_block = dsd_block.astype(np.float32, copy=False)

                    agg_blocks.append(dsd_block)
                    agg_samples += dsd_block.shape[0]

                    if agg_samples >= target_dsd_samples_per_ch:
                        big_block = np.concatenate(agg_blocks, axis=0)

                        # 初段 CIC
                        x1 = process_block_cic_decimate(big_block, cic_state)
                        if x1.size != 0:
                            # 後段 FIR + decim
                            pcm_block = process_block_fir_decimate(
                                x1, fir_taps, fir_state, int(D_fir)
                            )
                            if pcm_block.size != 0:
                                out_f.write(pcm_block.astype(np.float32))

                        agg_blocks.clear()
                        agg_samples = 0

                # 端数を flush
                if agg_blocks:
                    big_block = np.concatenate(agg_blocks, axis=0)
                    if big_block.dtype != np.float32:
                        big_block = big_block.astype(np.float32, copy=False)

                    x1 = process_block_cic_decimate(big_block, cic_state)
                    if x1.size != 0:
                        pcm_block = process_block_fir_decimate(
                            x1, fir_taps, fir_state, int(D_fir)
                        )
                        if pcm_block.size != 0:
                            out_f.write(pcm_block.astype(np.float32))

        finally:
            reader.close()

        # タグコピー
        try:
            copy_tags_dsf_to_flac(src, dst)
        except Exception as tag_exc:  # noqa: BLE001
            return ConversionResult(
                True,
                src,
                dst,
                f"Converted, but failed to copy tags: {tag_exc}",
            )

        return ConversionResult(True, src, dst, "OK")

    except Exception as exc:  # noqa: BLE001
        return ConversionResult(False, src, None, f"Error: {exc}")
