from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, Future

from .dsf_reader import DsfReader
from .dsp import design_kaiser_lowpass, fir_decimate_chunk_stateless
from .tagging import copy_tags_dsf_to_flac


@dataclass(frozen=True)
class ConversionSettings:
    output_dir: Path
    pcm_samplerate: int
    stopband_hz: float
    stopband_atten_db: float
    max_workers : int


@dataclass(frozen=True)
class ConversionResult:
    success: bool
    src_path: Path
    dst_path: Path | None
    message: str


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
            #   ・基本は 0.5 秒
            #   ・ただし DSD256 のような高 Fs では、チャンクのバイト数が
            #     32MB を超えないように制限
            # -----------------------------
            target_chunk_bytes = 32 * 1024 * 1024  # 32MB くらいを目安
            bytes_per_sample = 4 * channels       # float32 × channels

            chunk_dsd_samples_time = int(fs_dsd * 0.5)  # 0.5 秒
            chunk_dsd_samples_mem = target_chunk_bytes // bytes_per_sample

            chunk_dsd_samples = min(chunk_dsd_samples_time, chunk_dsd_samples_mem)
            if chunk_dsd_samples <= overlap:
                chunk_dsd_samples = overlap * 2

            with sf.SoundFile(
                dst,
                mode="w",
                samplerate=fs_pcm,
                channels=channels,
                format="FLAC",
                subtype="PCM_24",
            ) as out_f, ThreadPoolExecutor(
                max_workers= settings.max_workers
            ) as executor:

                # 直前までの末尾 overlap サンプル（float32）
                tail = np.zeros((overlap, channels), dtype=np.float32)

                # グローバル DSD インデックス
                global_index = 0

                agg_blocks: list[np.ndarray] = []
                agg_count = 0

                # チャンク ID と書き出し順管理
                next_chunk_id = 0       # 次に submit するチャンクの ID
                next_write_id = 0       # 次に out_f に書き出すべきチャンク ID
                pending: dict[int, Future[np.ndarray]] = {}

                def submit_chunk(main: np.ndarray, tail_arr: np.ndarray, g_start: int) -> None:
                    nonlocal next_chunk_id
                    chunk_id = next_chunk_id
                    next_chunk_id += 1

                    # [tail; main] を作成
                    dsd_ext = np.concatenate([tail_arr, main], axis=0).astype(
                        np.float32, copy=False
                    )

                    fut: Future[np.ndarray] = executor.submit(
                        fir_decimate_chunk_stateless,
                        dsd_ext,
                        taps,
                        decim,
                        g_start,
                        overlap,
                    )
                    pending[chunk_id] = fut

                def drain_completed(block: bool = False) -> None:
                    """
                    chunk_id の昇順で、完了済みのチャンクを out_f に書き出す。

                    block=False: すぐ終わらないものは飛ばす（軽く流す）
                    block=True : next_write_id が完了するまで（または pending が空になるまで）待つ
                    """
                    nonlocal next_write_id

                    while True:
                        fut = pending.get(next_write_id)
                        if fut is None:
                            return

                        if not fut.done():
                            if not block:
                                return
                            # block=True の場合はここで待つ
                            pcm_block = fut.result()
                        else:
                            pcm_block = fut.result()

                        if pcm_block.size != 0:
                            out_f.write(pcm_block)

                        del pending[next_write_id]
                        next_write_id += 1

                # DSD ブロックを読みながらチャンク分割して、その場で FIR+decimate する
                for blk in reader.iter_blocks():
                    if blk.size == 0:
                        continue
                    if blk.dtype != np.float32:
                        blk = blk.astype(np.float32, copy=False)

                    agg_blocks.append(blk)
                    agg_count += blk.shape[0]

                    while agg_count >= chunk_dsd_samples:
                        # チャンク本体を取り出す
                        big = np.concatenate(agg_blocks, axis=0)
                        main = big[:chunk_dsd_samples, :]
                        rest = big[chunk_dsd_samples:, :]

                        agg_blocks = [rest] if rest.size > 0 else []
                        agg_count = rest.shape[0] if rest.size > 0 else 0

                        # ★ このチャンクをスレッドプールに投げる
                        submit_chunk(main, tail, global_index)

                        # tail 更新（次チャンク用）: ここは入力 DSD だけで決まるので
                        # 計算結果は待たなくてよい
                        concat_for_tail = np.concatenate([tail, main], axis=0)
                        if concat_for_tail.shape[0] >= overlap:
                            tail = concat_for_tail[-overlap:, :].astype(np.float32, copy=False)
                        else:
                            pad = overlap - concat_for_tail.shape[0]
                            new_tail = np.zeros((overlap, channels), dtype=np.float32)
                            new_tail[pad:, :] = concat_for_tail.astype(np.float32, copy=False)
                            tail = new_tail

                        global_index += main.shape[0]

                        # ★ 溜めすぎ防止: pending が多くなったら少し捌く
                        if len(pending) >= 2 * (settings.max_workers):
                            drain_completed(block=True)  # 少なくとも一つは書き出す

                        else:
                            # 軽く流す（完了済みがあれば書き出す）
                            drain_completed(block=False)

                # 余りチャンクがあれば最後に処理（これもスレッドプールに投げる）
                if agg_count > 0:
                    big = np.concatenate(agg_blocks, axis=0)
                    main = big.astype(np.float32, copy=False)

                    submit_chunk(main, tail, global_index)

                # ★ すべてのチャンクが終わるまで待って順番に書き出す
                drain_completed(block=True)
                # （executor は with ブロックを抜けると自動で shutdown(wait=True)）

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
