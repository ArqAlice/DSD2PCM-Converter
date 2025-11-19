from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Iterator

import numpy as np


class DsfReader:
    """DSF (DSD Stream File) をストリーミング読み出しするクラス。

    BitsPerSample == 1 の 1bit DSD のみサポート。
    サンプルは [-1.0, +1.0] の float64 で返す。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        self._parse_header()

    # ------------------------------------------------------------------ #
    # 低レベルヘルパ
    # ------------------------------------------------------------------ #
    def _read_exact(self, n: int) -> bytes:
        data = self._fh.read(n)
        if len(data) != n:
            raise IOError("Unexpected end of file while reading DSF header.")
        return data

    @staticmethod
    def _u32_le(data: bytes) -> int:
        return struct.unpack("<I", data)[0]

    @staticmethod
    def _u64_le(data: bytes) -> int:
        return struct.unpack("<Q", data)[0]

    def _read_u32(self) -> int:
        return self._u32_le(self._read_exact(4))

    def _read_u64(self) -> int:
        return self._u64_le(self._read_exact(8))

    # ------------------------------------------------------------------ #
    # ヘッダ解析
    # ------------------------------------------------------------------ #
    def _parse_header(self) -> None:
        # DSD chunk ------------------------------------------------------
        if self._read_exact(4) != b"DSD ":
            raise ValueError(f"{self.path} is not a DSF file (missing 'DSD ' chunk).")

        dsd_chunk_size = self._read_u64()
        total_file_size = self._read_u64()
        metadata_ptr = self._read_u64()  # ID3v2 タグへのポインタ（0 の場合もあり）

        # fmt chunk ------------------------------------------------------
        if self._read_exact(4) != b"fmt ":
            raise ValueError("Missing 'fmt ' chunk in DSF file.")

        fmt_chunk_size = self._read_u64()
        fmt_version = self._read_u32()
        fmt_id = self._read_u32()
        channel_type = self._read_u32()
        channel_num = self._read_u32()
        sample_rate = self._read_u32()
        bits_per_sample = self._read_u32()
        sample_count = self._read_u64()  # per channel
        block_size_per_channel = self._read_u32()
        reserved = self._read_u32()

        # fmt chunk の残り
        bytes_consumed = (
            4    # header
            + 8  # chunk_size
            + 4  # fmt_version
            + 4  # fmt_id
            + 4  # channel_type
            + 4  # channel_num
            + 4  # sample_rate
            + 4  # bits_per_sample
            + 8  # sample_count
            + 4  # block_size_per_channel
            + 4  # reserved
        )
        remaining = int(fmt_chunk_size) - bytes_consumed
        if remaining > 0:
            self._fh.read(remaining)

        # data chunk -----------------------------------------------------
        if self._read_exact(4) != b"data":
            raise ValueError("Missing 'data' chunk in DSF file.")

        data_chunk_size = self._read_u64()
        data_start = self._fh.tell()
        # data_chunk_size = n + 12 (ヘッダ+サイズフィールドを含む)
        data_n = int(data_chunk_size) - 12

        # 属性保存
        self.channels = int(channel_num)
        self.sample_rate = int(sample_rate)
        self.bits_per_sample = int(bits_per_sample)
        self.sample_count = int(sample_count)  # per channel
        self.block_size_per_channel = int(block_size_per_channel)
        self.data_start = data_start
        self.data_size = data_n
        self.metadata_ptr = int(metadata_ptr)
        self.total_file_size = int(total_file_size)

        if self.bits_per_sample != 1:
            raise NotImplementedError("Only 1-bit DSF (BitsPerSample == 1) is supported.")

        if self.block_size_per_channel <= 0:
            raise ValueError("Invalid block size per channel in DSF file.")

        if self.channels <= 0:
            raise ValueError("Invalid channel count in DSF file.")

        # サンプル読み出し準備
        self._fh.seek(self.data_start)

        # 派生値
        self._bits_per_block_per_ch = self.block_size_per_channel * 8
        self._blocks_total = int(
            math.ceil(self.sample_count / float(self._bits_per_block_per_ch))
        )

    # ------------------------------------------------------------------ #
    # サンプル読み出し
    # ------------------------------------------------------------------ #
    def iter_blocks(self) -> Iterator[np.ndarray]:
        """DSD サンプルをブロック単位で yield する。

        各ブロックは shape=(N, channels) の numpy 配列（float64、[-1,1]）を返す。
        最後のブロックは短くなる場合がある。
        """
        samples_done = 0
        total_samples = self.sample_count
        bpc = self.block_size_per_channel
        ch = self.channels

        while samples_done < total_samples:
            # 1 DSF ブロック（全チャネル分）読み出し
            raw = self._fh.read(bpc * ch)
            if len(raw) != bpc * ch:
                raise IOError("Unexpected end of file while reading DSF sample data.")

            remaining = total_samples - samples_done
            samples_in_block = min(self._bits_per_block_per_ch, remaining)

            dsd_channels = []
            for ci in range(ch):
                ch_bytes = raw[ci * bpc : (ci + 1) * bpc]
                byte_arr = np.frombuffer(ch_bytes, dtype=np.uint8)
                # DSF は BitsPerSample==1 のとき LSB-first で格納される
                bits = np.unpackbits(byte_arr, bitorder="little")
                bits = bits[:samples_in_block].astype(np.float32)
                # {0,1} -> {-1.0, +1.0}
                dsd = np.float32(2.0) * bits - np.float32(1.0)
                dsd_channels.append(dsd)

            block = np.stack(dsd_channels, axis=1)  # (samples_in_block, channels)
            yield block
            samples_done += samples_in_block

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "DsfReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
