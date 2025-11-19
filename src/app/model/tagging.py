from __future__ import annotations

from pathlib import Path

from mutagen.dsf import DSF
from mutagen.flac import FLAC, Picture
from mutagen.id3 import ID3Tags


def _copy_text_frames(id3: ID3Tags, flac: FLAC) -> None:
    """代表的なテキストフレームを FLAC の Vorbis コメントにコピーする。"""
    if id3 is None:
        return

    frame_map = {
        "TIT2": "TITLE",        # Title/songname/content description
        "TPE1": "ARTIST",       # Lead performer/soloist
        "TPE2": "ALBUMARTIST",  # Band/orchestra/accompaniment
        "TALB": "ALBUM",        # Album title
        "TCON": "GENRE",        # Genre
        "TRCK": "TRACKNUMBER",  # Track number
        "TPOS": "DISCNUMBER",   # Disc number
        "TCOM": "COMPOSER",
        "TDRC": "DATE",         # Recording time (v2.4)
        "TYER": "DATE",         # Year (v2.3)
    }

    for frame_id, flac_key in frame_map.items():
        frames = id3.getall(frame_id)
        if not frames:
            continue
        values: list[str] = []
        for frame in frames:
            text = getattr(frame, "text", None)
            if text:
                values.extend(str(t) for t in text)
        if values:
            flac[flac_key] = values

    # コメント (COMM)
    comm_frames = id3.getall("COMM")
    if comm_frames:
        comments: list[str] = []
        for frame in comm_frames:
            text = getattr(frame, "text", None)
            if text:
                comments.extend(str(t) for t in text)
        if comments:
            flac["COMMENT"] = comments


def _copy_pictures(id3: ID3Tags, flac: FLAC) -> None:
    """APIC (Attached Picture) を FLAC の Picture ブロックにコピーする。"""
    if id3 is None:
        return

    apic_frames = id3.getall("APIC")
    if not apic_frames:
        return

    for apic in apic_frames:
        pic = Picture()
        pic.data = apic.data
        pic.mime = apic.mime
        pic.type = apic.type
        pic.desc = apic.desc or ""
        # 幅・高さなどは不明であれば 0 のまま
        pic.width = getattr(apic, "width", 0) or 0
        pic.height = getattr(apic, "height", 0) or 0
        pic.depth = getattr(apic, "depth", 0) or 0
        pic.colors = 0
        flac.add_picture(pic)


def copy_tags_dsf_to_flac(dsf_path: str | Path, flac_path: str | Path) -> None:
    """DSF ファイルの ID3 タグを FLAC ファイルにコピーする。

    タイトル・アーティスト・アルバムなど代表的な項目とアルバムアートをコピーする。
    必要に応じて拡張可能。
    """
    dsf_path = Path(dsf_path)
    flac_path = Path(flac_path)

    dsf = DSF(dsf_path)
    id3 = dsf.tags
    if id3 is None:
        return

    flac = FLAC(flac_path)
    if flac.tags is None:
        flac.add_tags()

    _copy_text_frames(id3, flac)
    _copy_pictures(id3, flac)

    flac.save()
