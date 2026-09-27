"""Minimal DICOM reader for the perfusion intake path.

WHY THIS EXISTS
---------------
Stream 3's specification asks for perfusion-weighted MRI, and the only place a
real rCBV map can come from here is a raw DSC series downloaded from TCIA. That
series is DICOM, so something has to read DICOM -- and this environment has no
pydicom, no SimpleITK and no nibabel (`danon/imaging.py` already writes NIfTI by
hand for the same reason). So this module reads the small, well-specified subset
the archive actually serves and refuses everything else by name:

  * transfer syntaxes it can decode: uncompressed explicit/implicit VR little
    endian, and JPEG 2000 (lossless and lossy), which UPENN-GBM's DSC series use;
  * transfer syntaxes it refuses: everything else, named, rather than decoded
    wrongly -- a JPEG-baseline or RLE series must fail loudly, because a
    mis-decoded perfusion curve looks like a plausible one.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No sequences are traversed (the tags this path needs are all top-level), no
retired VRs, no big-endian, no deflated datasets. A file that needs any of that
raises `DicomError` with the tag it choked on. That is the honest boundary: this
is enough to read a DSC series and not enough to be a DICOM library.
"""

from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DicomError",
    "SUPPORTED_TRANSFER_SYNTAXES",
    "DicomSlice",
    "read_slice",
    "read_series",
    "series_slice_paths",
]

#: Transfer syntaxes this reader decodes. The JPEG 2000 entries are encapsulated
#: (pixel data arrives as fragments) and go through Pillow; the two uncompressed
#: entries are read as a flat buffer. Anything absent raises on read.
SUPPORTED_TRANSFER_SYNTAXES: Dict[str, str] = {
    "1.2.840.10008.1.2": "implicit VR little endian",
    "1.2.840.10008.1.2.1": "explicit VR little endian",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 lossless",
    "1.2.840.10008.1.2.4.91": "JPEG 2000",
}

_ENCAPSULATED = ("1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91")

#: VRs whose value length is a 4-byte field preceded by 2 reserved bytes.
_LONG_VRS = frozenset({"OB", "OW", "OF", "SQ", "UT", "UN", "OD", "OL", "OV"})

#: The tags this path reads, by (group, element).
TAG_TRANSFER_SYNTAX = (0x0002, 0x0010)
TAG_ROWS = (0x0028, 0x0010)
TAG_COLUMNS = (0x0028, 0x0011)
TAG_NUMBER_OF_FRAMES = (0x0028, 0x0008)
TAG_BITS_ALLOCATED = (0x0028, 0x0100)
TAG_PIXEL_REPRESENTATION = (0x0028, 0x0103)
TAG_SAMPLES_PER_PIXEL = (0x0028, 0x0002)
TAG_TR = (0x0018, 0x0080)
TAG_TE = (0x0018, 0x0081)
TAG_INSTANCE_NUMBER = (0x0020, 0x0013)
TAG_IMAGE_POSITION = (0x0020, 0x0032)
TAG_ACQUISITION_TIME = (0x0008, 0x0032)
TAG_SERIES_DESCRIPTION = (0x0008, 0x103E)
TAG_PROTOCOL_NAME = (0x0018, 0x1030)
TAG_MODALITY = (0x0008, 0x0060)
TAG_PIXEL_SPACING = (0x0028, 0x0030)
TAG_SLICE_THICKNESS = (0x0018, 0x0050)
TAG_SPACING_BETWEEN_SLICES = (0x0018, 0x0088)
TAG_PIXEL_DATA = (0x7FE0, 0x0010)

#: Tags whose stored value is a binary unsigned short. Needed for IMPLICIT VR
#: files, which carry no VR at all: reading a 2-byte Rows value as text yields
#: '\x01' and the pixel decode then fails on a file that is perfectly valid.
#: (Found exactly that way: UPENN-GBM's CaPTk-processed series are implicit VR.)
_IMPLICIT_US_TAGS = frozenset(
    {
        TAG_ROWS,
        TAG_COLUMNS,
        TAG_SAMPLES_PER_PIXEL,
        TAG_BITS_ALLOCATED,
        TAG_PIXEL_REPRESENTATION,
        TAG_SLICE_THICKNESS,
        TAG_SPACING_BETWEEN_SLICES,
    }
)

#: Tags carried through as text or numbers, in the order they are reported.
_SCALAR_TAGS: Dict[Tuple[int, int], str] = {
    TAG_MODALITY: "modality",
    TAG_SERIES_DESCRIPTION: "series_description",
    TAG_PROTOCOL_NAME: "protocol_name",
    TAG_TR: "tr_ms",
    TAG_TE: "te_ms",
    TAG_INSTANCE_NUMBER: "instance_number",
    TAG_ACQUISITION_TIME: "acquisition_time",
    TAG_IMAGE_POSITION: "image_position",
    TAG_NUMBER_OF_FRAMES: "number_of_frames",
    TAG_ROWS: "rows",
    TAG_COLUMNS: "columns",
    TAG_BITS_ALLOCATED: "bits_allocated",
    TAG_PIXEL_REPRESENTATION: "pixel_representation",
    TAG_SAMPLES_PER_PIXEL: "samples_per_pixel",
    TAG_PIXEL_SPACING: "pixel_spacing",
    TAG_SLICE_THICKNESS: "slice_thickness_mm",
    TAG_SPACING_BETWEEN_SLICES: "spacing_between_slices_mm",
}


class DicomError(RuntimeError):
    """A file this reader will not decode, with the reason it will not."""


@dataclass(frozen=True)
class DicomSlice:
    """One file's header plus its decoded frames.

    ``frames`` is ``[n_frames, rows, columns]`` in float32. ``tags`` holds the
    raw header values this path consumes; string-valued tags keep their DICOM
    spelling (e.g. ``image_position`` is ``"x\\y\\z"``).
    """

    path: str
    transfer_syntax: str
    transfer_syntax_name: str
    tags: Dict[str, object] = field(default_factory=dict)
    frames: Optional[np.ndarray] = None

    @property
    def z_position(self) -> Optional[float]:
        """Slice location along the patient axis, or None if not stated."""
        raw = self.tags.get("image_position")
        if not isinstance(raw, str):
            return None
        parts = [p for p in raw.replace(",", "\\").split("\\") if p.strip()]
        if len(parts) < 3:
            return None
        try:
            return float(parts[2])
        except ValueError:
            return None

    @property
    def pixel_spacing(self) -> Optional[Tuple[float, float]]:
        """(row_mm, column_mm) from (0028,0030), or None if absent/unnumbered."""
        raw = self.tags.get("pixel_spacing")
        if not isinstance(raw, str):
            return None
        parts = [p for p in raw.replace(",", "\\").split("\\") if p.strip()]
        if len(parts) < 2:
            return None
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            return None

    @property
    def time_key(self) -> Tuple[float, int]:
        """Ordering key within a slice: acquisition time, then instance number.

        Acquisition time alone is not always distinct between frames of one
        slice, so the instance number is the tie-break rather than the primary
        key: files that share a timestamp still have a defined order.
        """
        raw_time = self.tags.get("acquisition_time")
        seconds = float("inf")
        if isinstance(raw_time, str):
            digits = "".join(ch for ch in raw_time if ch.isdigit() or ch == ".")
            try:
                seconds = float(digits) if digits else float("inf")
            except ValueError:
                seconds = float("inf")
        raw_inst = self.tags.get("instance_number")
        try:
            instance = int(raw_inst)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            instance = 0
        return (seconds, instance)


# ---------------------------------------------------------------------------
# Element walking
# ---------------------------------------------------------------------------


def _read_element(buf: bytes, pos: int, explicit: bool):
    """Read one element header. Returns (tag, vr, length, value_pos), or None."""
    if pos + 8 > len(buf):
        return None
    group, element = struct.unpack_from("<HH", buf, pos)
    if explicit:
        vr = buf[pos + 4 : pos + 6].decode("ascii", "replace")
        if vr in _LONG_VRS:
            if pos + 12 > len(buf):
                return None
            length = struct.unpack_from("<I", buf, pos + 8)[0]
            return (group, element), vr, length, pos + 12
        length = struct.unpack_from("<H", buf, pos + 6)[0]
        return (group, element), vr, length, pos + 8
    length = struct.unpack_from("<I", buf, pos + 4)[0]
    return (group, element), None, length, pos + 8


@dataclass(frozen=True)
class _Element:
    """One top-level element: its header span, its value span, where it ends."""

    tag: Tuple[int, int]
    vr: Optional[str]
    length: int  # 0xFFFFFFFF means undefined (encapsulated)
    start: int
    value_pos: int
    end: int


def _skip_sequence(buf: bytes, pos: int, explicit: bool) -> int:
    """Advance past a sequence declared with undefined length.

    This reader does not enter sequences -- the tags the perfusion path needs
    are all top-level -- but UPENN-GBM's MR headers put one before the pixel
    data, so it has to be stepped over exactly rather than guessed at. Both
    delimiters close one level: (FFFE,E00D) an item, (FFFE,E0DD) the sequence.
    A structure this cannot follow refuses instead of resynchronising.
    """
    depth = 1
    while pos + 8 <= len(buf):
        group, element = struct.unpack_from("<HH", buf, pos)
        if (group, element) == (0xFFFE, 0xE0DD) or (group, element) == (0xFFFE, 0xE00D):
            length = struct.unpack_from("<I", buf, pos + 4)[0]
            if length != 0:
                raise DicomError(
                    f"delimiter {group:04X},{element:04X} at offset {pos} carries "
                    f"{length} bytes; this reader will not guess at it"
                )
            depth -= 1
            pos += 8
            if depth <= 0:
                return pos
            continue
        if (group, element) == (0xFFFE, 0xE000):  # item
            length = struct.unpack_from("<I", buf, pos + 4)[0]
            pos += 8
            if length == 0xFFFFFFFF:
                depth += 1
            else:
                pos += length
            continue
        got = _read_element(buf, pos, explicit)
        if got is None:
            break
        _tag, _vr, length, value_pos = got
        if length == 0xFFFFFFFF:
            depth += 1
            pos = value_pos
        else:
            if value_pos + length > len(buf):
                break
            pos = value_pos + length
    raise DicomError(f"undefined-length sequence at offset {pos} is never closed")


def _walk(buf: bytes, start: int, explicit: bool):
    """Yield each top-level element as ``_Element`` until the file runs out."""
    pos = start
    while True:
        got = _read_element(buf, pos, explicit)
        if got is None:
            return
        tag, vr, length, value_pos = got
        if length == 0xFFFFFFFF:
            if tag == TAG_PIXEL_DATA:
                # Encapsulated frames: the caller walks the items.
                yield _Element(tag, vr, length, pos, value_pos, value_pos), None
                return
            end = _skip_sequence(buf, value_pos, explicit)
            yield _Element(tag, vr, 0, pos, value_pos, end), None
            pos = end
            continue
        if value_pos + length > len(buf):
            raise DicomError(
                f"element {tag[0]:04X},{tag[1]:04X} runs past the end of the "
                f"file ({length} bytes declared at offset {value_pos})"
            )
        end = value_pos + length
        yield _Element(tag, vr, length, pos, value_pos, end), buf[value_pos:end]
        pos = end


def _decode_fragments(buf: bytes, value_pos: int, length: int) -> List[bytes]:
    """Split encapsulated pixel data into per-frame fragments.

    The first item of an encapsulated pixel data element is the Basic Offset
    Table, not image data: it is skipped here. Every remaining item is one
    frame's codestream. Reference (0,0) -- the BOT is *required* to be present,
    and may itself be empty -- marks the last fragment.
    """
    fragments: List[bytes] = []
    pos = value_pos
    end = value_pos + length if length != 0xFFFFFFFF else len(buf)
    first = True
    while pos + 8 <= end:
        item_group, item_element = struct.unpack_from("<HH", buf, pos)
        item_length = struct.unpack_from("<I", buf, pos + 4)[0]
        if item_length == 0xFFFFFFFF:
            break
        payload = buf[pos + 8 : pos + 8 + item_length]
        pos += 8 + item_length
        if first:
            first = False  # Basic Offset Table
            continue
        if (item_group, item_element) == (0xFFFE, 0xE0DD):  # sequence delimiter
            break
        if payload:
            fragments.append(payload)
    return fragments


def _decode_jpeg2000(fragment: bytes, path: str) -> np.ndarray:
    """Decode one JPEG 2000 codestream. Pillow is the codec; failure refuses."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dependency
        raise DicomError(
            f"{path}: pixel data is JPEG 2000 and Pillow is unavailable ({exc})"
        ) from exc
    try:
        with Image.open(io.BytesIO(fragment)) as img:
            return np.asarray(img).astype(np.float32)
    except Exception as exc:
        raise DicomError(
            f"{path}: JPEG 2000 frame did not decode ({type(exc).__name__}: {exc})"
        ) from exc


# ---------------------------------------------------------------------------
# Public reads
# ---------------------------------------------------------------------------


def read_slice(path: str, with_pixels: bool = True) -> DicomSlice:
    """Read one DICOM file's header (and pixel frames, unless told not to)."""
    with open(path, "rb") as fh:
        buf = fh.read()

    if len(buf) < 132 or buf[128:132] != b"DICM":
        raise DicomError(f"{path}: no DICM magic at offset 128")

    # Meta group is always explicit VR little endian.
    transfer_syntax: Optional[str] = None
    dataset_start: Optional[int] = None
    for element, value in _walk(buf, 132, True):
        if element.tag[0] != 0x0002:
            dataset_start = element.start
            break
        if element.tag == TAG_TRANSFER_SYNTAX and value is not None:
            transfer_syntax = value.decode("ascii", "replace").strip("\x00 ")
    if dataset_start is None:
        raise DicomError(f"{path}: meta group ran to end of file")

    if transfer_syntax is None:
        raise DicomError(f"{path}: no (0002,0010) transfer syntax in the meta group")
    if transfer_syntax not in SUPPORTED_TRANSFER_SYNTAXES:
        raise DicomError(
            f"{path}: transfer syntax {transfer_syntax} is not one this reader "
            f"decodes (supported: "
            f"{', '.join(sorted(SUPPORTED_TRANSFER_SYNTAXES))})"
        )
    syntax_name = SUPPORTED_TRANSFER_SYNTAXES[transfer_syntax]
    explicit = transfer_syntax != "1.2.840.10008.1.2"

    tags: Dict[str, object] = {}
    frames: Optional[np.ndarray] = None
    pixel_flat: Optional[bytes] = None
    pixel_length = 0
    rows = columns = None
    bits = 16
    signed = False

    for element, value in _walk(buf, dataset_start, explicit):
        tag, vr = element.tag, element.vr
        name = _SCALAR_TAGS.get(tag)
        if name is not None and value is not None:
            if vr in ("US", "SS") or (vr is None and tag in _IMPLICIT_US_TAGS):
                tags[name] = struct.unpack_from("<H", value, 0)[0]
            elif vr == "UL":
                tags[name] = struct.unpack_from("<I", value, 0)[0]
            else:
                tags[name] = value.decode("ascii", "replace").strip("\x00 ")
        if tag == TAG_PIXEL_DATA:
            if value is None:
                # Encapsulated: fragments begin at the value position.
                pixel_fragments = _decode_fragments(
                    buf, element.value_pos, element.length
                )
                frames = np.stack(
                    [_decode_jpeg2000(f, path) for f in pixel_fragments], axis=0
                )
                tags["n_fragments"] = len(pixel_fragments)
            else:
                pixel_flat = value
                pixel_length = len(value)
            break

    if frames is None:
        if pixel_flat is None:
            raise DicomError(f"{path}: no pixel data element")
        rows = int(tags.get("rows") or 0)
        columns = int(tags.get("columns") or 0)
        bits = int(tags.get("bits_allocated") or 16)
        signed = int(tags.get("pixel_representation") or 0) == 1
        if rows <= 0 or columns <= 0:
            raise DicomError(f"{path}: pixel data present but Rows/Columns absent")
        dtype = {8: "u1", 16: "i2" if signed else "u2", 32: "i4" if signed else "u4"}.get(
            bits
        )
        if dtype is None:
            raise DicomError(f"{path}: BitsAllocated={bits} is not 8/16/32")
        per_frame = rows * columns
        if pixel_length % per_frame:
            raise DicomError(
                f"{path}: {pixel_length} pixel bytes is not a whole number of "
                f"{rows}x{columns} frames"
            )
        flat = np.frombuffer(pixel_flat, dtype=np.dtype(dtype))
        frames = flat.reshape(-1, rows, columns).astype(np.float32)

    return DicomSlice(
        path=path,
        transfer_syntax=transfer_syntax,
        transfer_syntax_name=syntax_name,
        tags=tags,
        frames=frames if with_pixels else None,
    )


def series_slice_paths(path: str) -> List[str]:
    """Every file in a series directory, or the single path given."""
    if os.path.isdir(path):
        names = [
            os.path.join(path, n)
            for n in sorted(os.listdir(path))
            if n.lower().endswith((".dcm", ".dicom", ".ima"))
        ]
        if not names:
            raise DicomError(f"{path}: directory holds no .dcm/.dicom/.ima files")
        return names
    return [path]


def read_series(path: str) -> List[DicomSlice]:
    """Read every file of a series, headers and pixels."""
    return [read_slice(p) for p in series_slice_paths(path)]
