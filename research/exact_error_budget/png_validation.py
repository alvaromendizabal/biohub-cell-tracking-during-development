import json,base64,struct,zlib,re,hashlib
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise RuntimeError(message)

def png_integrity(data: bytes) -> dict:
    """Check bounded, non-interlaced PNGs using the Python standard library.

    Validates signature, chunk framing/CRCs, IHDR, IDAT inflation, scanline
    lengths and filter bytes. This is not a general image decoder. The existing
    Plotly renderer emits non-interlaced PNGs; other encodings fail explicitly.
    """
    import struct
    require(isinstance(data, bytes) and 45 <= len(data) <= (12 << 20), "PNG_SIZE_INVALID")
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "PNG_SIGNATURE_INVALID")
    pos = 8; header = None; pieces = []; seen_idat = False; closed_idat = False
    palette_entries = None; ended = False; chunks = 0
    allowed_depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
    while pos < len(data):
        require(pos + 12 <= len(data), "PNG_CHUNK_TRUNCATED")
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos+4:pos+8]; end = pos+12+length
        require(end <= len(data), "PNG_CHUNK_TRUNCATED")
        require(re.fullmatch(rb"[A-Za-z]{2}[A-Z][A-Za-z]", kind) is not None, "PNG_CHUNK_TYPE_INVALID")
        body = data[pos+8:pos+8+length]
        recorded = struct.unpack_from(">I", data, pos+8+length)[0]
        require((zlib.crc32(kind + body) & 0xffffffff) == recorded, "PNG_CHUNK_CRC_INVALID")
        if chunks == 0: require(kind == b"IHDR", "PNG_IHDR_NOT_FIRST")
        if kind == b"IHDR":
            require(header is None and chunks == 0 and length == 13, "PNG_IHDR_INVALID")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
            require(0 < width <= 8192 and 0 < height <= 8192 and width*height <= 16_000_000, "PNG_DIMENSIONS_OUTSIDE_BOUND")
            require(color in allowed_depths and depth in allowed_depths[color], "PNG_COLOR_DEPTH_INVALID")
            require(compression == 0 and filtering == 0, "PNG_COMPRESSION_OR_FILTER_METHOD_INVALID")
            require(interlace == 0, "PNG_INTERLACE_UNSUPPORTED_BY_RENDERER_CONTRACT")
            header = (width, height, depth, color)
        elif kind == b"PLTE":
            require(header is not None and not seen_idat and palette_entries is None, "PNG_PALETTE_ORDER_INVALID")
            require(0 < length <= 768 and length % 3 == 0 and header[3] not in (0, 4), "PNG_PALETTE_INVALID")
            palette_entries = length // 3
            if header[3] == 3: require(palette_entries <= 2**header[2], "PNG_PALETTE_TOO_LARGE")
        elif kind == b"IDAT":
            require(header is not None and not closed_idat, "PNG_IDAT_ORDER_INVALID")
            require(header[3] != 3 or palette_entries is not None, "PNG_PALETTE_MISSING")
            seen_idat = True; pieces.append(body)
        elif kind == b"IEND":
            require(length == 0 and seen_idat and end == len(data), "PNG_IEND_INVALID")
            ended = True; pos = end; break
        else:
            require(kind[0] & 32, "PNG_UNKNOWN_CRITICAL_CHUNK")
        if seen_idat and kind != b"IDAT": closed_idat = True
        chunks += 1; require(chunks <= 10000, "PNG_CHUNK_COUNT_OUTSIDE_BOUND")
        pos = end
    require(ended and header is not None and pieces, "PNG_DATA_OR_TRAILER_MISSING")
    width, height, depth, color = header
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    row_bytes = (width * depth * channels + 7) // 8
    expected = height * (row_bytes + 1)
    require(expected <= (64 << 20), "PNG_INFLATED_SIZE_OUTSIDE_BOUND")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(b"".join(pieces), expected + 1)
    except zlib.error as e:
        raise RuntimeError("PNG_ZLIB_INVALID") from e
    require(len(raw) == expected and decompressor.eof and not decompressor.unconsumed_tail and not decompressor.unused_data, "PNG_SCANLINE_DATA_INVALID")
    require(all(b <= 4 for b in raw[::row_bytes+1]), "PNG_SCANLINE_FILTER_INVALID")
    return {"width": width, "height": height, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "chunk_crc_checked": True, "scanline_structure_checked": True}

def validate_notebook(path, expected_png, sentinel):
    """Validate saved evidence without Pillow, using JSON and PNG structure."""
    nb = read(path); counts = []; streams = []; sizes = []; images = []; code_cells = []
    png = html = native = errors = 0
    def stream_text(value):
        return "".join(value) if isinstance(value, list) else str(value)
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code": continue
        code_cells.append(c); counts.append(c.get("execution_count"))
        for o in c.get("outputs", []):
            data = o.get("data", {})
            png += int("image/png" in data); html += int("text/html" in data)
            native += int("application/vnd.plotly.v1+json" in data); errors += int(o.get("output_type") == "error")
            if "image/png" in data:
                encoded = stream_text(data["image/png"])
                require(len(encoded) <= (20 << 20), "PNG_BASE64_SIZE_OUTSIDE_BOUND")
                try: image_bytes = base64.b64decode("".join(encoded.split()), validate=True)
                except (ValueError, base64.binascii.Error) as e: raise RuntimeError("PNG_BASE64_INVALID") from e
                info = png_integrity(image_bytes)
                require(info["width"] >= 850 and info["height"] >= 400, "PLOT_DIMENSIONS_TOO_SMALL")
                sizes.append([info["width"], info["height"]]); images.append(info)
            if o.get("output_type") == "stream": streams.append(stream_text(o.get("text", "")))
    require(len(counts) >= 2 and counts == list(range(1, len(counts)+1)), "NOTEBOOK_EXECUTION_COUNTS_INVALID")
    require(png == expected_png and html == native == errors == 0, "NOTEBOOK_MIME_OR_ERROR_CONTRACT_FAILED")
    text = "".join(streams)
    final_stream = "".join(stream_text(o.get("text", "")) for o in code_cells[-1].get("outputs", []) if o.get("output_type") == "stream")
    require(sentinel in final_stream, "NOTEBOOK_SENTINEL_MISSING_OR_NOT_LAST")
    require(not re.search(r"\b(?:UserWarning|RuntimeWarning|FutureWarning|DeprecationWarning):", text), "NOTEBOOK_UNRESOLVED_WARNING")
    return {"status": "passed", "execution_counts": counts, "png_outputs": png, "html_outputs": html,
            "native_plotly_outputs": native, "error_outputs": errors, "warning_outputs": 0,
            "png_dimensions": sizes, "png_integrity": images, "sha256": sha(path),
            "validation_backend": "python_standard_library_no_pillow"}
