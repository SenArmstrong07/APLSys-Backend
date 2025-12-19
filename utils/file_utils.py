def detect_file_type(content: bytes) -> str | None:
    """
    Detect file type from bytes.
    Returns:
        'pdf' | 'jpg' | 'png' | 'webp' | 'tiff' | 'bmp' | 'gif' | None
    """
    if not content or len(content) < 8:
        return None

    # PDF
    if content.startswith(b"%PDF-"):
        return "pdf"

    # JPEG
    if content.startswith(b"\xff\xd8\xff"):
        return "jpg"

    # PNG
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # GIF
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
        return "gif"

    # WEBP
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "webp"

    # TIFF
    if content.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"

    # BMP
    if content.startswith(b"BM"):
        return "bmp"

    return None
