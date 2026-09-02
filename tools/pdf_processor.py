import io
import threading
import requests
import fitz  # pymupdf
from PIL import Image
from typing import List
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import config

# Reused session for connection keep-alive (faster when downloading multiple files).
_session: requests.Session | None = None
_session_lock = threading.Lock()

# Browser-like User-Agent; some CDNs serve faster or avoid throttling for common clients.
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; rera-document-parser/1.0)",
}


S3_ORIGIN = "xbyte-rera-documents-housing.s3.ap-south-1.amazonaws.com"
CLOUDFRONT_CDN = "d38irxr1xt4zpo.cloudfront.net"


def _resolve_s3_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == S3_ORIGIN:
        return urlunparse(parsed._replace(netloc=CLOUDFRONT_CDN))
    return url


def _resolve_dropbox_url(url: str) -> str:
    parsed = urlparse(url)
    if "dropbox.com" not in parsed.netloc:
        return url
    params = parse_qs(parsed.query, keep_blank_values=True)
    params["dl"] = ["1"]
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(netloc="dl.dropboxusercontent.com", query=new_query))


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                _session.headers.update(DOWNLOAD_HEADERS)
    return _session


def download_pdf(url: str) -> bytes:
    url = url.replace("\\", "")
    url = _resolve_s3_url(url)
    url = _resolve_dropbox_url(url)
    session = _get_session()
    resp = session.get(url, timeout=(10, 60), allow_redirects=True, stream=True)
    resp.raise_for_status()
    chunks = []
    for chunk in resp.iter_content(chunk_size=1024 * 256):
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


def convert_to_images(pdf_bytes: bytes, max_pages: int = config.MAX_PAGES) -> List[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        images.append(Image.open(io.BytesIO(pix.tobytes("jpeg"))))
    doc.close()
    return images
