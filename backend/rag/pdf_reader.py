from pypdf import PdfReader
from pathlib import Path


def read_pdf(upload_file):
    upload_file.file.seek(0)

    reader = PdfReader(upload_file.file)
    text = ""

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- PAGE {i+1} ---\n"
            text += page_text + "\n"

    print("===== RAW PDF TEXT START =====")
    print(text[:3000])
    print("===== RAW PDF TEXT END =====")

    return text


# ✅ NEW: adapter for background worker
class FileLike:
    def __init__(self, file_obj):
        self.file = file_obj


def read_pdf_from_path(path: Path) -> str:
    with open(path, "rb") as f:
        fake_upload = FileLike(f)
        return read_pdf(fake_upload)

