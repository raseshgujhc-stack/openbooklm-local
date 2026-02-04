from pypdf import PdfReader

#def read_pdf(upload_file):
#    upload_file.file.seek(0)
#    reader = PdfReader(upload_file.file)

#    pages_text = []

#    for i, page in enumerate(reader.pages):
#        text = page.extract_text() or ""

        # Normalize whitespace
#        text = text.replace("\xa0", " ")
#        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        # Force page markers (important for headers)
#        page_block = f"\n\n=== PAGE {i+1} ===\n{text}"
#        pages_text.append(page_block)

#    return "\n".join(pages_text)

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
    print(text[:3000])  # first 3k chars
    print("===== RAW PDF TEXT END =====")

    return text
