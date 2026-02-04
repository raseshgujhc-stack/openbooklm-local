import uuid

# notebook_id -> vector store
NOTEBOOKS = {}

def create_notebook(filename: str):
    notebook_id = str(uuid.uuid4())
    NOTEBOOKS[notebook_id] = {
        "filename": filename,
        "vectors": [],   # (text, embedding)
    }
    return notebook_id

def get_notebook(notebook_id: str):
    return NOTEBOOKS.get(notebook_id)

def delete_notebook(notebook_id: str):
    if notebook_id in NOTEBOOKS:
        del NOTEBOOKS[notebook_id]
