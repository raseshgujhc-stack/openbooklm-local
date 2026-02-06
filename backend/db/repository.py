class MetadataRepository:
    def insert(self, data: dict):
        raise NotImplementedError
        
    def fetch_by_collection(self, collection_id):
        raise NotImplementedError
