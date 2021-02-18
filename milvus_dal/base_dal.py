import os
from milvus import Milvus, DataType

# host = os.getenv("MILVUS_HOST")
# port = int(os.getenv("MILVUS_PORT"))

host = 'localhost'
port = '19530'


milvus_client = Milvus(host, port)


class BaseDAL(object):
    def __init__(self, collection_name: str):
        self.milvus_client = milvus_client
        self.collection_name = collection_name

    def create_collections(self, name: str):
        pass

    def create_partition(self, name: str):
        pass

    def insert_entities(self, data: dict, partition_name: str):
        pass

    def delete_entities(self, ids: list):
        self.milvus_client.delete_entity_by_id(self.collection_name, ids)

    def get_entities(self, ids: list):
        pass

    def search_vector(self, list_facial_vector: list, topk: int, metric_type: str):
        pass
