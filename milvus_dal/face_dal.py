# from app.setting import *
# import pymongo
import time

from milvus import Milvus, DataType
from pprint import pprint
from milvus_dal.base_dal import BaseDAL
import numpy as np
import os

collection_name = "identities_matching"
# collection_name = "test"


class FaceDAL(BaseDAL):
    def __init__(self):
        super().__init__(collection_name)
        self.partition_identities_matching = "identities_matching"
        self.partition_objects = "objects"

    @staticmethod
    def define_collection_param_faces():
        collection_param_faces = {
            "fields": [
                {
                    "name": "person_id",
                    "type": DataType.INT32
                },
                {
                    "name": "facial_vector",
                    "type": DataType.FLOAT_VECTOR,
                    "params": {"dim": 512}
                },
            ],
            "segment_row_limit": 4096,
            "auto_id": True
        }
        return collection_param_faces

    def create_collections(self, collection_param):
        if self.collection_name in self.milvus_client.list_collections():
            self.milvus_client.drop_collection(self.collection_name)
        self.milvus_client.create_collection(self.collection_name, collection_param)
        print(self.milvus_client.list_collections())

    def delete_colection(self):
        if self.collection_name in self.milvus_client.list_collections():
            self.milvus_client.drop_collection(self.collection_name)

    def create_partition(self, partition_name):
        self.milvus_client.create_partition(self.collection_name, partition_name)
        pprint("Have partitions {} in collection {}".format(self.milvus_client.list_partitions(self.collection_name),
                                                            self.collection_name))

    def create_table_for_face_rec(self):
        collection_param_faces = self.define_collection_param_faces()
        self.create_collections(collection_param_faces)
        # self.create_partition(self.partition_objects)
        # self.create_partition(self.partition_identities)

    def insert_entities(self, feature_list, list_person_id):
        faces_entities = [
            {"name": "person_id", "values": list_person_id, "type": DataType.INT32},
            {"name": "facial_vector", "values": feature_list, "type": DataType.FLOAT_VECTOR},
        ]
        ids = self.milvus_client.insert(self.collection_name, faces_entities)
        return ids

    def search_vector(self, list_facial_vector, topk, metric_type="L2"):
        dsl = {
            "bool": {
                "must": [
                    {
                        "vector": {
                            "facial_vector": {"topk": topk, "query": list_facial_vector, "metric_type": metric_type}
                        }
                    }
                ]
            }
        }
        results = self.milvus_client.search(self.collection_name, dsl, fields=["person_id"])
        print("\n----------search----------")
        list_results = []
        for entities in results:
            top_results = []
            for top_k in entities:
                top_results.append({
                    "id": top_k.id,
                    "dis": top_k.distance,
                    "person_id": top_k.entity.person_id
                })
                # print("- id: {}".format(top_k.id))
                # print("- distance: {}".format(top_k.distance))
                # print("- person_id: {}".format(top_k.entity.person_id))
            list_results.append(top_results)
        return list_results

    def get_entities(self, ids: list):
        result = self.milvus_client.get_entity_by_id(self.collection_name, ids)
        list_vector_id_track_search = []
        for body_info in result:
            if body_info is not None:
                print(body_info.id)
                print(body_info.get("id_track"))
                list_vector_id_track_search.append(body_info.get("facial_vector"))

        return list_vector_id_track_search


if __name__ == '__main__':
    face_dal = FaceDAL()

    face_dal.create_table_for_face_rec()
    os.remove('/home/vuong/Desktop/GG_Project/glover_function_test/data/person.json')
    # face_dal.create_table_for_face_rec()
    # test(face_dal)
