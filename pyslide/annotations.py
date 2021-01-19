
import os
import json


class Annotations()
    def __init__(self, labels, paths, file_type, keys):
        self.labels=labels
        self.paths=paths 
        self.type = file_type
        self.keys = keys

    def generate_annotations(self):

        with open(json_path) as json_file:
            json_annotations=json.load(json_file)
            keys = list(json_annotations.keys())

        for k in keys:
            if k not in class_key:
                del json_annotations[k]

        annotations = {class_key[k]: [[[int(i['x']), int(i['y'])] for i in v2] for
                      k2, v2 in v.items()]for k, v in json_annotations.items()}
    
        return annotations


