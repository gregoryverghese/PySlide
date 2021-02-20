

class Annotations():
    def __init__(self, paths, file_type):
        self.paths=paths 
        self.type = file_type


    def generate_annotations(self, labels):

        with open(self.paths) as json_file:
            json_annotations=json.load(json_file)
            keys = list(json_annotations.keys())

        for k in keys:
            if k not in labels:
                del json_annotations[k]
        
        annotations = {labels[k]: [[[int(i['x']), int(i['y'])] for i in v2] for
                      k2, v2 in v.items()]for k, v in json_annotations.items()}
    
        return annotations


