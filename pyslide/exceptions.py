class MissingPatches(Exception):
    def __init__(self,patch_names,message="patches missing"):
        self.patch_names=patch_names
        self.message=message
        super().__init__(self.message)

    def __str__(self):
        num_missing=len(self.patch_names)
        return f'{num_missing} -> {self.message}'



