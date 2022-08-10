







class TFRecordRead(self):
    def __init__(self):
        pass


    @property
    def num(self):
        pass

    @static_method
    def _print_progress(self,i):
        pass

    
    def _wrap_Int_64(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def wrap_float(value):
    '''
    convert to tf float
    returns:
    '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def wrap_bytes(value):
    '''
    convert value to bytes
    param: value: image
    returns: 
    '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def convert():
        writer=tf.io.TFRecordWriter(tfRecordPath)
        for i, (img, m) in enumerate(images):
            self._print_progress(i)
            #image_name=os.path.basename
            image = tf.keras.preprocessing.image.load_img(image_path)
            image = tf.keras.preprocessing.image.img_to_array(image,dtype=np.uint8)
            image = tf.image.encode_png(image)
            
            data = {
                'image': wrap_bytes(image),
                'name': wrap_bytes(name),
                'dims': wrap_int64(dims[0]) 
                }
               
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)

