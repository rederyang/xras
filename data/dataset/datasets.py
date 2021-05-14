import keras
import tensorflow as tf

def cifar10(wrapped=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train.astype('float32')
    x_test.astype('float32')
    x_train = x_train / 255.
    x_test = x_test / 255.

    x_train = (x_train - [0.4914, 0.4822, 0.4465]) / [0.247, 0.243, 0.261]
    x_test = (x_test - [0.4914, 0.4822, 0.4465]) / [0.247, 0.243, 0.261]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if wrapped:
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        return ds_train, ds_test
    else:
        return x_train, y_train, x_test, y_test


import pathlib
import os
import xml.etree.ElementTree as ET

class ObjAnnotation():
  def __init__(self, *args, **kwargs):
    '''
    content is a dict which has:
    name: a string, the name of the object
    difficult: an integer
    coords: a numpy array with shape (4,)
    '''
    self.obj = kwargs
  
  def get_name(self):
    return self.obj['name']

  def to_numpy(self, class_map, one_not=False):
    '''
    augs:
      class_map, a dict to map class name to an integer
    return:
      a numpy array which contatins:
      digitized name, difficult, coords
    '''
    # name = np.array([class_map[self.obj['name']]], np.float32)
    # difficult = np.array([self.obj['difficult']], np.float32)
    # coords = np.array(self.obj['coords'], np.float32)

    name = float(class_map[self.obj['name']])
    difficult = float(self.obj['difficult'])
    coords = self.obj['coords']

    return (name, difficult, coords)

class VOCDataReader():
  '''
  input:
    data_root (trainval or test)
    e.g. './voc2007/trainval/'
  output:
    tf.data.dataset 
  '''

  def __init__(self, path):
    self.data_root = pathlib.Path(path)
    self.class_map = {'bgd':0}

  def check_map(self, class_name):
    '''
    check if the class name exists in the class_map, 
    if not, add the name
    '''
    if not (class_name in self.class_map.keys()):
      self.class_map[class_name] = max(self.class_map.values()) + 1
  
  def read_images(self):
    '''
    return:
      image dataset
    '''
    all_image_paths = list(self.data_root.glob('JPEGImages/*'))
    all_image_paths = sorted([str(image_path) for image_path in all_image_paths])

    def preprocess_image(image):
      image = tf.image.decode_jpeg(image, channels=3)
      image = tf.image.resize(image, [300, 300])
      image /= 255.
      return image

    def load_and_preprocess(path):
      image = tf.io.read_file(path)
      return preprocess_image(image)
      
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return image_ds

  def read_annotations(self):
    '''
    input:
      data_root 
    return:
      annotation dataset
    '''
    all_annotation_paths = list(self.data_root.glob('Annotations/*'))
    all_annotation_paths = sorted([str(annotation_path) for annotation_path in all_annotation_paths])

    def parse_xml(xml_path):
      '''
      input:
        path of the xml file
      output:
        a list of ObjAnnotation objects corresponding to a xml file (an image)
      '''
      tree = ET.ElementTree(file=xml_path)
      width = float(tree.find('size/width').text)
      height = float(tree.find('size/height').text)

      annotation = []

      for elem in tree.findall('object'):
        # class name
        name = elem.find('name').text
        self.check_map(name)
        # difficult
        difficult = int(elem.find('difficult').text)
        # coords
        bndbox = elem.find('bndbox')
        abs_coords = [float(coord.text) for coord in bndbox] # xmin, ymin, xmax, ymax
        ral_coords = (abs_coords[0]/width, abs_coords[1]/height, abs_coords[2]/width, abs_coords[3]/height)
        # generate an ObjAnnotation object
        annotation.append(ObjAnnotation(name=name, difficult=difficult, coords=ral_coords)) 
      return annotation

    def preprocess(all_annotations):
      '''
      args:
        all_annotations: a list contains lists of ObjAnnotation objects
      return:
        a list contains lists of lists of numpy array
      '''
      obj_map = lambda obj : obj.to_numpy(self.class_map)
      return [tuple(map(obj_map, one_image)) for one_image in all_annotations]

    all_annotations = [parse_xml(annotation_path) for annotation_path in all_annotation_paths] # a list of lists of ObjAnnotation objects
    all_annotations = preprocess(all_annotations)
    # annotation_ds = tf.data.Dataset.from_tensor_slices(all_annotations)
    # annotation_ds = annotation_ds.map(lambda x: tf.ragged.constant(x), num_parallel_calls=tf.data.AUTOTUNE)

    # return annotation_ds
    return all_annotations

    '''
    TODO:
    一个图片对应的target应该是怎样的形式？
    构建Dataset
    '''