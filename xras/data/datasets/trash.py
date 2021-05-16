import pathlib
import os
import xml.etree.ElementTree as ET

import tensorflow as tf

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
  args:
    data_root (trainval or test)
    e.g. './voc2007/trainval/'
  return:
    tf.data.dataset 
  '''

  def __init__(self, path, image_size=300):

    self.image_size = 300
    
    self.data_root = pathlib.Path(path)

    self.class_map = {'bgd':0}

    self.image_size = image_size

    # for each feature map
    self.fm_scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
    self.fm_dimensions = [38, 19, 10, 5, 3, 1]
    self.fm_subset = [True, False, False, False, True, True] # whether to use the subset of default boxes

    # for each tile in a fm
    self.box_aspect = [1., 2., 0.5, 3., 1/3]

    self.default_boxes = self.get_default_boxes()

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
      image = tf.image.resize(image, [self.image_size, self.image_size])
      image /= 255.
      return image

    def load_and_preprocess(path):
      image = tf.io.read_file(path)
      return preprocess_image(image)
      
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return image_ds

  def get_default_boxes(self):
    '''
    args:
      scales, how large each tile of the fm reperent in the image
      fm_dimensions, the spatial dimension of the fm
    return:
      a numpy matrix with shape == (8732, 4), which represents the 8732 default boxes,
      and the boxes coresponding to the same feature map from the pred convs are neighbors
    '''
    def fm_boxes(dimension, scale, extra_scale, subset=True):
      tiles = []
      aspects = self.box_aspect[:3] if subset else self.box_aspect
      w = [scale * math.sqrt(a) for a in aspects] + [extra_scale]
      h = [scale / math.sqrt(a) for a in aspects] + [extra_scale]
      tails = zip(w, h) # width and height pairs
      for tail in tails:
        for i in range(dimension):
          for j in range(dimension):
            # center sized coords [c_x, c_y, w, h] 
            head = [(i + 0.5) / dimension, (j + 0.5) / dimension]
            tiles.append(head + list(tail))
      # the order matters, 'i' and 'j' (tiles in the feature map) must be iterated at the end, 
      # to ensure that the default boxes in the same feature map are beighbours
      return tiles

    default_boxes = []
    scales = self.fm_scales + [1.0]
    for fm in range(len(self.fm_scales)):
      extra_scale = math.sqrt(scales[fm] * scales[fm+1])
      default_boxes += fm_boxes(self.fm_dimensions[fm], self.fm_scales[fm], extra_scale, self.fm_subset[fm])

    return np.array(default_boxes, np.float32)

  # def get_class_map(self):


  def read_annotations(self):
    '''
    input:
      data_root 
    return:
      annotation dataset
    '''

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
    
    def bound2center(center):
      '''
      args:
        center, a list contains [x_min, y_min, x_max, y_max]
      return:
        a list contains [x_c, y_c, w, h]
      '''
      x_min, y_min, x_max, y_max = center
      x_c = (x_min + x_max) / 2.
      y_c = (y_min + y_max) / 2.
      w = (x_min - x_max) / 2.
      h = (y_min - y_max) / 2.

      return [x_c, y_c, w, h]
    
    def center2bound(bound):
      x_c, y_c, w, h = bound
      x_min = x_c - w / 2.
      y_min = y_c - h / 2.
      x_max = x_c + w / 2.
      y_max = y_c + h / 2.

      return [x_min, y_min, x_max, y_max]

    def cal_iou(a, b):
      '''
      center coords
      '''
      if a[0] == b[0]: # x方向中心不可分
        dx = min(a[2], b[2])
      else:
        left = a if a[0] < b[0] else b
        right = a if left == b else b
        dx = left[0] + left[2] / 2 - right[0] + right[2] / 2

      if a[1] == b[1]: # y方向中心不可分
        dy = min(a[3], b[3])
      else:
        up = a if a[1] < b[1] else b
        down = a if up == b else b
        dy = up[1] + up[3] / 2 - down[1] + down[3] / 2

      if dx > 0 and dy > 0:
        return dx * dy / (a[2] * a[3] + b[2] * b[3] - dx * dy)
      
      return 0

    def cal_offsets(src, target):
      '''
      center size coords
      src: the priors
      target: the gt box
      '''
      offsets = [
                 (target[0] - src[0]) / src[0],
                 (target[1] - src[1]) / src[1],
                 math.log(target[2] / target[2]),
                 math.log(target[3] / target[3])
      ]

      return offsets

      
    def match(gt):
      '''
      match default boxes to ground truth boxes
      args:
        gt, the groud truth annotation in a single image
      return:
        a numpy array like [match_type, target_label, if_difficult, target_coords] * 8732
      '''

      c_gt = [[t, d, bound2center(c)] for t, d, c in gt]

      result = []

      # calculate IoU and select best match for each priors
      iou = np.zeros([8732, len(c_gt)], np.float32)
      for i in range(iou.shape[0]):
        for j in range(iou.shape[1]):
          iou[i, j] = cal_iou(tuple(self.default_boxes[i]), tuple(c_gt[j][2]))
        maybe_target = np.argmax(iou[i, :])
        if iou[i, maybe_target] >= 0.5:
          new_row = (1., c_gt[maybe_target][0], c_gt[maybe_target][1], cal_offsets(self.default_boxes[i], c_gt[maybe_target][2]))
        else:
          new_row = (0., 0., 0., (0., 0., 0., 0.))
        result.append(new_row)

      # select the best match for each GT
      for j in range(iou.shape[1]):
        selected = np.argmax(iou[:, j])
        if result[selected][0] == 0.: # only select when the best matched is not a positive one
          result[selected] = (1., c_gt[j][0], c_gt[j][1], cal_offsets(self.default_boxes[selected], c_gt[j][2]))

      # print(result)

      # result = np.array(result, np.float32)

      return result
      
    def preprocess(all_annotations):
      '''
      args:
        all_annotations: a list contains lists of ObjAnnotation objects
      return:
        a list contains lists of lists of numpy array
      '''
      obj_map = lambda obj : obj.to_numpy(self.class_map)
      return [tuple(map(obj_map, one_image)) for one_image in all_annotations]

    all_annotation_paths = list(self.data_root.glob('Annotations/*'))
    all_annotation_paths = sorted([str(annotation_path) for annotation_path in all_annotation_paths])

    all_annotations = [parse_xml(annotation_path) for annotation_path in all_annotation_paths] # a list of lists of ObjAnnotation objects
    all_annotations = preprocess(all_annotations)
    match_result = [match(gt) for gt in tqdm(all_annotations)]
    match_ds = tf.data.Dataset.from_tensor_slices(match_result)

    return match_ds