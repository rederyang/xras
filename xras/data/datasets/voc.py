import tensorflow_datasets as tfds
import math

'''TODO:
1. finish cal_iou
2. finish match
'''

def load_voc_dataset(sub=True):
    if sub:
        ds_train = tfds.load('voc/2007', split='train+validation', shuffle_files=True)
        ds_val = tfds.load('voc/2007', split='test')
    else:
        ds_train_a = tfds.load('voc/2007', split='train+validation', shuffle_files=True)
        ds_train_b = tfds.load('voc/2012', split='train+validation', shuffle_files=True)
        ds_train = ds_train_b.concat(ds_train_a)
        ds_val = tfds.load('voc/2007', split='test')

    return ds_train, ds_val


'''TODO:
    1. check where to add the 'bgd' class - might be in the match function
    
'''
def decode(elem):
    ''' to decode an element from voc dataset provided by tfds
    args:
        elem
            an element from voc dataset provided by tfdd
    returns:
        image 
            tf.Tensor, the image of the elem
        gt
            a dict, gt['label'] == the label of objects,
            gt['bbox'] == the bouding box of objects, in the
            format of center-size coords
    '''
    image = elem['image']
    gt = {
        'bbox': elem['object']['bbox'],
        'label': elem['object']['label']
    }

    return image, gt


def cc2bc(cc):
    '''convert a center-size coords to a boundary coords'''
    x_c, y_c, w, h = cc
    x_min = x_c - w / 2.
    x_max = x_min + w
    y_min = y_c - h / 2.
    y_max = y_min + h
    
    return [x_min, y_min, x_max, y_max]


def cc2bc(bc):
    '''convert a boundary coords to a center-size coords'''
    x_min, y_min, x_max, y_max = bc
    x_c = (x_max - x_min) / 2.
    y_c = (y_max - y_min) / 2.
    w = x_max - x_min
    h = y_max - y_min
    
    return [x_c, y_c, w, h]


def cal_iou(bboxes_a, bboxes_b):
    '''calculate iou betwwen two groups of bboxes
    args:
        bboxes_a
            a list of bbox coords
    returns:
        a matrix of shape [size(bboxes_a), size(bboxes_b)]
    '''
    

def match(anchors, gts):
    '''map anchors to gts
    args:
        anchors
            a list of dicts, as defined in the SSDAnchorGenerator
        gts
            a list of dicts, as defined in the decode function
    returns:
        targets
            a list of dicts, in the same form of gts, for each target
            target['offset'] is [g_c_x, g_c_y, g_w, g_h]
            target['label'] is the target class
    '''

    anchor_bboxes = [anchor['bbox'] for anchor in anchors]
    gt_bboxes = [gt['bbox'] for gt in gts]

    ious = cal_iou(anchor_bboxes, gt_bboxes)



    # two rules to do the match depending on ious
    # 1. anchor-wise 2. gt-wise
    
    # anchor-wise
    for anchor in anchors:
        pass
        

'''TODO:
    1. make sure the order of each anchor whether correspond
    to the order of flattened feature map, for func
        make_anchors_for_one_fm
        make_anchors_for_multi_fm
    2. for SSDAnchorGenerator, in case we want to change the default
    settings
'''
class SSDAnchorGenerator:
    ''' generate anchors defined in SSD'''
    
    def __init__(self, default=True):
        if default:
            # for each feature map
            self.fm_scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
            self.fm_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
            self.fm_subset = [True, False, False, False, True, True]

            # for each tile in a feature map
            self.bbox_aspects = [1., 2., 0.5, 3., 1./3.]
        else:
            pass    

    def make_anchors_for_one_fm(self, size, scale, extra_scale, subset):
        ''' make anchors for a particular feature map
        args:
            size
                tuple (H, W), hight and width of the feature map
            scale
                float, the scale of each anchor, the area in the image is scale * scale
            extra_scale
                float, the extra scale for each anchor with aspect == 1:1
            subset
                bool, whether to use the simple aspects
        returns:
            a list of dict
                each dict['bbox'] is the coords of the anchor,  in the order of different setting, width, height
        '''
        anchors = []

         # width and height wrt each aspect
        aspects = self.bbox_aspects[:3] if subset else self.bbox_aspects
        w = [scale * math.sqrt(a) for a in aspects] + [extra_scale]
        h = [scale / math.sqrt(a) for a in aspects] + [extra_scale]
        w_h_pairs = zip(w, h)

        # the order matters, 'i' and 'j' (tiles in the feature map) must be iterated after the aspects,
        # to ensure that neighbour anchors have the same aspect
        for w_h in w_h_pairs:
            for j in range(size[0]): # size is in the order of (H, W), height direction
                for i in range(size[1]): # size is in the order of (H, W), width direction
                    c_x_y = [(i + 0.5) / size[1], (j + 0.5) / size[0]] 
                    anchors.append({'bbox':c_x_y + list(w_h)}) # [x_c, y_c, w, h]
        return anchors

    def make_anchors_for_multi_fm(self):
        '''make anchors for multiple feature maps in different stages
        returns:
            a list of dict
                each dict['bbox'] is the coords of the anchor, in the order of different stage, setting, 
                width, height
        '''
        anchors = []

        # add 1.0 in the end to calculate extra scales
        scales = self.fm_scales + [1.0]

        for i in range(len(self.fm_scales)):
            extra_scale = math.sqrt(scales[i] * scales[i+1]) # the extra scale for each stage of feature map
            anchors += self.make_anchors_for_one_fm(self.fm_sizes[i], self.fm_scales[i], extra_scale, self.fm_subset[i])

        return anchors
