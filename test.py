import matplotlib.pyplot as plt
from xras.data.datasets import load_voc_dataset, SSDAnchorGenerator



if __name__ == '__main__':
    ds_train, ds_test = load_voc_dataset()

    for example in ds_train.take(5):
        print(example)


    # test_gen = SSDAnchorGenerator()
    # anchors = test_gen.make_anchors_for_multi_fm()
    # for i in range(10, 100, 10):
    #     print(anchors[i])
    # print(len(anchors))






