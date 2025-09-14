import os
import random
import pandas as pd
from natsort import natsorted

random.seed(0)

def split_InteractMove():
    train_id = []
    test_id = []
    anno = pd.read_csv(os.path.join('./data/InteractMove/contact_motion/anno.csv'))
    for i in range(len(anno)):
        scene_id = anno.loc[i]['scene_id']

        if int(scene_id[5:9]) < 600:
            train_id.append(i)
        else:
            test_id.append(i)

    with open('./data/InteractMove/train.txt', 'w') as f:
        for i in train_id:
            f.write(f'{i:0>6d}\n')
    with open('./data/InteractMove/test.txt', 'w') as f:
        for i in test_id:
            f.write(f'{i:0>6d}\n')
    with open('./data/InteractMove/all.txt', 'w') as f:
        for i in range(len(anno)):
            f.write(f'{i:0>6d}\n')

if __name__ == '__main__':
    split_InteractMove()
