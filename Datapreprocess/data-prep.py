import numpy as np
import json

#from img information to get pure img id
val_img_list = json.load(open('test_img_list.json'))
val_img_list_str = []
for i in range(len(val_img_list)):
    a = list(val_img_list[i][-16:-4])
    aa = [x.encode('UTF8') for x in a]
    idx = next((i for i, x in enumerate(aa) if x != '0'), None)
    str1 = ''.join(aa[idx:])
    val_img_list_str.append(str1)
print(len(val_img_list_str))
json.dump(val_img_list_str, open('test_img_list_new.json', 'w'))


#rcnn feature for train val test samples
val_img_list = json.load(open('test_img_list_new.json'))
val_img_feature_dic = json.load(open('image/rcnn_test2015_1_2048.json'))
O = np.zeros((len(val_img_list),2048))
for i in range(len(val_img_list)):
    O[i,:] = val_img_feature_dic[val_img_list[i]]
print(O.shape) 
np.save("rcnn_test_2048_img_feature", O)


#resnet feature for train val test samples
val_img_list = json.load(open('train_img_list.json'))
O = np.zeros((len(val_img_list),2048))
prefix = '/home/ubuntu/efs/vqa1/image/vqa_train_res5c/large/train2014/'

for i in range(len(val_img_list)):
    a = val_img_list[i][8:]+'.npz'
    a = a.encode('ascii','ignore')
    aa = np.load(prefix+a)['x']
    aa = np.mean(aa, axis = (1,2))
    O[i,:] = aa
print(O.shape)
np.save("resnet_train_2048_img_feature", O)


#prepare question_id for test
import json
imgs_test = json.load(open('vqa_raw_test.json', 'r'))
thefile = open('test_question_id.txt', 'w')
for i, img in enumerate(imgs_test):
    thefile.write("%s\n" % img['ques_id'])

