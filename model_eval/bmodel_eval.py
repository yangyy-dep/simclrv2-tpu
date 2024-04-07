import pickle
import numpy as np
import json
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#标签
test_dict = unpickle(r'dataset\test_batch')
testdata = test_dict[b'data'].astype('float')
testlabels = np.array(test_dict[b'labels'])
filename = (test_dict[b'filenames'])

# 从 JSON 文件中读取数据
with open(r'model_simclr_1b_f16.bmodel_test_batch_opencv_python_result.json', 'r') as file:
    data = json.load(file)

true_img = 0
total = 0
i = 0
print(testlabels)
for prediction,label in zip(data,testlabels):
    actual_label = label
    predicted_label = prediction['prediction']
    if actual_label == predicted_label:
        true_img = true_img + 1
    total = total+1
acc = true_img / total
print('准确率：%02.3f %%' % (100.0 * true_img / total))

