from datetime import datetime
import joblib

import numpy as np
import torch
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import *


from datasets import ModelNet40
from model import AutoEncoder
from utils import to_one_hots


def prepare_data(autoencoder, split='train', one_hot=False):
    DATASET_PATH = '/home/rico/Workspace/Dataset/modelnet/modelnet40_hdf5_2048'
    train_dataset = ModelNet40(root=DATASET_PATH, npoints=2048, split=split, data_augmentation=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

    latent_features = list()
    lbs = list()
    print('\033[32mStart loading {} data\033[0m'.format(split))
    for i, data in enumerate(train_dataloader):
        pcs, labels, _ = data
        pcs = pcs.permute(0, 2, 1)
        bottleneck = autoencoder.encoder(pcs)
        latent_features.append(bottleneck.detach().numpy())
        if one_hot:
            lbs.append(to_one_hots(labels.squeeze(), 40).detach().numpy())
        else:
            lbs.append(labels.squeeze().detach().numpy())
        print(' ==> Finish load batch', i)

    x = np.concatenate(latent_features, axis=0)
    y = np.concatenate(lbs)

    print('Finishing loading {} data, saving it to file...'.format(split))

    np.save('data/{}_x.npy'.format(split), x)
    if one_hot:
        np.save('data/{}_y_onehot.npy'.format(split), y)
    else:
        np.save('data/{}_y.npy'.format(split), y)

    print('Finishing saving!')

    return x, y


def load_data(split='train', one_hot=False):
    if one_hot:
        x, y = np.load('data/{}_x.npy'.format(split)), np.load('data/{}_y_onehot.npy'.format(split))
    else:
        x, y = np.load('data/{}_x.npy'.format(split)), np.load('data/{}_y.npy'.format(split))
    return x, y


if __name__ == '__main__':
    # ae = AutoEncoder()
    # ae.load_state_dict(torch.load('log/model_lowest_cd_loss.pth'))
    # ae.eval()
    # train_x, train_y = prepare_data(ae, split='train')
    # test_x, test_y = prepare_data(ae, split='test')
    # print(train_x.shape, train_y.shape)
    # print(test_x.shape, test_y.shape)

    # data preprocessing
    train_x, train_y = load_data(split='train')
    test_x, test_y = load_data(split='test')
    train_y = label_binarize(train_y, classes=range(40))
    test_y = label_binarize(test_y, classes=range(40))
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    model = OneVsRestClassifier(svm.LinearSVC(random_state=0, verbose=1, max_iter=10000))
    start_time = datetime.now()
    model.fit(train_x, train_y)
    print('\033[32mFinish training SVM. It cost totally {} s.\033[0m'.format((datetime.now() - start_time).total_seconds()))

    y_pred = model.predict(test_x)
    print('\033[32mAccuracy Overall: {}\033[0m'.format(np.sum(test_y.argmax(axis=1) == y_pred.argmax(axis=1)) / test_x.shape[0]))
    confusion_matrix(test_y.argmax(axis=1), y_pred.argmax(axis=1))  # 需要0、1、2、3而不是OH编码格式

    print('Precision:', precision_score(test_y, y_pred, average='micro'))
    print('Recall:', recall_score(test_y, y_pred, average='micro'))
    print('F1 Score:', f1_score(test_y, y_pred, average='micro'))

    print(classification_report(test_y, y_pred, digits=4))

    joblib.dump(model, 'log/LinearSVC.pkl')
