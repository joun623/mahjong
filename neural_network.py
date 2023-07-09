import os
import numpy as np
import chainer
from chainer import Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from config import FIRST_LAYER_NUM, SECOND_LAYER_NUM, OUTPUT_LAYER_NUM, DROP_OUT_RATIO, L2_WEIGHT, MODEL_PATH, STATE_PATH


class NeuralNet(chainer.Chain):
    """
        ニューラルネットのモデル構築
        参考:http://pao2.hatenablog.com/entry/2018/07/26/203813
    """
    def __init__(self):
        super().__init__(
            l1 = L.Linear(None, FIRST_LAYER_NUM),
            l2 = L.Linear(FIRST_LAYER_NUM, FIRST_LAYER_NUM),
            l3 = L.Linear(FIRST_LAYER_NUM, FIRST_LAYER_NUM),
            l4 = L.Linear(FIRST_LAYER_NUM, SECOND_LAYER_NUM),
            l5 = L.Linear(SECOND_LAYER_NUM, SECOND_LAYER_NUM),
            l6 = L.Linear(SECOND_LAYER_NUM, OUTPUT_LAYER_NUM),
            bn1 = L.BatchNormalization(FIRST_LAYER_NUM),
            bn2 = L.BatchNormalization(FIRST_LAYER_NUM),
            bn3 = L.BatchNormalization(FIRST_LAYER_NUM),
            bn4 = L.BatchNormalization(SECOND_LAYER_NUM),
            bn5 = L.BatchNormalization(SECOND_LAYER_NUM),
            bn6 = L.BatchNormalization(SECOND_LAYER_NUM)
        )
        

    def __call__(self, x):
        x = Variable(x)
        # h1 = F.dropout(F.sigmoid(self.l1(x)), ratio=DROP_OUT_RATIO)
        # h2 = F.dropout(F.sigmoid(self.l2(h1)), ratio=DROP_OUT_RATIO)

        """ 
        h1 = F.dropout(F.relu(self.l1(x)), ratio=DROP_OUT_RATIO)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio=DROP_OUT_RATIO)
        h3 = F.dropout(F.relu(self.l3(h2)), ratio=DROP_OUT_RATIO)
        """ 
        h1 = F.dropout(F.relu(self.bn1(self.l1(x))), ratio=DROP_OUT_RATIO)
        h2 = F.dropout(F.relu(self.bn2(self.l2(h1))), ratio=DROP_OUT_RATIO)
        h3 = F.dropout(F.relu(self.bn3(self.l3(h2))), ratio=DROP_OUT_RATIO)
        h4 = F.dropout(F.relu(self.bn4(self.l4(h3))), ratio=DROP_OUT_RATIO)
        h5 = F.dropout(F.relu(self.bn5(self.l5(h4))), ratio=DROP_OUT_RATIO)

        return self.l6(h5)

        """
        super().__init__(
            l1 = L.Linear(None, FIRST_LAYER_NUM),
            l2 = L.Linear(FIRST_LAYER_NUM, FIRST_LAYER_NUM),
            l3 = L.Linear(FIRST_LAYER_NUM, FIRST_LAYER_NUM),
            l4 = L.Linear(FIRST_LAYER_NUM, SECOND_LAYER_NUM),
            l5 = L.Linear(SECOND_LAYER_NUM, SECOND_LAYER_NUM),
            l6 = L.Linear(SECOND_LAYER_NUM, OUTPUT_LAYER_NUM)
        )
        

    def __call__(self, x):
        x = Variable(x)
        h1 = F.dropout(F.relu(self.l1(x)), ratio=DROP_OUT_RATIO)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio=DROP_OUT_RATIO)
        h3 = F.dropout(F.relu(self.l3(h2)), ratio=DROP_OUT_RATIO)
        h4 = F.dropout(F.relu(self.l4(h3)), ratio=DROP_OUT_RATIO)
        h5 = F.dropout(F.relu(self.l5(h4)), ratio=DROP_OUT_RATIO)

        return self.l6(h5)
        """

    def backward(self, pred_label, y, optimizer):
        y = Variable(y)
        loss = F.softmax_cross_entropy(pred_label, y)
        loss.backward()
        optimizer.update()

        return loss


def prepare_model(is_retrain=True):
    # モデル準備
    model = NeuralNet()
    model.compute_accuracy = False 
    # optimizer = optimizers.MomentumSGD()
    # optimizer = optimizers.RMSprop()
    # optimizer = optimizers.AdaDelta()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    optimizer.add_hook(chainer.optimizer.WeightDecay(L2_WEIGHT))

    """
    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)
    xp = cuda.cupy
    """
    if all([os.path.exists(path) for path in [MODEL_PATH, STATE_PATH]]) and is_retrain:
        serializers.load_npz(MODEL_PATH, model)
        serializers.load_npz(STATE_PATH, optimizer)


    return model, optimizer

def calc_accuracy(model, data, label):
    # data_variable = Variable(data)    
    with chainer.using_config("train", False):
        y = model(data)
        y = F.softmax(y)
        pred_label = np.argmax(y.data, 1)
        acc = float(np.sum(pred_label == label) / pred_label.size)

    return acc, pred_label, y

def save_model_optimizer(model, optimizer):
    serializers.save_npz(MODEL_PATH, model)
    serializers.save_npz(STATE_PATH, optimizer)


