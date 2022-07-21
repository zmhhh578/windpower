import numpy as np
import pandas as pd

pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)
import meta_util
import meta_learner
import 数据处理
data=数据处理.data_sum
Y_train,Y_test,X_train_y,X_train_p,X_test_y,X_test_p=meta_util.data_mehtalearning_prepare(data,7)
basetrainpool,basetestpool=meta_util.get_prediction('pooling',7)
basetrain,basetest=meta_util.get_prediction('individual',7)

base_train_pool,base_test_pool,bias_pool=meta_util.base_forecast_meta_prepare(basetrainpool,basetestpool,Y_train)
base_train,base_test,bias=meta_util.base_forecast_meta_prepare(basetrain,basetest,Y_train)


def acc_h(pred,real,base):
    sMAPE=np.mean(np.abs(real-pred)/np.abs(real+pred),axis=0)*100
    arelmae=pd.DataFrame((np.abs(pred-real)/np.abs(base-real))).apply(lambda x:np.exp(np.log(x[np.isfinite(np.log(x))]).mean()),axis=0).values
    acc_result = pd.DataFrame(index=['sMAPE', 'arelmae'], columns=[str(i) + 'Day' for i in range(1, 8)])
    acc_result.iloc[0, :] = sMAPE
    acc_result.iloc[1, :] = arelmae

    return acc_result

def acc_all(pred,real,base):
    def g_mean(x):
        return np.exp(np.log(x[np.isfinite(np.log(x))]).mean())
    ME=((np.sum(pred-real,axis=1)/np.sum(real,axis=1))*100).mean()
    sMAPE=(np.mean(np.abs(real-pred)/np.abs(real+pred),axis=0)*100).mean()
    arelmae=g_mean(np.mean(np.abs(pred-real),axis=0)/np.mean(np.abs(base-real),axis=0))
    acc_result = pd.DataFrame(index=['sMAPE', 'arelmae','ME'], columns=[str(1)+'-'+str(7)+'Day'])
    acc_result.iloc[0, :] = sMAPE
    acc_result.iloc[1, :] = arelmae
    acc_result.iloc[2,:]=ME
    return acc_result

def acc_best(pred,real):
    acc_mse = np.zeros(shape=(pred.shape[0], pred.shape[1]))
    for i in range(pred.shape[1]):
        acc_mse[:, i] = np.mean(np.square(pred[:, i, :] - real), axis=1)
    acc_min = np.min(acc_mse, axis=1)
    #转化为 1和0
    acc_one =pd.DataFrame(acc_mse).apply(lambda x: x == acc_min, axis=0).astype('int').values

    return acc_one

def acc_err(pred,real):
    acc_mse=np.zeros(shape=(pred.shape[0],pred.shape[1]))
    for i in range(pred.shape[1]):
        acc_mse[:,i]=np.mean(np.abs(pred[:,i,:]-real),axis=-1)
    return acc_mse