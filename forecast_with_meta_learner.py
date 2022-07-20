import numpy as np
import pandas as pd
import meta_util
import meta_learner
Y_train,Y_test,X_train_y,X_train_p,X_test_y,X_test_p=meta_util.data_mehtalearning_prepare(data,7)
pd.DataFrame(Y_train).to_csv('/Users/zhongming/Desktop/windresult/Y_train.csv')
pd.DataFrame(Y_test).to_csv('/Users/zhongming/Desktop/windresult/Y_test.csv')

basetrain,basetest=meta_util.get_prediction('mixed',7)
base_train,base_test,bias_all=meta_util.base_forecast_meta_prepare(basetrain,basetest,Y_train)
# L_train=meta_util.get_scaler(base_train,Y_train)
# L_test=meta_util.get_scaler(base_test,Y_test)
horizons=base_train.shape[2]
mds=base_train.shape[1]
y_shape=X_train_y.shape[1:3]
p_shape=X_train_p.shape[1:3]
L_train=meta_util.get_scaler(base_train,Y_train)
print(L_train.shape)
L_test=meta_util.get_scaler(base_test,Y_test)
train_input=[X_train_y,X_train_p,base_train,L_train]
test_input=[X_test_y,X_test_p,base_test,L_test]
nn=10
M0_forecast=Y_test
predictions=np.zeros(shape=(Y_test.shape[0],Y_test.shape[1],nn))
for i in range(nn):
    model=meta_learner.M0(y_shape,p_shape,mds,horizons,train_input,Y_train,test_input,Y_test)
    predictions[:,:,i]=model.predict(test_input).reshape(-1,7)
for i in range(7):
    M0_forecast[:,i]=np.mean(predictions[:,i,:],axis=1)
