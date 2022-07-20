#元学习部分
import numpy as np
import  tensorflow
from keras import Model
from keras.layers import GlobalAveragePooling1D,Dense,Permute,multiply,Conv1D,ReLU,Dropout,RepeatVector,Lambda,concatenate,Input
import keras.backend as K
def scale_mse(scaler):

    def wind_loss(y_true,y_pred):
    #return K.mean((K.mean(K.square(y_pred-y_true),axis=-1,keepdims=True)))
        return K.mean(K.mean(K.square(y_pred-y_true),axis=-1,keepdims=True)/scaler)
    #return wind_loss
    return wind_loss

def fcn_block(tensor):
    def squeeze_excite_block(tensor, ratio=16):
        init = tensor
        filters = init.shape[2]

        se = GlobalAveragePooling1D()(init)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        x = multiply([init, se])

        return x

    fcn = Conv1D(64, 2, padding='same', kernel_initializer='he_normal',activation='relu')(tensor)
    #fcn = ReLU()(fcn)
    fcn = squeeze_excite_block(fcn)
    fcn = Conv1D(128, 4, padding='same', kernel_initializer='he_normal',activation='relu')(fcn)
    #fcn = ReLU()(fcn)
    fcn = squeeze_excite_block(fcn)
    fcn = Conv1D(64, 8, padding='same', kernel_initializer='he_normal',activation='relu')(fcn)
    #fcn = ReLU()(fcn)
    fcn = GlobalAveragePooling1D()(fcn)
    return fcn


def M0(Y_shape, P_shape, mds, horizons, train_input, Y_train, test_input, Y_test, XX=True,ep=100):
    ##这个元学习器用来训练 M0 M2 M3 and M4
    ##XX:是否在影响因素中提取特征
    ##Y_shape:目标变量的shape
    ##P_shape:影响因素的shape
    ##mds:基础预测器的数量
    ##horizons:预测的范围
    X_input_y = Input(shape=Y_shape)
    X_input_p = Input(shape=P_shape)
    X_layer_y = fcn_block(X_input_y)
    X_layer_p = fcn_block(X_input_p)
    y_input = Input(shape=(mds, horizons))
    L_input=Input(shape=(None,))
    if XX:
        x = concatenate([X_layer_y, X_layer_p])
        X_layer = Dropout(0.9)(x)
        X_layer = Dense(units=mds, activation='softmax')(X_layer)
        X_layer = RepeatVector(horizons)(X_layer)
    else:
        X_layer=Dropout(0.9)(X_layer_y)
        X_layer = Dense(units=mds, activation='softmax')(X_layer)
        X_layer = RepeatVector(horizons)(X_layer)

    y_input_p = Permute([2, 1])(y_input)

    def multi_sum(args):
        return K.sum(args, axis=-1)

    output = multiply(inputs=[y_input_p, X_layer])
    output = Lambda(multi_sum)(output)
    model = Model([X_input_y, X_input_p, y_input,L_input], output)
    model.compile(optimizer='adam', loss=scale_mse(L_input))
    model.fit(train_input, Y_train, validation_data=[test_input, Y_test], epochs=ep, batch_size=512)
    return model
### M5
def M5(y_shape,p_shape,mds,train_input,Y_train_bestone,test_input,Y_test_bestone,ep=100):

    X_input_y=Input(shape=y_shape)
    X_input_p=Input(shape=p_shape)
    X_layer_y = fcn_block(X_input_y)
    X_layer_p = fcn_block(X_input_p)

    x=concatenate([X_layer_y,X_layer_p])
    X_layer = Dropout(0.8)(x)
    X_layer = Dense(units=mds, activation='softmax')(X_layer)
    model=Model([X_input_y,X_input_p],X_layer)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
    model.fit(train_input,Y_train_bestone,validation_data=[test_input,Y_test_bestone],epochs=ep,batch_size=4096)
    return model
#M6
def M6(y_shape,p_shape,mds,train_input,Y_train_err,test_input,Y_test_err,ep=50):
    X_input_y = Input(shape=y_shape)
    X_input_p = Input(shape=p_shape)
    X_layer_y = fcn_block(X_input_y)
    X_layer_p = fcn_block(X_input_p)
    x = concatenate([X_layer_y, X_layer_p])
    X_layer = Dropout(0.8)(x)
    X_layer = Dense(units=mds, activation='relu')(X_layer)
    model=Model([X_input_y,X_input_p],X_layer)
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(train_input,Y_train_err,validation_data=[test_input,Y_test_err],epochs=ep,batch_size=4096)
    return model



