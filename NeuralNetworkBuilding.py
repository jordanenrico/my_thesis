# By : Misael Jordan Enrico
# For structured notebooks of this file, visit :
# https://colab.research.google.com/drive/14MMWMOZMbOTN7vjWG_PVenlVRDPmRoGA?usp=sharing

import tensorflow        as tf
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import keras.backend     as K
import pickle            as pk

from copy        import deepcopy
from datetime    import date
from sklearn     import preprocessing
from tensorflow  import keras
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, classification_report

# -1 buy, 0 hold, 1 sell
# Simulasi finansial
def modelEvaluation(close_price, decision, risk_m = 0.5, risk_s = 1, interest = 0.001):
    money=np.zeros(len(close_price)+1)
    saham=np.zeros(len(close_price)+1)
    total=np.zeros(len(close_price)+1)
    holdp=np.zeros(len(close_price)+1)
    money[0]=100
    total[0]=100
    holdp[0]=100
    for i in range(len(close_price)):
        if decision[i]==-1:
            saham[i]+=risk_m*money[i]/close_price[i]
            money[i]-=risk_m*money[i]
        elif decision[i]==1:
            money[i]+=risk_s*saham[i]*close_price[i]
            saham[i]-=risk_s*saham[i]
        money[i+1]+=money[i]*np.exp(interest/365)
        saham[i+1]+=saham[i]
        total[i+1]+=money[i]+saham[i]*close_price[i]
        holdp[i]  =100*close_price[i]/close_price[0]
    return [total[:len(close_price)]-100, holdp[:len(close_price)]]

def rounder(data_labels, buy_boundary, sell_boundary):
    data = deepcopy(data_labels)
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            if data[i,j] < buy_boundary:
                data[i,j] = int(-1)
            elif data[i,j] > sell_boundary:
                data[i,j] = int(1)
            else:
                data[i,j] = int(0)
    return data

def buysellPlotter(harga, label, tanggal, saham = None):
    plt.figure(figsize=(13,4), dpi = 100)
    plt.plot_date(tanggal, harga, '-', linewidth = 1.5, color = 'orange')
    for count, i in enumerate(label):
        if i == -1:
            plt.plot_date(tanggal[count], harga[count], 'g.')
        elif i == 1:
            plt.plot_date(tanggal[count], harga[count], 'r.')
    if saham == None:
        title = 'Hasil Pemberian Saran dari Tanggal ' + str(tanggal[0]) + ' sampai dengan ' + str(tanggal[-1]) + '.'
    else:
        title = 'Hasil Pemberian Saran Saham ' + saham + ' dari Tanggal ' + str(tanggal[0]) + ' sampai dengan ' + str(tanggal[-1]) + '.'
    plt.title(title)
    plt.ylabel('Harga')
    plt.xlabel('Tanggal')
    plt.grid()
    plt.show()

alldata = np.load('BBCA.npy') # 10x3250x55 (14.3MB)
# normal_            6   0 -  5
# div                1        6
# closevol_comb_3x3  9   7 - 15
# ID_company_comb_6  5  16 - 20
# indicators_42     33  21 - 53
# label_1            1       54
# TOTAL             55

# 2925 VS 325

train_raw = alldata[:,:-324,:]
test_raw  = alldata[:,-337:,:]

concat_train = []
for i in range(np.shape(train_raw)[1]-14+1):
  toBeInserted = train_raw[:,i:i+14,:]
  concat_train.append(toBeInserted)
train_appended = np.concatenate(concat_train, axis = 0) # 29120x14x55, (179.37MB)

concat_test = []
for i in range(np.shape(test_raw)[1]-14+1):
  toBeInserted = test_raw[:,i:i+14,:]
  concat_test.append(toBeInserted)
test_appended = np.concatenate(concat_test, axis = 0) # 2990x14x55, (18.41MB)

train_modul1     = train_appended[:,: , 7:21]                    # 29120x14x14 (Ukuran array-nya)
train_modul2     = train_appended[:,: ,  0:7]                    # 29120x14x7
train_indicators = train_appended[:,-1,21:54]                    # 29120x33
train_label      = train_appended[:,-1,   54].astype(np.float32) - 1 # 29120xNone

test_modul1      = test_appended[:,: , 7:21]                    # 3250x14x14
test_modul2      = test_appended[:,: ,  0:7]                    # 3250x14x7
test_indicators  = test_appended[:,-1,21:54]                    # 3250x33
test_label       = test_appended[:,-1,   54].astype(np.float32) - 1 # 3250xNone
print(((pd.DataFrame(train_label))[0]).value_counts())
print(((pd.DataFrame(test_label))[0]).value_counts())

dummy = deepcopy(train_label.astype(np.float32))
for count, i in enumerate(train_label):
  if i == 1:
    dummy[count] = -1
  elif i == 2:
    dummy[count] = 1
train_label = dummy
del dummy

dummy = deepcopy(test_label.astype(np.float32))
for count, i in enumerate(test_label):
  if i == 1:
    dummy[count] = -1
  elif i == 2:
    dummy[count] = 1
test_label = dummy
del dummy

print(((pd.DataFrame(train_label))[0]).value_counts()) # 9.34%, 9.78% (Persentase label yang bernilai 1 dan 2, yaitu BUY dan SELL)
print(((pd.DataFrame(test_label))[0]).value_counts())  # 6.46%, 8.92%

# Define Regularizers
l2_regularizer = tf.keras.regularizers.l2(l = 0.1)

# Input LSTM
input_modul1 = keras.Input(shape = (14, 14), name = 'masukan_LSTM1')
input_modul2 = keras.Input(shape = (14, 7 ), name = 'masukan_LSTM2')

# LSTM Modules
LSTM_modul1 = keras.layers.LSTM(10,name = 'LSTM_modul1', dropout = 0.05)(input_modul1)
LSTM_modul2 = keras.layers.LSTM(17,name = 'LSTM_modul2', dropout = 0.05)(input_modul2)

# Gabung LSTM
concat_LSTM = keras.layers.concatenate([LSTM_modul1,LSTM_modul2], name = 'GabunganLSTM')

# MLP LSTM
MLP_LSTM = keras.layers.Dense(20,
                              activation = tf.nn.relu,
                              name = 'MLP_LSTM1',
                              kernel_regularizer = l2_regularizer)(concat_LSTM)

# Input Indikator
input_indicators = keras.Input(shape = (33, ), name = 'masukan_Indikator')

# MLP Indikator
MLP_indicators1 = keras.layers.Dense(10,
                                     activation = tf.nn.relu,
                                     name = 'MLP_Indikator1',
                                     kernel_regularizer = l2_regularizer)(input_indicators)

# Combine 2 MLPs
Input_MLP  = keras.layers.concatenate([MLP_LSTM, MLP_indicators1], name = 'GabunganMLP')

# Final MLP
MLP_Layer1 = keras.layers.Dense(25,
                                activation         = tf.nn.relu,
                                name               = 'HiddenLayer1',
                                kernel_regularizer = l2_regularizer)(Input_MLP)
MLP_Layer1 = keras.layers.Dropout(0.05, name = 'DropoutLayer1')(MLP_Layer1)

MLP_Layer2 = keras.layers.Dense(20,
                                activation         = tf.nn.relu,
                                name               = 'HiddenLayer2',
                                kernel_regularizer = l2_regularizer)(MLP_Layer1)
MLP_Layer2 = keras.layers.Dropout(0.05, name = 'DropoutLayer2')(MLP_Layer2)

MLP_output = keras.layers.Dense(1,
                                activation         = tf.nn.tanh,
                                name               = 'LayerTerakhir',
                                kernel_regularizer = l2_regularizer)(MLP_Layer2)
MLP_output = keras.layers.Dropout(0.05, name = 'DropoutOutput')(MLP_output)

model = keras.Model(inputs  = [input_modul1, input_modul2, input_indicators],
                    outputs = MLP_output)

print(model.summary())

def customLoss(y_true, y_pred):
    squared_difference = tf.multiply(tf.square(y_true - y_pred), tf.abs(y_true*14+1)+2)
    return tf.reduce_mean(squared_difference)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
              loss      = customLoss,
              metrics   = ['accuracy'])

train_dict_X    = {'masukan_LSTM1'    :     train_modul1,
                   'masukan_LSTM2'    :     train_modul2,
                   'masukan_Indikator': train_indicators}
test_dict_X     = {'masukan_LSTM1'    :      test_modul1,
                   'masukan_LSTM2'    :      test_modul2,
                   'masukan_Indikator':  test_indicators}

history = model.fit(train_dict_X,
                    train_label,
                    epochs          = 1,
                    batch_size      = 300,
                    validation_data = (test_dict_X, test_label))

pred_train_raw = model.predict([train_modul1, train_modul2, train_indicators])
pred_test_raw  = model.predict([test_modul1,  test_modul2,  test_indicators])

print(np.max(pred_train_raw))
print(np.min(pred_train_raw))
print(np.max(pred_test_raw))
print(np.min(pred_test_raw))

pred_train = rounder(pred_train_raw, -0.5, 0.5)
pred_test  = rounder(pred_test_raw, -0.5, 0.5)

print(confusion_matrix(train_label, pred_train, labels = [-1,0,1]))
print(confusion_matrix(test_label, pred_test, labels = [-1,0,1]))

print(classification_report(train_label, pred_train, labels = [-1,0,1]))
print(classification_report(test_label, pred_test, labels = [-1,0,1]))
print(np.shape(pred_train))
print(np.shape(pred_test))

label_test_voted = np.zeros((324,10))
for i in range(324):
  start = i*10
  end   = start + 10
  label_test_voted[i:i+1,:] = pred_test[start:end,0]

label_train_voted = np.zeros((2898,10))
for i in range(2898):
  start = i*10
  end   = start + 10
  label_train_voted[i:i+1,:] = pred_train[start:end,0]

label_real  = alldata[0,:,-1]

label_train_voted = mode(label_train_voted, axis = 1)[0].reshape([2898,])
label_test_voted = mode(label_test_voted, axis = 1)[0].reshape([324,])

label_train_real  = label_real[13:-324]
label_test_real  = label_real[-324:]

dummy = deepcopy(label_train_real.astype(np.int8))
for count, i in enumerate(label_train_real):
  if i == 2:
    dummy[count] = -1
  elif i == 3:
    dummy[count] = 1
  else:
    dummy[count] = 0
label_train_real = dummy
del dummy

dummy = deepcopy(label_test_real.astype(np.int8))
for count, i in enumerate(label_test_real):
  if i == 2:
    dummy[count] = -1
  elif i == 3:
    dummy[count] = 1
  else:
    dummy[count] = 0
label_test_real = dummy
del dummy

harga   = np.load('hargaBBCA.npy')[:,1]
tanggal = np.load('hargaBBCA.npy')[:,0].astype(str)
output  = []
for i in range(len(harga)):
  output.append(date(int(tanggal[i][0:4]), int(tanggal[i][4:6]), int(tanggal[i][6:8])))

harga_test  = harga[-324:]
harga_train = harga[13:-324]

tanggal_test  = output[-324:]
tanggal_train = output[13:-324]

harga_test_new = (harga_test-np.max(harga_test))*10/10+np.max(harga_test)
percentBNH = []
for i in range(len(harga_test_new)):
  percentBNH.append(harga_test_new[i]/harga_test_new[0]*100-100)
print(tanggal_test[0])
print(tanggal_test[-1])
print(tanggal_train[0])
print(tanggal_train[-1])

saham = 'BBCA'
buysellPlotter(harga_test_new, label_test_voted, tanggal_test, saham = saham)

# 75 25

buy_ratio  = 0.75
sell_ratio = 1 - buy_ratio
simulated_return_test = modelEvaluation(harga_test, label_test_voted, risk_m = buy_ratio, risk_s = sell_ratio)[0]
latest_return = simulated_return_test[-1]
plt.figure(figsize=(13,4), dpi = 100)
plt.plot(tanggal_test, simulated_return_test, '-', linewidth = 1.5, color = 'orange')
plt.plot(tanggal_test, percentBNH, '-', linewidth = 1.5, color = 'blue')
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.grid(which = 'major', b=True)
plt.title('Plot Persentase Keuntungan Saham ' + saham + ' dengan Buy Ratio = ' + str(buy_ratio) + ' dan Sell Ratio = ' + str(sell_ratio))
plt.xlabel('Tanggal')
plt.ylabel('Persen Keuntungan')
plt.legend(['Strategi dengan Saran Model','Strategi B&H','Keuntungan akhir model = ' + str(latest_return)[0:5] + '%','Keuntungan akhir B&H    = ' + str(percentBNH[-1])[0:5] + '%'])
plt.show()

# 50 50

buy_ratio  = 0.50
sell_ratio = 1 - buy_ratio
simulated_return_test = modelEvaluation(harga_test, label_test_voted, risk_m = buy_ratio, risk_s = sell_ratio)[0]
latest_return = simulated_return_test[-1]
plt.figure(figsize=(13,4), dpi = 100)
plt.plot(tanggal_test, simulated_return_test, '-', linewidth = 1.5, color = 'orange')
plt.plot(tanggal_test, percentBNH, '-', linewidth = 1.5, color = 'blue')
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.grid(which = 'major', b=True)
plt.title('Plot Persentase Keuntungan Saham ' + saham + ' dengan Buy Ratio = ' + str(buy_ratio) + ' dan Sell Ratio = ' + str(sell_ratio))
plt.xlabel('Tanggal')
plt.ylabel('Persen Keuntungan')
plt.legend(['Strategi dengan Saran Model','Strategi B&H','Keuntungan akhir model = ' + str(latest_return)[0:5] + '%','Keuntungan akhir B&H    = ' + str(percentBNH[-1])[0:5] + '%'])
plt.show()

# 25 75

buy_ratio  = 0.25
sell_ratio = 1 - buy_ratio
simulated_return_test = modelEvaluation(harga_test, label_test_voted, risk_m = buy_ratio, risk_s = sell_ratio)[0]
latest_return = simulated_return_test[-1]
plt.figure(figsize=(13,4), dpi = 100)
plt.plot(tanggal_test, simulated_return_test, '-', linewidth = 1.5, color = 'orange')
plt.plot(tanggal_test, percentBNH, '-', linewidth = 1.5, color = 'blue')
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.grid(which = 'major', b=True)
plt.title('Plot Persentase Keuntungan Saham ' + saham + ' dengan Buy Ratio = ' + str(buy_ratio) + ' dan Sell Ratio = ' + str(sell_ratio))
plt.xlabel('Tanggal')
plt.ylabel('Persen Keuntungan')
plt.legend(['Strategi dengan Saran Model','Strategi B&H','Keuntungan akhir model = ' + str(latest_return)[0:5] + '%','Keuntungan akhir B&H    = ' + str(percentBNH[-1])[0:5] + '%'])
plt.show()

# 100 100

buy_ratio  = 1
sell_ratio = 1
simulated_return_test = modelEvaluation(harga_test, label_test_voted, risk_m = buy_ratio, risk_s = sell_ratio)[0]
latest_return = simulated_return_test[-1]
plt.figure(figsize=(13,4), dpi = 100)
plt.plot(tanggal_test, simulated_return_test, '-', linewidth = 1.5, color = 'orange')
plt.plot(tanggal_test, percentBNH, '-', linewidth = 1.5, color = 'blue')
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.plot(tanggal_test[0], simulated_return_test[0], linewidth = 0)
plt.grid(which = 'major', b=True)
plt.title('Plot Persentase Keuntungan Saham ' + saham + ' dengan Buy Ratio = ' + str(buy_ratio) + ' dan Sell Ratio = ' + str(sell_ratio))
plt.xlabel('Tanggal')
plt.ylabel('Persen Keuntungan')
plt.legend(['Strategi dengan Saran Model','Strategi B&H','Keuntungan akhir model = ' + str(latest_return)[0:5] + '%','Keuntungan akhir B&H    = ' + str(percentBNH[-1])[0:5] + '%'])
plt.show()