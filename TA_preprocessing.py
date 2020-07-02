import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd

from itertools             import combinations
from math                  import factorial
from datetime              import date
from sklearn.preprocessing import StandardScaler
from copy                  import deepcopy
from ta.volatility         import AverageTrueRange, BollingerBands           , KeltnerChannel # ATR
from ta.trend              import EMAIndicator    , MACD                     , AroonIndicator     , CCIIndicator
from ta.volume             import MFIIndicator    , ChaikinMoneyFlowIndicator, ForceIndexIndicator # FII
from ta.momentum           import ROCIndicator    , RSIIndicator             , WilliamsRIndicator , StochasticOscillator

def heatmap_corr(corr_matrix, col_names, save = False, title = 'Correlation Coefficient Between Variables', filename = None):
    corr_matrix = np.around(corr_matrix.to_numpy(), 3)
    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)

    ax.set_xticks(np.arange(len(col_names)))
    ax.set_yticks(np.arange(len(col_names)))

    ax.set_xticklabels(col_names)
    ax.set_yticklabels(col_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(col_names)):
        for j in range(len(col_names)):
            text = ax.text(j, i, corr_matrix[i, j],
                           ha="center", va="center", color="black", fontsize = 10)

    ax.set_title(title)
    fig.tight_layout()
    if save == True:
        plt.savefig(filename, dpi=1000)
    plt.show()

def plot_features(data, features = ['close']):
    output = np.zeros((np.shape(data)[0],)).tolist()
    tanggal = deepcopy(data['date']).to_numpy().astype(str)
    for i in range(len(output)):
        output[i] = date(int(tanggal[i][0:4]), int(tanggal[i][4:6]), int(tanggal[i][6:8]))

    for i in features:
        plt.plot_date(output, data[i],'-', linewidth = 0.9)
    plt.legend(features)
    plt.show()

def plot_all(multiple_company_prices, company_names, feature = 'close', save = False, title = 'Graph of Closing Price Over Time Across Mutliple Assets', filename = None):
    output = np.zeros((np.shape(multiple_company_prices[0])[0],)).tolist()
    tanggal = deepcopy(multiple_company_prices[0]['date']).to_numpy().astype(str)
    for i in range(len(output)):
        output[i] = date(int(tanggal[i][0:4]), int(tanggal[i][4:6]), int(tanggal[i][6:8]))

    for i, var in enumerate(multiple_company_prices):
        plt.plot_date(output, var[feature], '-', linewidth=0.7)
    plt.legend(company_names, markerscale = 50)
    plt.title(title)
    if save == True:
        plt.savefig(filename, dpi=1000)
    plt.show()

class transform(object):
    def __init__(self, data, split_info = None, dividend_info = None, train_ratio = None, company_names = None):
        self.data          = data
        self.split_info    = split_info
        self.dividend_info = dividend_info
        self.train_ratio   = train_ratio
        self.company_names = company_names

    def Standarinator(self):
        self.columns = self.data.columns

        self.index_train     = int(np.floor(np.shape(self.data)[0]*self.train_ratio))
        self.finalResult     = deepcopy(self.data)
        self.standarizedData = self.finalResult.iloc[:,1:-1]
        Scaler = StandardScaler()
        Scaler.fit(self.standarizedData.iloc[:self.index_train,:])

        self.finalResult.iloc[:,1:-1] = Scaler.transform(self.standarizedData)
        self.finalResult.columns = self.columns
        return self.finalResult

    def unsplitter(self):
        self.n_split = np.shape(self.split_info)[0]
        output = deepcopy(self.data)
        for i in range(self.n_split):
            date   = self.split_info[i,0]
            factor = self.split_info[i,1]
            index_date = np.where(self.data[:,0] == date)[0][0]
            output[index_date+1:,1:5] = self.data[index_date+1:,1:5]/factor
            output[index_date+1:,5:6] = self.data[index_date+1:,5:6]*factor

            self.data = output
        return output

    def div_adder(self):
        self.output = np.zeros((np.shape(self.data)[0], 1))
        for i in range(len(self.dividend_info)):
            index = np.where(self.data[:,0] == self.dividend_info[i,0])[0][0]-10
            if index < 0:
                self.output[0,0] = self.dividend_info[i,1]
            else:
                self.output[index,0] = self.dividend_info[i,1]
        return np.hstack((self.data,self.output))

    def OneHotCompany(self):
        self.col_names = self.data[0].columns
        self.insertion = np.where(self.col_names == 'dividend')[0][0]

        allText = []
        for i in range(len(self.data)):
            allText.append('if_' + self.company_names[i])

        for i in range(len(self.data)):
            for j in range(len(self.data)):
                text = 'if_' + self.company_names[j]
                if i == j:
                    self.data[i][text] = 1
                else:
                    self.data[i][text] = 0
        self.new_col_sequence = np.concatenate((self.col_names[:self.insertion+1], allText, self.col_names[self.insertion+1:])).tolist()
        for i in range(len(self.data)):
            self.data[i] = self.data[i].reindex(self.new_col_sequence, axis = 1)
        return self.data

    def logger(self):
        self.price_data  = self.data[['open','high','low','close']]
        self.output_data = deepcopy(self.price_data)
        for i in range(1,np.shape(self.output_data)[0]):
            self.output_data.iloc[i:i+1,:] = np.log(self.price_data.iloc[i:i+1,:].to_numpy()/self.price_data.iloc[i-1:i,:].to_numpy())
        self.data[['open','high','low','close']] = self.output_data
        return self.data

class produce(object):
    def applyIndicator(self, full_company_price):
        self.data = full_company_price

        high   = self.data['high']
        low    = self.data['low']
        close  = self.data['close']
        volume = self.data['volume']

        EMA12      = EMAIndicator(close, 12 , fillna = False)
        EMA30      = EMAIndicator(close, 20 , fillna = False)
        EMA60      = EMAIndicator(close, 60 , fillna = False)
        MACD1226   = MACD(close, 26, 12, 9 , fillna = False)
        MACD2452   = MACD(close, 52, 24, 18, fillna = False)
        ROC12      = ROCIndicator(close, 12, fillna = False)
        ROC30      = ROCIndicator(close, 30, fillna = False)
        ROC60      = ROCIndicator(close, 60, fillna = False)
        RSI14      = RSIIndicator(close, 14, fillna = False)
        RSI28      = RSIIndicator(close, 28, fillna = False)
        RSI60      = RSIIndicator(close, 60, fillna = False)
        AROON25    = AroonIndicator(close, 25, fillna = False)
        AROON50    = AroonIndicator(close, 50, fillna = False)
        AROON80    = AroonIndicator(close, 80, fillna = False)
        MFI14      = MFIIndicator(high, low, close, volume, 14, fillna = False)
        MFI28      = MFIIndicator(high, low, close, volume, 28, fillna = False)
        MFI80      = MFIIndicator(high, low, close, volume, 80, fillna = False)
        CCI20      = CCIIndicator(high, low, close, 20 , 0.015, fillna = False)
        CCI40      = CCIIndicator(high, low, close, 40 , 0.015, fillna = False)
        CCI100     = CCIIndicator(high, low, close, 100, 0.015, fillna = False)
        WILLR14    = WilliamsRIndicator(high, low, close, 14, fillna = False)
        WILLR28    = WilliamsRIndicator(high, low, close, 28, fillna = False)
        WILLR60    = WilliamsRIndicator(high, low, close, 60, fillna = False)
        BBANDS20   = BollingerBands(close, 20, 2, fillna = False)
        KC20       = KeltnerChannel(high, low, close, 20, 10, fillna=False)
        STOCH14    = StochasticOscillator(high, low, close, 14, 3 , fillna = False)
        STOCH28    = StochasticOscillator(high, low, close, 28, 6 , fillna = False)
        STOCH60    = StochasticOscillator(high, low, close, 60, 12, fillna = False)
        CMI20      = ChaikinMoneyFlowIndicator(high, low, close, volume, 20 , fillna = False)
        CMI40      = ChaikinMoneyFlowIndicator(high, low, close, volume, 40 , fillna = False)
        CMI100     = ChaikinMoneyFlowIndicator(high, low, close, volume, 100, fillna = False)

        self.data['ema12']          = (close - EMA12.ema_indicator())/close
        self.data['ema30']          = (close - EMA30.ema_indicator())/close
        self.data['ema60']          = (close - EMA60.ema_indicator())/close
        self.data['macd1226']       = MACD1226.macd() - MACD1226.macd_signal()
        self.data['macd2452']       = MACD2452.macd() - MACD2452.macd_signal()
        self.data['roc12']          = ROC12.roc()
        self.data['roc30']          = ROC30.roc()
        self.data['roc60']          = ROC60.roc()
        self.data['rsi14']          = RSI14.rsi()
        self.data['rsi28']          = RSI28.rsi()
        self.data['rsi60']          = RSI60.rsi()
        self.data['aroon25']        = AROON25.aroon_indicator()
        self.data['aroon50']        = AROON50.aroon_indicator()
        self.data['aroon80']        = AROON80.aroon_indicator()
        self.data['mfi14']          = MFI14.money_flow_index()
        self.data['mfi28']          = MFI28.money_flow_index()
        self.data['mfi80']          = MFI80.money_flow_index()
        self.data['cci20']          = CCI20.cci()
        self.data['cci40']          = CCI40.cci()
        self.data['cci100']         = CCI100.cci()
        self.data['willr14']        = WILLR14.wr()
        self.data['willr28']        = WILLR28.wr()
        self.data['willr60']        = WILLR60.wr()
        self.data['bband20up']      = (BBANDS20.bollinger_hband() - close)/close
        self.data['bband20down']    = (close - BBANDS20.bollinger_lband())/close
        self.data['stoch14']         = STOCH14.stoch()
        self.data['stoch28']         = STOCH28.stoch()
        self.data['stoch60']         = STOCH60.stoch()
        self.data['cmi20' ]          = CMI20.chaikin_money_flow()
        self.data['cmi40' ]          = CMI40.chaikin_money_flow()
        self.data['cmi100']          = CMI100.chaikin_money_flow()
        self.data['kc20up']         = (KC20.keltner_channel_hband() - close)/close
        self.data['kc20down']       = (close - KC20.keltner_channel_lband())/close
        return self.data

    def corrMatrix(self, companies, column = 'close'):
        self.companies = companies

        corr = self.companies[0][column]
        for i in range(1,len(self.companies)):
            corr = pd.concat((corr,self.companies[i][column]), axis = 1)
        return corr.corr()

    def applyLabel(self, company, n_window = 11):
        self.finalResult = company
        self.data    = company['close']
        self.nWindow = n_window
        self.output  = np.ones((np.shape(self.data)[0],)).tolist()
        for j in range(np.shape(self.data)[0]-self.nWindow+1):
            self.examinedWindow = self.data[j:j+self.nWindow]
            self.midWindowVal   = self.data[j+(self.nWindow-1)/2]

            self.maxWindow      = np.max(self.examinedWindow)
            self.minWindow      = np.min(self.examinedWindow)
            if self.midWindowVal == self.minWindow:
                self.output[j+int(np.around((self.nWindow-1)/2,0))] = 2 # Buy
            elif self.midWindowVal == self.maxWindow:
                self.output[j+int(np.around((self.nWindow-1)/2,0))] = 3 # Sell
        # for i in range(np.shape(self.output)[0]-1):
        #     self.output[i] = self.output[i+1]
        self.finalResult['label'] = self.output
        return self.finalResult

class aug(object):
    def __init__(self, data):
        self.data   = data
        self.nComp  = len(self.data)
        self.nObs   = np.shape(data[0])[0]
        self.nFeat  = np.shape(data[0])[1]

        self.insertionIndex = np.where(self.data[0].columns == 'if_BBNI')[0][0]

        self.dicData    = {}
        self.dicIndices = []
        for i in range(self.nComp):
            self.dicData[i] = self.data[i]
            self.dicIndices.append(i)

    def singleAug(self, target, nChosen):
        self.nCombs = int(factorial(self.nComp-1)/factorial(nChosen)/factorial(self.nComp-nChosen-1))
        self.output = np.zeros((self.nCombs, self.nObs, self.nFeat + 3*nChosen + len(self.data))) # value of 2 refers to 'high', 'low' and 'close'

        del self.dicIndices[target]
        self.indicesCombined = list(combinations(self.dicIndices, nChosen))

        for counter, i in enumerate(self.indicesCombined):
            self.toBeInserted = []
            self.markerToBeInserted = np.zeros((self.nObs,self.nComp))
            for j in i:
                self.toBeInserted.append(self.dicData[j][['high','low','close']].to_numpy(dtype = np.float64))
                self.markerToBeInserted[:,j]= 1
            self.toBeInserted = np.concatenate(self.toBeInserted, axis = 1)
            self.toBeInserted = np.concatenate([self.toBeInserted, self.markerToBeInserted], axis = 1)
            self.output[counter,:,:] = np.concatenate((self.dicData[target].iloc[:,0:self.insertionIndex+1].to_numpy(np.float64), self.toBeInserted, self.dicData[target].iloc[:,self.insertionIndex+1:].to_numpy(np.float64)), axis = 1)
        return self.output