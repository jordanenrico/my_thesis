import numpy             as np
import pandas            as pd
import TA_preprocessing  as procie

stock_names        = ['BBCA', 'BBRI', 'TLKM', 'BMRI', 'UNVR', 'ASII', 'HMSP', 'BBNI']
stock_names_split  = ['BBCA', 'BBRI', 'TLKM', 'BMRI', 'UNVR', 'ASII', 'HMSP']

for i in ['Read and preliminary data processing']:
    # Read stock data
    BBCA = pd.read_csv('all_raw_data/stock_data/BBCA.csv', header = None).to_numpy()
    BBRI = pd.read_csv('all_raw_data/stock_data/BBRI.csv', header = None).to_numpy()
    TLKM = pd.read_csv('all_raw_data/stock_data/TLKM.csv', header = None).to_numpy()
    BMRI = pd.read_csv('all_raw_data/stock_data/BMRI.csv', header = None).to_numpy()
    UNVR = pd.read_csv('all_raw_data/stock_data/UNVR.csv', header = None).to_numpy()
    ASII = pd.read_csv('all_raw_data/stock_data/ASII.csv', header = None).to_numpy()
    HMSP = pd.read_csv('all_raw_data/stock_data/HMSP.csv', header = None).to_numpy()
    BBNI = pd.read_csv('all_raw_data/stock_data/BBNI.csv', header = None).to_numpy()

    # Read dividend information
    BBCA_DIV = pd.read_csv('all_raw_data/dividend_info/BBCA_DIV.csv', header = None).to_numpy()
    BBRI_DIV = pd.read_csv('all_raw_data/dividend_info/BBRI_DIV.csv', header = None).to_numpy()
    TLKM_DIV = pd.read_csv('all_raw_data/dividend_info/TLKM_DIV.csv', header = None).to_numpy()
    BMRI_DIV = pd.read_csv('all_raw_data/dividend_info/BMRI_DIV.csv', header = None).to_numpy()
    UNVR_DIV = pd.read_csv('all_raw_data/dividend_info/UNVR_DIV.csv', header = None).to_numpy()
    ASII_DIV = pd.read_csv('all_raw_data/dividend_info/ASII_DIV.csv', header = None).to_numpy()
    HMSP_DIV = pd.read_csv('all_raw_data/dividend_info/HMSP_DIV.csv', header = None).to_numpy()
    BBNI_DIV = pd.read_csv('all_raw_data/dividend_info/BBNI_DIV.csv', header = None).to_numpy()

    # Read stock split information and adjust stock data using stock split information
    BBCA_SS = pd.read_csv('all_raw_data/stock_split_info/BBCA_SS.csv', header = None).to_numpy()
    BBCA    = procie.transform(BBCA, split_info = BBCA_SS).unsplitter()
    BBRI_SS = pd.read_csv('all_raw_data/stock_split_info/BBRI_SS.csv', header = None).to_numpy()
    BBRI    = procie.transform(BBRI, split_info = BBRI_SS).unsplitter()
    TLKM_SS = pd.read_csv('all_raw_data/stock_split_info/TLKM_SS.csv', header = None).to_numpy()
    TLKM    = procie.transform(TLKM, split_info = TLKM_SS).unsplitter()
    BMRI_SS = pd.read_csv('all_raw_data/stock_split_info/BMRI_SS.csv', header = None).to_numpy()
    BMRI    = procie.transform(BMRI, split_info = BMRI_SS).unsplitter()
    UNVR_SS = pd.read_csv('all_raw_data/stock_split_info/UNVR_SS.csv', header = None).to_numpy()
    UNVR    = procie.transform(UNVR, split_info = UNVR_SS).unsplitter()
    ASII_SS = pd.read_csv('all_raw_data/stock_split_info/ASII_SS.csv', header = None).to_numpy()
    ASII    = procie.transform(ASII, split_info = ASII_SS).unsplitter()
    HMSP_SS = pd.read_csv('all_raw_data/stock_split_info/HMSP_SS.csv', header = None).to_numpy()
    HMSP    = procie.transform(HMSP, split_info = HMSP_SS).unsplitter()

    # Rearrange dates from prior to latter date
    BBCA = np.flip(BBCA, axis = 0)
    BBRI = np.flip(BBRI, axis = 0)
    TLKM = np.flip(TLKM, axis = 0)
    BMRI = np.flip(BMRI, axis = 0)
    UNVR = np.flip(UNVR, axis = 0)
    ASII = np.flip(ASII, axis = 0)
    HMSP = np.flip(HMSP, axis = 0)
    BBNI = np.flip(BBNI, axis = 0)

    # Add dividend information to stock data
    BBCA = procie.transform(BBCA, dividend_info = BBCA_DIV).div_adder()
    BBRI = procie.transform(BBRI, dividend_info = BBRI_DIV).div_adder()
    TLKM = procie.transform(TLKM, dividend_info = TLKM_DIV).div_adder()
    BMRI = procie.transform(BMRI, dividend_info = BMRI_DIV).div_adder()
    UNVR = procie.transform(UNVR, dividend_info = UNVR_DIV).div_adder()
    ASII = procie.transform(ASII, dividend_info = ASII_DIV).div_adder()
    HMSP = procie.transform(HMSP, dividend_info = HMSP_DIV).div_adder()
    BBNI = procie.transform(BBNI, dividend_info = BBNI_DIV).div_adder()

    # Convert stock data in numpy array into pandas DataFrame and add column names
    column_names = ['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'dividend']
    BBCA = pd.DataFrame(BBCA)
    BBCA.columns = column_names
    BBRI = pd.DataFrame(BBRI)
    BBRI.columns = column_names
    TLKM = pd.DataFrame(TLKM)
    TLKM.columns = column_names
    BMRI = pd.DataFrame(BMRI)
    BMRI.columns = column_names
    UNVR = pd.DataFrame(UNVR)
    UNVR.columns = column_names
    ASII = pd.DataFrame(ASII)
    ASII.columns = column_names
    HMSP = pd.DataFrame(HMSP)
    HMSP.columns = column_names
    BBNI = pd.DataFrame(BBNI)
    BBNI.columns = column_names


# Calculates correlation coefficient
corr  = procie.produce().corrMatrix([BBCA, BBRI, TLKM, BMRI, UNVR, ASII, HMSP, BBNI],'close') # All stocks
corr6 = procie.produce().corrMatrix([BBCA,BBRI,TLKM,BMRI,UNVR,BBNI],                 'close') # 6 highest correlation coefficient

# Plots correlation coefficient heatmap
procie.heatmap_corr(corr,
                    stock_names,
                    save = False,
                    title = 'Koefisien Korelasi Antara Harga Penutupan Saham Beberapa Perusahaan',
                    filename = 'corr.png')
# Plots correlation coefficient heatmap (6 highest)
procie.heatmap_corr(corr6,
                    ['BBCA', 'BBRI','TLKM','BMRI','UNVR','BBNI'],
                    save = False,
                    title = 'Koefisien Korelasi Antara Harga Penutupan Saham dari 6 Perusahaan',
                    filename = 'corr6.png')
# Plots stock closing price
procie.plot_all([BBCA, BBRI, TLKM, BMRI, UNVR, ASII, HMSP, BBNI],
                stock_names,
                'close',
                save = False,
                title = 'Grafik Harga Penutupan Beberapa Saham Perusahaan',
                filename = 'closing.png')
# Plots stock closing price (6 highest correlation coefficient)
procie.plot_all([BBCA, BBRI, TLKM, BMRI, UNVR, BBNI],
                ['BBCA', 'BBRI', 'TLKM', 'BMRI', 'UNVR', 'BBNI'],
                'close',
                save = False,
                title = 'Grafik Harga Penutupan 6 Saham Perusahaan yang Berkorelasi Tinggi',
                filename = 'closing6.png')

del ASII, HMSP, ASII_DIV, HMSP_DIV # Deleting H.M.Sampoerna and Astra International

# Applies technical indicators
BBCA = procie.produce().applyIndicator(BBCA)
BBRI = procie.produce().applyIndicator(BBRI)
BMRI = procie.produce().applyIndicator(BMRI)
BBNI = procie.produce().applyIndicator(BBNI)
UNVR = procie.produce().applyIndicator(UNVR)
TLKM = procie.produce().applyIndicator(TLKM) # 3455x41

# Generates heuristic trading desicions for several trading days
n_window = 11
BBCA = procie.produce().applyLabel(BBCA, n_window = n_window)
BBRI = procie.produce().applyLabel(BBRI, n_window = n_window)
BMRI = procie.produce().applyLabel(BMRI, n_window = n_window)
BBNI = procie.produce().applyLabel(BBNI, n_window = n_window)
UNVR = procie.produce().applyLabel(UNVR, n_window = n_window)
TLKM = procie.produce().applyLabel(TLKM, n_window = n_window) # 3455x42

# Plots selected technical indicator
procie.plot_features(BBCA, ['close','kc20up','kc20down'])

# Stores date and closing price of Bank Central Asia
BBCA_real_close = BBCA[['date','close']]

# Transforms price into log-return
BBCA            = procie.transform(BBCA).logger()
BBRI            = procie.transform(BBRI).logger()
TLKM            = procie.transform(TLKM).logger()
BMRI            = procie.transform(BMRI).logger()
UNVR            = procie.transform(UNVR).logger()
BBNI            = procie.transform(BBNI).logger() # 3455x42, 4price_LOGRETURN, 8main+33indicators+label
BBCA_real_close = BBCA_real_close.drop(labels = range(200), axis = 0).reset_index(drop = True)

# Drops first 200 observations
BBCA = BBCA.drop(labels = range(200), axis = 0).reset_index(drop = True)
BBRI = BBRI.drop(labels = range(200), axis = 0).reset_index(drop = True)
BMRI = BMRI.drop(labels = range(200), axis = 0).reset_index(drop = True)
BBNI = BBNI.drop(labels = range(200), axis = 0).reset_index(drop = True)
UNVR = UNVR.drop(labels = range(200), axis = 0).reset_index(drop = True)
TLKM = TLKM.drop(labels = range(200), axis = 0).reset_index(drop = True)  #3255 x 42, 8main+33indicators+label
BBCA_real_close = BBCA_real_close.drop(labels = range(3235,3255), axis = 0).reset_index(drop = True)
BBCA_real_close = BBCA_real_close.drop(labels = range(200), axis = 0).reset_index(drop = True)

# Drops 20 latest observations
BBCA            = BBCA.drop(labels = range(3235,3255), axis = 0).reset_index(drop = True)
BBRI            = BBRI.drop(labels = range(3235,3255), axis = 0).reset_index(drop = True)
BMRI            = BMRI.drop(labels = range(3235,3255), axis = 0).reset_index(drop = True)
BBNI            = BBNI.drop(labels = range(3235,3255), axis = 0).reset_index(drop = True)
UNVR            = UNVR.drop(labels = range(3235,3255), axis = 0).reset_index(drop = True)
TLKM            = TLKM.drop(labels = range(3235,3255), axis = 0).reset_index(drop = True)  #3235 x 42, 8main+33indicators+label

# Standarizes data by taking train-test split into account
train_ratio = 0.9 # 2911 observations [31 October 2006] [16 November 2018] , 324 latest observations [17 November 2018] [11 February 2020]
BBCA = procie.transform(BBCA, train_ratio = train_ratio).Standarinator()
BBRI = procie.transform(BBRI, train_ratio = train_ratio).Standarinator()
BMRI = procie.transform(BMRI, train_ratio = train_ratio).Standarinator()
BBNI = procie.transform(BBNI, train_ratio = train_ratio).Standarinator()
UNVR = procie.transform(UNVR, train_ratio = train_ratio).Standarinator()
TLKM = procie.transform(TLKM, train_ratio = train_ratio).Standarinator() #3235 x 42, 8main+33indicators+label, NORMALIZED except first and last column

# Drops date
BBCA = BBCA.drop(labels = ['date'], axis = 1)
BBRI = BBRI.drop(labels = ['date'], axis = 1)
BMRI = BMRI.drop(labels = ['date'], axis = 1)
BBNI = BBNI.drop(labels = ['date'], axis = 1)
UNVR = UNVR.drop(labels = ['date'], axis = 1)
TLKM = TLKM.drop(labels = ['date'], axis = 1) #3250 x 41, 7main+33indicators+label

# Adds company name (Encoded)
BBCA, BBRI, TLKM, BMRI, UNVR, BBNI = procie.transform(data          = [BBCA,BBRI,TLKM,BMRI,UNVR,BBNI],
                                                      company_names = ['BBCA','BBRI','TLKM','BMRI','UNVR','BBNI']).OneHotCompany() #3250 x 47, 7main+6if_Company+33indicators+label

# Applying combinations https://www.sciencedirect.com/science/article/abs/pii/S0957417418304342
BBCA_combined = procie.aug(data = [BBCA,BBRI,BMRI,BBNI,UNVR,TLKM]).singleAug(target = 0, nChosen = 3) # 10x3250x62
'''
BBRI_combined = procie.aug(data = [BBCA,BBRI,BMRI,BBNI,UNVR,TLKM]).singleAug(target = 1, nChosen = 3)
BMRI_combined = procie.aug(data = [BBCA,BBRI,BMRI,BBNI,UNVR,TLKM]).singleAug(target = 2, nChosen = 3)
BBNI_combined = procie.aug(data = [BBCA,BBRI,BMRI,BBNI,UNVR,TLKM]).singleAug(target = 3, nChosen = 3)
UNVR_combined = procie.aug(data = [BBCA,BBRI,BMRI,BBNI,UNVR,TLKM]).singleAug(target = 4, nChosen = 3)
TLKM_combined = procie.aug(data = [BBCA,BBRI,BMRI,BBNI,UNVR,TLKM]).singleAug(target = 5, nChosen = 3)
ALLDATA = np.concatenate((BBCA_combined,BBRI_combined,BMRI_combined,BBNI_combined,UNVR_combined,TLKM_combined), axis = 0) # 60x3250x62 (96.72MB)
np.save('allcompany.npy', ALLDATA)
'''
# normal             6       0  -  5
# div                1             6
# ID_company         6       7  - 12
# closevol_comb_3x3  9      13  - 21
# ID_company_comb_6  6      22  - 27
# indicators_42     33      28  - 60
# label_1            1            61
# TOTAL             62

# Prepares for the final output
partition_1 = BBCA_combined[:, :, 0:7]   # Basic BBCA info
partition_2 = BBCA_combined[:, :, 13:22] # Combined augmented data
partition_3 = BBCA_combined[:, :, 23:62] # ID_company + technical indicator + label
BBCA = np.concatenate((partition_1, partition_2, partition_3), axis = 2) # 10x3250x55 (14.3MB)
# normal_            6
# div                1
# closevol_comb_3x3  9
# ID_company_comb_6  5
# indicators_42     33
# label_1            1
# TOTAL             55

# Final output
np.save('BBCA.npy', BBCA) # Main data with log-return price
np.save('hargaBBCA.npy', BBCA_real_close) # Date and real closing price for trading simulation