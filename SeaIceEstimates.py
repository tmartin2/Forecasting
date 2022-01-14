# Sea Ice Extent in 2023, 2025, 2030
# https://www.metaculus.com/questions/7939/minimum-antarctic-sea-ice-extent-in-2023/
# Author: Trevor Martin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA #https://www.statsmodels.org/stable/api.html
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from scipy import stats
#import pmdarima as pm
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman"}
)
# look at https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-two/


def create_ds(ds, look_back=1):
    dX, dY = [], []
    for i in range(len(ds)-look_back-1):
        p = ds[i:(i+look_back), 0]
        dX.append(p)
        dY.append(ds[i+look_back, 0])
    return np.array(dX), np.array(dY)

def use_BiLSTM(years, ts_data, name):
    # over LSTM: https://par.nsf.gov/servlets/purl/10186554
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds = scaler.fit_transform(np.asarray(ts_data).reshape(-1, 1))
    train = ds[:int(len(ts_data) * 0.8)]
    test = ds[int(len(ts_data) * 0.8):]
    trainX, trainY = create_ds(train)
    testX, testY = create_ds(test)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(125, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    # ======== uncomment for training ========
    model.compile(loss='mean_squared_error', optimizer='adam')
    # modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    #     monitor='val_loss',
    #     filepath='model_checkpoint3.h5',
    #     verbose=1,
    #     save_weights_only=True,
    #     save_best_only=True,
    # )
    # history = model.fit(
    #     trainX, trainY, epochs=50, batch_size=5, validation_data=(testX, testY),
    #     callbacks=[
    #         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    #         modelckpt_callback], verbose=1, shuffle=False
    # )
    #
    # fig = plt.figure(figsize=(7.5, 5))
    # ax = fig.add_subplot()
    # ax.set_title(f'LSTM Train/Loss for {name} Sea Ice Extent', fontsize=16)
    # ax.set_xlabel(f'Epochs', fontsize=14)
    # ax.set_ylabel(f'Train/Loss Percentage', fontsize=14)
    # ax.plot(
    #     history.history['loss'],
    #     color='blue',
    #     linewidth=2.0,
    #     linestyle='-',
    #     label='Train Loss'
    # )
    # ax.plot(
    #     history.history['val_loss'],
    #     color='red',
    #     linewidth=2.0,
    #     linestyle='-',
    #     label='Test Loss'
    # )
    # ======== uncomment for training ========
    model.load_weights('model_checkpoint2.h5')
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)
    # invert
    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform([trainY])  # one row
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])
    # # future
    # pred = testX[0][-1].reshape(1, 1, 1)
    # preds = []
    # for timestep in list(range(1,11)):
    #     print(pred)
    #     pred = model.predict(pred)
    #     pred = pred.reshape(1,1,1)
    #     preds.append(pred)
    # preds = np.asarray([scaler.inverse_transform(pred) for pred in preds])
    # ax.plot(
    #     list(range(len(all_pred[:,0]), len(all_pred[:,0])+11)),
    #     preds.reshape(len(preds),1).reshape(len(preds)),
    #     color='yellow'
    # )
    fig2 = plt.figure(figsize=(7.5, 5))
    ax = fig2.add_subplot()
    all = np.concatenate((trainY, testY), axis=1)
    all_pred = np.concatenate((train_predict, test_predict), axis=0)
    ax.plot(
        list(range(1, len(all_pred[:,0])+1)),
        all_pred[:,0],
        color='blue',
        label='predicted',
        markerfacecolor='orange',
        marker='o',
        markersize=5.0,
        linewidth=2.0,
        linestyle='--',
    )
    ax.plot(
        list(range(1, len(all[0])+1)), # years?
        all[0],
        color='black',
        label='actual',
        marker='D',
        markerfacecolor='red',
        markersize=5.0,
        linewidth=2.0,
    )
    # ax.set_xlabel('Years', fontsize=14)
    # ax.set_ylabel(r'Extent ($\displaystyle10^6$ sq km)', fontsize=14)
    # unitx = (max(years)-min(years))/len(years)
    # ax.set_xlim(years[0]-unitx, years[-1]+unitx)
    # unity = (max(ts_data)-min(ts_data))/len(ts_data)
    # ax.set_ylim(min(ts_data)-unity, max(ts_data)+unity)
    # step_size = round((max(ts_data)-min(ts_data))/10, 2)
    # ax.set_yticks(np.arange(min(ts_data), max(ts_data), step_size))
    plt.legend()
    model.summary()
    plt.show()

def differentiated_sea_ice(name, x, y, marker, markerfacecolor):
    y = np.asarray(y)
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8.5, 6))
    fig.suptitle(f'{name} Sea Ice Autocorrelation Check', fontsize=16.0)
    ax[0,0].plot(
        y,
        color='black',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=markerfacecolor,
    )
    ax[0,0].set_title('Actual', fontsize=14.0)
    ax[0,0].grid()
    plot_acf(y, ax=ax[0,1])
    ax[0,1].grid()
    ax[1,0].plot(
        np.diff(np.asarray(y)),
        color='green',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=markerfacecolor,
    )
    ax[1,0].set_title('1st Order Diff.', fontsize=14.0)
    ax[1,0].grid()
    plot_acf(np.diff(np.asarray(y)), ax=ax[1,1])
    ax[1,1].grid()

    ax[2,0].plot(
        np.diff(np.diff(np.asarray(y))),
        color='blue',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=markerfacecolor,
    )
    ax[2,0].set_title('2nd Order Diff.', fontsize=14.0)
    ax[2,0].grid()
    plot_acf(np.diff(np.diff(np.asarray(y))), ax=ax[2,1])
    ax[2,1].grid()
    plt.show()

def plot_ice_extent(name, x, y, marker, markerfacecolor):
    fig = plt.figure(figsize=(7.5, 5))
    ax = fig.add_subplot()
    ax.set_title(f'{name} Sea Ice Extent {x[0]}-{x[-1]}', fontsize=16)
    ax.set_xlabel('Years', fontsize=14)
    ax.set_ylabel(r'Extent ($\displaystyle10^6$ sq km)', fontsize=14)
    unitx = (max(x)-min(x))/len(x)
    # ax.set_xlim(x[0]-unitx, 2030+unitx)
    ax.set_xlim(x[0]-unitx, x[-1]+unitx)
    unity = (max(y)-min(y))/len(y)
    ax.set_ylim(min(y)-unity, max(y)+unity)
    step_size = round((max(y)-min(y))/10, 2)
    ax.set_yticks(np.arange(min(y), max(y), step_size))
    ax.grid() # if grid is not on use annotate
    ax.plot(
        x, y,
        color='black',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=markerfacecolor,
    )
    # model = pm.auto_arima(
    #     y,
    #     start_p=1,
    #     d=None,
    #     start_q=0,
    #     max_p=3,
    #     max_q=3,
    #     start_P=0,
    #     D=0,
    #     start_Q=0,
    #     stationary=True,
    #     information_criterion='bic',
    #     test='adf',
    #     stepwise=True,
    #     trend=None,
    #     trace=True,
    #     random_state=np.random.seed(1243),
    #     error_action='ignore',
    #     suppress_warnings=True,
    # )
    # print(model.summary())
    # steps_2030 = 9
    # f2030, ci2030 = model.predict(n_periods=steps_2030, return_conf_int=True)
    #plot_acf(model.resid(), 20)
    #print(model.get_params())

    # ax.plot(
    #     list(range(2022, 2023+1)), f2023,
    #     color='k',
    #     linewidth=2.0,
    #     marker=marker,
    #     markersize=2.5,
    #     alpha=0.75,
    #     markerfacecolor=markerfacecolor,
    #
    # )
    # ax.plot(
    #     list(range(2022, 2025+1)), f2025,
    #     color='k',
    #     linewidth=2.0,
    #     marker=marker,
    #     markersize=2.5,
    #     alpha=0.50,
    #     markerfacecolor=markerfacecolor,
    #
    # )
    # ax.plot(
    #     list(range(2022, 2030+1)), f2030,
    #     color='k',
    #     linewidth=2.0,
    #     marker=marker,
    #     markersize=2.5,
    #     alpha=0.25,
    #     markerfacecolor=markerfacecolor,
    #
    # )
    # plt.fill_between(
    #     list(range(2022, 2023+1)),
    #     ci2023[:, 0],
    #     ci2023[:, 1],
    #     color='k',
    #     alpha=.65
    # )
    # plt.fill_between(
    #     list(range(2022, 2025+1)),
    #     ci2025[:, 0],
    #     ci2025[:, 1],
    #     color='k',
    #     alpha=.45
    # )
    # plt.fill_between(
    #     list(range(2022, 2030+1)),
    #     ci2030[:, 0],
    #     ci2030[:, 1],
    #     color='k',
    #     alpha=.15
    # )

    # for i, v in enumerate(y):
    #     ax.text(x[i], v, "%d" %v, ha="center", color='blue', fontsize=4)
    # for i, v in enumerate(y):
    #     ax.annotate(str(round(v,2)), xy=(x[i],v), xytext=(-7,7), textcoords='offset points', color='red', fontsize=6)
    #plt.savefig(f'{name}')
    plt.show()


def smoothing(years, ts_data, name):
    # run simple exponential smoothing
    alphas = np.arange(0.1, 1.0, 0.1)
    print(ts_data)
    for alpha in alphas:
        simp_smoothed = SimpleExpSmoothing(ts_data, initialization_method='heuristic').fit(smoothing_level=alpha, optimized=False)
        plt.plot(years, simp_smoothed.fittedvalues, label=f'Alpha-0.3', color='red', alpha=alpha, linewidth=0.85, marker='o', markersize=2.5)
        print(simp_smoothed.initialvalues)
        print([elt for elt in simp_smoothed.fittedvalues])
        #exp_smoothed = ExponentialSmoothing(endog=ts_data).fit(smoothing_level=alpha, optimized=False)
        #plt.plot(years, exp_smoothed.fittedvalues, label=f'Alpha-{alpha}', color='blue', alpha=alpha, linewidth=0.85, marker='o', markersize=2.5)
    plt.plot(years, ts_data, color='black', label='Actual', alpha=1.0, linewidth=0.85, marker='o', markersize=2.5)
    plt.xticks(np.arange(years[0], years[-1], 5))
    plt.yticks(np.arange(min(ts_data), max(ts_data), 0.75))
    plt.title(f'{name} Sea Ice Extent {years[0]}-{years[-1]}', fontsize=20, **TMFONT)
    plt.xlabel('Years', fontsize=14, **TMFONT)
    plt.ylabel('Sea Ice Extent', fontsize=14, **TMFONT)
    #plt.legend()


def check_if_stationary(years, ts_data):
    # run Augmented Dickey-Fuller test
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
    for ice_val in ['Antarctic Maximum', 'Antarctic Minimum', 'Arctic Maximum', 'Arctic Minimum']:
        print(f'Augmented Dickey-Fuller Test for {ice_val} says...')
        adftest = adfuller(ts_data[ice_val], autolag='AIC')
        print(f'The test statistic is {adftest[0]}')
        print(f'The p-value is {adftest[1]}') # p > 0.5 Accept the null hypothesis (H0), the data has a unit root and is non-stationary.
        print(f'The # of lags used was {adftest[2]}')
        print(f'The # of observations used was {adftest[3]}')
        if adftest[1] > 0.05:
            print(f'With p={adftest[1]} > 0.5=alpha, we fail to reject the null hypothesis\
    that \"a unit root is present in an autoregressive time series model,\
    which means we can conclude that the {ice_val} data is likely non-stationary.\"')
            adftest_1diff = adfuller(np.diff(np.asarray(ts_data[ice_val])), autolag='AIC')
            if adftest_1diff[1] > 0.05:
                print(f'With p={adftest_1diff[1]} > 0.5=alpha, we fail to reject the null hypothesis\
        that \"a unit root is present in an autoregressive time series model,\
        which means we can conclude that the 1st order differentiated {ice_val} data is likely non-stationary.\"')
            else:
                print(f'1st order differentiating for {ice_val} data makes it stationary')
        else:
            print(f'At p={adftest[1]} >= 0.5=alpha, we can reject the null hypothesis\
    that \"a unit root is present in an autoregressive time series model,\
    which means we can conclude that the {ice_val} data is likely stationary.\"')
        print()

def check_if_gaussian(years, ts_data):
    # run Augmented Dickey-Fuller test
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
    for ice_val in ['Antarctic Maximum', 'Antarctic Minimum', 'Arctic Maximum', 'Arctic Minimum']:
        print(f'A Test for Normality for {ice_val} says...')
        stat, p = stats.normaltest(ts_data[ice_val])
        alphas = [0.0001, 0.001, 0.01, 0.05]
        not_stat_significant_alphas = []
        for alpha in alphas:
            if p > alpha:
                not_stat_significant_alphas.append(alpha)
            else:
                pass
        kurtosis = stats.kurtosis(ts_data[ice_val]) # Gaussian close to 0
        skew = stats.skew(ts_data[ice_val]) # between -0.5 and 0.5, the data are fairly symmetrical
        print(f'...that at alpha levels of {not_stat_significant_alphas}, the {ice_val} data appears Gaussian.')
        print(f'Kurtosis for {ice_val}: {kurtosis}')
        print(f'Skew for {ice_val}: {skew}')
        print()

def assemble_dataset(df, func):
    if func == 'max':
        new_df = df.groupby(['Year']).max()
    else:
        new_df = df.groupby(['Year']).min()
    new_df.rename(
        columns = dict([(elt, elt.replace(' ', '')) for elt in new_df.columns]),
        inplace = True
    )
    return [float(elt.replace(' ', '')) for elt in new_df['Extent'].to_list()[1:-1]]

def main():
    # read data; comes from the "Data" tab in https://nsidc.org/data/seaice_index/archives
    antarctic_data = pd.read_csv('Data_Southern-(Antarctic)_Hemisphere_Daily_Sea_Ice_Extent.csv')
    arctic_data = pd.read_csv('Data_Northern-(Arctic)_Hemisphere_Daily_Sea_Ice_Extent.csv')

    years = list(range(1979, 2021+1))
    antarctic_max_ts = assemble_dataset(antarctic_data, 'max')
    antarctic_min_ts = assemble_dataset(antarctic_data, 'min')
    arctic_max_ts = assemble_dataset(arctic_data, 'max')
    arctic_min_ts = assemble_dataset(arctic_data, 'min')
    assert len(years)==len(antarctic_max_ts)==len(antarctic_min_ts)==len(arctic_max_ts)==len(arctic_min_ts)
    sea_ice = {
        'Antarctic Maximum': antarctic_max_ts,
        'Antarctic Minimum': antarctic_min_ts,
        'Arctic Maximum': arctic_max_ts,
        'Arctic Minimum': arctic_min_ts,
    }
    # plot the sea ice data
    plot_ice_extent('Antarctic Maximum', years, antarctic_max_ts, 'D', 'red')
    plot_ice_extent('Antarctic Minimum', years, antarctic_min_ts, 's', 'cyan')
    plot_ice_extent('Arctic Maximum', years, arctic_max_ts, 'o', 'orange')
    plot_ice_extent('Arctic Minimum', years, arctic_min_ts, 'H', 'pink')

    differentiated_sea_ice('Antarctic Maximum', years, antarctic_max_ts, 'D', 'red')
    differentiated_sea_ice('Antarctic Minimum', years, antarctic_min_ts, 's', 'cyan')
    differentiated_sea_ice('Arctic Maximum', years, arctic_max_ts, 'o', 'orange')
    differentiated_sea_ice('Arctic Minimum', years, arctic_min_ts, 'H', 'pink')
    #differentiated_sea_ice('Arctic Maximum', years, arctic_max_ts, 'o', 'orange')

    #use_BiLSTM(years, antarctic_max_ts, 'Antarctic Maximum')

    # check stationary-ness of data
    check_if_stationary(years, sea_ice)
    # check if Gaussian
    check_if_gaussian(years, sea_ice)

    # look at a smoothed plot
    #smoothing(years, sea_ice['Antarctic Minimum'], 'Antarctic Minimum')


    # TODO
    # LTSM forecast method, understand LSTM
    # Dickey Fuller read-up:
    #   https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test
    #   https://stats.stackexchange.com/questions/24072/interpreting-rs-ur-df-dickey-fuller-unit-root-test-results
    # Transformations of stationary
    # Time Series forecasting statsmodels
    # Statsmodels smoothing
    # Website re-read / scanning



if __name__ == '__main__':
    main()
