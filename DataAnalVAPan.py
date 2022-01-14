# Author: Rodeo Flagellum
# see https://www.metaculus.com/tournament/realtimepandemic/
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r''
plt.rcParams["font.family"] = "Times New Roman"


def differentiated_VA_COVID(cov_data, start, title, y_name, marker, color):
    current_date = start + dt.timedelta(days=len(cov_data))
    days = mdates.drange(start, current_date, dt.timedelta(days=1))
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8.5, 6))
    fig.suptitle(f'{title} Autocorrelation Check', fontsize=16.0)
    ax[0,0].plot(
        cov_data,
        color='black',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=color,
    )
    ax[0,0].set_title('Actual', fontsize=14.0)
    ax[0,0].grid()
    plot_acf(cov_data, ax=ax[0,1])
    ax[0,1].grid()
    ax[1,0].plot(
        np.diff(np.asarray(cov_data)),
        color='green',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=color,
    )
    ax[1,0].set_title('1st Order Diff.', fontsize=14.0)
    ax[1,0].grid()
    plot_acf(np.diff(np.asarray(cov_data)), ax=ax[1,1])
    ax[1,1].grid()

    ax[2,0].plot(
        np.diff(np.diff(np.asarray(cov_data))),
        color='blue',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=color,
    )
    ax[2,0].set_title('2nd Order Diff.', fontsize=14.0)
    ax[2,0].grid()
    plot_acf(np.diff(np.diff(np.asarray(cov_data))), ax=ax[2,1])
    ax[2,1].grid()
    plt.show()


def test_for_stationary(cov_data, name):
    # use Dickey Fuller test at alpha=0.5
    # want to make the time series data stationary, so that
    # trend and seasonality are removed
    adftest = adfuller(cov_data, autolag='AIC')
    print(f'Dickey Fuller test statistic: {adftest[0]}')
    print(f'Dickey Fuller test p-value: {adftest[1]}')
    print(f'Dickey Fuller lag count: {adftest[2]}')
    print(f'Dickey Fuller observation count: {adftest[3]}')
    if adftest[1] > 0.05:
        print(f'With p={adftest[1]} > 0.5=alpha, we fail to reject the null hypothesis\
that \"a unit root is present in an autoregressive time series model\",\
which means we can conclude that the {name} data is likely non-stationary.')
    else:
        print(f'At p={adftest[1]} >= 0.5=alpha, we can reject the null hypothesis\
that \"a unit root is present in an autoregressive time series model\",\
which means we can conclude that the {name} data is likely stationary.')
    print()


def plot_VA_COVID_futures(cov_data, start, title, y_name, marker, color):
    current_date = start + dt.timedelta(days=len(cov_data))
    print(current_date)
    days = mdates.drange(start, dt.date(2022, 1, 22), dt.timedelta(days=1)) # jan 21
    diff = abs(dt.date(2022, 1, 22) - current_date).days
    fig = plt.figure(figsize=(7.5, 5))
    ax = fig.add_subplot()
    ax.grid()
    ax.set_title(title, color='black', fontsize=16.0)
    ax.set_ylabel(y_name, color='black', fontsize=14.0)
    ax.set_xlabel('Dates', color='black', fontsize=14.0)
    ax.plot(
        days[:-diff],
        cov_data,
        color='black',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=color,
        label='Actual Values'
    )
    # slope
    slope = (cov_data[-1] - cov_data[0])/len(cov_data)
    new_data = []
    future_value = cov_data[-1]
    for _ in [slope]*diff:
        future_value += _
        new_data.append(future_value)
    new_data = np.asarray(new_data)
    print(f'Taking the current value substracting it from the first value and dividing this quantity\
by the number of days gives {new_data[-1]} for January 21st 2022')
    # ax.plot(
    #     days[-diff:],
    #     new_data,
    #     color='red',
    #     linewidth=2.0,
    #     marker=marker,
    #     markersize=5.0,
    #     markerfacecolor=color,
    #     label='(Last-First)/Count'
    # )
    # line of best fit
    x = np.asarray(list(range(1, len(days[:-diff])+1)))
    print(len(x), len(cov_data))
    a, b = np.polyfit(x, cov_data, 1)
    new_x = np.asarray(list(range(1, len(days)+1)))
    slope = (a*new_x) + b
    print(len(days), len(slope))
    ax.plot(days, slope, color='blue', linewidth=2.0, linestyle='--', label='Best Fit Line')
    print(f'Taking line of best fit for the current data gives {new_data[-1]} for January 21st 2022')
    # ARIMA model
    model = pm.auto_arima(
        cov_data,
        start_p=1,
        start_q=1,
        test='adf',
        max_p=3,
        max_q=3,
        D=0,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=False,
    )
    steps = diff
    forecast, confidence_int_95 = model.predict(n_periods=steps, alpha=0.05, return_conf_int=True)
    forecast2, confidence_int_85 = model.predict(n_periods=steps, alpha=0.15, return_conf_int=True)
    forecast3, confidence_int_75 = model.predict(n_periods=steps, alpha=0.25, return_conf_int=True)

    ax.plot(
        days[-diff:],
        forecast,
        color='gray',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=color,
    )
    # 95%
    ax.fill_between(
        days[-diff:],
        confidence_int_95[:, 0],
        confidence_int_95[:, 1],
        label='95\% conf. int.',
        color='red',
        alpha=.15)

    # 85%
    ax.fill_between(
        days[-diff:],
        confidence_int_85[:, 0],
        confidence_int_85[:, 1],
        color='red',
        label='85\% conf. int.',
        alpha=.25)
    # 75%
    ax.fill_between(
      days[-diff:],
      confidence_int_75[:, 0],
      confidence_int_75[:, 1],
      color='red',
      label='75\% conf. int.',
      alpha=.35
    )


    print(f'The current ARIMA forecast:\n{forecast}')
    #print(f'The current confidence intervals at each future point:\n{confidence_int_95}')
    print(confidence_int_85)
    print(model.summary())

    all_data = np.concatenate((cov_data, new_data, slope, confidence_int_95[:, 1]), axis=0)
    step_size = round(((max(all_data)-min(all_data))/10), 2)
    ax.set_yticks(np.arange(min(all_data), max(all_data)+step_size, step_size))
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval=int(len(all_data)/12))
    ax.xaxis.set_major_locator(locator)
    fig.autofmt_xdate()
    name = '_'.join(y_name.split()[-3:])
    plt.legend()
    plt.show()
    #plt.savefig(f'{name}.png')

def plot_VA_COVID(cov_data, start, title, y_name, marker, color):
    current_date = start + dt.timedelta(days=len(cov_data))
    days = mdates.drange(start, current_date, dt.timedelta(days=1))
    fig = plt.figure(figsize=(7.5, 5))
    ax = fig.add_subplot()
    ax.grid()
    ax.set_title(title, color='black', fontsize=16.0)
    ax.set_ylabel(y_name, color='black', fontsize=14.0)
    ax.set_xlabel('Dates', color='black', fontsize=14.0)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval=int(len(cov_data)/8))
    ax.xaxis.set_major_locator(locator)
    fig.autofmt_xdate()
    #unity = (max(cov_data)-min(cov_data))/len(cov_data)
    #ax.set_ylim(bottom=min(cov_data)-unity, top=max(cov_data)+unity)
    step_size = round(((max(cov_data)-min(cov_data))/10), 2)
    ax.set_yticks(np.arange(min(cov_data), max(cov_data), step_size))
    ax.plot(
        days,
        cov_data,
        color='black',
        linewidth=2.0,
        marker=marker,
        markersize=5.0,
        markerfacecolor=color,
    )
    plt.show()

def main():
    # I have uninterrupted data starting on Dec. 1 2021
    child_vac_start = dt.date(2021, 12, 1)
    child_vac = np.array(
        [22.5, 22.9, 23.3, 23.7, 24.2,
         24.5, 24.6, 25.0, 25.4, 25.7,
         26.1, 26.3, 26.6, 26.8, 27.1,
         27.4, 27.8, 28.1, 28.5, 28.7,
         29.0, 29.3, 29.7, 30.1, 30.2,
         30.3, 30.3, 30.4, 30.6, 31.0,
         31.5, 31.8, 32.0, 32.1, 32.2,
         32.4, 32.6, 32.8, 33.1, 33.3,
         33.7, 34.0, 34.2, 34.4, 34.6]
    )
    # I have uninterrupted data starting on Dec. 6 2021
    booster_start = dt.date(2021, 12, 6)
    booster = np.array(
        [25.40, 25.53, 26.07, 26.42, 27.02,
         27.46, 27.91, 28.19, 28.54, 29.03,
         29.40, 30.04, 30.57, 31.00, 31.41,
         31.68, 32.28, 32.82, 33.47, 33.71,
         33.82, 33.89, 34.05,34.34, 34.80,
         35.42, 35.68, 35.93, 36.03, 36.25,
         36.43, 36.69, 37.00, 37.32, 37.55,
         38.07, 38.42, 38.69, 39.00, 39.31]
    )
    # I have uninterrupted data starting on Nov. 29 2021
    one_dose_start = dt.date(2021, 11, 29)
    one_dose = np.array(
        [74.3, 74.4, 74.5, 74.6, 74.8,
         74.9, 74.9, 74.9, 74.9, 75.0,
         75.1, 75.3, 75.4, 75.5, 75.5,
         75.6, 75.7, 75.8, 76.0, 76.1,
         76.2, 76.3, 76.3, 76.5, 76.6,
         76.8, 76.8, 76.9, 76.9, 76.9,
         77.0, 77.1, 77.3, 77.4, 77.4,
         77.5, 77.5, 77.6, 77.7, 77.8,
         77.8, 77.9, 78.0, 78.1, 78.2,
         78.3, 78.4]
    )
    test_for_stationary(child_vac, '\% of VA 5-11 yo Population w/ Vaccine')
    test_for_stationary(booster, '\% of VA Population w/ Booster/3rd Dose')
    test_for_stationary(one_dose, '\% of VA Population w/ >=1 Dose of Vaccination')

    plot_q = input('Plot? (y, n): ')
    if plot_q == 'y':
        which_to_plot = int(input('Child Vaccinations: 1\nBooster/Third Dose: 2\n>=1 Dose: 3\n: '))
        if (which_to_plot) == 1:
            plot_VA_COVID_futures(
                child_vac,
                child_vac_start,
                r'\% of VA 5-11 yo Population w/ Vaccine',
                r'\% 5-11 yo\'s Vaccinated',
                'o',
                'orange'
            )
            differentiated_VA_COVID(
                child_vac,
                child_vac_start,
                r'\% of VA 5-11 yo Population w/ Vaccine',
                r'\% 5-11 yo\'s Vaccinated',
                'o',
                'orange'
            )
        if (which_to_plot) == 2:
            plot_VA_COVID_futures(
                booster,
                booster_start,
                r'\% of VA Population w/ Booster/3rd Dose',
                r'\% w Booster\\3rd Dose',
                's',
                'cyan'
            )
            differentiated_VA_COVID(
                booster,
                booster_start,
                r'\% of VA Population w/ Booster/3rd Dose',
                r'\% w Booster\\3rd Dose',
                's',
                'cyan'
            )
        if (which_to_plot) == 3:
            plot_VA_COVID_futures(
                one_dose,
                child_vac_start,
                r'\% of VA Population w/ $\displaystyle\geq 1$ Vaccination',
                r'\% w Vaccine',
                'D',
                'green'
            )
            differentiated_VA_COVID(
                one_dose,
                child_vac_start,
                r'\% of VA Population w/ $\displaystyle\geq 1$ Vaccination',
                r'\% w Vaccine',
                'D',
                'green'
            )

if __name__ == '__main__':
    main()




# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
# ax.set_xlim((days[0], days[-1]))
# plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# goals, learn how to accurately play around with
# dates on xaxis, forecasting the next values,
# as well as changing the fonts, plot sizes, and
# tick locations

# check if 5-11 yo, booster, and at least values are correlated
# then use them as features for RNN times series
