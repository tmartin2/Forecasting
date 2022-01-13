# Author: Rodeo Flagellum
# Flu Tournament: https://www.metaculus.com/tournament/flusight-challenge/
# Timelines: Sum, jan16-22, feb13-19, jan30-feb5
# Data: https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh
# Resources:
#   https://gis.cdc.gov/grasp/FluView/FluHospRates.html
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime as dt
import pmdarima as pm #https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
plt.rcParams["text.usetex"] == True,
plt.rcParams["font.family"] == "Times New Roman"

def state_check(ts_data, state):
    # get data by state
    state_data = ts_data[ts_data['state'] == state[0]]
    state_data = state_data.sort_values(
        by='date',
        axis=0,
        ignore_index=True
    )
    state_data = state_data.dropna(subset=['previous_day_admission_influenza_confirmed'])

    dates = state_data['date']
    x = [dt.datetime.strptime(elt, '%Y/%m/%d') for elt in dates]
    daily = state_data['previous_day_admission_influenza_confirmed']
    cumulative = np.cumsum(state_data['previous_day_admission_influenza_confirmed'])

    # setup plots
    fig, ax = plt.subplots(2, sharex=True, figsize=(8.5, 6))
    fig.suptitle(f'{state[1]} Confirmed Influenza Hospitalizations', fontsize=16.0)
    ax[0].grid()
    ax[1].grid()
    ax[1].set_xlabel('Dates', color='black', fontsize=14.0)
    ax[0].set_ylabel('Cumulative', color='black', fontsize=14.0)
    ax[1].set_ylabel('Daily', color='black', fontsize=14.0)

    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax[1].xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval=int(len(dates)/12))
    ax[1].xaxis.set_major_locator(locator)
    fig.autofmt_xdate()

    # plot values
    ax[0].plot(
        x, cumulative,
        color='black',
        linewidth=1.0,
    )
    ax[1].plot(
        x, daily,
        color='black',
        linewidth=1.0,
    )
    # # #ARIMA model
    # model = pm.auto_arima(
    #     daily,
    #     start_p=1,
    #     start_q=1,
    #     test='adf',
    #     max_p=3,
    #     max_q=3,
    #     trace=True,
    #     error_action='ignore',
    #     suppress_warnings=True,
    #     stepwise=False,
    # )
    # when1 = dt.date(2022, 1, 22) # jan 16 - 22
    # step1 = int((when1 - x[-1].date()).days) # inclusive e.g., end is 11, this is 12-22
    #
    # forecast, confidence_int_95 = model.predict(n_periods=step1, alpha=0.05, return_conf_int=True)
    # forecast2, confidence_int_85 = model.predict(n_periods=step1, alpha=0.15, return_conf_int=True)
    # forecast3, confidence_int_75 = model.predict(n_periods=step1, alpha=0.25, return_conf_int=True)
    #
    # #[x[-1].date() - datetime.timedelta(days=x) for x in range(step1)]
    # ax[1].plot(
    #     pd.date_range(x[-1], periods=step1).tolist(),
    #     forecast,
    #     color='gray',
    #     label='Forecast',
    #     linewidth=1.0,
    # )
    # # 95%
    # ax[1].fill_between(
    #     pd.date_range(x[-1], periods=step1).tolist(),
    #     confidence_int_95[:, 0],
    #     confidence_int_95[:, 1],
    #     label='95\% conf. int.',
    #     color='red',
    #     alpha=.15)
    # print(forecast)
    # print(confidence_int_95)

    plt.savefig(f'011222_{state[1]}')



def main():
    ts_data = pd.read_csv('./ITData/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv')
    # totals to date
    totals_by_date = ts_data.groupby(by='date').sum()
    #plt.plot(list(range(totals_by_date.shape[0])), totals_by_date['previous_day_admission_influenza_confirmed'])
    # states
    state_params = {
        'VT':['VT', 'Vermont', 'red'],
        'FL':['FL', 'Florida', 'blue'],
        'NY':['NY', 'New York', 'black'],
        'OK':['OK', 'Oklahoma', 'black'],
        'CA':['CA', 'California', 'black'],
        'WY':['WY', 'Wyoming', 'black'],
        'WI':['WI', 'Wisconsin', 'black'],
    }
    state_check(ts_data, state_params['WI'])

if __name__ == '__main__':
    main()




#================================================================================================
# COLUMNS
#================================================================================================
# state
# date
# critical_staffing_shortage_today_yes
# critical_staffing_shortage_today_no
# critical_staffing_shortage_today_not_reported
# critical_staffing_shortage_anticipated_within_week_yes
# critical_staffing_shortage_anticipated_within_week_no
# critical_staffing_shortage_anticipated_within_week_not_reported
# hospital_onset_covid
# hospital_onset_covid_coverage
# inpatient_beds
# inpatient_beds_coverage
# inpatient_beds_used
# inpatient_beds_used_coverage
# inpatient_beds_used_covid
# inpatient_beds_used_covid_coverage
# previous_day_admission_adult_covid_confirmed
# previous_day_admission_adult_covid_confirmed_coverage
# previous_day_admission_adult_covid_suspected
# previous_day_admission_adult_covid_suspected_coverage
# previous_day_admission_pediatric_covid_confirmed
# previous_day_admission_pediatric_covid_confirmed_coverage
# previous_day_admission_pediatric_covid_suspected
# previous_day_admission_pediatric_covid_suspected_coverage
# staffed_adult_icu_bed_occupancy
# staffed_adult_icu_bed_occupancy_coverage
# staffed_icu_adult_patients_confirmed_and_suspected_covid
# staffed_icu_adult_patients_confirmed_and_suspected_covid_coverage
# staffed_icu_adult_patients_confirmed_covid
# staffed_icu_adult_patients_confirmed_covid_coverage
# total_adult_patients_hospitalized_confirmed_and_suspected_covid
# total_adult_patients_hospitalized_confirmed_and_suspected_covid_coverage
# total_adult_patients_hospitalized_confirmed_covid
# total_adult_patients_hospitalized_confirmed_covid_coverage
# total_pediatric_patients_hospitalized_confirmed_and_suspected_covid
# total_pediatric_patients_hospitalized_confirmed_and_suspected_covid_coverage
# total_pediatric_patients_hospitalized_confirmed_covid
# total_pediatric_patients_hospitalized_confirmed_covid_coverage
# total_staffed_adult_icu_beds
# total_staffed_adult_icu_beds_coverage
# inpatient_beds_utilization
# inpatient_beds_utilization_coverage
# inpatient_beds_utilization_numerator
# inpatient_beds_utilization_denominator
# percent_of_inpatients_with_covid
# percent_of_inpatients_with_covid_coverage
# percent_of_inpatients_with_covid_numerator
# percent_of_inpatients_with_covid_denominator
# inpatient_bed_covid_utilization
# inpatient_bed_covid_utilization_coverage
# inpatient_bed_covid_utilization_numerator
# inpatient_bed_covid_utilization_denominator
# adult_icu_bed_covid_utilization
# adult_icu_bed_covid_utilization_coverage
# adult_icu_bed_covid_utilization_numerator
# adult_icu_bed_covid_utilization_denominator
# adult_icu_bed_utilization
# adult_icu_bed_utilization_coverage
# adult_icu_bed_utilization_numerator
# adult_icu_bed_utilization_denominator
# geocoded_state
# previous_day_admission_adult_covid_confirmed_18-19
# previous_day_admission_adult_covid_confirmed_18-19_coverage
# previous_day_admission_adult_covid_confirmed_20-29
# previous_day_admission_adult_covid_confirmed_20-29_coverage
# previous_day_admission_adult_covid_confirmed_30-39
# previous_day_admission_adult_covid_confirmed_30-39_coverage
# previous_day_admission_adult_covid_confirmed_40-49
# previous_day_admission_adult_covid_confirmed_40-49_coverage
# previous_day_admission_adult_covid_confirmed_50-59
# previous_day_admission_adult_covid_confirmed_50-59_coverage
# previous_day_admission_adult_covid_confirmed_60-69
# previous_day_admission_adult_covid_confirmed_60-69_coverage
# previous_day_admission_adult_covid_confirmed_70-79
# previous_day_admission_adult_covid_confirmed_70-79_coverage
# previous_day_admission_adult_covid_confirmed_80+
# previous_day_admission_adult_covid_confirmed_80+_coverage
# previous_day_admission_adult_covid_confirmed_unknown
# previous_day_admission_adult_covid_confirmed_unknown_coverage
# previous_day_admission_adult_covid_suspected_18-19
# previous_day_admission_adult_covid_suspected_18-19_coverage
# previous_day_admission_adult_covid_suspected_20-29
# previous_day_admission_adult_covid_suspected_20-29_coverage
# previous_day_admission_adult_covid_suspected_30-39
# previous_day_admission_adult_covid_suspected_30-39_coverage
# previous_day_admission_adult_covid_suspected_40-49
# previous_day_admission_adult_covid_suspected_40-49_coverage
# previous_day_admission_adult_covid_suspected_50-59
# previous_day_admission_adult_covid_suspected_50-59_coverage
# previous_day_admission_adult_covid_suspected_60-69
# previous_day_admission_adult_covid_suspected_60-69_coverage
# previous_day_admission_adult_covid_suspected_70-79
# previous_day_admission_adult_covid_suspected_70-79_coverage
# previous_day_admission_adult_covid_suspected_80+
# previous_day_admission_adult_covid_suspected_80+_coverage
# previous_day_admission_adult_covid_suspected_unknown
# previous_day_admission_adult_covid_suspected_unknown_coverage
# deaths_covid
# deaths_covid_coverage
# on_hand_supply_therapeutic_a_casirivimab_imdevimab_courses
# on_hand_supply_therapeutic_b_bamlanivimab_courses
# on_hand_supply_therapeutic_c_bamlanivimab_etesevimab_courses
# previous_week_therapeutic_a_casirivimab_imdevimab_courses_used
# previous_week_therapeutic_b_bamlanivimab_courses_used
# previous_week_therapeutic_c_bamlanivimab_etesevimab_courses_used
# icu_patients_confirmed_influenza
# icu_patients_confirmed_influenza_coverage
# previous_day_admission_influenza_confirmed
# previous_day_admission_influenza_confirmed_coverage
# previous_day_deaths_covid_and_influenza
# previous_day_deaths_covid_and_influenza_coverage
# previous_day_deaths_influenza
# previous_day_deaths_influenza_coverage
# total_patients_hospitalized_confirmed_influenza
# total_patients_hospitalized_confirmed_influenza_and_covid
# total_patients_hospitalized_confirmed_influenza_and_covid_coverage
# total_patients_hospitalized_confirmed_influenza_coverage
#================================================================================================
