# Common imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns

sns.set(color_codes=True)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta


def preprocessing(Data, Ground_Truth, entire_period_months, entire_period_days, training_period, testing_period):
    Ground_Truth = pd.to_datetime(Ground_Truth)

    Data.Date = pd.to_datetime(Data.Date)

    Data2 = Data[Data.Date >= Ground_Truth - relativedelta(months=entire_period_months) - relativedelta(
        days=entire_period_days)]  # start date

    Data3 = Data2[['CumVar_1', 'Date']]

    X_train_withdate = Data3[Data3.Date < min(Data3.Date) + relativedelta(months=training_period)]
    X_test_withdate = Data3[Data3.Date >= min(Data3.Date) + relativedelta(months=training_period)]
    X_test_withdate = X_test_withdate[
        X_test_withdate.Date < max(X_train_withdate.Date) + relativedelta(days=testing_period + 2)]
    train_last = X_train_withdate.tail(29)
    X_test_withdate = pd.concat([train_last, X_test_withdate])

    X_test = X_test_withdate.drop(columns=['Date'])
    X_train = X_train_withdate.drop(columns=['Date'])
    # X_train = X_train.fillna(0)
    # X_test = X_test.fillna(0)

    window_size = 30
    X_train = np.array(X_train)
    window_train = X_train[np.arange(window_size)[None, :] + np.arange(X_train.shape[0] - window_size)[:, None]]

    X_test = np.array(X_test)
    window_test = X_test[np.arange(window_size)[None, :] + np.arange(X_test.shape[0] - window_size)[:, None]]
    print(window_train.shape, window_test.shape, )
    print(min(X_train_withdate.Date), max(X_train_withdate.Date))
    print(min(X_test_withdate.Date), max(X_test_withdate.Date))
    return window_train, window_test, X_train_withdate, X_test_withdate


def sliding_window_regression(elements, window_size):
    regression_filtered_2 = []
    for i in range(0, len(elements) - window_size + 1):
        model = LinearRegression()
        model.fit(elements["Day"].iloc[i:i + window_size].values.reshape(-1, 1),
                  elements["median_filtered"].iloc[i:i + window_size].values)
        regression_filtered_2.append(model.predict(elements["Day"].iloc[i].reshape(-1, 1)))
    return regression_filtered_2


def calculate(data):
    from sklearn.metrics import mean_absolute_error
    import statistics
    k = 3
    t = 1
    x_data = data.iloc[:, 0].values
    y_data = data.iloc[:, 1].values
    x = data.iloc[0:k, 0].values
    y = data.iloc[0:k, 1].values
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    tube_s_pred = model.predict(x.reshape(-1, 1))
    tube_dev = mean_absolute_error(y, tube_s_pred)
    tol_factor = 70
    tol_min = 40
    tol_max = 70
    tol = statistics.median([tol_factor * tube_dev, tol_min, tol_max])
    tube_up = tube_s_pred + tol
    tube_low = tube_s_pred - tol
    tube_s_whole = model.predict(x_data.reshape(-1, 1))
    tube_up_whole = tube_s_whole + tol
    tube_low_whole = tube_s_whole - tol
    Output = dict()
    Output['tube_no'] = np.array([t, t, t])
    Output['tube_up'] = tube_up
    Output['tube_low'] = tube_low
    Output['tube_slope'] = tube_s_pred
    print(Output)
    counter = 0
    count_out = 0
    count_within = 0
    for j in range(k, len(data)):
        print("j:", j)
        if tube_up_whole[j] > y_data[j] > tube_low_whole[j]:
            print("Within")
            counter = 0
            count_within += 1
            tube_slope = np.append(Output['tube_slope'], tube_s_whole[j])
            tube_up = np.append(tube_up, tube_up_whole[j])
            tube_low = np.append(tube_low, tube_low_whole[j])
            Output['tube_slope'] = tube_slope
            Output['tube_up'] = tube_up
            Output['tube_low'] = tube_low
            Output['tube_no'] = np.append(Output['tube_no'], t)
            print("count_out", count_out)
            print("count_within", count_within)
            print("count total", count_out + count_within)
            print(Output['tube_up'])
            print(Output['tube_low'])
            print(Output['tube_slope'])
            print(Output['tube_no'])
            print(len(Output['tube_no']))
        else:
            counter += 1
            count_out += 1
            count_within = 0
            print("counter", counter)
            if 1 <= counter < k + 1:
                Output['tube_no'] = np.append(Output['tube_no'], t)
                tube_slope = np.append(Output['tube_slope'], tube_s_whole[j])
                tube_up = np.append(tube_up, tube_up_whole[j])
                tube_low = np.append(tube_low, tube_low_whole[j])
                Output['tube_slope'] = tube_slope
                Output['tube_up'] = tube_up
                Output['tube_low'] = tube_low
                print("count_out", count_out)
                print("count_within", count_within)
                print("count total", count_out + count_within)
                print(Output['tube_up'])
                print(Output['tube_low'])
                print(Output['tube_slope'])
                print(Output['tube_no'])
                print(len(Output['tube_no']))
            if counter == k + 1:
                print("Other condition")
                print("in counter", k + 1)
                t = t + 1
                Output['tube_no'] = np.append(Output['tube_no'][:-3], np.repeat(t, counter))
                x = data.iloc[j - counter + 1:(j + 1), 0].values
                y = data.iloc[j - counter + 1:(j + 1), 1].values
                print(x)
                model = LinearRegression()
                model.fit(x.reshape(-1, 1), y)
                tube_s_pred_new = model.predict(x.reshape(-1, 1))
                tube_dev_new = mean_absolute_error(y, tube_s_pred_new)
                tol_new = statistics.median([tol_factor * tube_dev_new, tol_min, tol_max])
                print("new model")
                tube_up_new2 = tube_s_pred_new + tol_new
                tube_low_new2 = tube_s_pred_new - tol_new
                tube_s_whole = model.predict(x_data.reshape(-1, 1))
                tube_up_whole = tube_s_whole + tol
                tube_low_whole = tube_s_whole - tol
                tube_slope = np.append(Output['tube_slope'][:-counter + 1], tube_s_pred_new)
                tube_up = np.append(tube_up[:-counter + 1], tube_up_new2)
                tube_low = np.append(tube_low[:-counter + 1], tube_low_new2)
                Output['tube_slope'] = tube_slope
                Output['tube_up'] = tube_up
                Output['tube_low'] = tube_low
                counter = 0
                count_within = 0
                count_out = 0
                print(Output['tube_no'])
                print(len(Output['tube_no']))
                print(Output)
    return Output


def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return indices


def TUBE(data, tank_name="tank"):
    Processed = []
    for l in range(44, -2, -1):
        for m in (15, 0):
            a = preprocessing(data, Ground_Truth="2021-09-02", entire_period_months=int(l),
                              entire_period_days=int(m), training_period=3, testing_period=15)
            Processed.append(a)

    window_size = 30
    All = {}
    for n in range(0, len(Processed) - 1):
        All[n] = np.concatenate((Processed[n][2], Processed[n][3][window_size - 1:]))

    Alarm_Dict = dict()
    All_Slopes = dict()
    All_length = dict()
    for i in range(1, len(All)):
        Period = pd.DataFrame(All[i], columns=["CumVar", "Day"])
        Period['Day'] = Period.index
        Period.Day += 1
        Period["median_filtered"] = Period["CumVar"].rolling(5, min_periods=1).median()
        regression_filtered_2 = sliding_window_regression(Period, 5)
        regression_filtered_2 = np.array(regression_filtered_2).flatten()
        regression_filtered_2 = np.append(regression_filtered_2, np.array(Period['median_filtered'][-4:]))
        Period["regression_filtered"] = regression_filtered_2
        Period = Period.drop(columns=['CumVar', 'median_filtered'])
        Tube = calculate(Period)
        Tube_up = np.hstack(Tube['tube_up'])
        Tube_low = np.hstack(Tube['tube_low'])
        Tube_slope = np.hstack(Tube['tube_slope'])
        print(i)
        print(len(Tube_up))
        print(len(Tube_low))
        print(len(Tube_slope))
        data_tube = Period.copy()
        data_tube['Tube_up'] = Tube_up
        data_tube['Tube_low'] = Tube_low
        data_tube['Tube_slope'] = Tube_slope
        data_tube['Tube_No'] = Tube['tube_no']
        Slopes = []
        Length = []
        for j in range(1, max(data_tube['Tube_No'] + 1)):
            print(j)
            index = which(data_tube['Tube_No'] == j)
            length = (max(index) - min(index)) + 1
            model = LinearRegression()
            model.fit(data_tube.Day[:max(index) + 1].values.reshape(-1, 1),
                      data_tube.regression_filtered[:max(index) + 1])
            model1_Slope = model.coef_[0]
            Slopes.append(model1_Slope)
            Length.append(int(length))
        All_Slopes[i] = Slopes
        All_length[i] = Length
        plt.plot(data_tube['Day'], Period['regression_filtered'])
        plt.title(r"Period: %s" % str(i))
        plt.plot(data_tube['Day'], data_tube['Tube_low'])
        plt.plot(data_tube['Day'], data_tube['Tube_up'])
        plt.plot(data_tube['Day'], data_tube['Tube_slope'])
        plt.show()

        Slopes_Dict = pd.DataFrame.from_dict(All_Slopes, orient='index')
        Length_Dict = pd.DataFrame.from_dict(All_length, orient='index')
        Alarm = []
        for r in range(0, len(Length_Dict)):
            if (Length_Dict.iloc[r].dropna()[-1:] >= 15).bool() & (abs(Slopes_Dict.iloc[r].dropna()[-1:]) > 18.2).bool():
                Alarm.append(1)
            elif (sum(Length_Dict.iloc[r].dropna()[-2:]) >= 15) & (abs(Slopes_Dict.iloc[r].dropna()[-2:].mean()) > 18.2):
                Alarm.append(1)
            elif (sum(Length_Dict.iloc[r].dropna()[-3:]) >= 15) & (abs(Slopes_Dict.iloc[r].dropna()[-3:].mean()) > 18.2):
                Alarm.append(1)
            elif (sum(Length_Dict.iloc[r].dropna()[-4:]) >= 15) & (abs(Slopes_Dict.iloc[r].dropna()[-4:].mean()) > 18.2):
                Alarm.append(1)
            else:
                Alarm.append(0)

        Alarm_Dict[i] = Alarm
    To_csv = pd.DataFrame(Alarm)
    To_csv.to_csv(r"%s_Alarm.csv" %tank_name)
