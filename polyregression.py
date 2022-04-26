from cgi import test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  #data visualization library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures  # evaluation metrics

def read_windpower_data(csv_name):
    df = pd.read_csv(csv_name)

    df.columns=['1','2', 'datetime', '3', 'wind power']
    df = df.dropna(axis=0)
    data = df

    data = data.drop(['1','2','3'], axis=1)
    data = data[['datetime','wind power']]
    data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True)

    return data

def parse_windpower_data(data, date_range):
    startdate = pd.to_datetime(date_range[0], dayfirst=True)
    enddate = pd.to_datetime(date_range[1], dayfirst=True)

    mask = (data['datetime'] > startdate) & (data['datetime'] <= enddate)
    time_range = data.loc[mask]

    wind = time_range['wind power']
    time = time_range['datetime']
    Y_data = np.array(wind).reshape(len(wind),1)
    X_data = np.array(time)

    return X_data, Y_data

def read_weather_data(csv_name):
    df = pd.read_csv(csv_name)

    df.columns=['year','month', 'day', 'time', 'time_zone','wind speed']
    df = df.dropna(axis=0)
    data = df.assign(datetime = df["day"].astype(str)+"-"+df["month"].astype(str)+"-"+df["year"].astype(str) + " " +df["time"].astype(str))

    data = data.drop(['year','month','day','time_zone'], axis=1)
    data = data[['datetime','wind speed']]
    data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True)

    return data


def parse_weather_data(data, date_range):
    startdate = pd.to_datetime(date_range[0], dayfirst=True)
    enddate = pd.to_datetime(date_range[1], dayfirst=True)

    mask = (data['datetime'] > startdate) & (data['datetime'] <= enddate)
    time_range = data.loc[mask]

    wind = time_range['wind speed']
    time = time_range['datetime']
    Y_data = np.array(wind).reshape(len(wind),1)
    X_data = np.array(time)

    return X_data, Y_data

def combine_weather_data(datasets):
    return sum(datasets) * 1/len(datasets)


period =  ["16-01-2022 00:00", "28-02-2022 00:00"]
[time, wind0] = parse_weather_data(read_weather_data("2022\kalajoki_ulkokalla.csv"), period)
[time, wind1] = parse_weather_data(read_weather_data("2022\kemi_ajos.csv"), period)
[time, wind2] = parse_weather_data(read_weather_data("2022\sodankyla_tahtela.csv"), period)
[time, wind3] = parse_weather_data(read_weather_data("2022\pori_tahkoluoto.csv"), period)
[time, wind4] = parse_weather_data(read_weather_data("2022\suomussalmi_pesio.csv"), period)

[time, y] = parse_windpower_data(read_windpower_data("2022\windpower.csv"), period)
y = y/1000

X = combine_weather_data([wind0, wind1, wind2, wind3, wind4])

#val_error, tr_error, x_train, x_val, y_train, y_val, y_pred_train, y_pred_val, full_prediction, lin_reg, Poly = predict_power_output(X, y, 4, 0.33)

[X_train, X_rem, y_train, y_rem] = train_test_split(X, y, test_size=0.4, random_state=42)
[X_val, X_test, y_val, y_test] = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

tr_errors  = []
val_errors = []

degrees = range(1, 11)
for i in degrees:
    lin_reg = LinearRegression()
    Poly = PolynomialFeatures(degree = i)

    X_train_poly = Poly.fit_transform(X_train)
    X_val_poly   = Poly.transform(X_val)
    lin_reg.fit(X_train_poly, y_train)

    y_pred_train = lin_reg.predict(X_train_poly)  
    y_pred_val = lin_reg.predict(X_val_poly)

    val_error = mean_squared_error(y_val, y_pred_val)
    tr_error = mean_squared_error(y_train, y_pred_train)   

    val_errors.append(val_error)
    tr_errors.append(tr_error)

    print("val {i}: {val:.8f}, tr {tr:.8f}".format(i = i, val = val_error, tr = tr_error))


print("train set size: {}, val set size: {}, test set size: {}".format(len(X_train),len(X_val), len(X_test)))
print("Min validation error: {:.6f}, min train error: {:.6f}".format(min(val_errors), min(tr_errors)))

lin_reg = LinearRegression() # NOTE: "fit_intercept=False" as we already have a constant iterm in the new feature X_poly
 
Poly = PolynomialFeatures(degree=4)    
X_train_poly = Poly.fit_transform(X_train)    # fit the raw features
lin_reg.fit(X_train_poly, y_train) 

X_test_poly = Poly.fit_transform(X_test) # transform the raw features for the test data 
y_pred_test = lin_reg.predict(X_test_poly) # predict values for the test data using the linear model 
test_error = mean_squared_error(y_test, y_pred_test) # calculate the test error
print("test error: {:.6f}".format(test_error))


b = "cornflowerblue"; r = "salmon"

fig, ax = plt.subplots(1, 2, figsize=(15,5))
X_fit = np.linspace(0, 14, 100) 
ax[0].scatter(X_train, y_train, color=b, s=5, label="Train data points")
ax[0].scatter(X_val, y_val, color=r, s=5, label="Validation data points")
ax[0].plot(X_fit, lin_reg.predict(Poly.transform(X_fit.reshape(-1, 1))),
label="Fitted model (degree 4)", color = "black") 
ax[0].set_ylabel("Power (GW)")
ax[0].set_xlabel("Wind speed (m/s)")
ax[0].xaxis.set_major_locator(plt.MaxNLocator(8))
ax[0].yaxis.set_major_locator(plt.MaxNLocator(10))
ax[0].legend(loc = 'lower right')
ax[0].set_title('Wind power output vs wind speed')

ax[1].plot(degrees, tr_errors, label = 'Train error', color=b)
ax[1].plot(degrees, val_errors,label = 'Validation error', color=r)
ax[1].scatter(degrees, val_errors, color=r)
ax[1].scatter(degrees, tr_errors, color=b)
ax[1].legend(loc = 'lower right')


ax[1].set_xlabel('Polynomial degree')
ax[1].set_ylabel('Mean squared loss')
ax[1].set_title('Train vs validation loss')
ax[1].xaxis.set_major_locator(plt.MaxNLocator(10))

ax[0].autoscale(enable=True, axis='x', tight=True)
ax[1].autoscale(enable=True, axis='x', tight=True)
#ax.xaxis.set_major_locator(plt.MaxNLocator(10))
#plt.show()

fig, ax = plt.subplots(figsize=(15,7))

ax.plot(time, y, label="True power output", color=b)
ax.set_ylabel("Power (GW)",size=15)
ax.set_xlabel("Time ",size=15)
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
ax.yaxis.set_major_locator(plt.MaxNLocator(10))

ax.plot(time, lin_reg.predict(Poly.transform(X)),label="Predicted power output", color=r)
ax.set_ylabel("Power (GW)",size=15)
ax.set_xlabel("Time ",size=15)
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
ax.yaxis.set_major_locator(plt.MaxNLocator(10))


ax.autoscale(enable=True, axis='x', tight=True)
ax.legend(loc = 'lower left')

plt.show()