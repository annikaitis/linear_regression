import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('bottle.csv', delimiter=",", low_memory=False)
df_binary = df[['Salnty', 'T_degC']].dropna()
df_binary = df_binary[:1000]


#df of the only two columns you want in correlation plot and plot them with regression line
df_binary.columns = ['Sal', 'Temp']
df_binary.head(20)


# WORKING WITH SMALLER DATASET
# Selecting the 1st 500 rows of the data
df_binary500 = df_binary[:][:1000]

sns.lmplot(x="Sal", y="Temp", data=df_binary500,
           order=2, ci=None).set_titles(title='hoi')
df_binary500.fillna(method ='ffill', inplace = True)

X = np.array(df_binary500['Sal']).reshape(-1, 1)
y = np.array(df_binary500['Temp']).reshape(-1, 1)

df_binary500.dropna(inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

regr = LinearRegression()
regr.fit(X_train, y_train)
print(f'''regression score:
{regr.score(X_test, y_test)}
''')

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='r')
plt.title('titleyouknow')
plt.legend(title="hoi")
plt.show()

#get the metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = root_mean_squared_error(y_true=y_test,y_pred=y_pred)
r_squared = r2_score(y_true=y_test,y_pred=y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R_squared", r_squared)

#test: 0.10 and 500rows
# regression score: 0.8943178652066541
# MAE: 0.6837191617892615
# MSE: 0.7598720332165375
# RMSE: 0.8717063916345558
# R_squared 0.8943178652066541

#test: 0.30 and 500rows
# regression score: 0.858852326071481
# MAE: 0.8925636779115579
# MSE: 1.253163275564423
# RMSE: 1.119447754727492
# R_squared 0.858852326071481

#test: 0.10 and 1000rows
# regression score:0.7882615653134978
# MAE: 1.2063477626881374
# MSE: 2.3264182274688134
# RMSE: 1.5252600524070685
# R_squared 0.7882615653134978

#test: 0.30 and 1000rows
# regression score: 0.727402352998848
# MAE: 1.2801427486899293
# MSE: 2.7546837460875016
# RMSE: 1.6597239969607904
# R_squared 0.727402352998848

new_sal = np.array([[35.0]])  # Let op: dubbele haakjes
predicted_temp = regr.predict(new_sal)

print(f"Voorspelde temperatuur bij salinity 35.0 = {predicted_temp[0][0]:.2f} °C")

#Voorspelde temperatuur bij salinity 35.0 = 1.31 °C
