import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

df = pd.read_csv('bottle.csv', delimiter=",", low_memory=False)
df_binary = df[['Salnty', 'T_degC', 'STheta']][:1000]

#df of the only two columns you want in correlation plot and plot them with regression line
df_binary.columns = ['Sal', 'Temp', 'STheta']
print(df_binary[df_binary['STheta'].notna()].head(20))

sns.regplot(x='Sal', y='Temp', data=df_binary)
plt.title("plot1")
plt.show()

sns.regplot(x='Sal', y='STheta', data=df_binary)
plt.title("plot1")
plt.show()

model = smf.ols(formula='Salnty ~ T_degC + STheta', data=df).fit()
print(model.summary())

