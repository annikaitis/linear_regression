import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('bottle.csv', delimiter=",", low_memory=False)
df_binary = df[['Salnty', 'T_degC', 'STheta']].dropna()
df_binary = df_binary[:1000]

print(np.isinf(df_binary).sum())

print(df_binary.isna().sum())        # check op NaNs
print(df_binary.describe())          # check op extreme waardes

# Trainingsdata
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(df_binary[['Salnty']].values.astype(np.float32))
y = scaler_y.fit_transform(df_binary[['T_degC']].values.astype(np.float32))

# Model maken (1 neuron met 1 input)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),         # expliciete inputlaag
    tf.keras.layers.Dense(units=1)
])

# Compileer het model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss='mean_squared_error')

# Train het model
history = model.fit(x, y, epochs=1000, verbose=0)
print(history)

# Gewichten ophalen
weights = model.layers[0].get_weights()
print("Gewichten:", weights)
w, b = weights[0][0][0], weights[1][0]

# Voorspellen
predictions = model.predict(x)
print(predictions)

# Visualisatie
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predictions, label='Fitted line')
plt.title('Linear Regression with Keras')
plt.legend()
plt.show()

# Toon resultaat
print(f"Training loss: {model.evaluate(x, y, verbose=0)}")
print(f"Gewicht (W): {w:.4f}, Bias (b): {b:.4f}")

# 1. Standaardiseer input net als bij training
new_sal = np.array([[35.0]], dtype=np.float32)
new_sal_scaled = scaler_x.transform(new_sal)

# 2. Doe voorspelling met geschaalde input
prediction_scaled = model.predict(new_sal_scaled)

# 3. Terugtransformeer de output naar echte temperatuur
prediction_real = scaler_y.inverse_transform(prediction_scaled)

# 4. Print
print(f"Voorspelling voor Salinity 35.0 = {prediction_real[0][0]:.2f} graden C")
# Voorspelling voor Salinity 35.0 = 1.31 graden C
