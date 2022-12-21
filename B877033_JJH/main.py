import tensorflow as tf
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = 'lemonade.csv'
lemonade = pd.read_csv(path)
lemonade.head()

independent = lemonade[['온도']]
dependent = lemonade[['판매량']]
print(independent.shape, dependent.shape)

X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(independent, dependent, epochs=1000, verbose=0)
model.fit(independent, independent, epochs=10)

print(model.predict(independent))
print(model.predict([[15]]))
