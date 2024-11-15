# model.py
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# بارگذاری و پیش‌پردازش داده‌ها
current_dir = os.path.dirname(__file__)  # پوشه فعلی که model_tark.py در آن قرار دارد
csv_path = os.path.join(current_dir, 'student_math.csv') 

df = pd.read_csv('student_math.csv', sep=';', engine='python')


df['attendance_rate'] = 100 - (df['absences'] / df['absences'].max() * 100)
df['average_grade'] = df[['G1', 'G2', 'G3']].mean(axis=1)
df['disciplinary_actions'] = df['failures']
df['extracurricular_participation'] = df['activities'].map({'no': 0, 'yes': 1})
df['gender'] = df['sex'].map({'F': 0, 'M': 1})
df['economic_status'] = df[['Medu', 'Fedu']].mean(axis=1)
df['parental_support'] = df['famsup'].map({'no': 0, 'yes': 1})
# تعریف قاعده‌ای برای تعیین `dropped_out`
df['dropped_out'] = (
    (df['attendance_rate'] < 75) |                # حضور کمتر از 75%
    (df['average_grade'] < 12) |                  # میانگین نمرات کمتر از 12
    (df['disciplinary_actions'] > 2) |            # بیشتر از 2 اقدام انضباطی
    (df['parental_support'] == 0) |               # عدم حمایت والدین
    (df['economic_status'] < 2) |                 # وضعیت اقتصادی ضعیف
    (df['age'] > 15) |                            # سن بیشتر از 18
    (df['gender'] == 1)                           # جنسیت مذکر
).astype(int)


features = ['attendance_rate', 'average_grade', 'disciplinary_actions', 'extracurricular_participation', 
            'age', 'gender', 'economic_status', 'parental_support']


# تعریف ویژگی‌ها و هدف
X = df[['attendance_rate', 'average_grade', 'disciplinary_actions', 'extracurricular_participation', 
        'age', 'gender', 'economic_status', 'parental_support']]
y = df['dropped_out']




# تقسیم داده‌ها به مجموعه آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف و اعمال مقیاس‌کننده
scaler = StandardScaler()
X = scaler.fit_transform(X)

# تغییر شکل داده‌ها برای استفاده در LSTM
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ذخیره مقادیر میانگین و انحراف معیار برای استفاده در آینده
np.save('scaler.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)


model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# کامپایل و آموزش مدل با استفاده از EarlyStopping و ReduceLROnPlateau
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# ارزیابی مدل
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # تعریف مدل
# model = Sequential()
# model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # کامپایل کردن مدل
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # آموزش مدل
# model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# تابع ذخیره‌سازی مدل و مقیاس‌کننده
model.save('dropout_model.h5')
#ذخیره دقت
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

np.save('train_accuracy.npy', train_accuracy)
np.save('test_accuracy.npy', test_accuracy)

def predict_dropout(X_new):
    # Load scaler and model
    scaler_mean = np.load('scaler.npy')
    scaler_scale = np.load('scaler_scale.npy')
    X_new = (X_new - scaler_mean) / scaler_scale
    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
    prediction = model.predict(X_new)[0][0]
    return prediction * 100
