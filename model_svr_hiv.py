import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def train_and_predict(data):
    df = data.copy()

    # Encode kota
    df['kode_kota'] = df['nama_kabupaten_kota'].astype('category').cat.codes

    X = df[['tahun', 'kode_kota']]
    y = df['jumlah_kasus']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVR(kernel='rbf')
    model.fit(X_scaled, y)

    df['prediksi'] = model.predict(X_scaled)

    return df, model
