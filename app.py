import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_svr_hiv import train_and_predict

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Dashboard Prediksi HIV",
    layout="wide",
    page_icon="📊"
)

# Header dengan gambar atau emoji
st.title("📊 Dashboard Analisis & Prediksi Kasus HIV di Jawa Barat")
st.markdown("""
Selamat datang di dashboard interaktif untuk analisis dan prediksi kasus HIV berdasarkan data Dinas Kesehatan Jawa Barat.
Dashboard ini menggunakan **Support Vector Regression (SVR)** untuk memprediksi tren kasus HIV.
""")

# Tambahkan pemisah
st.divider()

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("dinkes-od_18510_jumlah_kasus_hiv_berdasarkan_kabupatenkota_v3_data.csv")
    # Pastikan kolom tahun sebagai integer
    df['tahun'] = df['tahun'].astype(int)
    return df

df = load_data()

# ===============================
# SIDEBAR FILTER
# ===============================
st.sidebar.header("🔍 Filter Data")
st.sidebar.markdown("Pilih tahun dan kabupaten/kota untuk analisis:")

tahun_pilih = st.sidebar.multiselect(
    "Pilih Tahun",
    sorted(df['tahun'].unique()),
    default=sorted(df['tahun'].unique())
)

kota_pilih = st.sidebar.multiselect(
    "Pilih Kabupaten/Kota",
    sorted(df['nama_kabupaten_kota'].unique()),
    default=sorted(df['nama_kabupaten_kota'].unique())
)

df_filter = df[
    (df['tahun'].isin(tahun_pilih)) &
    (df['nama_kabupaten_kota'].isin(kota_pilih))
]

# Info data
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Info Data")
st.sidebar.write(f"Total data: {len(df_filter)} baris")
st.sidebar.write(f"Rentang tahun: {min(tahun_pilih)} - {max(tahun_pilih)}")
st.sidebar.write(f"Jumlah kab/kota: {len(kota_pilih)}")

# ===============================
# KPI RINGKASAN
# ===============================
st.subheader("📌 Ringkasan Data")

total_kasus = int(df_filter['jumlah_kasus'].sum())
kota_tertinggi = (
    df_filter.groupby('nama_kabupaten_kota')['jumlah_kasus']
    .sum()
    .idxmax()
)
rata_rata_kasus = round(df_filter['jumlah_kasus'].mean(), 2)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Kasus HIV", f"{total_kasus:,}")
with col2:
    st.metric("Kab/Kota Tertinggi", kota_tertinggi)
with col3:
    st.metric("Jumlah Kab/Kota", df_filter['nama_kabupaten_kota'].nunique())
with col4:
    st.metric("Rata-rata Kasus per Entry", rata_rata_kasus)

st.divider()

# ===============================
# GRAFIK 1: TREN TAHUNAN
# ===============================
st.subheader("📈 Tren Kasus HIV dari Tahun ke Tahun")

kasus_tahunan = df_filter.groupby('tahun')['jumlah_kasus'].sum()

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(kasus_tahunan.index, kasus_tahunan.values, marker='o', color='blue', linewidth=2, markersize=8)
ax1.set_title("Tren Kasus HIV Tahunan", fontsize=16, fontweight='bold')
ax1.set_xlabel("Tahun", fontsize=12)
ax1.set_ylabel("Jumlah Kasus", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#f9f9f9')

st.pyplot(fig1)

# ===============================
# GRAFIK 2: RANKING KOTA
# ===============================
st.subheader("🏙️ 10 Kabupaten/Kota dengan Kasus HIV Tertinggi")

ranking = (
    df_filter.groupby('nama_kabupaten_kota')['jumlah_kasus']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ranking.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
ax2.set_title("Top 10 Kabupaten/Kota Kasus HIV Tertinggi", fontsize=16, fontweight='bold')
ax2.set_ylabel("Total Kasus", fontsize=12)
ax2.set_xlabel("Kabupaten/Kota", fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)
ax2.set_facecolor('#f9f9f9')

# Tambahkan nilai di atas bar
for i, v in enumerate(ranking.values):
    ax2.text(i, v + max(ranking.values)*0.01, f'{int(v):,}', ha='center', va='bottom', fontsize=10)

st.pyplot(fig2)

# ===============================
# PROSES MODEL SVR
# ===============================
st.subheader("🤖 Hasil Prediksi Menggunakan SVR")

with st.spinner("Memproses model SVR..."):
    df_model, svr_model = train_and_predict(df)

st.success("Model SVR berhasil diproses!")

# ===============================
# GRAFIK 3: AKTUAL vs PREDIKSI
# ===============================
st.markdown("### 📊 Perbandingan Kasus Aktual vs Prediksi")

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(df_model['jumlah_kasus'], df_model['prediksi'], alpha=0.6, color='green', s=50)
ax3.plot(
    [df_model['jumlah_kasus'].min(), df_model['jumlah_kasus'].max()],
    [df_model['jumlah_kasus'].min(), df_model['jumlah_kasus'].max()],
    linestyle='--', color='red', linewidth=2, label='Garis Ideal'
)
ax3.set_title("Aktual vs Prediksi Kasus HIV", fontsize=16, fontweight='bold')
ax3.set_xlabel("Kasus Aktual", fontsize=12)
ax3.set_ylabel("Kasus Prediksi", fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_facecolor('#f9f9f9')

st.pyplot(fig3)

# ===============================
# GRAFIK 4: TREN PREDIKSI PER TAHUN
# ===============================
st.markdown("### 📈 Tren Prediksi Kasus HIV per Tahun")

prediksi_tahun = df_model.groupby('tahun')['prediksi'].sum()

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(prediksi_tahun.index, prediksi_tahun.values, marker='s', color='orange', linewidth=2, markersize=8, label='Prediksi')
ax4.set_title("Tren Prediksi Kasus HIV Tahunan", fontsize=16, fontweight='bold')
ax4.set_xlabel("Tahun", fontsize=12)
ax4.set_ylabel("Prediksi Jumlah Kasus", fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_facecolor('#f9f9f9')

st.pyplot(fig4)

# ===============================
# TABEL DATA
# ===============================
st.subheader("📄 Data Lengkap")

# Tambahkan tombol download
csv = df_filter.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Data sebagai CSV",
    data=csv,
    file_name='data_hiv_filtered.csv',
    mime='text/csv',
    help="Unduh data yang telah difilter dalam format CSV"
)

st.dataframe(df_filter, width='stretch')

# Footer
st.divider()
st.markdown("**Dashboard dibuat dengan Streamlit | Data dari Dinas Kesehatan Jawa Barat**")
