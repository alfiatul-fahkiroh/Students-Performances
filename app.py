import streamlit as st
import pandas as pd
import joblib # Untuk memuat model .pkl

# --- Judul Aplikasi ---
st.title('Prediksi Kategori Waktu Lulus Mahasiswa')
st.write('Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus "On Time" atau "Late" berdasarkan beberapa faktor.')

# --- Memuat Model ---
# Pastikan 'model_graduation.pkl' ada di direktori yang sama
try:
    model_graduation = joblib.load('model_graduation.pkl')
    st.success('Model berhasil dimuat!')
except FileNotFoundError:
    st.error("Error: File 'model_graduation.pkl' tidak ditemukan. Pastikan model berada di direktori yang sama dengan aplikasi.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# --- Input Pengguna ---
st.header('Masukkan Data Mahasiswa:')

new_ACT = st.number_input('Nilai ACT Composite Score:', min_value=1.0, max_value=36.0, value=25.0, step=0.1)
new_SAT = st.number_input('Nilai SAT Total Score:', min_value=400.0, max_value=1600.0, value=1200.0, step=1.0)
new_GPA = st.number_input('Nilai Rata-rata SMA (GPA):', min_value=0.0, max_value=4.0, value=3.0, step=0.01)
new_income = st.number_input('Pendapatan Orang Tua (USD):', min_value=0.0, value=50000.0, step=1000.0)
new_education = st.number_input('Tingkat Pendidikan Orang Tua (numerik, misal: 1=SD, 2=SMP, dst.):', min_value=0.0, max_value=20.0, value=12.0, step=1.0)

# --- Tombol Prediksi ---
if st.button('Prediksi Kategori Kelulusan'):
    try:
        # Buat DataFrame dari input baru
        new_data_df = pd.DataFrame(
            [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
            columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
        )

        # Lakukan prediksi
        # Asumsi 'nb' dalam kode asli Anda adalah 'model_graduation' yang dimuat di sini
        predicted_code = model_graduation.predict(new_data_df)[0]

        # Konversi hasil prediksi ke label asli
        label_mapping = {1: 'On Time', 0: 'Late'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.success(f"**Prediksi Kategori Masa Studi adalah: {predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# --- Informasi Tambahan (Opsional) ---
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini menggunakan model Machine Learning yang telah dilatih untuk memprediksi kategori waktu kelulusan.")