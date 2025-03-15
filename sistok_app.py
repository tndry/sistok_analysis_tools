import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime
import matplotlib.pyplot as plt
import os
import gdown
from sklearn.linear_model import LinearRegression




# Konfigurasi layout Streamlit
st.set_page_config(
    page_title="Sistok App",
    page_icon="🐟",
    layout="wide"
)
# # Inisialisasi OpenAI Client
# client = OpenAI(
#     api_key=st.secrets.DEEPSEEK,
#     base_url="https://api.deepseek.com"
           
#                 )

# Data ASLIII
# ID file Googel Drive
# file_id = '1eACQIHOn3oS96V8rHzN6VlMuKtNX5raz'
# drive_url = f'https://drive.google.com/uc?id={file_id}'


# Data DEMOO
# # ID file Googel Drive
# file_id = '1wXxn-GJtVfEaZJTH-IPe4svur9uLbZIs' 
# drive_url = f'https://drive.google.com/uc?id={file_id}'



# Fungsi untuk memuat data dari database atau file CSV
@st.cache_data
def load_data():
    try:
        # Download file CSV
        # ASLI
        # file_path = 'data_bersih.csv'  
        #DEMO
        file_path = 'data/data_bersih_demo.csv'
        # gdown.download(drive_url, file_path, quiet=False)
        # Baca file CCSV
        df = pd.read_csv(file_path,  low_memory=False)
        # Konversi tanggal ke tipe datetime
        df['tanggal_berangkat'] = pd.to_datetime(df['tanggal_berangkat'], errors='coerce')
        df['tanggal_kedatangan'] = pd.to_datetime(df['tanggal_kedatangan'], errors='coerce')
        df['tahun'] = df['tanggal_kedatangan'].dt.year

        
        return df

    except FileNotFoundError:
        st.error("File tidak ditemukan. Pastikan file 'data_bersih.csv' ada di folder './data/'." )
        return pd.DataFrame()

# Fungsi filter data
def filter_data(df, pelabuhan_kedatangan_id, nama_ikan_id, start_year, end_year, time_frame):

    # Filter data berdasarkan pelabuhan kedatangan
    if pelabuhan_kedatangan_id:
        df = df[df['pelabuhan_kedatangan_id'] == pelabuhan_kedatangan_id]

    # Filter data berdasarkan nama ikan
    if nama_ikan_id:
        df = df[df['nama_ikan_id'].isin(nama_ikan_id)]

    # Filter berdasarkan tahun
    if start_year:
        df = df[df['tahun'] >= start_year]
    if end_year:
        df = df[df['tahun'] <= end_year]

    # Filter berdasarkan time frame
    if time_frame == 'Daily':
        df['time_period'] = df['tanggal_kedatangan'].dt.date
    elif time_frame == 'Weekly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('W').astype(str)
    elif time_frame == 'Monthly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('M').astype(str)
    elif time_frame == 'Yearly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('Y').astype(str)


    return df

# Function to get OpenAI chat response

def analyze_fishing_data(query, filtered_data):
    """
    Fungsi untuk menganalisis data perikanan berdasarkan query pengguna
    """
    query = query.lower()
    response = ""
    
    try:
        # Analisis total tangkapan
        if 'total tangkapan' in query or 'berapa tangkapan' in query:
            # Filter berdasarkan jenis ikan jika disebutkan
            for fish in filtered_data['nama_ikan_id'].unique():
                if fish.lower() in query:
                    specific_data = filtered_data[filtered_data['nama_ikan_id'].str.lower() == fish.lower()]
                    
                    # Filter tahun jika disebutkan
                    for year in filtered_data['tahun'].unique():
                        if str(year) in query:
                            year_data = specific_data[specific_data['tahun'] == year]
                            total = year_data['berat'].sum()
                            return f"Total tangkapan {fish} pada tahun {year} adalah {total:,.2f} Kg"
                    
                    # Jika tahun tidak disebutkan, tampilkan semua tahun
                    yearly_data = specific_data.groupby('tahun')['berat'].sum()
                    response = f"Total tangkapan {fish} per tahun:\n"
                    for year, total in yearly_data.items():
                        response += f"Tahun {year}: {total:,.2f} Kg\n"
                    return response

        # Analisis alat tangkap
        elif 'alat tangkap' in query or 'jenis alat' in query:
            alat_tangkap = filtered_data.groupby('jenis_api')['berat'].sum().sort_values(ascending=False)
            response = "Alat tangkap yang digunakan (berdasarkan total tangkapan):\n"
            for alat, total in alat_tangkap.items():
                response += f"{alat}: {total:,.2f} Kg\n"
            return response

        # Analisis tren tahunan
        elif 'tren' in query or 'perkembangan' in query:
            yearly_trend = filtered_data.groupby('tahun')['berat'].sum()
            max_year = yearly_trend.idxmax()
            min_year = yearly_trend.idxmin()
            
            response = "Analisis tren tangkapan:\n"
            response += f"Tahun dengan tangkapan tertinggi: {max_year} ({yearly_trend[max_year]:,.2f} Kg)\n"
            response += f"Tahun dengan tangkapan terendah: {min_year} ({yearly_trend[min_year]:,.2f} Kg)\n"
            
            # Hitung pertumbuhan year-over-year
            yoy_growth = yearly_trend.pct_change() * 100
            response += "\nPertumbuhan year-over-year:\n"
            for year, growth in yoy_growth.items():
                if not pd.isna(growth):
                    response += f"{year}: {growth:,.1f}%\n"
            return response

        # Analisis nilai produksi
        elif 'nilai produksi' in query or 'nilai ekonomi' in query:
            if 'tahun' in query:
                for year in filtered_data['tahun'].unique():
                    if str(year) in query:
                        year_data = filtered_data[filtered_data['tahun'] == year]
                        total_value = year_data['nilai_produksi'].sum()
                        return f"Total nilai produksi tahun {year}: Rp {total_value:,.2f}"
            
            total_value = filtered_data['nilai_produksi'].sum()
            avg_value = filtered_data.groupby('tahun')['nilai_produksi'].mean()
            response = f"Total nilai produksi: Rp {total_value:,.2f}\n"
            response += "Rata-rata nilai produksi per tahun:\n"
            for year, value in avg_value.items():
                response += f"Tahun {year}: Rp {value:,.2f}\n"
            return response

        # Analisis jenis ikan
        elif 'jenis ikan' in query or 'ikan apa' in query:
            top_fish = filtered_data.groupby('nama_ikan_id')['berat'].sum().sort_values(ascending=False).head(5)
            response = "5 jenis ikan dengan tangkapan terbanyak:\n"
            for fish, total in top_fish.items():
                response += f"{fish}: {total:,.2f} Kg\n"
            return response

        # Default response
        else:
            return """Saya dapat membantu Anda menganalisis data perikanan. Anda dapat bertanya tentang:
                1 . Total tangkapan (per jenis ikan/tahun)
                2. Alat tangkap yang digunakan
                3. Tren tangkapan tahunan
                4. Nilai produksi
                5. Jenis ikan dominan

                Contoh: 'Berapa total tangkapan cumi tahun 2022?' atau 'Apa saja alat tangkap yang digunakan?'"""

    except Exception as e:
        return f"Maaf, terjadi kesalahan dalam menganalisis data: {str(e)}"

# Ganti fungsi get_openai_response dengan fungsi ini
def get_openai_response(query, filtered_data):
    return analyze_fishing_data(query, filtered_data)

# st.write('Kolom yang ada:', data.columns)


# CSS
st.markdown(
    """
    <style>
    .metric-box{
      border: 1px solid #ccc;
      padding: 10px;
      border-radius: 5px;
    
      margin: 5px;
      text-align: center;
    }
    </style>
""", unsafe_allow_html=True
)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown("<h1 style='text-align: center; '>SISTOK</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Fish Stock Analysis Tools</h2>", unsafe_allow_html=True)



# Menu
menu = option_menu(None, ['Dashboard', 'Analysis', 'About'],
    icons= ['house', 'graph-up', 'book'],
    menu_icon='cast', default_index=0, orientation='horizontal')

# # Sidebar untuk navigasi
# menu = st.sidebar.radio('Navigasi', ['Dashboard', 'Analysis', 'About'])

# Memuat data
data = load_data()

if menu == 'Dashboard':
    st.title('Dashboard')
     

    # Filter
    st.sidebar.subheader("Filter Data")
    pelabuhan = st.sidebar.selectbox("Pilih Pelabuhan", options=[None] + list(data['pelabuhan_kedatangan_id'].unique()))

    jenis_ikan = st.sidebar.multiselect("Pilih Jenis Ikan", options=list(data['nama_ikan_id'].unique()), default=[])

    start_year = st.sidebar.number_input('Start Year', min_value = int(data['tahun'].min()), max_value=int(data['tahun'].max()), value=int(data['tahun'].min()), step=1)
    
    end_year = st.sidebar.number_input('End Year', min_value=start_year, max_value=int(data['tahun'].max()), value=int(data['tahun'].max()), step=1)

    time_frame = st.sidebar.selectbox('Time Frame', ['Daily', 'Weekly', 'Monthly', 'Yearly'])

    
    # Filter data
    filtered_data = filter_data(data, pelabuhan, jenis_ikan, start_year, end_year, time_frame )

    # Chatbot
    st.sidebar.markdown('---')
    st.sidebar.subheader('Ask AI')

    # Chat input
    user_input = st.sidebar.text_input('Ask about the Data:', key='chat_input')

    # Send Button
    if st.sidebar.button('Send', key='send_button'):
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})

            # Get bot response
            with st.spinner('Thinking...'):
                bot_response = get_openai_response(user_input, filtered_data)
            # Add bot response to history
            st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response})

    #  Display chat history
    st.sidebar.markdown("### Riwayat Chat")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.sidebar.markdown(f"**Anda:** {message['content']}")
        else:
            st.sidebar.markdown(f"**Assistant:** {message['content']}")
        st.sidebar.markdown("---")

    # Clear chat history button
    if st.sidebar.button("Hapus Riwayat Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

    

 # Cek kelengkapan data tahun 2024
    if 2024 in range(start_year, end_year+1):
        if not filtered_data.empty:
            data_tahun_2024 = filtered_data[filtered_data['tahun'] == 2024]
            if data_tahun_2024.empty or data_tahun_2024['berat'].sum() == 0:
                st.warning("⚠️ Data tahun 2024 belum lengkap. Mohon diperhatikan!")
            else:
                st.success("✅ Data tahun 2024 sudah lengkap.")
        else:
            st.warning("Data tidak tersedia. Silahkan periksa kembali filter Anda.")


    # Rename column
    columns_to_rename ={
        
        'nilai_produksi': 'Nilai Produksi',
        'jumlah_hari': 'Jumlah Hari',
        'pelabuhan_kedatangan_id': 'Pelabuhan Kedatangan',
        'pelabuhan_keberangkatan_id': 'Pelabuhan Keberangkatan',
        'kelas_pelabuhan': 'Port Class',
        'provinsi': 'Provinsi',
        'tanggal_berangkat': 'Tanggal Berangkat',
        'tanggal_kedatangan': 'Tanggal Kedatangan',
        
    }
    #  rename kolom yang ada di dataframe
    filtered_data = filtered_data.rename(columns={k: v for k, v in columns_to_rename.items() if k in filtered_data.columns})

    # Ringkasan data
    with st.expander('PREVIEW DATASET'):
        showData= st.multiselect('Filter: ', filtered_data.columns, default=filtered_data.columns)
        st.dataframe(filtered_data[showData], use_container_width=True)
     
    #  compute top analytics
    if not filtered_data.empty:
        total_tangkapan = float(pd.Series(filtered_data['berat']).sum())
        total_nilai_produksi = float(pd.Series(filtered_data['Nilai Produksi']).sum())
        total_hari = filtered_data['Jumlah Hari'].sum()
        total_ikan = filtered_data['nama_ikan_id'].nunique()
    else:
        total_tangkapan = total_nilai_produksi = total_hari = total_ikan = 0


    # Display top analytics
    total1, total2, total3, total4 = st.columns(4, gap='small')
    with total1:
        st.markdown(f'<div class="metric-box"> 🐟<br>Total Tangkapan<br><b>{total_tangkapan:,.0f} Kg </b></div>', unsafe_allow_html=True)
    with total2:
        st.markdown(f"<div class='metric-box'>💵<br>Nilai Produksi<br><b>{total_nilai_produksi:,.0f} IDR</b></div>", unsafe_allow_html=True)
    with total3:
        st.markdown(f"<div class='metric-box'>📆<br>Total Hari<br><b>{total_hari}</b></div>", unsafe_allow_html=True)
    with total4:
        st.markdown(f"<div class='metric-box'>🎣<br>Jenis Ikan<br><b>{total_ikan}</b></div>", unsafe_allow_html=True)

    # Graph
    
    
    # Grafik 1 : Data Tangkapan per Tahun
    # st.subheader('Tangkapan per Tahun')
    tangkapan_tahunan = filtered_data.groupby('tahun').agg({'berat':'sum'}).reset_index()
    fig_tangkapan = px.line(
        tangkapan_tahunan, 
        x='tahun', 
        y='berat', 
        orientation= 'v',
        title='TOTAL BERAT TANGKAPAN',
        color_discrete_sequence = ['#0083b8']*len(tangkapan_tahunan),
        template='plotly_dark',
    )
    fig_tangkapan.update_layout(
        xaxis=dict(tickmode='linear'),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=(dict(showgrid=False))
    )
    

     # Grafik 2: 10 Jenis Tangkapan Terbanyak
    tangkapan_dominan = (
        filtered_data.groupby('nama_ikan_id').agg({'berat':'sum'}).reset_index().sort_values(by='berat', ascending=False).head(10))
    fig_tangkapan_dominan = px.bar(
        tangkapan_dominan,
        x='berat', 
        y= 'nama_ikan_id',
        orientation='h',
        title="<b> JENIS TANGKAPAN TERBANYAK </b>",
        color_discrete_sequence=['#0083b8']*len(tangkapan_dominan),
        template='plotly_dark'
        )
    fig_tangkapan_dominan.update_layout(
        plot_bgcolor = 'rgba(0,0,0,0)',
        font=dict(color='black'),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#cecdcd',
            categoryorder= 'total ascending'),
        paper_bgcolor= 'rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#cecdcd'),
        )
    
    left,right,center=st.columns(3)
    left.plotly_chart(fig_tangkapan,use_container_width=True)
    right.plotly_chart(fig_tangkapan_dominan, use_container_width=True)
   
    with center:
    # Pie chart
        alat_tangkap_dominan = filtered_data.groupby('jenis_api').agg({'berat':'sum'}).reset_index().sort_values(by='berat', ascending=False).head(10)
        fig_alat_tangkap = px.pie(alat_tangkap_dominan, names='jenis_api', values='berat', title='ALAT TANGKAP DOMINAN')
        fig_alat_tangkap.update_layout(legend_title='Alat Tangkap', legend_y=0.9)
        fig_alat_tangkap.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_alat_tangkap, use_container_width=True)
          

elif menu == 'Analysis':
    st.title('Analysis')
   
    # Display Upload Button
   
    df_sample = pd.read_csv('./data/data_kembung_karangantu.csv')

    

    
    try:
        with open('./data/data_kembung_karangantu.csv', 'r') as file:
            sample_csv_content = file.read()
        st.download_button(
            label='Download Sample CSV',
            data=sample_csv_content,
            file_name='sample_data.csv',
            mime='text/csv'
        )
    except FileNotFoundError:
        st.error('Sample data tidak ditemukan. Silahkan periksa kembali file Anda.')

    # Komponen upload file
    uploaded_file = st.file_uploader(
        'Choose a file',
        type=['csv'],
        help='Limit: 200MB per file'
    )  

    # Proses jika file diupload
    if uploaded_file is not None:
        # Membaca file yang diupload
        user_data = pd.read_csv(uploaded_file)
        st.success('File uploaded succesfully!')
        with st.expander('Your Dataset:'):
            st.dataframe(user_data)

        # Analisis: Data Tangkapan per Tahun
        if 'tahun' in user_data.columns:

            if 'Nilai Produksi' in user_data.columns:
            
            # Grafik 1: Data Tangkapan per Tahun
            # Mengelompokkan data berdasarkan tahun dan menjumlahkan berat
                data_per_year = user_data.groupby('tahun').agg({'berat': 'sum', 'Nilai Produksi': 'sum'}).reset_index()

            # Menghitung rata-rata nilai produksi dan nilai produksi
                data_per_year['Harga rata-rata nilai produksi'] = data_per_year['Nilai Produksi'] / data_per_year['berat']
                data_per_year['Produksi (Ton)'] = data_per_year['berat'] / 1000
                data_per_year['Nilai Produksi'] = data_per_year['Produksi (Ton)'] * data_per_year['Harga rata-rata nilai produksi'] 
            else:
                data_per_year = user_data.groupby('tahun').agg({'berat': 'sum'}).reset_index()
                data_per_year['Produksi (Ton)'] = data_per_year['berat'] / 1000
                data_per_year['Harga rata-rata nilai produksi'] = None
                data_per_year['Nilai Produksi'] = None
            
            
            # Membuat grafik garis menggunakan plotly
            fig_data_per_year = px.line(
                data_per_year, 
                x='tahun', 
                y='berat', 
                orientation='v',
                title='TOTAL BERAT TANGKAPAN PER TAHUN', 
                template='plotly_dark'
            )
            
            # Memperbarui layout grafik
            fig_data_per_year.update_layout(
                xaxis=dict(tickmode='linear'),
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(showgrid=False)
            )

            # Grafik 2: 
            api_dominan = user_data.groupby('jenis_api').agg({'berat':'sum'}).reset_index().sort_values(by='berat', ascending=False).head(10)
            fig_api_dominan = px.bar(
                api_dominan,
                x='berat', 
                y= 'jenis_api',
                orientation='h',
                title="<b> JENIS API DOMINAN </b>",
                color_discrete_sequence=['#0083b8']*len(api_dominan),
                template='plotly_dark'
                )
            fig_api_dominan.update_layout(
                plot_bgcolor = 'rgba(0,0,0,0)',
                font=dict(color='black'),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='#cecdcd',
                    categoryorder= 'total ascending'),
                paper_bgcolor= 'rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#cecdcd'),
                )
            
            
           
            left, right = st.columns(2)
            left.plotly_chart(fig_data_per_year, user_container_width=True)
            right.plotly_chart(fig_api_dominan, user_container_width=True)

            with st.expander('DATA PRODUKSI DAN NILAI PRODUKSI PER TAHUN'):
                st.dataframe(data_per_year[['tahun', 'Produksi (Ton)', 'Harga rata-rata nilai produksi', 'Nilai Produksi']].reset_index(drop=True), use_container_width=True)
            

        
        
        st.markdown(
            """
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
            <hr>

           <div class="card mb-3" style='background-color: black; color: white;'>
                <div class="card">
                <div class="card-body">
                    <h3 class="card-title"style="color:#007710;"><strong>📈 ANALISIS LANJUTAN: MODEL PRODUKSI SURPLUS</strong></h3>
                    <p class="card-text">Model Produksi Surplus adalah salah satu model yang digunakan dalam analisis stok ikan untuk mengukur kelimpahan stok ikan dan hubungannya dengan upaya penangkapan (effort). Model ini membantu memprediksi tingkat eksploitasi optimal untuk menjaga keberlanjutan sumber daya perikanan. </p>
                    <p class="card-text"><small class="text-body-secondary"> </small></p>
                </div>
                </div>
                </div>
                <style>
                    [data-testid=stSidebar] {
                        color: white;
                        background-color: black;
                        text-size:24px;
                    }
                    .card{
                        background-color: black;
                        border: 1px solid #444
                    }
                    .card-body{
                        color: white;
                    }
                </style>
                """,unsafe_allow_html=True
                )
        

        # Hasil Tangkapan per Alat Tangkap 
        with st.expander('⬇ Hasil Tangkapan per Alat Tangkap'):	
            if {'jenis_api', 'tahun', 'berat'}.issubset(user_data.columns):
                tangkapan_per_tahun = user_data.groupby(['jenis_api', 'tahun']).agg({'berat': 'sum'}).reset_index()
                
                tangkapan_pivot = tangkapan_per_tahun.pivot(index='jenis_api', columns='tahun', values='berat').fillna(0)

                # Tambahkan kolom total untuk tiap alat tangkap
                tangkapan_pivot['Total'] = tangkapan_pivot.sum(axis=1)

                # Tambahkan baris jumlah total untuk tiap alat tangkap
                tangkapan_pivot.loc['Jumlah'] = tangkapan_pivot.sum()

                # Reset index untuk tampilkan tabel
                tangkapan_pivot = tangkapan_pivot.reset_index()

                # Tampilkan tabel
                st.write('Hasil Tangkapan per Alat Tangkap')
                st.dataframe(tangkapan_pivot, use_container_width=True)

                # Membuat grafik batang hasil tangkapan
            fig_tangkapan_total = px.bar(
                tangkapan_pivot[tangkapan_pivot['jenis_api'] != 'Jumlah'],
                x='jenis_api',
                y='Total',
                # orientation='h',
                title="<b>Hasil Tangkapan per Alat Tangkap</b>",
                color='jenis_api',
                template='plotly_dark'
            )
            fig_tangkapan_total.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#cecdcd'),
                yaxis=dict(categoryorder='total ascending')
            )
            st.plotly_chart(fig_tangkapan_total, use_container_width=True)

        # Jumlah Trip per Alat Tangkap
        with st.expander('⬇ Jumlah Trip per Alat Tangkap'):
            if {'jenis_api', 'tahun', 'Jumlah Hari'}.issubset(user_data.columns):

                effort_per_tahun = user_data.groupby(['jenis_api', 'tahun']).agg({'Jumlah Hari': 'sum'}).reset_index()
                effort_pivot = effort_per_tahun.pivot(index='jenis_api', columns='tahun', values='Jumlah Hari').fillna(0)

                # Tambahkan kolom total untuk tiap alat tangkap
                effort_pivot['Total'] = effort_pivot.sum(axis=1)
                # Tambahkan baris jumlah total untuk tiap alat tangkap
                effort_pivot.loc['Jumlah'] = effort_pivot.sum()

                # Reset index untuk tampilkan tabel
                effort_pivot = effort_pivot.reset_index()
                

                # Tampilkan tabel
                st.write('Jumlah Trip per Alat Tangkap')
                st.dataframe(effort_pivot, use_container_width=True)

                # Membuat grafik batang jumlah trip
                fig_trip_per_alat = px.bar(
                    effort_pivot[effort_pivot['jenis_api'] != 'Jumlah'],
                    x='jenis_api',
                    y='Total',
                    # orientation='h',
                    title="<b>Jumlah Trip per Alat Tangkap</b>",
                    color='jenis_api',
                    template='plotly_dark'
                )
                fig_trip_per_alat.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(categoryorder='total ascending')
                )
                st.plotly_chart(fig_trip_per_alat, use_container_width=True)

            
        # # Analisis Lanjutan: Model Produksi Surplus
        with st.expander('⬇ CPUE'):
            # st.write('CPUE (Catch Per Unit Effort) adalah rasio antara jumlah tangkapan ikan dengan upaya penangkapan yang dilakukan.')

        
            if 'jenis_api' in user_data.columns and 'berat' in user_data.columns and 'Jumlah Hari' in user_data.columns:
                user_data.rename(columns={'jenis_api' : 'Alat Tangkap', 'berat': 'catch (ton)', 'Jumlah Hari': 'effort (hari)'}, inplace=True)

                # Konversi tangkapan ke ton
                user_data['catch (ton)'] = user_data['catch (ton)'] / 1000 
                # Kelompokkan data berdasarkan jenis alat tangkap
                alat_tangkap_group = user_data.groupby('Alat Tangkap').agg({'catch (ton)': 'sum', 'effort (hari)': 'sum'}).reset_index()

                # Menambah kolom CPUE
                alat_tangkap_group['CPUE'] = alat_tangkap_group['catch (ton)'] / alat_tangkap_group['effort (hari)']

                # Hitung persentase kontribusi setiap alat tangkap
                total_catch = alat_tangkap_group['catch (ton)'].sum()
                alat_tangkap_group['percentage'] = (alat_tangkap_group['catch (ton)'] / total_catch) *100

                # Urutkan data berdasarkan persentase tangkapan
                alat_tangkap_group = alat_tangkap_group.sort_values(by='percentage', ascending=False)


                # Threshold untuk alat tangkap dominan (80%)
                DOMINANCE_THRESHOLD = 50

                # Threshold untuk kontribusi signifikan alat tangkap lain (20%)
                SIGNIFICANT_THRESHOLD = 20

                # Cek apakah ada alat tangkap yang dominan (>=80%)
                dominant_gear = alat_tangkap_group.iloc[0]
                is_dominant = dominant_gear['percentage'] >= DOMINANCE_THRESHOLD

                if is_dominant:
                    # Jika ada alat tangkap dominan (>=80%)
                    st.write(f"Alat tangkap dominan adalah {dominant_gear['Alat Tangkap']}" f" dengan persentase tangkapan {dominant_gear['percentage']:.2f}% dari total tangkapan")

                    # Gunakan langsung sebagai alat tangkap standar
                    alat_tangkap_dominan = pd.DataFrame([dominant_gear])
                    alat_tangkap_dominan['FPI'] = 1.0

                else:
                    # Jika tidak ada alat tangkap dominan, gunakan threshold signifikansi
                    significant_gear = alat_tangkap_group[alat_tangkap_group['percentage'] >= SIGNIFICANT_THRESHOLD]

                    st.write(f"Tidak ada alat tangkap dominan (>=50%)." f"Melakukan standarisasi untuk {len(significant_gear)} alat tangkap yang berkontribusi >=20%.")

                    # Lakukan standarisasi untuk alat tangkap yang signifikan
                    alat_tangkap_dominan = significant_gear.copy()
                    cpue_max = alat_tangkap_dominan['CPUE'].max()
                    alat_tangkap_dominan['FPI'] = alat_tangkap_dominan['CPUE'] / cpue_max
                    alat_tangkap_dominan.loc[alat_tangkap_dominan['CPUE'] == cpue_max, 'FPI'] = 1

                # Tampilkan data dalam tabel
                st.write('Data CPUE per Alat Tangkap:')
                display_cols = ['Alat Tangkap', 'catch (ton)', 'effort (hari)', 'CPUE', 'percentage', 'FPI']
                st.table(alat_tangkap_dominan[display_cols])

                # # Pilih 2 alat tangkap dominan
                # alat_tangkap_dominan = alat_tangkap_group.head(2)

                # # Cari CPUE tertinggi
                # cpue_max = alat_tangkap_dominan['CPUE'].max()

                # # Kolom FPI
                # alat_tangkap_dominan['FPI'] = alat_tangkap_dominan['CPUE'] / cpue_max

                # alat_tangkap_dominan.loc[alat_tangkap_dominan['CPUE'] == cpue_max, 'FPI'] = 1

                # # Tampilkan data dalam tabel
                # st.write('Data CPUE per Alat Tangkap:')
                # st.table(alat_tangkap_dominan)

                # Buat grafik batang
                fig_cpue = px.bar(
                    alat_tangkap_dominan,
                    x='Alat Tangkap',
                    y='CPUE',
                    text='CPUE',	
                    title='CPUE per Alat Tangkap',
                    labels={'Alat Tangkap': 'Jenis Alat Tangkap', 'CPUE': 'CPUE'},
                    template='plotly_dark',
                    color= 'Alat Tangkap'
                )
                fig_cpue.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_cpue.update_layout(showlegend=False)

                st.plotly_chart(fig_cpue, use_container_width=True)

                if is_dominant:
                    # Jika ada alat tangkap dominan, gunakan effort dari alat tangkap tersebut
                    yearly_effort = user_data[user_data['Alat Tangkap'] == dominant_gear['Alat Tangkap']].groupby('tahun')['effort (hari)'].sum().reset_index()
                else:
                    # Jika tidak ada yg dominan, standarisasi effort dengan FPI
                    standardized_efforts = []
                    for gear in alat_tangkap_dominan['Alat Tangkap']:
                        gear_data = user_data[user_data['Alat Tangkap'] == gear]
                        gear_fpi = alat_tangkap_dominan.loc[alat_tangkap_dominan['Alat Tangkap'] == gear, 'FPI'].iloc[0]

                        # Hitung effort terstandarisasi untuk setiap alat tangkap
                        gear_effort = gear_data.groupby('tahun')['effort (hari)'].sum() * gear_fpi
                        standardized_efforts.append(pd.DataFrame({
                            'tahun' : gear_effort.index,
                            'effort_std' : gear_effort.values
                        }))

                    # Gabungkan semua effort yang sudah distandarisasi
                    combined_efforts = pd.concat(standardized_efforts)
                    yearly_effort = combined_efforts.groupby('tahun')['effort_std'].sum().reset_index()
                    yearly_effort.columns = ['tahun', 'effort (hari)']

                # Hitung total cactch per tahun
                yearly_catch = user_data.groupby('tahun')['catch (ton)'].sum().reset_index()

                # Gabungkan data catch dan effort
                yearly_data = pd.merge(yearly_catch, yearly_effort, on='tahun')
                yearly_data['CPUE'] = yearly_data['catch (ton)'] /yearly_data['effort (hari)']

                # Tampilkan data tahunan
                st.write('Data Tahunan:')
                st.table(yearly_data)

                # Function untuk menghitung R-squared
                def calculate_r2(y_true, y_pred):
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / ss_tot)
                    return r2

                # PERHITUNGAN MODEL SCHAEFER
                def calculate_schaefer(data):
                    X = data['effort (hari)'].values.reshape(-1, 1)
                    Y = data['CPUE']


                    # Regresi Linear
                    model = LinearRegression()
                    model.fit(X, Y)

                    # Parameter model Schaefer
                    a = model.intercept_  #intercept
                    b = model.coef_[0] #slope

                    # Prediksi CPUE untuk R^2
                    Y_pred = model.predict(X)
                    r2 = calculate_r2(Y, Y_pred)


                    # Hitung MSY dan Eopt
                    Eopt = -a / (2*b)
                    CMSY = -(a ** 2) / (4 * b)

                    return {
                        'name' : 'Schaefer',
                        'a': a,
                        'b': b,
                        'Eopt': Eopt,
                        'CMSY': CMSY,
                        'R2': r2

                    }

                # PERHITUNGAN MODEL FOX
                def calculate_fox(data):
                    X = data['effort (hari)'].values.reshape(-1, 1)
                    Y = np.log(data['CPUE']) #Menggunakan Ln(CPUE)

                    # Regresi linear
                    model = LinearRegression()
                    model.fit(X ,Y)

                    # Parameter model
                    # c = np.exp(model.intercept_) 
                    c = model.intercept_
                    d = model.coef_[0]

                    # Prdiksi Ln(CPUE ) untuk R2
                    Y_pred = model.predict(X)
                    r2 = calculate_r2(Y, Y_pred)

                    # Hitung MSY dan Eopt untuk Fox
                    Eopt = -1 / d
                    CMSY = -(1/d) * np.exp(c-1) 

                    # Buat array untuk menyimpan nilai effort dan catch
                    n_points = 20
                    effort_points = np.zeros(n_points)
                    catch_points = np.zeros(n_points)

                    # Isi nilai data ke-1 dengan nilai 0
                    effort_points[0] = 0 
                    catch_points[0] = 0

                    # Data ke 2 sampapi ke-2 menggunakan rumus: data sebelumnya + Eopt*0.1
                    for i in range(1, n_points):
                        effort_points[i] = effort_points[i-1] + (Eopt * 0.1)
                        # Hitung catch menggunakan rumus Ct = Et * Exp(c+ dEt)
                        catch_points[i] = effort_points[i] * np.exp(c + d * effort_points[i])

                    return {
                        'name': 'Fox',
                        'c': c, 
                        'd': d,
                        'Eopt': Eopt,
                        'CMSY': CMSY,
                        'R2': r2,
                        'effort_range' : effort_points,
                        'catch_pred': catch_points
                    }

                # Hitung kedua model
                schaefer_results = calculate_schaefer(yearly_data)
                fox_results = calculate_fox(yearly_data)

                # Tampilkan hasil perhitungan kedua model
                st.write('### Hasil Perhitungan Model')
                comparison_df = pd.DataFrame({
                    'Parameter' : ['a', 'b', 'R2', 'Eopt', 'MSY'],
                    'Model Schaefer': [
                        f"{schaefer_results['a']:.6f}",
                        f"{schaefer_results['b']:.6f}",	
                        f"{schaefer_results['R2']:.4f}",
                        f"{schaefer_results['Eopt']:.2f} hari",
                        f"{schaefer_results['CMSY']:.2f} ton"
                    ],
                    'Model Fox': [
                        f"{fox_results['c']:.4f}",
                        f"{fox_results['d']:.4f}",
                        f"{fox_results['R2']:.4f}",
                        f"{fox_results['Eopt']:.2f} hari",
                        f"{fox_results['CMSY']:.2f} ton"
                    ]
                })
                st.table(comparison_df)

                # Pilih model untuk visualisasi
                selected_model = st.radio(
                    "Pilih model yang akan ditampilkan (berdasarkan nilai R^2):",
                    [f"Model Schaefer(R² = {schaefer_results['R2']:.4f})",
                    f"Model Fox (R² = {fox_results['R2']:.4f})"
                    ]
                )

                # Visualisasi model yang dipilih 
                if 'Schaefer' in selected_model:
                    model_results = schaefer_results

                    # Buat 20 titik data dari 0 sampai 2 Eopt
                    n_points = 20
                    effort_range = np.linspace(0, 2 * model_results['Eopt'], n_points)
                    # Hitung catch menggunakan schaefer
                    catch_pred = effort_range * (model_results['a'] + model_results['b'] * effort_range)

                else:
                    model_results = fox_results
                    # Buat 20 titik data sesuai panduan buku
                    n_points = 100
                    
                    # Inisialisasi array untuk effort dan catch
                    effort_range = np.zeros(n_points)
                    catch_pred = np.zeros(n_points)
                    
                    # Data pertama = 0
                    effort_range[0] = 0
                    catch_pred[0] = 0
                    
                    # Hitung effort dan catch untuk titik 2-20
                    for i in range(1, n_points):
                        # Effort = data sebelumnya + (Eopt * 0.1)
                        effort_range[i] = effort_range[i-1] + (model_results['Eopt'] * 0.1)
                        # Catch = Et * exp(a + b*Et)
                        catch_pred[i] = effort_range[i] * np.exp(model_results['c'] + model_results['d'] * effort_range[i])

                # Setelah kedua model selesai, buat dataframe
                model_df = pd.DataFrame({
                    'Effort': effort_range,
                    'Catch': catch_pred
                })

                # Pastikan nilai catch di titik awal = 0
                model_df.loc[0, 'Catch'] = 0

                # Buat dataframe untuk data aktual
                actual_df = yearly_data[['effort (hari)', 'catch (ton)']]

                # Visualisasi 
                fig_surplus = go.Figure()

                # Tambahkan kurva model 
                fig_surplus.add_trace(
                    go.Scatter(
                        x=model_df['Effort'],
                        y=model_df['Catch'],
                        mode='lines',
                        name=f"Model {model_results['name']}",
                        line=dict(color='blue', width=2)
                    )
                )

                # Tambahkan data aktual
                fig_surplus.add_trace(
                    go.Scatter(
                        x=actual_df['effort (hari)'],
                        y=actual_df['catch (ton)'],
                        mode='markers',
                        name='Data Aktual',
                        marker=dict(color='red', size=8)
                    )
                )

                # Tambahkan titik MSY
                fig_surplus.add_trace(
                    go.Scatter(
                        x=[model_results['Eopt']],
                        y=[model_results['CMSY']],
                        mode='markers',
                        name='MSY',
                        marker=dict(color='green', size=12, symbol='star')
                    )
                )

                # Update layout
                fig_surplus.update_layout(
                    title=f"Hubungan Hasil Tangkapan dan Upaya Penangkapan ({model_results['name']})",
                    xaxis_title='Upaya Penangkapan (hari)',
                    yaxis_title='Hasil Tangkapan (ton)',
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        orientation="h"
                    ),
                    # Pastikan sumbu x dan y dimulai dari 0
                    xaxis=dict(range=[0, max(effort_range)], zeroline=True, linewidth=2),
                    yaxis=dict(range=[0, max(catch_pred) * 1.2], zeroline=True, linewidth=2)
                )

                st.plotly_chart(fig_surplus, use_container_width=True)


                # Grafik Hubungan CPUE dengan Effort
                st.write('### Grafik Hubungan CPUE dengan Effort')

                # Buat figure baru untuk CPUE-Effort
                if 'Schaefer' in selected_model:
                    # Hitung CPUE prediksi untuk model Schaefer
                    cpue_pred = model_results['a'] + model_results['b'] * effort_range
                    
                    fig_cpue = go.Figure()
                    
                    # Tambahkan garis regresi
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=effort_range,
                            y=cpue_pred,
                            mode='lines',
                            name='Model Schaefer',
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    # Tambahkan data aktual
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=yearly_data['effort (hari)'],
                            y=yearly_data['CPUE'],
                            mode='markers',
                            name='Data Aktual',
                            marker=dict(color='red', size=8)
                        )
                    )
                    
                    # Update layout
                    fig_cpue.update_layout(
                        title='Hubungan CPUE dengan Effort (Model Schaefer)',
                        xaxis_title='Upaya Penangkapan (hari)',
                        yaxis_title='CPUE (ton/hari)',
                        template='plotly_white',
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            orientation="h"
                        ),
                        xaxis=dict(zeroline=True, linewidth=2),
                        yaxis=dict(zeroline=True, linewidth=2)
                    )
                    
                else:
                    # Hitung ln(CPUE) prediksi untuk model Fox
                    ln_cpue_pred = model_results['c'] + model_results['d'] * effort_range
                    
                    fig_cpue = go.Figure()
                    
                    # Tambahkan garis regresi
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=effort_range,
                            y=ln_cpue_pred,
                            mode='lines',
                            name='Model Fox',
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    # Tambahkan data aktual
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=yearly_data['effort (hari)'],
                            y=np.log(yearly_data['CPUE']),  # menggunakan ln(CPUE) untuk data aktual
                            mode='markers',
                            name='Data Aktual',
                            marker=dict(color='red', size=8)
                        )
                    )
                    
                    # Update layout
                    fig_cpue.update_layout(
                        title='Hubungan ln(CPUE) dengan Effort (Model Fox)',
                        xaxis_title='Upaya Penangkapan (hari)',
                        yaxis_title='ln(CPUE)',
                        template='plotly_white',
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            orientation="h"
                        ),
                        xaxis=dict(zeroline=True, linewidth=2),
                        yaxis=dict(zeroline=True, linewidth=2)
                    )

                # Tampilkan grafik
                st.plotly_chart(fig_cpue)

                # Tambahkan tabel data yang digunakan dalam grafik
                st.write('### Data yang Digunakan dalam Grafik')
                if 'Schaefer' in selected_model:
                    plot_data = pd.DataFrame({
                        'Effort': yearly_data['effort (hari)'],
                        'CPUE': yearly_data['CPUE'],
                        'CPUE_predicted': model_results['a'] + model_results['b'] * yearly_data['effort (hari)']
                    })
                    st.write('Model Schaefer:')
                else:
                    plot_data = pd.DataFrame({
                        'Effort': yearly_data['effort (hari)'],
                        'ln(CPUE)': np.log(yearly_data['CPUE']),
                        'ln(CPUE)_predicted': model_results['c'] + model_results['d'] * yearly_data['effort (hari)']
                    })
                    st.write('Model Fox:')

                st.table(plot_data.round(4))


                # Tampilkan beberapa titik penting dari model
                st.write(f'### Titik-titik penting Model {model_results['name']}')
                important_points = pd.DataFrame({
                    'Effort': [0, model_results['Eopt']/2, model_results['Eopt'], model_results['Eopt']*1.5, model_results['Eopt']*2],
                    'Catch': [0, model_results['CMSY']*0.75, model_results['CMSY'], 
                                model_results['CMSY']*0.75, 0]
                }).round(2)
                important_points['% dari MSY'] = (important_points['Catch'] / model_results['CMSY'] * 100).round(2)
                st.table(important_points)



           	
        
            
        # # Analisis Lanjutan: Penghitungan CPUE
        # if 'Jumlah Hari' in user_data.columns and 'berat' in user_data.columns:
        #     st.subheader('Analisis Lanjutan: CPUE (Catch Per Unit Effort)')

        #     # Menghitung CPUE
        #     user_data['CPUE'] = user_data['berat'] / user_data['Jumlah Hari']
        #     cpue_per_year = user_data.groupby('tahun').agg({'CPUE': 'mean'}).reset_index()

        #     # Menampilkan Data CPUE dalam bentuk Tabel
        #     st.write('Data CPUE per Tahun:')
        #     st.dataframe(cpue_per_year)

        #     # Membuat Grafik CPUE per Tahun
        #     fig_cpue = px.line(
        #         cpue_per_year,
        #         x='tahun',
        #         y='CPUE',
        #         title='CPUE per Tahun',
        #         template='plotly_dark'
        #     )
        #     fig_cpue.update_layout(
        #         plot_bgcolor='rgba(0,0,0,0)',
        #         xaxis=dict(tickmode='linear'),
        #         yaxis=dict(showgrid=True, gridcolor='#cecdcd'),
        #         paper_bgcolor='rgba(0,0,0,0)'
        #     )

        #     # Menampilkan Grafik CPUE
        #     st.plotly_chart(fig_cpue, use_container_width=True)

           
        # else:
        #     st.warning('Kolom "tahun" atau "berat" tidak ditemukan. Pastikan file Anda memiliki kolom tersebut.')
    else:
        st.info('Please upload a CSV file to proceed.')
        



elif menu == 'About':
    # st.title('About this App')
    
    # Main Description
    st.markdown ("""
    ## What is SISTOK ?
    SISTOK (Sistem Informasi Stok Perikanan) is a comprehensive web-based application designed to help fisheries researchers, managers, and stakeholders analysize and understand fish stock data effectively. This tool provides various analytical capabilities to supports sustainable fisheries management.     
                 """)
    
    # Key Features
    st.markdown("## ✨ Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Data Management
        - CSV file upload support
        - Real-time data processing
        - Interactive data filtering
        
        ### 📈 Visualization
        - Dynamic charts and graphs
        - Catch statistics visualization
        - Temporal trend analysis
        """)
        
    with col2:
        st.markdown("""
        ### 🎯 Advanced Analytics
        - Surplus Production Models (Schaefer & Fox)
        - CPUE Analysis
        - Fishing effort standardization
        
        ### 📆 Time Series Analysis
        - Multiple time frame options
        - Trend identification
        - Seasonal pattern analysis
        """)

    # How to use
    st.markdown("## 🔍 How to Use SISTOK")

    with st.expander("STEP-BY-STEP GUIDE"):
        st.markdown("""
        1. **Data Upload**
            - Navigate to the Analysis section
            - Upload your CSV file containing fishing data
            - Ensure your data includes required columns (date, catch, effort)
                    
        2. **Data Exploration**
            - Use the Dashboard to view data
            - Apply filters to focus on specific time or Ports or Fish
            - Examine catch trends and patterns
                
        3. **Analysis**
            - Calculate CPUE for different fishing gears
            - Apply surplus production models
            - Estimate MSY and optimal fishing effort
                    
        4. **Results Interpretation**
            - Review visualizations and statistics
            - Download or export yor analytics results
            - Make informed management decisions
                    
    """)
        # Technical Requirements
        st.markdown("## 💻 Technical Requirements")
        st.info("""
        Your data should be in CSV fortmat with the following columns:
        - Date columns (tanggal_berangkat, tanggal_kedatangan
        - Catch data (berat)
        - Effort data (Jumlah hari)
        - Fishing gear (jenis_api)
        - Additional metadata as needed
    """)
        
        # Contact Information
        st.markdown("## 📬 Contact & Support")
        st.markdown("""
        - 📧 Email: tandrysimamora@gmail.com
        - 📱 Phone: +62 822 6160-6428
    """)
        # Version
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Version Info")
        st.sidebar.info("SISTOK v1.0.0")
        

else:
    st.error('Data tidak tersedia. Silahkan periksa kembali file Anda.')




    