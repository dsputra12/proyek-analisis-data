import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

hour_df = pd.read_csv('hour.csv')
day_df = pd.read_csv('day.csv')


st.title("Proyek Belajar Analisis Data Dicoding")

st.markdown(
    """
    <div style='font-size: 18px;'>
        <span style='color: #9bd4e4;'><strong>Nama:</strong></span> 
        <span style='color: white;'><strong>Darmawan Setyaputra Purba</strong></span><br>
        <span style='color: #9bd4e4;'><strong>Learning Path:</strong></span>
        <span style='color: white;'><strong>Machine Learning (ML-38)</strong></span><br>
        <span style='color: #9bd4e4;'><strong>Dataset:</strong></span>
        <span style='color: white;'><strong>Bike Sharing Dataset</strong></span><br>
    </div>
    """,
    unsafe_allow_html=True
)

st.header("Source")
st.markdown(
    """
    <a href="https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset/data" style="color:#FF6347; font-size:20px; text-decoration:none;">
        Kaggle Bike Sharing Dataset
    </a>
    """, 
    unsafe_allow_html=True
)
st.header("Pertanyaan bisnis")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Pertanyaan  1**", "**Pertanyaan 2**", "**Pertanyaan 3**", "**Pertanyaan 4**", "**Pertanyaan 5**"])
 
with tab1:
    st.markdown("<h4 style='margin: 0;'>1. Bagaimana tingkat peminjaman sepeda pada tahun 2011 dan 2012?</h4>", unsafe_allow_html=True)
    with st.expander("**Lihat Kode Program**"):
        st.code(
        """
        total_0 = day0_df['cnt'].sum()
        total_1 = day1_df['cnt'].sum()
        
        tahun = [0, 1]
        total = [total_0, total_1]
        
        plt.bar(x=tahun, height=total, color=['skyblue', 'orange'])
        plt.xticks(ticks=[0, 1], labels=['2011', '2012'])
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Peminjaman (dalam juta)')
        plt.title('Total Peminjaman Sepeda di tahun 2011 dan 2012')
        plt.show()
        """, language='python')
   
    with st.expander("**Lihat Visualisasi**"):
        day0_df = day_df[day_df['yr'] == 0] 
        day1_df = day_df[day_df['yr'] == 1]  
        total_0 = day0_df['cnt'].sum()
        total_1 = day1_df['cnt'].sum()

        tahun = [0, 1] 
        total = [total_0, total_1]  
        plt.bar(x=tahun, height=total, color=['skyblue', 'orange'])
        plt.xticks(ticks=[0, 1], labels=['2011', '2012'])
        plt.xlabel('Tahun')
        plt.ylabel('Jumlah Peminjaman (dalam juta)')
        plt.title('Total Peminjaman Sepeda di Tahun 2011 dan 2012')
        
        st.pyplot(plt)
        plt.clf()  

    with st.expander("**Lihat Insight**"):
        st.write(
        """ 
        Melalui visualisasi Bar Graph, ditemukan fakta bahwa jumlah peminjaman sepeda mengalami peningkatan yang cukup signifikan dari tahun 2011 ke 
        tahun 2012, yaitu sekitar 750000 peminjam
        """
        )
    
 
with tab2:
    st.markdown("<h4 style='margin: 0;'>2. Pada bulan apa jumlah peminjaman sepeda terbesar berhasil dicapai untuk setiap tahunya?</h4>", unsafe_allow_html=True)
    with st.expander("**Lihat Kode Program**"):
        st.code(
        """
        month_names = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
            7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }
        months_2011 = monthly_stats0.index.get_level_values('mnth')
        rentals_2011 = monthly_stats0[('cnt', 'sum')]
        months_2012 = monthly_stats1.index.get_level_values('mnth')
        rentals_2012 = monthly_stats1[('cnt', 'sum')]
        
        month_labels_2011 = [month_names[mnth] for mnth in months_2011]
        month_labels_2012 = [month_names[mnth] for mnth in months_2012]
        
        bar_width = 0.35  
        x1 = range(len(months_2011))  
        x2 = [pos + bar_width for pos in x1]  
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x1, rentals_2011, width=bar_width, label='2011', color='skyblue')
        ax.bar(x2, rentals_2012, width=bar_width, label='2012', color='orange')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Jumlah Peminjaman')
        ax.set_title('Jumlah Peminjaman Bulanan selama Tahun 2011 dan 2012')
        ax.set_xticks([r + bar_width / 2 for r in x1])  
        ax.set_xticklabels(month_labels_2011)  
        
        ax.legend()
        
        plt.xticks(rotation = 35)
        plt.show()
        """, language='python')
   
    with st.expander("**Lihat Visualisasi**"):
        monthly_stats0 = day0_df.groupby(["mnth", "yr"]).agg({
            "cnt": ["sum", "max", "min", "mean", "std"]
        })

        monthly_stats1 = day1_df.groupby(["mnth", "yr"]).agg({
            "cnt": ["sum", "max", "min", "mean", "std"]
        })

        month_names = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
            7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }  
        months_2011 = monthly_stats0.index.get_level_values('mnth')
        rentals_2011 = monthly_stats0[('cnt', 'sum')]
        months_2012 = monthly_stats1.index.get_level_values('mnth')
        rentals_2012 = monthly_stats1[('cnt', 'sum')]

        month_labels_2011 = [month_names[mnth] for mnth in months_2011]
        month_labels_2012 = [month_names[mnth] for mnth in months_2012]
        
        bar_width = 0.35  
        x1 = range(len(months_2011))  
        x2 = [pos + bar_width for pos in x1]  
        
        fig, ax = plt.subplots(figsize=(12, 6))        
        ax.bar(x1, rentals_2011, width=bar_width, label='2011', color='skyblue')
        ax.bar(x2, rentals_2012, width=bar_width, label='2012', color='orange')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Jumlah Peminjaman')
        ax.set_title('Jumlah Peminjaman Bulanan selama Tahun 2011 dan 2012')
        ax.set_xticks([r + bar_width / 2 for r in x1])  
        ax.set_xticklabels(month_labels_2011)  
        
        ax.legend()
        
        plt.xticks(rotation = 35)
        st.pyplot(plt)
        plt.clf()  

    with st.expander("**Lihat Insight**"):
        st.write(
            """ 
           Peminjaman sepeda  di tahun 2011 meningkat secara konsisten dari awal tahun dan berpuncak di bulan Juni, sedangkan pada tahun 2012, peminjaman
           secara konsisten meningkat dari awal tahun dan berpuncak di bulan September
            """
        )

with tab3:
    st.markdown("<h4 style='margin: 0;'>3. Apakah keadaan cuaca dan kondisi alam mempengaruhi jumlah peminjaman sepeda pada setiap harinya?</h4>", unsafe_allow_html=True)
    with st.expander("**Lihat Kode Program**"):
        st.code(
        """
        revised_day_df = day_df.rename(columns={'cnt': 'jumlah', 'temp': 'temperatur', 'hum': 'kelembapan',  'windspeed': 'kecepatan_angin'},inplace=False)

        weather_corr = revised_day_df[['weathersit', 'jumlah']].corr()
        nature_corr = revised_day_df[['jumlah', 'temperatur', 'kelembapan', 'kecepatan_angin']].corr()
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))  
        
        sns.heatmap(weather_corr, annot=True, cmap='coolwarm', ax=ax[0], cbar=False)
        ax[0].set_title('Korelasi antara Kondisi Cuaca dan Jumlah Pinjaman Sepeda')
        ax[0].set_xticklabels(['Kondisi Cuaca', 'Jumlah Pinjaman'], rotation=35)
        ax[0].set_yticklabels(['Kondisi Cuaca', 'Jumlah Pinjaman'], rotation=0)
        
        sns.heatmap(nature_corr, annot=True, cmap='coolwarm', ax=ax[1])
        ax[1].set_title('Korelasi antara Faktor Alam dan Jumlah Peminjaman Sepeda')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=35)
        ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.show()
        """, language='python')
   
    with st.expander("**Lihat Visualisasi**"):
        revised_day_df = day_df.rename(columns={'cnt': 'jumlah', 'temp': 'temperatur', 'hum': 'kelembapan',  'windspeed': 'kecepatan_angin'}, inplace=False)

        weather_corr = revised_day_df[['weathersit', 'jumlah']].corr()
        nature_corr = revised_day_df[['jumlah', 'temperatur', 'kelembapan', 'kecepatan_angin']].corr()
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))  
        
        sns.heatmap(weather_corr, annot=True, cmap='coolwarm', ax=ax[0], cbar=False)
        ax[0].set_title('Korelasi antara Kondisi Cuaca dan Jumlah Pinjaman Sepeda')
        ax[0].set_xticklabels(['Kondisi Cuaca', 'Jumlah Pinjaman'], rotation=35)
        ax[0].set_yticklabels(['Kondisi Cuaca', 'Jumlah Pinjaman'], rotation=0)
        
        sns.heatmap(nature_corr, annot=True, cmap='coolwarm', ax=ax[1])
        ax[1].set_title('Korelasi antara Faktor Alam dan Jumlah Peminjaman Sepeda')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=35)
        ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  

    with st.expander("**Lihat Insight**"):
        st.write(
            """ 
           Melalui visualisasi Heatmap, kondisi cuaca yang meningkat mempengaruhi penurunan jumlah peminjaman sepeda, tetapi tidak cukup signifikan untuk 
           mampu menurunkan jumlah peminjaman sepeda. Dari 4 faktor alam yang diamati, faktor temperatur (suhu) merupakan satu-satunya faktor alam  yang
           memiliki pengaruh kuat terhadap
           jumlah peminjaman sepeda, dimana semakin tinggi suhu lingkungan, maka jumlah peminjaman sepeda juga akan meningkat
            """
        )


with tab4:
    st.markdown("<h4 style='margin: 0;'>4. Dalam satu hari, pada jam berapa peminjaman sepeda banyak terjadi?</h4>", unsafe_allow_html=True)
    with st.expander("**Lihat Kode Program**"):
        st.code(
        """
        hourly_stats = hour_df.groupby(by="hr").agg({
            'cnt' : ["sum", "max", "min", "mean", "std"]
        })
        
        hours = hourly_stats.index
        total_rentals = hourly_stats[('cnt', 'sum')]
        
        hour_labels = ['00.00', '01.00', '02.00', '03.00', '04.00', '05.00', '06.00', 
                       '07.00', '08.00', '09.00', '10.00', '11.00', '12.00', 
                       '13.00', '14.00', '15.00', '16.00', '17.00', '18.00', 
                       '19.00', '20.00', '21.00', '22.00', '23.00']
        
        plt.figure(figsize=(12, 6))
        plt.plot(hours, total_rentals, label='Total Peminjaman', marker='o')
        
        plt.xlabel('Jam Peminjaman')
        plt.ylabel('Jumlah Peminjaman')
        plt.title('Statistik Peminjaman Sepeda Berdasarkan Jam Peminjaman')
        plt.xticks(ticks=hours, labels=hour_labels, rotation = 35)
        plt.grid(visible=True) 
        plt.legend()
        
        plt.show()
        """, language='python')
   
    with st.expander("**Lihat Visualisasi**"):
        hourly_stats = hour_df.groupby(by="hr").agg({
            'cnt' : ["sum", "max", "min", "mean", "std"]
        })
        
        hours = hourly_stats.index
        total_rentals = hourly_stats[('cnt', 'sum')]
        
        hour_labels = ['00.00', '01.00', '02.00', '03.00', '04.00', '05.00', '06.00', 
                       '07.00', '08.00', '09.00', '10.00', '11.00', '12.00', 
                       '13.00', '14.00', '15.00', '16.00', '17.00', '18.00', 
                       '19.00', '20.00', '21.00', '22.00', '23.00']
        
        plt.figure(figsize=(12, 6))
        plt.plot(hours, total_rentals, label='Total Peminjaman', marker='o')
        
        plt.xlabel('Jam Peminjaman')
        plt.ylabel('Jumlah Peminjaman')
        plt.title('Statistik Peminjaman Sepeda Berdasarkan Jam Peminjaman')
        plt.xticks(ticks=hours, labels=hour_labels, rotation = 35)
        plt.grid(visible=True) 
        plt.legend()
        
        st.pyplot(plt)
        plt.clf()  

    with st.expander("**Lihat Insight**"):
        st.write(
            """ 
            Melalui visualisasi Line Chart, pemakaian jasa peminjaman sepeda terbanyak berada pada pukul 17.00  
            """
        )

    

with tab5:
    st.markdown("<h4 style='margin: 0;'>5. Tipe pelanggan yang seperti apa yang mendominasi penggunaan layanan peminjaman sepeda di hari kerja?</h4>", unsafe_allow_html=True)
    with st.expander("**Lihat Kode Program**"):
        st.code(
        """
        working_day_df = day_df[day_df['workingday'] == True]
        total_casual = working_day_df['casual'].sum()
        total_registered = working_day_df['registered'].sum()
        
        tipe_peminjam = ('Casual', 'Registered')
        total = (total_casual, total_registered)
        colors = ('#E69A8DFF', '#5F4B8BFF')
        explode = (0.1, 0.1)
        
        plt.pie(
            x=total,
            labels=tipe_peminjam,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode
        )
        
        plt.title("Proporsi Penggunaan Layanan Peminjaman Sepeda di Hari Kerja berdasarkan Tipe Pelanggan")
        plt.show()
        """, language='python')
   
    with st.expander("**Lihat Visualisasi**"):
        working_day_df = day_df[day_df['workingday'] == True]
        total_casual = working_day_df['casual'].sum()
        total_registered = working_day_df['registered'].sum()
        
        tipe_peminjam = ('Casual', 'Registered')
        total = (total_casual, total_registered)
        colors = ('#E69A8DFF', '#5F4B8BFF')
        explode = (0.1, 0.1)
        
        plt.pie(
            x=total,
            labels=tipe_peminjam,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode
        )
        
        plt.title("Proporsi Penggunaan Layanan Peminjaman Sepeda di Hari Kerja berdasarkan Tipe Pelanggan")
        st.pyplot(plt)
        plt.clf()  

    with st.expander("**Lihat Insight**"):
        st.write(
            """ 
            Melalui visualisasi Pie Chart, ditemukan fakta bahwa dari data harian peminjaman sepeda, layanan peminjaman sepeda banyak digunakan di hari kerja 
            oleh pengguna yang berlangganan dibandingkan pengguna biasa
            """
        )
st.header("Analisis Tambahan")
with st.expander("**Lihat Kode Program**"):
    st.code(
    """"
    data = hour_df[['cnt', 'casual', 'registered', 'hr', 'weekday']]
    
    hour_labels = ['00.00', '01.00', '02.00', '03.00', '04.00', '05.00', '06.00', 
                   '07.00', '08.00', '09.00', '10.00', '11.00', '12.00', 
                   '13.00', '14.00', '15.00', '16.00', '17.00', '18.00', 
                   '19.00', '20.00', '21.00', '22.00', '23.00']
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    hour_df['Cluster'] = kmeans.fit_predict(data_scaled)
    
    cluster_by_hour = hour_df[['hr', 'Cluster']]
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=cluster_by_hour, x='hr', hue='Cluster', palette='Set2')
    plt.title("Distribusi Cluster  Tipe Pelanggan berdasarkan Jam Peminjaman menggunakan KMeans")
    plt.xlabel("Jam Peminjaman")
    plt.ylabel("Banyaknya Peminjaman")
    plt.legend(title="Cluster", labels=['Casual (Cluster 0)', 'Commuter (Cluster 1)', 'Commuter dan Casual (Cluster 2)'])
    plt.xticks(ticks=range(len(hour_labels)), labels=hour_labels, rotation = 35)
    plt.tight_layout()  
    plt.show()
    """, language='python')

with st.expander("**Lihat Visualisasi**"):
    data = hour_df[['cnt', 'casual', 'registered', 'hr', 'weekday']]
    
    hour_labels = ['00.00', '01.00', '02.00', '03.00', '04.00', '05.00', '06.00', 
                   '07.00', '08.00', '09.00', '10.00', '11.00', '12.00', 
                   '13.00', '14.00', '15.00', '16.00', '17.00', '18.00', 
                   '19.00', '20.00', '21.00', '22.00', '23.00']
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    hour_df['Cluster'] = kmeans.fit_predict(data_scaled)
    
    cluster_by_hour = hour_df[['hr', 'Cluster']]
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=cluster_by_hour, x='hr', hue='Cluster', palette='Set2')
    plt.title("Distribusi Cluster  Tipe Pelanggan berdasarkan Jam Peminjaman menggunakan KMeans")
    plt.xlabel("Jam Peminjaman")
    plt.ylabel("Banyaknya Peminjaman")
    plt.legend(title="Cluster", labels=['Casual (Cluster 0)', 'Commuter (Cluster 1)', 'Commuter dan Casual (Cluster 2)'])
    plt.xticks(ticks=range(len(hour_labels)), labels=hour_labels, rotation = 35)
    plt.tight_layout()  
    st.pyplot(plt)
    plt.clf()  

with st.expander("**Lihat Insight**"):
    st.write(
        """ 
        Melalui Analisis Tambahan menggunakan metode Clustering, kita dapat mengidentifikasi karakteristik  pelanggan berdasarkan persebaran kluster di masing
        -masing jam dalam satu hari. Melalui Clustering yang dengan membagi data menjadi 3 Cluster, kita dapat melihat 3 karakteristik utama pengguna layanan
        peminjaman sepeda, yaitu tipe commuter, yang banyak menggunakan layanan di pagi hingga siang hari, casual user yang menggunakan layanan dari pagi
        hingga malam hari dengan jumlah yang lebih sedikit, dan tipe hybrid, yaitu casual dan commuter, yang menggunakan layanan peminjaman sepeda dari pagi
        hingga larut malam
        """
    )

st.header("Kesimpulan untuk masing-masing pertanyaan")
st.markdown(
    """
    <div style='font-size: 18px;'>
        <span style='color: white;'><strong>1. Tingkat peminjaman sepeda di tahun 2012 mengalami peningkatan dibandingkan dengan tahun 2011 </strong></span>
        <br>
        <span style='color: white;'><strong>2. Peminjaman sepeda  di tahun 2011 meningkat secara konsisten dari awal tahun dan berpuncak di bulan Juni,
        sedangkan
        pada tahun 2012, peminjaman secara konsisten meningkat dari awal tahun dan berpuncak di bulan September</strong></span><br>\
        <span style='color: white;'><strong>3. Peminjaman sepeda banyak terjadi di jam 17.00</strong></span><br>
        <span style='color: white;'><strong>4. Pelanggaan yang berlangganan di layanan sepeda mendominasi penggunaan layanan peminjaman sepeda dibandingkan 
        pengguna biasa</strong></span><br>
        <span style='color: white;'><strong>5. Melalui analisis tambahan, dapat diidentifikasi karakteristik pelanggan berdasarkan jam peminjaman yang
        dilakukan, yaitu tipe commuter, casual, maupun keduanya </strong></span><br>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption('Copyright (c) Dicoding 2023')