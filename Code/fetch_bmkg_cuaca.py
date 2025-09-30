import requests
import pandas as pd
import os
from datetime import datetime
import shutil
import schedule
import time

kelurahan_list = {
    "Kel.Manokwari Barat": "92.02.12.1001", 
    "Kel.Sanggeng": "92.02.12.1002",
    "Kel.Wosi": "92.02.12.1003",
    "Kel.Amban": "92.02.12.1004",
    "Kel.Padarni": "92.02.12.1005",
    "Kel.Manokwari Timur": "92.02.12.1006",
    "Kp.Udopi": "92.02.12.2007",
    "Kp.Ingramui": "92.02.12.2008",
    "Kp.Soribo": "92.02.12.2009",
    "Kp.Tanah Merah": "92.02.12.2010",
}

file_csv = "BMKG Data/dataset_cuaca_manokwari.csv"
backup_folder = "backup_cuaca"
os.makedirs(backup_folder, exist_ok=True)

header = ["kelurahan","datetime","suhu","kelembapan","curah_hujan",
          "kecepatan_angin","arah_angin","tutupan_awan","jarak_pandang","cuaca"]

if not os.path.exists(file_csv):
    pd.DataFrame(columns=header).to_csv(file_csv, index=False, encoding='utf-8')

def ambil_cuaca(kode_adm4, retries=3, delay=5):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_adm4}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            cuaca_list = []
            for cuaca_periode in data['data'][0]['cuaca']:
                for item in cuaca_periode:
                    cuaca_list.append({
                        "local_datetime": item['local_datetime'],
                        "t": item['t'],
                        "hu": item['hu'],
                        "tp": item['tp'],
                        "ws": item['ws'],
                        "wd": item['wd'],
                        "tcc": item['tcc'],
                        "vs_text": item['vs_text'],
                        "weather_desc": item['weather_desc']
                    })
            return cuaca_list
        except Exception as e:
            print(f"Error ambil data BMKG (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return []

def simpan_csv_efficient(kelurahan, cuaca_list):
    df = pd.read_csv(file_csv)
    count_new = 0
    count_updated = 0

    for item in cuaca_list:
        mask = (df['kelurahan'] == kelurahan) & (df['datetime'] == item['local_datetime'])
        new_row = {
            "kelurahan": kelurahan,
            "datetime": item['local_datetime'],
            "suhu": item['t'],
            "kelembapan": item['hu'],
            "curah_hujan": item['tp'],
            "kecepatan_angin": item['ws'],
            "arah_angin": item['wd'],
            "tutupan_awan": item['tcc'],
            "jarak_pandang": item['vs_text'],
            "cuaca": item['weather_desc']
        }

        if not mask.any():
            # Tambah data baru
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            count_new += 1
        else:
            # Cek apakah ada perbedaan
            idx = df[mask].index[0]
            row_old = df.loc[idx].to_dict()
            if any(row_old[col] != new_row[col] for col in header):
                df.loc[idx] = new_row  # update
                count_updated += 1
    df.to_csv(file_csv, index=False, encoding='utf-8')
    return count_new, count_updated

def clean_dataset():
    if not os.path.exists(file_csv):
        return
    df = pd.read_csv(file_csv, encoding="utf-8")
    df = df[header]
    df.drop_duplicates(subset=["kelurahan","datetime"], keep="last", inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.sort_values(by=["kelurahan","datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(file_csv, index=False, encoding="utf-8")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset berhasil dirapikan ({len(df)} baris).")


def backup_csv():
    if os.path.exists(file_csv):
        filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(file_csv, os.path.join(backup_folder, f"dataset_cuaca_{filename}.csv"))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Backup CSV dibuat: dataset_cuaca_{filename}.csv")

job_counter = 0

def job():
    global job_counter
    job_counter += 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n==== Putaran ke-{job_counter} | {now} ====")
    total_new = 0
    total_updated = 0
    for nama_kelurahan, kode_adm4 in kelurahan_list.items():
        cuaca = ambil_cuaca(kode_adm4)
        if cuaca:
            count_new, count_updated = simpan_csv_efficient(nama_kelurahan, cuaca)
            total_new += count_new
            total_updated += count_updated
            print(f"▶️  {nama_kelurahan}: {count_new} baru, {count_updated} diperbarui.🆗")
        else:
            print(f"▶️  {nama_kelurahan}: GAGAL ambil data 🆘")
    
    print(f"Total entri baru: {total_new} 🆕 | Total diperbarui: {total_updated} 🔄")
    print("=================================================================")
    clean_dataset()

schedule.every(30).minutes.do(job)
schedule.every(3).hours.do(backup_csv)

if __name__ == "__main__":
    job()
    start_time = time.time()
    while True:
        schedule.run_pending()
        elapsed = int(time.time() - start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"⏱️   Waktu berjalan: {hours:02d}:{minutes:02d}:{seconds:02d}", end="\r")
        time.sleep(1)
