import pandas as pd
import glob
import os
import re
from tqdm import tqdm

def clean_sensitive_information(df):
    # Loại bỏ các thông tin email
    df = df.replace(to_replace=r'\S+@\S+', value='', regex=True)
    
    # Loại bỏ các số điện thoại (mẫu cơ bản)
    df = df.replace(to_replace=r'\+?\d[\d -]{8,}\d', value='', regex=True)
    
    # Loại bỏ thông tin nhạy cảm khác nếu cần
    # Ví dụ: Loại bỏ các thông tin thẻ tín dụng (mẫu cơ bản)
    df = df.replace(to_replace=r'\b(?:\d[ -]*?){13,16}\b', value='', regex=True)
    
    return df

# Thay đổi đường dẫn này tới thư mục chứa các tệp CSV của bạn
input_directory = '/kaggle/input/textsummarizationvietnamese1/'

# Thay đổi tên của tệp CSV kết quả
output_file = 'combined_file_cleaned.csv'

# Tìm tất cả các tệp CSV trong thư mục
csv_files = glob.glob(os.path.join(input_directory, '*.csv'))

# Danh sách để chứa các DataFrame
dataframes = []

# Đọc từng tệp CSV, làm sạch dữ liệu và thêm vào danh sách
for file in tqdm(csv_files):
    df = pd.read_csv(file)
    df_cleaned = clean_sensitive_information(df)
    dataframes.append(df_cleaned)

# Kết hợp tất cả DataFrame trong danh sách thành một DataFrame duy nhất
combined_df = pd.concat(dataframes, ignore_index=True)

# Xuất DataFrame kết hợp ra tệp CSV mới
combined_df.to_csv(output_file, index=False)

print(f"Đã kết hợp tất cả các tệp CSV thành '{output_file}' và đã loại bỏ thông tin nhạy cảm.")
