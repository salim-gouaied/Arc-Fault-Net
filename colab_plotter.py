import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import io

# 1. Upload the file from your local machine
print("Please upload your CSV file (e.g., C2--exp13--IJL--LR--00066.csv)")
uploaded = files.upload()

# 2. Process and plot each uploaded file
for filename, content in uploaded.items():
    print(f"Processing {filename}...")
    
    try:
        # LeCroy CSVs typically have 5 lines of header metadata before the actual data
        data = pd.read_csv(
            io.BytesIO(content),
            skiprows=5,
            header=0,
            names=['Time', 'Ampl'],
            dtype={'Ampl': float, 'Time': float},
            usecols=['Time', 'Ampl']
        )
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue
        
    # 3. Plot the curve over time
    plt.figure(figsize=(16, 5))
    plt.plot(data['Time'], data['Ampl'], color='purple', linewidth=0.5)
    
    plt.title(f"Waveform: {filename}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude (V)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
