import zipfile
import pandas as pd
from typing import List

def load_csv_from_zip(zip_filename: str, csv_filenames: List[str], header=True) -> List[pd.DataFrame]:
  dataframes = []
  with zipfile.ZipFile(zip_filename, 'r') as z:
    for filename in csv_filenames:
      with z.open(filename) as csv_file:
        if not header:
          df = pd.read_csv(csv_file, header=None)
        else:
          df = pd.read_csv(csv_file)
        dataframes.append(df)
  return dataframes

