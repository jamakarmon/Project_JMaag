#Set snakemake
in_folder = snakemake.params[0]
out_csv = snakemake.output[0]

#Get all files
import os
files = [f for f in os.listdir(in_folder) if 'map_' in f]




#FUNCTION
#Load data and put it into pandas frame
import pandas as pd
def load_data(f):
    # ---
    # add your code here
    import re
    #change here names depending on what data is downloaded
    df = pd.read_csv(in_folder + f, header = 1, names = ['canton','search_wohnung'])
    date = re.search('\d{4}-\d{2}-\d{2}',f)
    df['date'] = date.group()
    # ---
    return df


#FUNCTION
#Load all files and add them together
# ---
# add your code here

google_raw = pd.DataFrame()

for f in files:
    df = load_data(f)
    google_raw = google_raw.append(df, ignore_index=True)

# ---


#Save google raw to csv
google_raw.to_csv(out_csv, sep=',')
