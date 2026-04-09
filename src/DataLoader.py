import os, zipfile, glob, requests, shutil

#checking if the user already has a csv file
if not os.path.exists("data/openpowerlifting.csv"):
    print("Downloading data...")
    
    url = "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
    
    #streams the download in chunks so it doesn't time out
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open("data.zip", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print("Extracting...")
    os.makedirs("data", exist_ok=True)
    with zipfile.ZipFile("data.zip") as z:
        z.extractall("data/")
    os.remove("data.zip")
    
    #move CSV to a clean path
    csv_files = glob.glob("data/**/*.csv", recursive=True)
    if len(csv_files) != 1:
        raise Exception(f"Expected 1 CSV, found {len(csv_files)}: {csv_files}")
    shutil.move(csv_files[0], "data/openpowerlifting.csv")
    print("Done!")