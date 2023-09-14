import requests
from pathlib import Path
from tqdm import tqdm
import time

def check_network_connection():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False 
def download_file(url, save_path):
    while True:
        if check_network_connection():
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # Adjust the block size as desired
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(save_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()
            print(f"File downloaded: {save_path}")
            break
            
        else:
            print("Network is disconnected. Retrying in 5 seconds...")
            time.sleep(3)  # Wait for 5 seconds before retrying

# List of URLs to download
    # "https://zenodo.org/record/4010759/files/Adjectives_1of8.zip"
    # "https://zenodo.org/record/4010759/files/Adjectives_2of8.zip"
    # ,"https://zenodo.org/record/4010759/files/Adjectives_3of8.zip"
    # ,"https://zenodo.org/record/4010759/files/Adjectives_4of8.zip"
    # ,"https://zenodo.org/record/4010759/files/Adjectives_5of8.zip"
    # ,"https://zenodo.org/record/4010759/files/Adjectives_6of8.zip"
    # ,"https://zenodo.org/record/4010759/files/Adjectives_7of8.zip"
    # ,"https://zenodo.org/record/4010759/files/Adjectives_8of8.zip"
    # ,"https://zenodo.org/record/4010759/files/Animals_1of2.zip"
    # ,"https://zenodo.org/record/4010759/files/Animals_2of2.zip"
    # ,"https://zenodo.org/record/4010759/files/Clothes_1of2.zip"
    # ,"https://zenodo.org/record/4010759/files/Clothes_2of2.zip"
    # ,"https://zenodo.org/record/4010759/files/Colours_1of2.zip"
    # ,"https://zenodo.org/record/4010759/files/Colours_2of2.zip"
    # ,"https://zenodo.org/record/4010759/files/Days_and_Time_1of3.zip"
    # ,"https://zenodo.org/record/4010759/files/Days_and_Time_2of3.zip"
    # ,"https://zenodo.org/record/4010759/files/Days_and_Time_3of3.zip"
url_list = [
    "https://zenodo.org/record/4010759/files/Electronics_1of2.zip"
    ,"https://zenodo.org/record/4010759/files/Electronics_2of2.zip"
    ,"https://zenodo.org/record/4010759/files/Greetings_1of2.zip"
    ,"https://zenodo.org/record/4010759/files/Greetings_2of2.zip"
    ,"https://zenodo.org/record/4010759/files/Home_1of4.zip"
    ,"https://zenodo.org/record/4010759/files/Home_2of4.zip"
    ,"https://zenodo.org/record/4010759/files/Home_3of4.zip"
    ,"https://zenodo.org/record/4010759/files/Home_4of4.zip"
    ,"https://zenodo.org/record/4010759/files/Jobs_1of2.zip"
    ,"https://zenodo.org/record/4010759/files/Jobs_2of2.zip"
    ,"https://zenodo.org/record/4010759/files/Means_of_Transportation_1of2.zip"
    ,"https://zenodo.org/record/4010759/files/Means_of_Transportation_2of2.zip"
    ,"https://zenodo.org/record/4010759/files/People_1of5.zip"
    ,"https://zenodo.org/record/4010759/files/People_2of5.zip"
    ,"https://zenodo.org/record/4010759/files/People_3of5.zip"
    ,"https://zenodo.org/record/4010759/files/People_4of5.zip"
    ,"https://zenodo.org/record/4010759/files/People_5of5.zip"
    ,"https://zenodo.org/record/4010759/files/Places_1of4.zip"
    ,"https://zenodo.org/record/4010759/files/Places_2of4.zip"
    ,"https://zenodo.org/record/4010759/files/Places_3of4.zip"
    ,"https://zenodo.org/record/4010759/files/Places_4of4.zip"
    ,"https://zenodo.org/record/4010759/files/Pronouns_1of2.zip"
    ,"https://zenodo.org/record/4010759/files/Pronouns_2of2.zip"
    ,"https://zenodo.org/record/4010759/files/Seasons_1of1.zip"
    ,"https://zenodo.org/record/4010759/files/Society_1of3.zip"
    ,"https://zenodo.org/record/4010759/files/Society_2of3.zip"
    ,"https://zenodo.org/record/4010759/files/Society_3of3.zip"
]

# Download files from the URL list
video_path = Path("Video")
for url in url_list:
    filename = url.split('/')[-1]  # Extract the filename from the URL
    save_path = video_path / f"{filename}"
    print(save_path)
    # Specify the desired save path
    download_file(url, save_path)