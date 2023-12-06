import os
import json
import requests
import concurrent.futures

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
}

def get_ids():
    id_url = "https://api.tosdr.org/all-services/v1/"
    r = requests.get(id_url, headers=headers)
    j = r.json()
    tod_ids = []
    for service in j['parameters']['services']:
        if not os.path.isfile(f"data/tos_{service['id']}.json"):
            tod_ids.append(service['id'])
    return tod_ids

def download_tos(tos_id):
    url = f'https://api.tosdr.org/rest-service/v2/{tos_id}.json'
    with requests.Session() as session:
        try:
            r = session.get(url, headers=headers)
            j = r.json()
            if j.get('error') == 193:
                return None
            
            directory = 'data'
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(f"{directory}/tos_{tos_id}.json", 'w') as outfile:
                json.dump(j, outfile)
            return tos_id
        except Exception as e:
            print(f"Error downloading ToS {tos_id}: {e}")
            return None

if __name__ == "__main__":
    ids = get_ids()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(download_tos, ids))

    downloaded_tos = [result for result in results if result is not None]
    print(f"Downloaded {len(downloaded_tos)} ToS.")
