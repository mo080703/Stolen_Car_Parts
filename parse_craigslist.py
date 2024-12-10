import csv
import json

def get_ids_images_tar():
    filepath = 'dataset_no_dup_images.csv'
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip headers
        posts = []
        for row in reader:
            id_images = tuple()
            id = row[0]
            images = row[9]
            tar_num = row[-2]

            if (images):
                #debugging for what image field looks like
                #print(f"Original Images data for post {id}: {images}")
                #processed_data_str = images.replace('""', '"')
                processed_data_str = images.replace("'", '"').replace('""', '"')
                try:
                    data_map = json.loads(processed_data_str)
                    file_names = list(data_map.values())
                    file_names = [x for x in file_names if x]
                    if len(file_names) > 0:
                        id_images = id, file_names, tar_num
                        posts.append(id_images)
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, log the error and continue
                    print(f"JSON decode error for post {id}: {e}")parse_craigslist.py
                    print(f"Malformed images data: {processed_data_str}")
                    continue  # Skip to the next row if JSON parsing fails
    return posts
