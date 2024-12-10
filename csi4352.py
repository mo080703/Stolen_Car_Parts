from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
from tar import TarFileReader
from parse_craigslist import get_ids_images_tar
import io
from PIL import UnidentifiedImageError
import pandas as pd
import os

# keep track of num of embeddings youve completed and print it out and then associate the post id w/ that image and save to ids text file of all that have already been processed

def create_model():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    return processor, model

# embed a singular image at a time
def embed_image(image_content, processor, model):
    if image_content is None:
        print("Warning: Image content is None. Skipping...")
        return None

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        cls_embedding = cls_embedding.reshape(1, -1)
        print(cls_embedding)
        print('done')
        return cls_embedding
    except UnidentifiedImageError as e:
        print(f"Warning: Unidentified image file - {e}. Skipping...")
        return None

# this embeds all the images found in a singular post
def embed_images_for_post(post, processor, model):
    id, images, tar_file = post
    tar_reader = TarFileReader('.assets/' + tar_file)  # Create TarFileReader instance
    embeddings = []
    print(len(images))
    for image_path in images:
        image_content = tar_reader.get_image(image_path)  # Use get_image method
        embedding = embed_image(image_content, processor, model)
        embeddings.append(embedding)
    return embeddings

def write_embedding(embeddings, postid):
    try:
        # get embeddings that arent none/corrupted and flatten good ones into 1D
        valid_embeddings = []
        for emb in embeddings:
            if emb is not None:
                valid_embeddings.append(emb.flatten())

        embeddings_df = pd.DataFrame(valid_embeddings) # create pandas dataframe

        embeddings_df['id'] = postid

        csv_file_path = 'embeddings.csv'
        write_header = not os.path.exists(csv_file_path) # don't write header every time
        embeddings_df.to_csv(csv_file_path, mode='a', index=False, header=write_header)
        return True
    except Exception as e:
        print(f"Error writing embeddings: {e}")
        return False

if __name__ == '__main__':
    processor, model = create_model()
    id_imagepaths_tar = get_ids_images_tar() # get ids, images, and tar for each post in dataset

    filtered_tar_posts = [x for x in id_imagepaths_tar if x[-1] == 'craisglist_images__62.tar'] # get only the posts that have images in this specific tar
    to_embed = filtered_tar_posts[6102:6104] # get the posts from this range



    #this is updated so just run python


    # get all the ids of posts that have already been processed
    with open('processedIds.txt', 'r') as f:
        ids = [line.strip() for line in f]

    # used to keep track of how many embeddings are successfully written to file
    posted = 0
    # embed these posts from that range
    for post in to_embed:
        curid, image, tar = post

        # embed and write only if this post hasnt been processed before
        if curid not in ids:
            # returns all embeddings for a post
            post_embedding = embed_images_for_post(post, processor, model)

            if(write_embedding(post_embedding, curid)):
                posted += 1
                print("Successful write: " + str(posted))

                # append the id of the post in the processedIds file
                with open('processedIds.txt', 'a') as f:
                    f.write(curid + '\n')
        else:
            print("Duplicate found w/ id: " + str(curid))
