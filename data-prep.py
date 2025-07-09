import pandas as pd
import numpy as np
from datetime import datetime
import os

# Utility to clean IDs and remove '.0' float artifacts
def clean_id(x):
    try:
        return str(int(float(x)))  # Removes .0 if present
    except:
        return str(x).strip().lower()

def preprocess_music_data():
    # Load your data
    interactions_df = pd.read_csv('interactions.csv')
    songs_df = pd.read_csv('songs.csv')
    print("Columns in interactions.csv:", interactions_df.columns.tolist())

    # Process interactions
    interactions_processed = interactions_df.copy()

    # Create implicit feedback rating
    interactions_processed['rating'] = 1.0
    interactions_processed.loc[interactions_processed['isSkipped'] == 1, 'rating'] = 0.5
    interactions_processed.loc[interactions_processed['liked'] == 1, 'rating'] = 2.0

    # Convert timestamp to unix timestamp (int)
    interactions_processed['timestamp'] = pd.to_datetime(
        interactions_processed['timestamp'], utc=True
    ).astype('int64') // 10**9  # Using astype instead of view

    # Prepare dataset folder
    dataset_dir = './dataset/music_dataset'
    os.makedirs(dataset_dir, exist_ok=True)

    # Create interaction file with proper column names and normalize IDs
    inter_file = interactions_processed[['user_id', 'song_id', 'rating', 'timestamp']].copy()

    # Normalize user_id and song_id strings
    inter_file['user_id'] = inter_file['user_id'].astype(str).str.strip().str.lower().map(clean_id)
    inter_file['song_id'] = inter_file['song_id'].map(clean_id)

    inter_file.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
    inter_file.to_csv(f'{dataset_dir}/music_dataset.inter', sep='\t', index=False)

    # Process item features
    songs_processed = songs_df.copy()
    songs_processed['artist_name'] = songs_processed['artist_name'].fillna('unknown').str.replace(' ', '_').str.lower()
    songs_processed['Genre_'] = songs_processed['Genre_'].fillna('unknown').str.replace(' ', '_').str.lower()
    songs_processed['ContentLanguage'] = songs_processed['ContentLanguage'].fillna('unknown').str.replace(' ', '_').str.lower()

    # Normalize IDs
    songs_processed['song_id'] = songs_processed['song_id'].map(clean_id)
    songs_processed['artist_id'] = songs_processed['artist_id'].map(clean_id)
    songs_processed['album_id'] = songs_processed['album_id'].map(clean_id)

    # Create item file
    item_columns = ['song_id', 'artist_id', 'artist_name', 'album_id', 'Genre_', 'ContentLanguage', 'year']
    item_file = songs_processed[item_columns].copy()
    item_file.columns = [
        'item_id:token', 'artist_id:token', 'artist_name:token',
        'album_id:token', 'genre:token', 'language:token', 'year:float'
    ]
    item_file.to_csv(f'{dataset_dir}/music_dataset.item', sep='\t', index=False)

    # Create knowledge graph relations
    kg_relations = []
    for _, row in songs_processed.iterrows():
        song_id = row['song_id']
        artist_id = row['artist_id']
        album_id = row['album_id']
        genre = row['Genre_']
        language = row['ContentLanguage']

        # Song-Artist relation
        kg_relations.append([song_id, 'performed_by', artist_id])
        kg_relations.append([artist_id, 'performs', song_id])

        # Song-Album relation
        kg_relations.append([song_id, 'belongs_to_album', album_id])
        kg_relations.append([album_id, 'contains_song', song_id])

        # Artist-Genre relation
        kg_relations.append([artist_id, 'has_genre', genre])
        kg_relations.append([genre, 'genre_of_artist', artist_id])

        # Song-Language relation
        kg_relations.append([song_id, 'in_language', language])
        kg_relations.append([language, 'language_of_song', song_id])

    kg_df = pd.DataFrame(kg_relations, columns=['head_id:token', 'relation_id:token', 'tail_id:token'])
    kg_df.drop_duplicates(inplace=True)

    # Normalize KG entity IDs
    kg_df['head_id:token'] = kg_df['head_id:token'].map(clean_id)
    kg_df['tail_id:token'] = kg_df['tail_id:token'].map(clean_id)

    kg_df.to_csv(f'{dataset_dir}/music_dataset.kg', sep='\t', index=False)

    # Link file: map items to entities (only those items that exist in KG entities)
    item_ids = set(inter_file['item_id:token'].unique())
    kg_entities = set(pd.concat([kg_df['head_id:token'], kg_df['tail_id:token']]).unique())

    # Debug print to help verify matches
    print("Sample interaction item IDs:", list(item_ids)[:10])
    print("Sample KG head entities:", list(kg_df['head_id:token'].unique())[:10])
    print("Sample KG tail entities:", list(kg_df['tail_id:token'].unique())[:10])

    linked_items = [item for item in item_ids if item in kg_entities]
    unmatched_items = item_ids - kg_entities

    print(f"Total items in interactions: {len(item_ids)}")
    print(f"Total unique KG entities: {len(kg_entities)}")
    print(f"Items linked to KG entities: {len(linked_items)}")
    print("Sample unmatched items:", list(unmatched_items)[:10])

    link_df = pd.DataFrame({'item_id:token': linked_items, 'entity_id:token': linked_items})
    link_df.to_csv(f'{dataset_dir}/music_dataset.link', sep='\t', index=False)

    print("Data preprocessing completed!")
    print(f"Interactions: {len(inter_file)}")
    print(f"Items: {len(item_file)}")
    print(f"KG triples: {len(kg_df)}")
    print(f"Link file entries: {len(link_df)}")

if __name__ == "__main__":
    preprocess_music_data()
