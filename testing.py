from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
import torch
import pandas as pd
from collections import deque
import torch.nn.functional as F

# === Load model and data ===
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='./checkpoint/RippleNet-Jun-15-2025_14-36-29.pth'
)

# === Load song metadata ===
item_metadata = pd.read_csv('songs.csv')
item_metadata['song_id'] = item_metadata['song_id'].astype(str).map(lambda x: str(int(float(x))))
item_id_to_name = dict(zip(item_metadata['song_id'], item_metadata['song_name']))
item_id_to_artist = dict(zip(item_metadata['song_id'], item_metadata['artist_name']))

# === Load KG data ===
kg_file_path = r'C:\Users\afarouk\Desktop\RecBole\dataset\music_dataset\music_dataset.kg'
try:
    kg_data = pd.read_csv(kg_file_path, sep='\t')
except FileNotFoundError:
    print(f"KG file not found at {kg_file_path}. Using dataset.kg_data if available.")
    kg_data = dataset.kg_data

# === Load interaction data ===
inter_file_path = r'C:\Users\afarouk\Desktop\RecBole\dataset\music_dataset\music_dataset.inter'
interaction_data = pd.read_csv(inter_file_path, sep='\t')

# === Optional relation mapping for explanation clarity ===
relation_id_to_name = {
    'performed_by': "performed by the same artist",
    'performs': "performed by the same artist",
    'belongs_to_album': "from the same album",
    'contains_song': "from the same album",
    'has_genre': "in the same genre",
    'genre_of_artist': "in the same genre",
    'in_language': "in the same language",
    'language_of_song': "in the same language"
}

def clean_id(x):
    try:
        return str(int(float(x)))
    except:
        return str(x).strip().lower()

def find_multi_hop_paths(kg_data, start_id, end_id, max_hops=2):
    queue = deque([(start_id, [])])
    visited = set()
    paths = []
    while queue:
        current_id, path = queue.popleft()
        if len(path) > max_hops:
            continue
        if current_id == end_id and path:
            paths.append(path)
            continue
        if current_id in visited:
            continue
        visited.add(current_id)
        neighbors = kg_data[
            (kg_data['head_id:token'] == current_id) | (kg_data['tail_id:token'] == current_id)
        ]
        for _, row in neighbors.iterrows():
            next_id = row['tail_id:token'] if row['head_id:token'] == current_id else row['head_id:token']
            relation = row['relation_id:token']
            queue.append((next_id, path + [(current_id, relation, next_id)]))
    explanations = []
    for path in paths:
        path_desc = []
        for head, rel, tail in path:
            rel_name = relation_id_to_name.get(rel, f"related via {rel}")
            tail_name = item_id_to_artist.get(tail, tail) if rel in ['performed_by', 'performs'] else tail
            path_desc.append(f"{head} → {rel_name} → {tail_name}")
        explanations.append(" and ".join(path_desc))
    return explanations if paths else []

def get_recommendations_with_similarity(user_id, model, dataset, kg_data, k=10, max_hops=2):
    model.eval()
    with torch.no_grad():
        cleaned_user_id = clean_id(user_id)
        try:
            user_idx = dataset.token2id('user_id', cleaned_user_id)
        except KeyError:
            try:
                user_idx = dataset.token2id('user_id', str(user_id))
                cleaned_user_id = str(user_id)
            except KeyError:
                return [(f"User {user_id} not found", 0.0, ["No recommendations possible."])]

        user_tensor = torch.full((dataset.item_num,), user_idx, dtype=torch.long)
        item_tensor = torch.arange(dataset.item_num, dtype=torch.long)
        interaction = Interaction({
            'user_id': user_tensor,
            'item_id': item_tensor
        }).to(model.device)
        scores = model.predict(interaction)
        top_scores, top_items = torch.topk(scores, k)

        # Items user interacted with
        user_interactions_df = interaction_data[interaction_data['user_id:token'] == cleaned_user_id]
        user_item_tokens = [clean_id(x) for x in user_interactions_df['item_id:token'].tolist()]

        recommended_items = []
        for item_idx, score in zip(top_items, top_scores):
            item_id = clean_id(dataset.id2token('item_id', item_idx.item()))
            song_name = item_id_to_name.get(item_id, f"Unknown Item {item_id}")
            explanations = []

            for user_item in user_item_tokens:
                # Direct KG link
                direct_paths = kg_data[
                    ((kg_data['head_id:token'] == user_item) & (kg_data['tail_id:token'] == item_id)) |
                    ((kg_data['head_id:token'] == item_id) & (kg_data['tail_id:token'] == user_item))
                ]
                for _, row in direct_paths.iterrows():
                    relation = relation_id_to_name.get(row['relation_id:token'], f"related via {row['relation_id:token']}")
                    user_song_name = item_id_to_name.get(user_item, user_item)
                    explanations.append(f"You liked '{user_song_name}' which is {relation} this song")

                # Multi-hop if no direct
                if not any(user_item in exp for exp in explanations):
                    multi_hop_explanations = find_multi_hop_paths(kg_data, user_item, item_id, max_hops)
                    for multi_exp in multi_hop_explanations:
                        user_song_name = item_id_to_name.get(user_item, user_item)
                        explanations.append(f"Connected to '{user_song_name}' via: {multi_exp}")

            if not explanations:
                if user_item_tokens:
                    explanations = ["Based on your listening history and RippleNet's learned patterns"]
                else:
                    explanations = ["Based on popular trends and RippleNet's recommendations for new users"]

            explanations = explanations[:3]  # Top 3 explanations
            recommended_items.append((song_name, score.item(), explanations))
        return recommended_items

# === Example Usage ===
user_id = "122189405492083872"
recommendations = get_recommendations_with_similarity(user_id, model, dataset, kg_data, k=10, max_hops=2)

print(f"\nTop 10 recommended songs for user {user_id}:")
for i, (song, score, explanations) in enumerate(recommendations, 1):
    print(f"{i}. {song} (Score: {score:.4f})")
    print("   Explanation:")
    for exp in explanations:
        print(f"   - {exp}")
