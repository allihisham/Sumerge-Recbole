import pandas as pd
import numpy as np

# Diagnostic function to understand why KG connections aren't being found
def diagnose_kg_connections(user_id, interaction_data, kg_data, item_id_to_name):
    """Diagnose why KG connections aren't being found between user interactions and recommendations"""
    
    # Get user interactions
    user_interactions_df = interaction_data[interaction_data['user_id:token'] == user_id]
    user_item_tokens = user_interactions_df['item_id:token'].tolist()
    
    print(f"=== DIAGNOSTIC REPORT FOR USER {user_id} ===")
    print(f"User interacted with {len(user_item_tokens)} items")
    print(f"Sample user interactions: {user_item_tokens[:10]}")
    
    # Check data types and formats
    print(f"\n=== DATA TYPE ANALYSIS ===")
    print(f"User interaction item types: {type(user_item_tokens[0])}")
    print(f"KG head_id types: {kg_data['head_id:token'].dtype}")
    print(f"KG tail_id types: {kg_data['tail_id:token'].dtype}")
    
    # Sample KG entities
    kg_entities = set(pd.concat([kg_data['head_id:token'], kg_data['tail_id:token']]).unique())
    sample_kg_entities = list(kg_entities)[:20]
    print(f"Sample KG entities: {sample_kg_entities}")
    
    # Check if user items exist in KG
    user_items_in_kg = [item for item in user_item_tokens[:50] if str(item) in kg_entities]
    print(f"\n=== USER ITEMS IN KG ===")
    print(f"User items found in KG (first 50 checked): {len(user_items_in_kg)}")
    print(f"Sample user items in KG: {user_items_in_kg[:10]}")
    
    # Convert user item tokens to strings for comparison
    user_item_strings = [str(item) for item in user_item_tokens[:100]]  # Check first 100
    user_items_in_kg_str = [item for item in user_item_strings if item in kg_entities]
    print(f"User items (as strings) in KG: {len(user_items_in_kg_str)}")
    
    # Check specific KG relations for user items
    if user_items_in_kg_str:
        sample_user_item = user_items_in_kg_str[0]
        print(f"\n=== KG RELATIONS FOR SAMPLE USER ITEM: {sample_user_item} ===")
        
        # Find all KG relations for this item
        item_relations = kg_data[
            (kg_data['head_id:token'] == sample_user_item) | 
            (kg_data['tail_id:token'] == sample_user_item)
        ]
        print(f"Number of KG relations for item {sample_user_item}: {len(item_relations)}")
        print("Sample relations:")
        for _, row in item_relations.head(10).iterrows():
            print(f"  {row['head_id:token']} --{row['relation_id:token']}--> {row['tail_id:token']}")
        
        # Get all connected entities
        connected_entities = set(item_relations['head_id:token'].tolist() + item_relations['tail_id:token'].tolist())
        connected_entities.discard(sample_user_item)  # Remove the item itself
        print(f"Connected entities: {len(connected_entities)}")
        print(f"Sample connected entities: {list(connected_entities)[:10]}")
        
    # Check relation types
    print(f"\n=== RELATION TYPES IN KG ===")
    relation_counts = kg_data['relation_id:token'].value_counts()
    print(relation_counts)
    
    return user_items_in_kg_str, sample_kg_entities

# Run diagnostic
user_id = "122189405492083872"
user_items_in_kg, sample_kg_entities = diagnose_kg_connections(
    user_id, interaction_data, kg_data, item_id_to_name
)

# Additional diagnostic: Check if recommended items have KG connections to user items
print(f"\n=== CHECKING RECOMMENDED ITEMS ===")

# Get some recommended items (you can replace these with actual recommended items)
recommended_items = ["haygeley mwgoaa", "kholsit el hekaya", "khalina zekra"]

# Find their IDs in the metadata
recommended_item_ids = []
for song_name in recommended_items:
    # Find song ID by name
    matching_songs = item_metadata[item_metadata['song_name'] == song_name]
    if not matching_songs.empty:
        song_id = str(int(float(matching_songs.iloc[0]['song_id'])))
        recommended_item_ids.append(song_id)
        print(f"Song '{song_name}' has ID: {song_id}")

print(f"Recommended item IDs: {recommended_item_ids}")

# Check if recommended items are in KG
kg_entities = set(pd.concat([kg_data['head_id:token'], kg_data['tail_id:token']]).unique())
recommended_in_kg = [item_id for item_id in recommended_item_ids if item_id in kg_entities]
print(f"Recommended items in KG: {recommended_in_kg}")

# Now check for direct connections between user items and recommended items
if user_items_in_kg and recommended_in_kg:
    print(f"\n=== CHECKING DIRECT CONNECTIONS ===")
    sample_user_item = user_items_in_kg[0]
    sample_recommended_item = recommended_in_kg[0]
    
    print(f"Checking connection between user item {sample_user_item} and recommended item {sample_recommended_item}")
    
    # Direct connections
    direct_connections = kg_data[
        ((kg_data['head_id:token'] == sample_user_item) & (kg_data['tail_id:token'] == sample_recommended_item)) |
        ((kg_data['head_id:token'] == sample_recommended_item) & (kg_data['tail_id:token'] == sample_user_item))
    ]
    
    print(f"Direct connections found: {len(direct_connections)}")
    if len(direct_connections) > 0:
        print("Direct connections:")
        for _, row in direct_connections.iterrows():
            print(f"  {row['head_id:token']} --{row['relation_id:token']}--> {row['tail_id:token']}")
    else:
        print("No direct connections found. Checking indirect connections...")
        
        # Find common entities
        user_item_connections = kg_data[
            (kg_data['head_id:token'] == sample_user_item) | 
            (kg_data['tail_id:token'] == sample_user_item)
        ]
        recommended_item_connections = kg_data[
            (kg_data['head_id:token'] == sample_recommended_item) | 
            (kg_data['tail_id:token'] == sample_recommended_item)
        ]
        
        user_connected_entities = set(user_item_connections['head_id:token'].tolist() + 
                                    user_item_connections['tail_id:token'].tolist())
        recommended_connected_entities = set(recommended_item_connections['head_id:token'].tolist() + 
                                           recommended_item_connections['tail_id:token'].tolist())
        
        common_entities = user_connected_entities.intersection(recommended_connected_entities)
        common_entities.discard(sample_user_item)
        common_entities.discard(sample_recommended_item)
        
        print(f"Common connected entities: {len(common_entities)}")
        print(f"Sample common entities: {list(common_entities)[:10]}")

print("\n=== DIAGNOSTIC COMPLETE ===")