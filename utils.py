import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def initialize_model(model_name="ViT-B/32", pretrained="openai", device="cpu"):
    model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess_val

def load_embeddings(pickle_file):
    try:
        return pd.read_pickle(pickle_file)
    except FileNotFoundError:
        raise Exception(f"Pickle file '{pickle_file}' not found. Ensure the file exists and the path is correct.")

def get_image_paths(image_folder):
    # Get all image paths and slice to first 10000
    image_paths = [
        os.path.join(image_folder, fname) 
        for fname in os.listdir(image_folder) 
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:10000]  # Hard limit to 10000 images
    return image_paths

def generate_embeddings(image_folder, model, device, batch_size=32):
    # Image transformations (lightweight)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Collect and slice image paths
    image_paths = get_image_paths(image_folder)
    print(f"Number of images to process: {len(image_paths)}")

    def preprocess_image(path):
        try:
            image = Image.open(path).convert("RGB")
            return transform(image)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def load_images_parallel(batch_paths):
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(preprocess_image, batch_paths))
        images = [img for img in images if img is not None]
        return torch.stack(images) if images else None

    results = []
    embeddings_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i + batch_size]
            images = load_images_parallel(batch_paths)
            if images is None:
                continue

            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            results.extend([os.path.basename(path) for path in batch_paths])
            embeddings_list.append(embeddings.cpu().numpy())

    all_embeddings = np.vstack(embeddings_list)
    return pd.DataFrame({"file_name": results, "embedding": list(all_embeddings)})

def encode_image(image_path, model, preprocess, device):
    try:
        if hasattr(image_path, 'read'):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = F.normalize(model.encode_image(image_tensor), p=2, dim=1)
        return image_embedding
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def encode_text(text_query, model, device):
    try:
        tokenizer = get_tokenizer('ViT-B-32')
        text = tokenizer([text_query]).to(device)
        with torch.no_grad():
            text_embedding = F.normalize(model.encode_text(text), p=2, dim=1)
        return text_embedding
    except Exception as e:
        raise Exception(f"Error processing text query: {str(e)}")

def combine_embeddings(image_embedding, text_embedding, lam=0.8):
    with torch.no_grad():
        hybrid_embedding = F.normalize(
            lam * text_embedding + (1.0 - lam) * image_embedding, 
            p=2, 
            dim=1
        )
    return hybrid_embedding

def calculate_similarity(query_embedding, df, image_folder, top_k=5):
    try:
        embeddings = np.stack(df['embedding'].to_numpy())
        query_embedding_np = query_embedding.squeeze().detach().cpu().numpy()
        cosine_similarities = np.dot(embeddings, query_embedding_np)

        top_indices = np.argsort(-cosine_similarities)[:top_k]
        results = [
            {
                "path": f"/images/{df.iloc[idx]['file_name']}",  # Updated to use the new route
                "score": float(round(cosine_similarities[idx], 3))  # Rounded for cleaner display
            } 
            for idx in top_indices
        ]
        return results
    except Exception as e:
        raise Exception(f"Error calculating similarity: {str(e)}")