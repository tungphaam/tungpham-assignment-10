from flask import Flask, render_template, request, flash, send_from_directory
import os
import torch
from utils import (
    initialize_model,
    load_embeddings,
    encode_image,
    encode_text,
    combine_embeddings,
    calculate_similarity,
    generate_embeddings
)

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configuration
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
pretrained = "openai"
pickle_file = "image_embeddings.pickle"
image_folder = "coco_images_resized"
batch_size = 32

# Load model and embeddings
try:
    print(f"Using device: {device}")
    model, preprocess = initialize_model(model_name, pretrained, device)
    
    # Check if embeddings exist, if not generate them
    if not os.path.exists(pickle_file):
        print("Generating embeddings for first 10000 images...")
        df = generate_embeddings(image_folder, model, device, batch_size)
        df.to_pickle(pickle_file)
        print(f"Embeddings saved to {pickle_file}")
    
    df = load_embeddings(pickle_file)
    print(f"Model and embeddings loaded successfully. Number of images: {len(df)}")
except Exception as e:
    print(f"Error initializing model or loading embeddings: {e}")
    raise e

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized', filename)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error_message = None
    try:
        if request.method == "POST":
            query_type = request.form.get("query_type")
            lam = float(request.form.get("weight", 0.8))

            # Handle text query
            if query_type == "text_query":
                text_query = request.form.get("text_query", "")
                if not text_query:
                    raise ValueError("Text query is empty.")
                query_embedding = encode_text(text_query, model, device)

            # Handle image query
            elif query_type == "image_query":
                uploaded_image = request.files.get("image_query")
                if not uploaded_image:
                    raise ValueError("No image file uploaded.")
                query_embedding = encode_image(uploaded_image, model, preprocess, device)

            # Handle hybrid query
            elif query_type == "hybrid_query":
                text_query = request.form.get("text_query", "")
                uploaded_image = request.files.get("image_query")
                if not text_query or not uploaded_image:
                    raise ValueError("Both text and image are required for a hybrid query.")

                text_embedding = encode_text(text_query, model, device)
                image_embedding = encode_image(uploaded_image, model, preprocess, device)
                query_embedding = combine_embeddings(image_embedding, text_embedding, lam=lam)

            else:
                raise ValueError("Invalid query type selected.")

            # Calculate results
            results = calculate_similarity(query_embedding, df, image_folder, top_k=5)
    except Exception as e:
        error_message = str(e)
        flash(error_message, "error")

    return render_template("index.html", results=results, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True, port=3000)