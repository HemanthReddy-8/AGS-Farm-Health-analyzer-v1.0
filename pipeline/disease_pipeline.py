import streamlit as st
import torch
import clip
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
import time

# -----------------------------------------------------------------
# MODEL 1: DINOv2 (for Crop & Disease Clustering)
# -----------------------------------------------------------------

@st.cache_resource
def load_dino_model(device="auto"):
    """
    Loads and caches the DINOv2 model and processor.
    This is used for both crop and disease clustering.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[INFO] Loading DINOv2 (facebook/dinov2-small) on {device}...")
    
    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        print("[INFO] DINOv2 model loaded successfully.")
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading DINOv2 model: {e}")
        return None, None, None

def extract_dino_features(image_batch: list[tuple[str, Image.Image]], model, processor, device, batch_size=32):
    """
    Extracts DINOv2 features from a list of (filename, PIL_Image) tuples.
    """
    print(f"[INFO] Extracting DINOv2 features for {len(image_batch)} images...")
    pil_images = [img for filename, img in image_batch]
    embeddings = []
    
    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i + batch_size]
        
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    feats = model(**inputs).pooler_output
            else:
                feats = model(**inputs).pooler_output
        
        embeddings.append(feats.detach().cpu().numpy())
    
    if not embeddings:
        return np.array([])
        
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    return embeddings

def cluster_images_kmeans(image_batch: list[tuple[str, Image.Image]], embeddings: np.ndarray, num_clusters: int):
    """
    Runs K-Means clustering and returns a dictionary of image groups.
    
    Returns:
        dict: {0: [(fname, img), ...], 1: [(fname, img), ...]}
    """
    print(f"[INFO] Running K-Means clustering (K={num_clusters})...")
    if len(image_batch) < num_clusters:
        print(f"[WARN] Fewer images ({len(image_batch)}) than clusters ({num_clusters}). Returning 1 cluster.")
        return {0: image_batch}
        
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(embeddings)
    
    # Group images by their assigned cluster ID
    groups = defaultdict(list)
    for img_tuple, cluster_id in zip(image_batch, cluster_ids):
        groups[int(cluster_id)].append(img_tuple)
        
    return dict(groups)

# -----------------------------------------------------------------
# MODEL 2: CLIP (for Health Classification)
# -----------------------------------------------------------------

class CLIPHealthClassifier:
    """
    Refactored version of your script.
    This class NO LONGER writes files. It takes a list of images
    and returns lists of images.
    """
    def __init__(self, confidence_threshold=0.25, device=None):
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print(f"CLIP model loaded on {self.device}")

        # Classes with descriptive prompts
        self.health_descriptions = {
            "Healthy": [
                "a perfect green leaf with uniform color and smooth surface",
                "a normal healthy leaf under sunlight or shade with no visible damage",
                "a clean leaf showing natural veins and edges, not deformed",
                "a vibrant green leaf with intact structure, no spots or trails",
            ],
            "Unhealthy": [
                "a leaf with thin winding white or gray trails across the surface (leaf miner damage)",
                "a leaf with distorted or curled shape from viral infection",
                "a leaf that appears wrinkled or twisted with uneven surface texture",
                "a leaf showing yellow patches, pale streaks, or discoloration",
                "a leaf with tiny holes or punctures from pests or disease",
                "a leaf with irregular lines, squiggly marks, or tunneling patterns",
                "a leaf affected by leaf curl virus showing rolled edges or twisted veins",
            ]
        }
        self.create_text_embeddings()

    def create_text_embeddings(self):
        print("Creating CLIP text embeddings...")
        self.text_embeddings = {}
        for label, descriptions in self.health_descriptions.items():
            embs = []
            for desc in descriptions:
                tokens = clip.tokenize([f"a photo of {desc}"]).to(self.device)
                with torch.no_grad():
                    e = self.model.encode_text(tokens)
                    e = e / e.norm(dim=-1, keepdim=True)
                    embs.append(e.squeeze(0))
            mean_emb = torch.stack(embs).mean(dim=0)
            mean_emb = mean_emb / mean_emb.norm()
            self.text_embeddings[label] = mean_emb.to(self.device)
        print("Text embeddings ready.")

    def classify_single_image(self, image: Image.Image):
        """Classifies a single PIL Image."""
        try:
            with torch.no_grad():
                img_input = self.preprocess(image).unsqueeze(0).to(self.device)
                img_emb = self.model.encode_image(img_input)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                sims = {}
                for label, text_emb in self.text_embeddings.items():
                    sim = torch.cosine_similarity(img_emb, text_emb.unsqueeze(0), dim=-1)
                    sims[label] = float(sim.item())

            best_label = max(sims, key=sims.get)
            best_conf = sims[best_label]
            return (best_label if best_conf >= self.confidence_threshold else "Unclassified",
                    best_conf)
        except Exception as e:
            print(f"Error classifying image: {e}")
            return "Unclassified", 0.0

    def classify_image_group(self, image_group: list[tuple[str, Image.Image]]):
        """
        Takes a list of (filename, PIL_Image) tuples and sorts them.
        Returns two lists: healthy_images and unhealthy_images.
        """
        print(f"CLIP: Classifying health for {len(image_group)} images...")
        groups = {
            "Healthy": [],
            "Unhealthy": [],
            "Unclassified": []
        }
        
        for filename, img in image_group:
            pred, conf = self.classify_single_image(img)
            groups[pred].append((filename, img))
        
        print(f"CLIP Results: {len(groups['Healthy'])} Healthy, {len(groups['Unhealthy'])} Unhealthy, {len(groups['Unclassified'])} Unclassified")
        
        # We will group "Unclassified" with "Unhealthy" for the next step
        unhealthy_combined = groups["Unhealthy"] + groups["Unclassified"]
        
        return groups["Healthy"], unhealthy_combined


@st.cache_resource
def load_clip_classifier():
    """
    Loads and caches the entire CLIPHealthClassifier object.
    This runs __init__ once and stores the ready-to-use object.
    """
    classifier = CLIPHealthClassifier()
    return classifier

# -----------------------------------------------------------------
# MAIN PIPELINE FUNCTIONS (to be called by app.py)
# -----------------------------------------------------------------

def run_crop_classification(image_batch: list[tuple[str, Image.Image]], dino_model, dino_processor, device) -> dict:
    """
    STEP 1: Classify a batch of images into 3 crop types using DINOv2.
    """
    print("\n--- BRANCH 2: STEP 1 (Crop Classification) ---")
    if not image_batch:
        return {}
        
    embeddings = extract_dino_features(image_batch, dino_model, dino_processor, device)
    if embeddings.size == 0:
        return {}
        
    # 1. Run K-Means with K=3
    # Output is {0: [imgs], 1: [imgs], 2: [imgs]}
    grouped_by_id = cluster_images_kmeans(image_batch, embeddings, num_clusters=3)
    
    # 2. Rename keys to be user-friendly for the report
    final_crop_groups = {
        "Crop 1": grouped_by_id.get(0, []),
        "Crop 2": grouped_by_id.get(1, []),
        "Crop 3": grouped_by_id.get(2, [])
    }
    
    print(f"Crop Results: C1={len(final_crop_groups['Crop 1'])}, C2={len(final_crop_groups['Crop 2'])}, C3={len(final_crop_groups['Crop 3'])}")
    return final_crop_groups


def run_health_classification(crop_name: str, crop_image_group: list, clip_classifier: CLIPHealthClassifier) -> dict:
    """
    STEP 2: Classify a single crop's images into Healthy/Unhealthy using CLIP.
    """
    print(f"\n--- BRANCH 2: STEP 2 (Health Classification for {crop_name}) ---")
    if not crop_image_group:
        return {"healthy": [], "unhealthy": []}
        
    healthy_images, unhealthy_images = clip_classifier.classify_image_group(crop_image_group)
    
    return {
        "healthy": healthy_images,
        "unhealthy": unhealthy_images
    }

def run_disease_classification(crop_name: str, unhealthy_batch: list, dino_model, dino_processor, device) -> dict:
    """
    STEP 3: Classify a batch of unhealthy images into 3 disease types using DINOv2.
    """
    print(f"\n--- BRANCH 2: STEP 3 (Disease Classification for {crop_name}) ---")
    if not unhealthy_batch:
        return {}
        
    embeddings = extract_dino_features(unhealthy_batch, dino_model, dino_processor, device)
    if embeddings.size == 0:
        return {}
        
    # 1. Run K-Means with K=3
    grouped_by_id = cluster_images_kmeans(unhealthy_batch, embeddings, num_clusters=3)
    
    # 2. Rename keys to be user-friendly
    # Note: These names are generic. You might want to pass in specific
    # disease names for each crop (e.g., {"Crop 1": ["Rust", "Blight", "Virus"]})
    final_disease_groups = {
        f"{crop_name} Disease A": grouped_by_id.get(0, []),
        f"{crop_name} Disease B": grouped_by_id.get(1, []),
        f"{crop_name} Disease C": grouped_by_id.get(2, [])
    }
    
    print(f"Disease Results: A={len(final_disease_groups[f'{crop_name} Disease A'])}, B={len(final_disease_groups[f'{crop_name} Disease B'])}, C={len(final_disease_groups[f'{crop_name} Disease C'])}")
    return final_disease_groups