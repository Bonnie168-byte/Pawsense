# =============================================================================
# ISOM5240 Group Project - Streamlit App
# PawSense — AI-Powered Dog Emotion Detection & Product Recommendation
# =============================================================================
#
# Deploy: Streamlit Cloud (Link GitHub Repo)
# Pipeline:
#     Pipeline A: image-classification (ViT, pre-trained) -> Breed Recognition
#     Pipeline B: image-classification (ViT, fine-tuned) -> Emotion Detection
#     Pipeline C: text2text-generation (bart-base, prompt engineering) -> Product Recommendation
# =============================================================================

import streamlit as st
from transformers import pipeline
from PIL import Image
import time

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="PawSense",
    page_icon="🐾",
    layout="wide",
)

# ============================
# Custom CSS
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.block-container { max-width: 1100px; padding-top: 2rem; }
.hero-title { font-size: 2.4rem; font-weight: 800; color: #6C3FC5; margin-bottom: 0.2rem; }
.hero-sub { font-size: 1.05rem; color: #888; margin-bottom: 1.5rem; }
.upload-box { border: 2px dashed #D1C4E9; border-radius: 16px; padding: 2.5rem 1rem; text-align: center; background: #FAF7FF; }
.result-card { background: #fff; border: 1px solid #EDE7F6; border-radius: 16px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 12px rgba(108,63,197,0.06); }
.result-label { font-size: 0.8rem; color: #999; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem; }
.result-value { font-size: 1.4rem; font-weight: 700; color: #333; }
.confidence-badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-left: 8px; }
.emo-happy   { background: #E8F5E9; color: #2E7D32; }
.emo-sad     { background: #E3F2FD; color: #1565C0; }
.emo-angry   { background: #FFEBEE; color: #C62828; }
.emo-relaxed { background: #F3E5F5; color: #7B1FA2; }
.score-row { display: flex; align-items: center; margin: 6px 0; }
.score-label { width: 80px; font-size: 0.85rem; color: #666; }
.score-bar-bg { flex: 1; height: 10px; background: #F3F0FA; border-radius: 5px; overflow: hidden; }
.score-bar { height: 100%; border-radius: 5px; transition: width .6s ease; }
.product-card { background: #fff; border: 1px solid #EDE7F6; border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 12px; box-shadow: 0 1px 6px rgba(108,63,197,0.05); }
.product-rank { width: 32px; height: 32px; border-radius: 50%; background: #7C4DFF; color: #fff; font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; flex-shrink: 0; }
.product-info { flex: 1; }
.product-name { font-weight: 700; color: #333; font-size: 0.95rem; }
.product-reason { font-size: 0.82rem; color: #888; margin-top: 2px; }
.product-meta { text-align: right; }
.product-price { font-weight: 700; color: #6C3FC5; font-size: 1rem; }
.product-cat { font-size: 0.72rem; color: #aaa; }
section[data-testid="stSidebar"] { background: #F8F5FF; }
.footer { text-align: center; color: #bbb; font-size: 0.8rem; margin-top: 3rem; padding: 1rem 0; border-top: 1px solid #F0ECF7; }
</style>
""", unsafe_allow_html=True)

# ============================
# Product Database (real Chewy.com products)
# ============================
CHEWY_CATALOG = {
    "sad": [
        {"name": "KONG Classic Dog Toy",              "category": "Toy",        "price": 13.99, "reason": "Stuff with treats to keep your pup mentally stimulated and distracted from the blues"},
        {"name": "Zesty Paws Calming Bites",          "category": "Supplement", "price": 29.97, "reason": "Chamomile & L-theanine formula helps promote natural relaxation"},
        {"name": "Snuggle Puppy Behavioral Aid Toy",  "category": "Comfort",    "price": 44.99, "reason": "Built-in heartbeat mimics a companion — great for separation sadness"},
    ],
    "angry": [
        {"name": "Benebone Wishbone Chew Toy",        "category": "Chew",       "price": 14.99, "reason": "Redirects aggressive energy into satisfying, safe chewing"},
        {"name": "KONG Extreme Chew Toy",             "category": "Chew",       "price": 15.99, "reason": "Ultra-durable rubber withstands the toughest chewers"},
        {"name": "Thundershirt Sport Anxiety Jacket",  "category": "Calming",    "price": 44.95, "reason": "Gentle constant pressure helps calm agitation and reactivity"},
    ],
    "relaxed": [
        {"name": "Frisco Orthopedic Dog Bed",         "category": "Bed",        "price": 59.99, "reason": "Premium memory-foam comfort to maintain that chill vibe"},
        {"name": "Milk-Bone Original Dog Biscuits",   "category": "Treat",      "price": 5.99,  "reason": "A classic crunchy reward to celebrate the good mood"},
        {"name": "Outward Hound Dog Puzzle Toy",      "category": "Enrichment", "price": 12.99, "reason": "Gentle mental enrichment keeps a relaxed dog happily engaged"},
    ],
    "happy": [
        {"name": "Chuckit! Ultra Ball Dog Toy",       "category": "Toy",        "price": 9.99,  "reason": "Match that high energy with an epic game of fetch"},
        {"name": "West Paw Zogoflex Tux Toy",         "category": "Toy",        "price": 18.95, "reason": "Bouncy, chewable, and treat-stuffable — triple the fun"},
        {"name": "Zuke's Mini Naturals Dog Treats",   "category": "Treat",      "price": 8.99,  "reason": "Pocket-sized rewards to keep the good times rolling"},
    ],
}

BEHAVIOR_META = {
    "happy":   {"emoji": "😊", "color": "#4CAF50", "css": "emo-happy",   "bar": "#66BB6A", "desc": "Your dog looks happy and content!"},
    "sad":     {"emoji": "😢", "color": "#1976D2", "css": "emo-sad",     "bar": "#42A5F5", "desc": "Your dog seems a bit down today."},
    "angry":   {"emoji": "😠", "color": "#D32F2F", "css": "emo-angry",   "bar": "#EF5350", "desc": "Your dog appears agitated or upset."},
    "relaxed": {"emoji": "😌", "color": "#7B1FA2", "css": "emo-relaxed", "bar": "#AB47BC", "desc": "Your dog is calm and relaxed."},
}

SMALL_BREEDS = {"chihuahua", "yorkshire terrier", "dachshund", "pomeranian", "maltese", "shih tzu", "toy poodle", "papillon"}
LARGE_BREEDS = {"golden retriever", "labrador retriever", "german shepherd", "boxer", "rottweiler", "doberman", "great dane"}

# ============================
# Helper Functions
# ============================
def get_size(breed):
    b = breed.lower()
    if any(s in b for s in SMALL_BREEDS): return "Small"
    if any(s in b for s in LARGE_BREEDS): return "Large"
    return "Medium"

def get_products(behavior, top_n=3):
    return CHEWY_CATALOG.get(behavior, CHEWY_CATALOG["happy"])[:top_n]

# ============================
# Load Pipelines (cached)
# ============================
@st.cache_resource
def load_breed_classifier():
    """Pipeline A: Breed Recognition (pretrained ViT)"""
    return pipeline("image-classification", model="wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")

@st.cache_resource
def load_emotion_classifier():
    """Pipeline B: Emotion Detection (fine-tuned ViT)"""
    return pipeline("image-classification", model="Bonnnz/CustomModel_dogemotion")

@st.cache_resource
def load_recommendation_generator():
    """Pipeline C: Product Recommendation (flan-t5-base)"""
    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150)

# ============================
# Pipeline Functions
# ============================
def classify_breed(image, classifier):
    """Pipeline A: Identify dog breed from image."""
    results = classifier(image, top_k=3)
    breed = results[0]["label"].replace("_", " ")
    confidence = results[0]["score"]
    return breed, confidence, results

def classify_emotion(image, classifier):
    """Pipeline B: Detect behavioral state from image."""
    results = classifier(image, top_k=4)
    emotion = results[0]["label"]
    confidence = results[0]["score"]
    return emotion, confidence, results

def generate_recommendation(breed, emotion, products, generator):
    """Pipeline C: Generate recommendation text using product catalog."""
    product_names = ", ".join(p["name"] for p in products)
    prompt = (
        f"Question: My {breed} dog is {emotion}. "
        f"Available Chewy products: {product_names}. "
        f"Write a short friendly message to the owner explaining "
        f"what this behavior means and why these products help.\n"
        f"Answer:"
    )
    result = generator(prompt)
    return result[0]["generated_text"]

# ============================
# Main App
# ============================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 🐾 PawSense")
        st.caption("Understand your dog. Love them better.")
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "1. 🔍 **Breed Recognition** — identifies your dog's breed\n"
            "2. 🧠 **Emotion Detection** — reads their current mood\n"
            "3. 📦 **Smart Recommendations** — picks the best Chewy.com products\n"
        )
        st.markdown("---")
        st.markdown("**Company:** [Chewy.com](https://www.chewy.com)")
        st.markdown("**Course:** ISOM5240 Group Project")

    # Hero Title
    st.markdown('<div class="hero-title">PawSense — Know Your Dog Better 🐾</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a photo of your dog — AI will identify the breed, read their mood, and recommend the perfect products from Chewy.com.</div>', unsafe_allow_html=True)

    # Image Upload
    uploaded_file = st.file_uploader(
        "Upload your dog's photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("💡 Tip: A clear front-facing photo gives the best results!")
    else:

    # Process image
    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Your dog's photo", use_container_width=True)

    with col_result:
        with st.spinner("🔍 Analyzing your dog..."):
            # Load all 3 pipelines
            breed_clf = load_breed_classifier()
            emotion_clf = load_emotion_classifier()
            rec_gen = load_recommendation_generator()

            # Pipeline A: Breed
            start_a = time.time()
            breed, breed_conf, breed_top3 = classify_breed(image, breed_clf)
            time_a = time.time() - start_a

            # Pipeline B: Emotion
            start_b = time.time()
            behavior, behavior_conf, all_behaviors = classify_emotion(image, emotion_clf)
            time_b = time.time() - start_b

        meta = BEHAVIOR_META.get(behavior, BEHAVIOR_META["happy"])
        size = get_size(breed)

        # Breed card
        st.markdown(f"""<div class="result-card">
            <div class="result-label">🐕 Dog Breed</div>
            <div class="result-value">{breed}
                <span class="confidence-badge" style="background:#F3E5F5;color:#6C3FC5;">{breed_conf*100:.1f}% confidence</span>
            </div>
            <div style="font-size:0.85rem;color:#999;margin-top:4px;">Size category: {size}</div>
        </div>""", unsafe_allow_html=True)

        # Show top 3 breeds if confidence is low
        if breed_conf < 0.5:
            with st.expander("🔍 Other possible breeds"):
                for r in breed_top3:
                    name = r["label"].replace("_", " ")
                    st.write(f"- {name}: {r['score']*100:.1f}%")

        # Emotion card
        st.markdown(f"""<div class="result-card">
            <div class="result-label">🧠 Behavior State</div>
            <div class="result-value">
                <span style="font-size:1.8rem;vertical-align:middle;">{meta['emoji']}</span>
                <span style="color:{meta['color']};vertical-align:middle;">{behavior.capitalize()}</span>
                <span class="confidence-badge {meta['css']}">{behavior_conf*100:.1f}%</span>
            </div>
            <div style="font-size:0.85rem;color:#666;margin-top:6px;">{meta['desc']}</div>
        </div>""", unsafe_allow_html=True)

    # Emotion Score Breakdown
    st.markdown("### 📊 Emotion Analysis")
    sc1, sc2 = st.columns([1.2, 1])

    with sc1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Behavior Confidence Scores</div>', unsafe_allow_html=True)
        for r in all_behaviors:
            m = BEHAVIOR_META.get(r["label"], BEHAVIOR_META["happy"])
            pct = r["score"] * 100
            st.markdown(f"""<div class="score-row">
                <span class="score-label">{m['emoji']} {r['label'].capitalize()}</span>
                <div class="score-bar-bg"><div class="score-bar" style="width:{pct}%;background:{m['bar']};"></div></div>
                <span style="width:55px;text-align:right;font-size:0.85rem;font-weight:600;color:#555;">{pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""<div class="result-card" style="text-align:center;padding:1.5rem;">
            <div style="font-size:3.5rem;">{meta['emoji']}</div>
            <div style="font-size:2rem;font-weight:800;color:{meta['color']};margin:0.3rem 0;">{behavior.capitalize()}</div>
            <div style="font-size:0.9rem;color:#999;">Detected behavior state</div>
        </div>""", unsafe_allow_html=True)

    # Pipeline C: AI Product Recommendation
    st.markdown("---")
    st.markdown("### 💬 AI Behavior Insight")

    products = get_products(behavior)

    with st.spinner("💡 Generating personalized insight..."):
        start_c = time.time()
        rec_text = generate_recommendation(breed, behavior, products, rec_gen)
        time_c = time.time() - start_c

    st.markdown(f"""<div class="result-card" style="border-left:4px solid {meta['color']};">
        <div style="font-size:0.95rem;line-height:1.7;color:#444;">{rec_text}</div>
    </div>""", unsafe_allow_html=True)

    # Product Cards
    st.markdown(f"### 🛒 Recommended Products for Your {breed}")
    st.caption(f"Curated based on **{behavior}** behavior state and **{size.lower()}** dog size")

    for i, p in enumerate(products):
        st.markdown(f"""<div class="product-card">
            <div class="product-rank">{i+1}</div>
            <div class="product-info">
                <div class="product-name">{p['name']}</div>
                <div class="product-reason">{p['reason']}</div>
            </div>
            <div class="product-meta">
                <div class="product-price">${p['price']:.2f}</div>
                <div class="product-cat">{p['category']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""<div class="footer">
    ⚠️ PawSense detects observable behavioral patterns only and does not constitute veterinary advice.<br>
    Please consult a professional veterinarian for health concerns.<br><br>
    <strong>PawSense</strong> — Understand your dog. Love them better. 🐾
</div>""", unsafe_allow_html=True)

# ============================
# Entry
# ============================
if __name__ == "__main__":
    main()
