# =============================================================================
# ISOM5240 Group Project - Streamlit App
# PawSense — AI-Powered Dog Behavior Detection & Product Recommendation
# =============================================================================
# 
# Deploy: Streamlit Cloud (Link GitHub Repo)
# Pipeline:
#   Pipeline A: image-classification (ViT, pre-trained) -> Breed Recognition
#   Pipeline B: image-classification (ViT, fine-tuned) -> Emotion Detection
#   Pipeline C: text2text-generation (bart-base, prompt engineering) -> Product Recommendation
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
# LOAD PIPELINES
# ============================
# use @st.cache_resource to cache pipeline

@st.cache_resource
def load_breed_classifier():
    """Load Pipeline A: Breed Recognition (pretrained ViT)"""
    return hf_pipeline("image-classification", model="wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")

@st.cache_resource
def load_emotion_classifier():
    """Load Pipeline B: Emotion Detection (fine-tuned ViT)"""
    return pipeline(
        "image-classification",
        model="Bonnnz/CustomModel_dogemotion",
    )

def load_recommendation_generator():
    """Load Pipeline C: product recommendation generator (bart-base)"""
    return pipeline(
        "text2text-generation",
        model="facebook/bart-base",
        max_length=150,
    )


# ============================
# Pipeline 函数 (老师要求: 拆成独立函数)
# ============================

def classify_behavior(image, classifier):
    """
    Pipeline A: 分析宠物行为状态
    
    输入: PIL Image
    输出: list of {label, score}
    """
    results = classifier(image)
    return results


def generate_recommendation(species, behavior, generator):
    """
    Pipeline B: 根据行为状态生成产品推荐
    
    输入: species (str), behavior (str)
    输出: str (推荐文本)
    """
    prompt = (
        f"A pet {species} is showing {behavior} behavioral state. "
        f"As a pet care expert at Chewy.com, recommend 3 specific product "
        f"categories that would help this pet. For each product, provide "
        f"the product type and a brief reason why it helps. "
        f"Format: 1. [Product] - [Reason] 2. [Product] - [Reason] 3. [Product] - [Reason]"
    )
    result = generator(prompt)
    return result[0]["generated_text"]


# ============================
# UI: 行为标签 → emoji 和颜色映射
# ============================
BEHAVIOR_INFO = {
    "happy":   {"emoji": "😊", "color": "#4CAF50", "desc": "Your pet looks happy and content!"},
    "sad":     {"emoji": "😢", "color": "#2196F3", "desc": "Your pet seems a bit down."},
    "angry":   {"emoji": "😠", "color": "#F44336", "desc": "Your pet appears agitated or upset."},
    "relaxed": {"emoji": "😌", "color": "#9C27B0", "desc": "Your pet is calm and relaxed."},
}


# ============================
# Main App
# ============================

def main():
    """主程序入口"""
    
    # ---------- Header ----------
    st.title("🐾 Chewy Pet Behavior AI")
    st.markdown(
        "Upload a photo of your pet to understand their current behavioral state "
        "and get personalized product recommendations from **Chewy.com**."
    )
    
    # ---------- Sidebar: Pet Species ----------
    st.sidebar.header("🐾 Pet Info")
    species = st.sidebar.radio(
        "What type of pet is this?",
        options=["dog", "cat"],
        index=0,
        help="Select your pet's species for more accurate recommendations."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("ISOM5240 Group Project")
    st.sidebar.caption("Company: [Chewy](https://www.chewy.com)")
    
    # ---------- Image Upload ----------
    st.header("📷 Upload Your Pet's Photo")
    
    uploaded_file = st.file_uploader(
        "Choose an image of your pet",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Your pet's photo", use_container_width=True)
        
        # ---------- 运行分析 ----------
        with col2:
            with st.spinner("🔍 Analyzing your pet's behavior..."):
                # 加载模型
                classifier = load_behavior_classifier()
                generator = load_recommendation_generator()
                
                # Pipeline A: 行为分类
                start_time = time.time()
                predictions = classify_behavior(image, classifier)
                classify_time = time.time() - start_time
                
                # 获取最高置信度的预测
                top_pred = predictions[0]
                behavior = top_pred["label"]
                confidence = top_pred["score"]
                
                info = BEHAVIOR_INFO.get(behavior, {"emoji": "❓", "color": "#666", "desc": ""})
            
            # ---------- 显示行为分析结果 ----------
            st.subheader("🧠 Behavior Analysis Result")
            
            # 主结果卡片
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {info['color']}22, {info['color']}11);
                    border-left: 4px solid {info['color']};
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                ">
                    <div style="font-size: 36px; margin-bottom: 8px;">{info['emoji']}</div>
                    <div style="font-size: 24px; font-weight: bold; color: {info['color']};">
                        {behavior.upper()}
                    </div>
                    <div style="font-size: 14px; color: #666; margin-top: 4px;">
                        Confidence: {confidence*100:.1f}%
                    </div>
                    <div style="font-size: 14px; margin-top: 8px;">
                        {info['desc']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # 置信度分布 (老师示例中用了 st.bar_chart)
            with st.expander("📊 See confidence scores for all behaviors"):
                chart_data = {p["label"]: p["score"] for p in predictions}
                st.bar_chart(chart_data)
                st.caption(f"Analysis completed in {classify_time:.2f}s")
        
        # ---------- Pipeline B: 产品推荐 ----------
        st.markdown("---")
        st.header("📦 Personalized Recommendations")
        st.markdown(f"Based on your **{species}**'s **{behavior}** behavioral state:")
        
        with st.spinner("💡 Generating personalized recommendations..."):
            start_time = time.time()
            recommendation = generate_recommendation(species, behavior, generator)
            rec_time = time.time() - start_time
        
        # 显示推荐结果
        st.success(recommendation)
        
        with st.expander("ℹ️ Technical details"):
            st.markdown(f"""
            - **Pipeline A** (Behavior Classification): Fine-tuned ViT model
            - **Pipeline B** (Recommendation): Flan-T5-small with prompt engineering
            - **Species detected as**: {species}
            - **Behavior classified as**: {behavior} ({confidence*100:.1f}%)
            - **Classification time**: {classify_time:.2f}s
            - **Recommendation time**: {rec_time:.2f}s
            """)
        
        # ---------- Disclaimer ----------
        st.caption(
            "⚠️ This tool provides behavioral observations only and does not "
            "constitute veterinary advice. Please consult a professional veterinarian "
            "for health concerns."
        )


# ============================
# 入口
# ============================
if __name__ == "__main__":
    main()
