# =============================================================================
# ISOM5240 Group Project - Streamlit App
# Chewy Pet Behavior Detection & Personalized Recommendation
# =============================================================================
# 
# Deploy: Streamlit Cloud (Link GitHub Repo)
# Pipeline:
#   Pipeline A: image-classification (ViT, fine-tuned)
#   Pipeline B: text2text-generation (Flan-T5, prompt engineering)
# =============================================================================

import streamlit as st
from transformers import pipeline
from PIL import Image
import time

# ============================
# Page Config (必须放在最前面)
# ============================
st.set_page_config(
    page_title="Chewy Pet Behavior AI",
    page_icon="🐾",
    layout="wide",
)

# ============================
# 模型加载 (缓存, 避免重复加载)
# ============================
# ⚠️ 老师要求: 用 @st.cache_resource 缓存 pipeline

@st.cache_resource
def load_behavior_classifier():
    """加载 Pipeline A: 宠物行为分类模型 (微调后的 ViT)"""
    # ⚠️ 替换为你的 HF 模型仓库名
    return pipeline(
        "image-classification",
        model="YOUR_USERNAME/chewy-pet-behavior-vit",
    )

@st.cache_resource
def load_recommendation_generator():
    """加载 Pipeline B: 产品推荐生成模型 (Flan-T5)"""
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=200,
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
