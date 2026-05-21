from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import re
import emoji
from underthesea import word_tokenize
from datetime import datetime


app = FastAPI(title="Threads Viral Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = 'lgbm_viral_prediction_pipeline.pkl'

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Load model error: {e}")
    model_pipeline = None



class PostRequest(BaseModel):
    text: str
    media_count: int
    scheduled_time: str
    is_reply: int
    is_repost: int


# ==============================
# 4. TEXT PROCESSING
# ==============================
vietnamese_stopwords = set([
    'là','và','của','thì','mà','trong','cho','với','những','các',
    'một','có','để','này','đó','ở','được','từ','cũng','đã','khi',
    'ra','về','như','lại','rất','hơn','nhiều','cái','hay','thế',
    'em','ạ','mình','không','làm','người','nhưng','nha','nào',
    'nữa','thôi','đi','vậy','rồi','chưa','các_bạn','mọi_người',
    'ko','k','khong','dc','đc','vs','nhé','ơi','à','ừ','nhỉ','lun',
    'rùi','kia','rứa','răng','tớ','cậu','nó','hắn','thể','ai','gì',
    'sao','bao_nhiêu','bây_giờ','năm','tháng','ngày','nên','vì','bởi',
    'nếu','tuy','tại','bị','sẽ','đang','vẫn','cứ','vào'
])


def clean_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s#]', ' ', text)  # ⚠️ giữ lại hashtag
    text = re.sub(r'\s+', ' ', text).strip()

    text = word_tokenize(text, format="text")

    words = text.split()
    words = [w for w in words if w not in vietnamese_stopwords and len(w) > 1]

    return ' '.join(words)


def get_time_bucket(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 23:
        return 'evening'
    else:
        return 'night'


# ==============================
# 5. MAIN API
# ==============================
@app.post("/api/predict")
async def predict_viral(req: PostRequest):

    if model_pipeline is None:
        return {"error": "Model not loaded"}

    # ========= TEXT =========
    text_clean_val = clean_text(req.text)

    # ⚠️ đếm hashtag chuẩn hơn
    hashtag_count_val = len(re.findall(r'#\w+', req.text))

    text_length = len(text_clean_val.split())

    # ========= MEDIA =========
    has_media_val = 1 if req.media_count > 0 else 0

    # ========= TIME =========
    try:
        dt = datetime.fromisoformat(req.scheduled_time.replace("Z", ""))
    except:
        return {"error": "Invalid datetime format"}

    post_hour = dt.hour
    post_day_of_week = dt.weekday()
    year = dt.year
    month = dt.month
    day = dt.day

    is_weekend = 1 if post_day_of_week >= 5 else 0
    time_bucket = get_time_bucket(post_hour)

    # ========= DATAFRAME =========
    input_df = pd.DataFrame([{
        "text_clean": text_clean_val,
        "media_count": req.media_count,
        "hashtag_count": hashtag_count_val,
        "post_hour": post_hour,
        "post_day_of_week": post_day_of_week,
        "year": year,
        "month": month,
        "day": day,
        "is_reply": req.is_reply,
        "is_repost": req.is_repost,
        "has_media": has_media_val,
        "is_weekend": is_weekend,
        "time_bucket": time_bucket
    }])

    # ========= MODEL =========
    probability = model_pipeline.predict_proba(input_df)[0][1]

    # ==============================
    # 🚀 RULE-BASED FIX (DEMO SAFE)
    # ==============================

    # Rule 1: Text quá ngắn
    if text_length < 3:
        probability *= 0.3

    # Rule 2: Có yếu tố cảm xúc / storytelling
    viral_keywords = ['mình', 'tôi', 'có ai', 'đã từng', 'vượt qua', 'câu hỏi']

    if any(k in req.text.lower() for k in viral_keywords):
        probability *= 1.3

    # Rule 3: Có hashtag
    if hashtag_count_val > 0:
        probability *= 1.1

    # Rule 4: Giờ xấu
    if 0 <= post_hour < 5:
        probability *= 0.5

    # Clamp
    probability = max(0, min(probability, 1))

    prediction = 1 if probability >= 0.5 else 0

    # ==============================
    # 🎯 GỢI Ý (BONUS)
    # ==============================
    tips = []

    if text_length < 5:
        tips.append("👉 Nội dung hơi ngắn, nên viết chi tiết hơn")

    if hashtag_count_val == 0:
        tips.append("👉 Nên thêm hashtag để tăng reach")

    if post_hour < 6 or post_hour > 23:
        tips.append("👉 Nên đăng vào buổi tối (18h–22h)")

    if not has_media_val:
        tips.append("👉 Thêm ảnh/video sẽ tăng tương tác")

    return {
        "is_viral": int(prediction),
        "viral_probability": round(probability * 100, 2),
        "message": "🔥 Bùng nổ!" if prediction == 1 else "👀 Bình thường",
        "tips": tips
    }


# ==============================
# 6. RUN SERVER
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)