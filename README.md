# 🚀 دليل تشغيل المشروع — خطوة بخطوة

---

## 🗺️ خريطة الملفات الأساسية

```
project/
│
├── manage.py                          ← نقطة تحكم Django
├── config/
│   ├── settings.py                    ← إعدادات كاملة (DB, Logging, MLflow)
│   └── urls.py                        ← كل الـ routes تحت /api/
│
├── recommender/                       ← تطبيق Django الرئيسي
│   ├── models.py                      ← Item, UserInteraction, UserBrowsingLog, SearchLog
│   ├── views.py                       ← dashboard, product_list, search, recommendations_api
│   ├── urls.py                        ← تعريف كل الصفحات والـ APIs
│   ├── services/
│   │   ├── item_service.py            ← get_items_by_ids, get_popular_items
│   │   └── interaction_service.py     ← log_interaction (ratings) → يكتب في interactions.log
│   ├── tracking/
│   │   └── browsing_tracker.py        ← log_event, log_search_event → يكتب في interactions.log
│   ├── api_helpers/
│   │   └── track_api.py               ← AJAX endpoints: /api/track/, /api/rate/
│   └── management/commands/
│       ├── import_data.py             ← يحشي DB بالـ items والـ interactions
│       ├── preprocess.py              ← يبني الـ matrices للـ ML
│       └── retrain.py                 ← يعيد تدريب كل الـ models + MLflow
│
├── ml/
│   ├── inference/
│   │   ├── predict.py                 ← SVDPredictor, ALSPredictor, ContentBasedPredictor, HybridPredictor
│   │   ├── recommender.py             ← Singleton: get_recommendations(), reload_predictor()
│   │   └── loader.py                  ← get_items_by_ids من DB
│   ├── preprocessing/
│   │   ├── global_encoders.py         ← يبني user_encoder.pkl + item_encoder.pkl
│   │   ├── explicit_transform.py      ← ينظف ratings وينورملايز
│   │   ├── explicit_interaction_matrix.py  ← يبني explicit_matrix.npz
│   │   └── implicit_interaction_matrix.py  ← يبني implicit_matrix.npz (browsing)
│   ├── training/
│   │   ├── train.py                   ← يدرب SVD + ALS + ContentBased
│   │   ├── evaluate.py                ← Precision/Recall/MAP/NDCG @5,10,20
│   │   └── tuning.py                  ← Grid search لأفضل params
│   └── pipelines/
│       ├── preprocess_pipeline.py     ← orchestrator للـ preprocessing
│       └── retrain_pipeline.py        ← orchestrator كامل (preprocess+train+eval+MLflow)
│
├── data/
│   ├── raw/                           ← ملفات Amazon الأصلية (gitignored)
│   ├── processed/                     ← parquet files بعد التنظيف
│   ├── artifacts/                     ← user_encoder.pkl, item_encoder.pkl, matrices
│   ├── models/                        ← svd_model.pkl, als_model.pkl, cb_*.pkl
│   └── reports/                       ← evaluation_report.json, retrain_meta.json
│
├── logs/
│   ├── interactions.log               ← كل click/view/search/rating من المستخدمين
│   ├── ml.log                         ← training, evaluation, pipeline logs
│   └── app.log                        ← Django app logs عامة
│
└── mlruns/                            ← MLflow experiments (يتنشأ أول مرة تعمل retrain)
```

---

## 📦 الخطوة 0 — تثبيت الـ Packages

```bash
pip install django numpy pandas scipy scikit-learn implicit mlflow openpyxl pyarrow
```

| Package      | بيستخدمه إيه                              |
|--------------|-------------------------------------------|
| django       | الـ framework الرئيسي                     |
| numpy/scipy  | الـ matrices والـ sparse arrays           |
| pandas       | قراءة الـ parquet/xlsx وتحويل البيانات    |
| scikit-learn | SVD, TF-IDF, LabelEncoder, metrics        |
| implicit     | ALS collaborative filtering               |
| mlflow       | تتبع experiments التدريب                  |
| openpyxl     | قراءة ملفات xlsx                          |
| pyarrow      | قراءة وكتابة ملفات parquet               |

---

## 🗄️ الخطوة 1 — إعداد قاعدة البيانات

```bash
cd <project_root>

# إنشاء جداول DB
python manage.py migrate
```

**الجداول اللي بتتنشأ:**
- `recommender_item` — المنتجات
- `recommender_userinteraction` — ratings/reviews
- `recommender_userbrowsinglog` — browsing events
- `recommender_searchlog` — عمليات البحث

---

## 📥 الخطوة 2 — استيراد البيانات في DB

```bash
# الملفات الموجودة (items.xlsx, interactions.xlsx) — بيانات تجريبية صغيرة
python manage.py import_data \
  --items-path data/processed/items.xlsx \
  --interactions-path data/processed/interactions.xlsx

# تحقق إن البيانات اتحملت
python manage.py shell -c "
from recommender.models import Item, UserInteraction
print('Items:', Item.objects.count())
print('Interactions:', UserInteraction.objects.count())
"
```

> **ملحوظة:** لو عندك `items.parquet` و`interactions.parquet` في `data/processed/`
> شغّل بدون arguments:
> ```bash
> python manage.py import_data
> ```

---

## 🧮 الخطوة 3 — بناء الـ ML Matrices

> ⚠️ **هذه الخطوة مش ضرورية لو ملفات `data/artifacts/` و`data/models/` موجودة بالفعل**
> (الـ zip اللي بعتهولك فيه كل الملفات دي جاهزة)

```bash
# بناء كل الـ matrices من أول وجديد
python manage.py preprocess

# أو خطوة بخطوة
python manage.py preprocess --step encoders
python manage.py preprocess --step explicit
python manage.py preprocess --step implicit
```

**الملفات اللي بتتنشأ:**
```
data/artifacts/user_encoder.pkl          ← يحول user_id لـ index
data/artifacts/item_encoder.pkl          ← يحول item_id لـ index
data/artifacts/explicit_matrix.npz      ← sparse matrix للـ ratings
data/artifacts/implicit_matrix.npz      ← sparse matrix للـ browsing
```

---

## 🌐 الخطوة 4 — تشغيل الـ Server

```bash
python manage.py runserver
```

### 🔗 روابط المشروع (كلها تحت /api/)

| الصفحة            | الرابط                                       |
|-------------------|----------------------------------------------|
| الرئيسية         | http://127.0.0.1:8000/api/                   |
| المنتجات          | http://127.0.0.1:8000/api/products/          |
| منتج محدد         | http://127.0.0.1:8000/api/products/<item_id>/|
| البحث             | http://127.0.0.1:8000/api/search/            |
| Dashboard         | http://127.0.0.1:8000/api/dashboard/         |
| Register          | http://127.0.0.1:8000/api/register/          |
| Login             | http://127.0.0.1:8000/api/login/             |
| Admin             | http://127.0.0.1:8000/admin/                 |

---

## 🧪 الخطوة 5 — اختبار الـ Recommendations

### A. عبر الـ API مباشرةً

```
http://127.0.0.1:8000/api/recommendations/?user_id=1
```

**Response ناجح:**
```json
{
  "user_id": "1",
  "count": 6,
  "recommendations": [
    {
      "item_id": "B001XK5L6O",
      "title": "...",
      "category": "Electronics",
      "avg_rating": 4.5,
      "price": 29.99,
      "image": "https://m.media-amazon.com/..."
    }
  ]
}
```

### B. عبر Django Shell

```bash
python manage.py shell -c "
from ml.inference.recommender import get_recommendations
recs = get_recommendations(user_id='1', user_item_ids=[], k=5)
print('Recommended IDs:', recs)
"
```

### C. Dashboard

1. روح على http://127.0.0.1:8000/api/register/ واعمل حساب
2. افتح 3-4 منتجات مختلفة
3. روح على http://127.0.0.1:8000/api/dashboard/
4. شوف قسم "Personalized Recommendations"

---

## 🔄 الخطوة 6 — إعادة التدريب (Retrain)

```bash
# شوف هتعمل إيه من غير ما تنفذ
python manage.py retrain --dry-run

# إعادة التدريب الكاملة (بيستخدم الـ matrices الموجودة)
python manage.py retrain --skip-preprocess

# إعادة التدريب الكاملة مع إعادة بناء الـ matrices
python manage.py retrain

# إعادة التدريب مع Grid Search للـ hyperparameters
python manage.py retrain --retune
```

**اللي بيحصل في كل خطوة:**

```
Step 1  → يحدّث user/item encoders
Step 2  → يعيد transform الـ explicit ratings
Step 3  → يعيد بناء explicit sparse matrix
Step 4  → يعيد بناء implicit matrix (browsing logs)
Step 5  → يدرب SVD + ALS + ContentBased
Step 6  → يحسب Precision/Recall/MAP/NDCG ويقارن بالـ run القديمة
Step 7  → يسجل كل شيء في MLflow
Step 8  → يعمل hot-reload للـ predictor (من غير restart للـ server)
```

---

## 📊 الخطوة 7 — MLflow Dashboard

بعد أول `retrain`:

```bash
mlflow ui --backend-store-uri mlruns/ --port 5000
```

روح على: **http://127.0.0.1:5000**

هتلاقي:
- كل الـ runs مع التواريخ
- مقارنة الـ params (n_factors, als_iterations...)
- مقارنة الـ metrics (NDCG@10 قبل وبعد)
- الـ evaluation report كـ artifact قابل للتحميل

---

## 📋 الخطوة 8 — مراقبة الـ Logs

افتح terminal منفصل وشغّل:

```bash
# كل تفاعلات المستخدمين (views, clicks, ratings, searches)
tail -f logs/interactions.log

# ML logs (training, inference, pipelines)
tail -f logs/ml.log

# Application logs (Django عامة)
tail -f logs/app.log
```

**شكل interactions.log:**
```
2025-05-16 14:23:01 | event=view                 user=3          item=B001XK5L6O   source=recommendation   device=desktop
2025-05-16 14:23:45 | event=search               user=3          query='wireless headphones' results=12
2025-05-16 14:24:10 | event=explicit_rating       user=3          item=B001XK5L6O   rating=4.0 verified=True
2025-05-16 14:25:00 | event=add_to_cart           user=3          item=B002GEN8I4   source=recommendation
```

---

## ✅ Checklist — كيف تعرف إن كل حاجة شغالة؟

| الاختبار | ✅ نجح لو |
|----------|-----------|
| `python manage.py check` | مفيش errors |
| `python manage.py migrate` | `No migrations to apply` |
| `Item.objects.count() > 0` | في Django shell |
| `/api/recommendations/?user_id=1` | `count > 0` في الـ JSON |
| Dashboard بعد تسجيل دخول | بيعرض recommendations مش فاضية |
| `python manage.py retrain --dry-run` | بيطبع الـ 8 steps |
| `logs/interactions.log` | بيتكتب فيه بعد أي click |
| `mlflow ui` | بيفتح ويعرض experiments |

---

## 🐛 مشاكل شائعة وحلها

| المشكلة | السبب | الحل |
|---------|-------|------|
| `No module named 'implicit'` | package مش متثبت | `pip install implicit` |
| Recommendations فاضية | DB فاضي | `python manage.py import_data` |
| `data/models/*.pkl not found` | models مش موجودة | `python manage.py retrain --skip-preprocess` |
| `404` على `/` | الـ routes تحت `/api/` | روح على `http://127.0.0.1:8000/api/` |
| `interactions.log` فاضي | مفيش events لسه | افتح أي منتج وهيتكتب تلقائياً |
| MLflow مش بيفتح | لسه ما اتعملش retrain | شغّل `python manage.py retrain` الأول |
