import pandas as pd
import gradio as gr
import random
import re
from rapidfuzz import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


data_path = r"C:\Users\DEEPIKA\Downloads\tamil_therapy_full350.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig").fillna("")
df.columns = df.columns.str.strip()

for col in ["input", "intent", "response"]:
    df[col] = df[col].str.strip()


tanglish_map = {
    "stress": "ஸ்டிரெஸ்", "stressed": "ஸ்டிரெஸ்", "sad": "வருத்தம்", "happy": "சந்தோஷம்",
    "lonely": "தனிமை", "hopeless": "நம்பிக்கையின்மை", "nervous": "பயந்தேன்", "angry": "கோபம்",
    "tired": "சோர்வு", "confused": "குழப்பம்", "excited": "உற்சாகம்", "bored": "பழுத்தது",
    "fear": "பயம்", "anxiety": "கவலை", "depressed": "மனச்சோர்ச்சி", "calm": "அமைதி",
    "depression": "மனச்சோர்ச்சி", "panic": "பயந்தேன்", "relaxed": "அமைதி"
}


def normalize_input(text):
    text = str(text).lower().strip()
    text = re.sub(r'([a-z])\1{2,}', r'\1', text)  
    text = re.sub(r'[^\w\s]', '', text)           
    text = re.sub(r'\s+', ' ', text)              
    words = text.split()
    mapped = [tanglish_map.get(w, w) for w in words]
    return " ".join(mapped)

df["norm_input"] = df["input"].apply(normalize_input)


response_counts = df["response"].value_counts()
df = df[df["response"].isin(response_counts[response_counts > 1].index)]


dataset_map = {row["norm_input"]: row["response"] for _, row in df.iterrows() if row["norm_input"].strip()}


X_train, X_test, y_train, y_test = train_test_split(
    df["norm_input"], df["response"], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\S+\b",
    ngram_range=(1,3),    
    sublinear_tf=True,
    min_df=1,
    max_df=0.95
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


clf = LinearSVC(class_weight='balanced')
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)
accuracy_percent = accuracy_score(y_test, y_pred) * 100
print(f" Model Accuracy: {accuracy_percent:.2f}%")


def chatbot(input, history=[]):
    norm_input = normalize_input(input)
    match = process.extractOne(norm_input, dataset_map.keys(), scorer=fuzz.token_sort_ratio)

    if match and match[1] >= 75:
        response = dataset_map[match[0]]
    else:
        try:
            user_vec = vectorizer.transform([norm_input])
            response = clf.predict(user_vec)[0]
        except:
            response = "மன்னிக்கவும், புரியவில்லை 😔. வேறு வார்த்தைகளில் முயற்சிக்கவும்."

    history.append((input, response))
    return history, history


def launch_chatbot():
    greeting = random.choice([
        f"வணக்கம்! 😊 உங்கள் உணர்ச்சியை பகிருங்கள் 🌸",
        f"வணக்கம்! 😊 இன்று எப்படி உணர்கிறீர்கள்?"
    ])
    return [("Bot", greeting)]

with gr.Blocks() as demo:
    gr.Markdown("##  Tamil Therapy Chatbot \nதமிழ், Tanglish -இல் பேசலாம் ")
    chatbot_ui = gr.Chatbot(value=launch_chatbot(), height=420)
    msg = gr.Textbox(placeholder="உங்கள் உணர்ச்சியை இங்கே எழுதுங்கள்...", label="Talk to Therapy Bot")
    clear = gr.Button("Clear Chat")
    msg.submit(chatbot, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    clear.click(lambda: launch_chatbot(), None, chatbot_ui, queue=False)

demo.launch()
