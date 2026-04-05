import streamlit as st
import google.generativeai as genai
import graphviz
import PyPDF2
import docx
from PIL import Image
import os
import datetime
import io
import time
import requests
import urllib3
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Импорты для генерации PDF ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

# Отключаем предупреждения SSL
urllib3.disable_warnings()

# === ГЛОБАЛЬНЫЕ НАСТРОЙКИ ===
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Ключ API не найден! Добавьте его в настройки Streamlit Cloud.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Настройка шрифтов
FONT_NAME = 'Helvetica'
try:
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'times.ttf'))
    FONT_NAME = 'TimesNewRoman'
except:
    pass

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="LexEntropy | AI Анализ", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Логика языков ---
languages = {'Русский': 'ru', 'Қазақша': 'kk', 'English': 'en'}
selected_lang_name = st.sidebar.selectbox("🌐 Язык интерфейса", list(languages.keys()))
lang_code = languages[selected_lang_name]

ui = {
    'ru': {
        "title": "Ввод Документов", "btn": "✨ Запустить Анализ", "loading": "AI парсит Adilet.zan.kz...",
        "report": "Отчет об энтропии", "upload_label": "Загрузить файл", "query_label": "Точечный запрос:", "download_pdf": "📥 Скачать PDF"
    },
    'kk': {
        "title": "Құжаттарды енгізу", "btn": "✨ Талдауды бастау", "loading": "AI Adilet.zan.kz өңдеуде...",
        "report": "Талдау есебі", "upload_label": "Файлды жүктеу", "query_label": "Нақты сұраныс:", "download_pdf": "📥 PDF жүктеу"
    },
    'en': {
        "title": "Document Input", "btn": "✨ Run Analysis", "loading": "AI is parsing Adilet.zan.kz...",
        "report": "Analysis Report", "upload_label": "Upload file", "query_label": "Query:", "download_pdf": "📥 Download PDF"
    }
}
_ = ui[lang_code]

# --- Функции извлечения текста ---
def extract_text(file):
    ext = file.name.lower().split('.')[-1]
    if ext == 'pdf':
        return "".join([p.extract_text() for p in PyPDF2.PdfReader(file).pages])
    elif ext == 'docx':
        return "\n".join([p.text for p in docx.Document(file).paragraphs])
    elif ext in ['png', 'jpg', 'jpeg']:
        return Image.open(file)
    else:
        return file.read().decode("utf-8")

def get_any_text_from_adilet(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, verify=False, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    # Собираем текст из всех основных блоков Adilet (p, div с текстом статей)
    content_blocks = soup.find_all(['p', 'div'], class_=['article', 'text', 'content'])
    if not content_blocks:
        content_blocks = soup.find_all('p')
    return ' '.join([b.text for b in content_blocks])

# --- ГЛАВНАЯ ЛОГИКА АНАЛИЗА ---
def retrieve_and_analyze(doc_data, is_image, query, target_lang):
    model = genai.GenerativeModel('gemini-2.5-flash')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 1. Определение закона и его языка
    st.toast("🔍 Распознаю документ и язык...")
    id_prompt = """Проанализируй текст и выведи: 
    1. Официальное название документа.
    2. Язык документа (напиши только 'rus' или 'kaz').
    Формат ответа: Название | язык"""
    
    id_res = model.generate_content(
        [id_prompt, doc_data] if is_image else id_prompt + f"\n\n{str(doc_data)[:2000]}").text.strip()
    
    try:
        law_name, law_lang = [x.strip() for x in id_res.split('|')]
    except:
        law_name, law_lang = id_res, "rus"

    # 2. Универсальный поиск по Adilet.zan.kz
    st.toast(f"🌐 Ищу '{law_name}' на Adilet...")
    # Формируем запрос строго под нужный язык портала
    lang_path = "rus" if "rus" in law_lang.lower() else "kaz"
    search_query = f"site:adilet.zan.kz/{lang_path}/docs {law_name}"
    
    url = None
    try:
        with DDGS() as ddgs:
            res = list(ddgs.text(search_query, max_results=1))
            if res: url = res[0]['href']
    except: pass

    if not url: return f"❌ Документ '{law_name}' не найден на портале Adilet.zan.kz."

    # 3. Парсинг и векторизация
    st.toast("📥 Загружаю эталонный текст...")
    try:
        full_legal_text = get_any_text_from_adilet(url)
        # Разбиваем на куски побольше для точности
        chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_text(full_legal_text)[:50]
        vdb = FAISS.from_texts(chunks[:15], embeddings)
        if len(chunks) > 15:
            time.sleep(2) # Защита от лимитов Google API
            vdb.add_texts(chunks[15:40])
    except Exception as e:
        return f"❌ Ошибка загрузки базы: {e}"

    # 4. Глубокий аудит коллизий
    st.toast("⚖️ Сверяю каждую букву...")
    search_context = str(query) if query else str(doc_data)[:1000]
    relevant_docs = vdb.similarity_search(search_context, k=5)
    context_text = "\n\n".join([d.page_content for d in relevant_docs])
    
    audit_prompt = f"""Ты - эксперт-юрист. Сравни ЗАГРУЖЕННЫЙ ДОКУМЕНТ с ОФИЦИАЛЬНЫМ ТЕКСТОМ из Adilet.zan.kz.
    Найди любые расхождения: измененные сроки, суммы, права или обязанности. 
    Если загруженный текст содержит ошибки по сравнению с оригиналом - укажи их четко.
    Язык ответа: {target_lang}.
    ОРИГИНАЛ ИЗ ADILET: {context_text}"""

    return model.generate_content(
        [audit_prompt, doc_data] if is_image else audit_prompt + f"\n\nЗАГРУЖЕННЫЙ ТЕКСТ:\n{str(doc_data)[:15000]}"
    ).text

# --- ГЕНЕРАЦИЯ PDF ---
def make_pdf(text, lang_code_pdf):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    if os.path.exists('logo.png'):
        try: p.drawImage('logo.png', 1*cm, height-2*cm, width=1.2*cm, height=1.2*cm, mask='auto')
        except: pass
    p.setFont(FONT_NAME, 9)
    p.setFillColorRGB(0.5, 0.5, 0.5)
    p.drawString(2.5*cm, height-1.25*cm, f"LexEntropy Official Audit | {datetime.datetime.now().strftime('%Y-%m-%d')}")
    p.line(1*cm, height-2.2*cm, width-1*cm, height-2.2*cm)
    
    style = ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=11, leading=15)
    y = height - 3*cm
    for line in text.split('\n'):
        if not line.strip(): y -= 10; continue
        clean = line.replace('**', '').replace('>', '').replace('<', '&lt;').strip()
        para = Paragraph(clean, style)
        w, h = para.wrap(width-3*cm, height)
        if y - h < 2*cm: p.showPage(); y = height - 2*cm
        para.drawOn(p, 1.5*cm, y-h)
        y -= (h + 6)
    p.save()
    buf.seek(0)
    return buf

# --- Интерфейс ---
st.title("📚 LexEntropy")
with st.sidebar:
    if os.path.exists('logo.png'): st.image('logo.png', use_container_width=True)
    st.info("Dynamic RAG Engine: Linked to Adilet.zan.kz")

up = st.file_uploader(_["upload_label"], type=["pdf", "docx", "png", "jpg", "txt"])
q = st.text_input(_["query_label"])

if st.button(_["btn"], type="primary") and up:
    with st.status(_["loading"]) as status:
        data = extract_text(up)
        res = retrieve_and_analyze(data, isinstance(data, Image.Image), q, selected_lang_name)
        status.update(label="Анализ готов!", state="complete")
    st.markdown("### " + _["report"])
    st.markdown(res)
    st.download_button(_["download_pdf"], make_pdf(res, lang_code), f"Audit_{datetime.datetime.now().strftime('%H%M')}.pdf")
