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
# Безопасное получение ключа (теперь он не утечет на GitHub)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Ключ API не найден! Добавьте его в .streamlit/secrets.toml или в настройки Streamlit Cloud.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Настройка шрифтов (Глобально, чтобы не было NameError)
FONT_NAME = 'Helvetica'
try:
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'times.ttf'))
    FONT_NAME = 'TimesNewRoman'
except:
    pass

# --- Конфигурация страницы ---
st.set_page_config(page_title="LexEntropy | AI Анализ", layout="wide")
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} 
    .stButton>button[kind="primary"] {background-color: #5b45ff; color: white; border-radius: 8px;}
    .expert-box {background-color: #2e2b70; color: white; padding: 20px; border-radius: 12px; margin-top: 15px;}
</style>
""", unsafe_allow_html=True)

# --- Логика языков ---
languages = {'Русский': 'ru', 'Қазақша': 'kk', 'English': 'en'}
selected_lang_name = st.sidebar.selectbox("🌐 Язык", list(languages.keys()))
lang_code = languages[selected_lang_name]

# ИСПРАВЛЕННЫЙ СЛОВАРЬ (Добавлены ключи для скачивания PDF)
ui = {
    'ru': {
        "title": "Ввод Документов",
        "btn": "✨ Запустить Анализ",
        "loading": "AI готовит базу...",
        "report": "Отчет об энтропии",
        "upload_label": "Загрузить файл (PDF, DOCX, TXT, JPG, PNG)",
        "query_label": "Точечный запрос (необязательно):",
        "download_pdf": "📥 Скачать PDF-отчет"
    },
    'kk': {
        "title": "Құжаттарды енгізу",
        "btn": "✨ Талдауды бастау",
        "loading": "AI базаны дайындауда...",
        "report": "Талдау есебі",
        "upload_label": "Файлды жүктеу (PDF, DOCX, TXT, JPG, PNG)",
        "query_label": "Нақты сұраныс (міндетті емес):",
        "download_pdf": "📥 PDF есепті жүктеп алу"
    },
    'en': {
        "title": "Document Input",
        "btn": "✨ Run Analysis",
        "loading": "AI is preparing database...",
        "report": "Analysis Report",
        "upload_label": "Upload file (PDF, DOCX, TXT, JPG, PNG)",
        "query_label": "Specific query (optional):",
        "download_pdf": "📥 Download PDF Report"
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


def get_text_from_adilet(url):
    resp = requests.get(url, verify=False, timeout=15)
    return ' '.join([p.text for p in BeautifulSoup(resp.content, 'html.parser').find_all('p')])


# --- ГЛАВНАЯ ЛОГИКА АНАЛИЗА ---
def retrieve_and_analyze(doc_data, is_image, query, target_lang):
    model = genai.GenerativeModel('gemini-2.5-flash')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 1. Определение закона
    st.toast("🔍 Определяю закон...")
    prompt_id = "Определи официальное название закона РК для этого текста. Только название."
    law_name = model.generate_content(
        [prompt_id, doc_data] if is_image else prompt_id + f"\n\n{doc_data[:2000]}").text.strip()

    # 2. Поиск ссылки
    COMMON = {
        "трудовой кодекс": "https://adilet.zan.kz/rus/docs/K1500000414",
        "уголовно-процессуальный кодекс": "https://adilet.zan.kz/rus/docs/K1400000231",
        "национальной безопасности": "https://adilet.zan.kz/rus/docs/Z1200000527"
    }
    url = next((v for k, v in COMMON.items() if k in law_name.lower()), None)
    if not url:
        with DDGS() as ddgs:
            res = list(ddgs.text(f"site:adilet.zan.kz/rus/docs {law_name}", max_results=1))
            if res: url = res[0]['href']

    if not url: return f"❌ Закон '{law_name}' не найден."

    # 3. Векторизация с защитой от 429
    st.toast("📥 Загрузка эталона (с защитой от лимитов)...")
    try:
        text = get_text_from_adilet(url)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)[:60]
        vdb = None
        for i in range(0, len(chunks), 15):
            batch = chunks[i:i + 15]
            if vdb is None:
                vdb = FAISS.from_texts(batch, embeddings)
            else:
                vdb.add_texts(batch)
            time.sleep(3)
    except Exception as e:
        return f"❌ Ошибка векторизации: {e}"

    # 4. Анализ
    st.toast("⚖️ Поиск противоречий...")
    context = "\n\n".join([d.page_content for d in vdb.similarity_search(str(query) if query else "коллизии", k=3)])
    final_prompt = f"Ты ИИ-аналитик. Найди 1 противоречие в документе с законом {law_name}.\nЯзык: {target_lang}.\nКонтекст: {context}"

    return model.generate_content(
        [final_prompt, doc_data] if is_image else final_prompt + f"\n\nДОК: {doc_data[:4000]}").text


# --- ИСПРАВЛЕННАЯ ГЕНЕРАЦИЯ PDF (Paragraph + Logo) ---
def make_pdf(text, lang_code_pdf):
    buf = io.BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Отрисовка логотипа
    if os.path.exists('logo.png'):
        try:
            p.drawImage('logo.png', 1 * cm, height - 2 * cm, width=1.2 * cm, height=1.2 * cm, mask='auto')
        except:
            pass

    # Шапка документа
    p.setFont(FONT_NAME, 9)
    p.setFillColorRGB(0.5, 0.5, 0.5)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p.drawString(2.5 * cm, height - 1.25 * cm, f"LexEntropy Report | Date: {date_str} | Lang: {lang_code_pdf.upper()}")
    p.line(1 * cm, height - 2.2 * cm, width - 1 * cm, height - 2.2 * cm)

    # Настройка стиля для переноса текста
    custom_style = ParagraphStyle(
        'CustomStyle',
        fontName=FONT_NAME,
        fontSize=11,
        leading=16,
        textColor='black'
    )

    y_position = height - 3 * cm
    margin_x = 1.5 * cm
    usable_width = width - 2 * margin_x

    # Отрисовка текста по абзацам
    for line in text.split('\n'):
        if not line.strip():
            y_position -= 10
            continue

        clean_line = line.replace('**', '').replace('>', 'ЦИТАТА:').replace('<', '&lt;').replace('>', '&gt;').strip()
        para = Paragraph(clean_line, custom_style)
        w, h = para.wrap(usable_width, height)

        if y_position - h < 1.5 * cm:
            p.showPage()
            p.setFont(FONT_NAME, 9)
            p.setFillColorRGB(0.5, 0.5, 0.5)
            p.drawString(width - 3 * cm, height - 1 * cm, "LexEntropy")
            p.line(1 * cm, height - 1.5 * cm, width - 1 * cm, height - 1.5 * cm)
            y_position = height - 2.5 * cm

        para.drawOn(p, margin_x, y_position - h)
        y_position -= (h + 8)

    p.save()
    buf.seek(0)
    return buf


# --- Интерфейс Streamlit ---
st.title("📚 LexEntropy")
with st.sidebar:
    # Отрисовка логотипа на сайте
    if os.path.exists('logo.png'):
        try:
            st.image('logo.png', use_container_width=True)
        except:
            pass
    st.markdown("<div class='expert-box'><b>RAG Агент активен</b><br>v3.8-stable</div>", unsafe_allow_html=True)

up = st.file_uploader(_["upload_label"], type=["pdf", "docx", "png", "jpg", "txt"])
q = st.text_input(_["query_label"])

if st.button(_["btn"], type="primary") and up:
    with st.status(_["loading"]) as status:
        data = extract_text(up)
        res = retrieve_and_analyze(data, isinstance(data, Image.Image), q, selected_lang_name)
        status.update(label="Готово!", state="complete")

    st.markdown("### " + _["report"])
    st.markdown(res)
    st.download_button(_["download_pdf"], make_pdf(res, lang_code),
                       f"LexEntropy_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
