import pickle
import re
import string
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "detik_berita.csv"
MODEL_PATH = BASE / "model_skipgram_b.pkl"

PARAM_SKIPGRAM = dict(
    vector_size=100,
    window=5,
    min_count=1,
    negative=5,
    sample=0,
    epochs=50,
    sg=1,
    seed=42,
    workers=1,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}

KAMUS_TIDAK_BAKU = {
    "nggak": "tidak", "enggak": "tidak", "gak": "tidak", "kagak": "tidak",
    "udah": "sudah", "gitu": "begitu", "gini": "begini",
    "bikin": "membuat", "biarin": "biarkan", "cuma": "hanya",
    "aja": "saja", "tuh": "itu", "kayaknya": "sepertinya", "kayak": "seperti",
    "ngaco": "kacau", "banget": "sangat", "makin": "semakin",
    "dibilang": "dikatakan", "bilang": "berkata",
    "cuan": "untung", "karir": "karier",
    "ojol": "ojek", "detikers": "pembaca", "gede": "besar",
    "emang": "memang", "gimana": "bagaimana", "pakai": "memakai",
    "kalo": "kalau", "mesti": "harus", "dipindahin": "dipindahkan",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "spt": "seperti", "krn": "karena",
    "perkeonomian": "perekonomian", "pemeritah": "pemerintah", "menyetujul": "menyetujui",
    "sih": "", "kok": "", "deh": "", "dong": "", "lah": "", "toh": "", "loh": "", "mah": "",
    "kan": "", "nih": "",
}

KAMUS_ASING_SUBSTITUSI = {
    "import": "impor", "point": "poin", "tennis": "tenis",
    "elite": "elit", "standard": "standar",
}

KATA_ASING = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "do", "does", "did", "for", "from", "had", "has", "have", "he", "her", "his",
    "if", "in", "into", "is", "it", "its", "may", "might", "must", "not", "of", "on",
    "or", "she", "should", "so", "than", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "own", "same", "too", "only", "just", "because", "until", "now",
    "very", "about", "after", "before", "between", "over", "under", "again", "further",
    "then", "once", "here", "men", "him", "us", "com", "www",
    "games", "champions", "league", "world", "race", "run", "set", "team", "start",
    "beach", "sport", "sports", "event", "fc", "super", "top", "pre", "center",
    "international", "approved", "online", "offline", "business", "price", "pricing",
    "market", "marketplace", "asset", "leveraging", "buy", "buyback", "corporate",
    "group", "location", "single", "streamlining", "commerce", "national", "investment",
    "artificial", "intelligence", "existing", "cost", "training", "tax", "village",
    "fame", "harmony", "marketing", "host", "buying", "panic", "tower", "fraud",
    "mission", "sprint", "tour", "time", "step", "running", "runner", "fun",
}

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "pada", "itu", "ini", "juga", "untuk",
    "dengan", "akan", "adalah", "ialah", "oleh", "karena", "agar", "sebagai",
    "para", "dalam", "serta", "atau", "masih", "sudah", "belum", "tidak",
    "bukan", "bisa", "dapat", "lebih", "saat", "ketika", "setelah", "sebelum",
    "selama", "melalui", "antara", "hingga", "kemudian", "lantas", "merupakan",
    "namun", "tetapi", "tapi", "terdapat", "ada", "adanya", "yaitu", "yakni",
    "terkait", "misalnya", "maupun", "seperti", "jika", "kalau", "maka", "harus",
    "hendak", "sangat", "paling", "semua", "setiap", "beberapa", "saya", "anda",
    "kami", "kita", "mereka", "dia", "ia", "kepada", "bagi", "pun",
    "saja", "mungkin", "lagi", "tersebut", "adapun", "beserta",
}

RE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F"
    "\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]+"
)

TANDA_BACA = string.punctuation + "«»“”‘’–—…"
RE_TANDA_BACA = re.compile(f"[{re.escape(TANDA_BACA)}]")
RE_KATA = re.compile(r"[a-z\u00e0-\u00ff]+")


def _normalisasi_kata(m):
    w = m.group()
    if len(w) < 2 or w in KATA_ASING:
        return " "
    return KAMUS_ASING_SUBSTITUSI.get(w, KAMUS_TIDAK_BAKU.get(w, w))


def tokenisasi_b(teks):
    t = RE_EMOJI.sub(" ", teks.casefold())
    t = re.sub(r"[a-z\u00e0-\u00ff]+", _normalisasi_kata, t)
    t = RE_TANDA_BACA.sub(" ", t)
    token = [w for w in RE_KATA.findall(t) if len(w) > 1]
    return [w for w in token if w not in STOPWORDS_ID]


def muat_dataset():
    df = pd.read_csv(CSV_PATH, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["id"] = df["id"].str.strip()
    df["isi_berita"] = df["isi_berita"].fillna("")
    return df


def latih_model():
    df = muat_dataset()
    label = df["tema"].to_numpy()
    indeks = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indeks, test_size=0.20, random_state=42, stratify=label
    )
    idx_train = np.sort(idx_train)
    idx_test = np.sort(idx_test)

    token = [tokenisasi_b(t) for t in df["isi_berita"]]
    korpus_train = [token[i] for i in idx_train]

    model = Word2Vec(sentences=korpus_train, **PARAM_SKIPGRAM)
    mv = model.wv
    X = np.array([np.mean([mv[w] for w in t if w in mv], axis=0) for t in token])

    clf = GaussianNB().fit(X[idx_train], label[idx_train])
    metrik = {}
    for nama, idx in (("training", idx_train), ("testing", idx_test)):
        pred = clf.predict(X[idx])
        metrik[nama] = {
            "jumlah": int(len(idx)),
            "accuracy": float(accuracy_score(label[idx], pred)),
            "precision": float(precision_score(label[idx], pred, average="macro", zero_division=0)),
            "recall": float(recall_score(label[idx], pred, average="macro", zero_division=0)),
            "f1": float(f1_score(label[idx], pred, average="macro", zero_division=0)),
        }

    bundle = {
        "wv": mv,
        "clf": clf,
        "kelas": [str(k) for k in clf.classes_],
        "vocabulary": len(mv),
        "metrik": metrik,
        "jumlah_data": int(len(df)),
        "jumlah_token_training": int(sum(len(t) for t in korpus_train)),
        "param": PARAM_SKIPGRAM,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def muat_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return latih_model()


@st.cache_resource(show_spinner=False)
def model_tercached():
    return muat_model()


def ambil_teks_berita(html):
    teks = trafilatura.extract(html) or ""
    if len(teks) >= 200:
        return teks
    soup = BeautifulSoup(html, "html.parser")
    paragraf = [
        p.get_text(" ", strip=True)
        for p in soup.select(".detail__body-text p, article p, .detail_text p")
    ]
    paragraf = [p for p in paragraf if len(p) > 50]
    return " ".join(paragraf)


def crawl(url):
    host = urlparse(url).netloc.lower()
    if not (host == "detik.com" or host.endswith(".detik.com")):
        raise ValueError("Link harus dari situs detik.com, contoh: "
                         "https://finance.detik.com/... atau https://sport.detik.com/...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        raise ValueError(f"Gagal mengambil halaman: {e}") from e
    if r.status_code != 200:
        raise ValueError(f"Halaman tidak ditemukan (HTTP {r.status_code}). "
                         "Pastikan link artikelnya benar dan masih aktif.")
    soup = BeautifulSoup(r.text, "html.parser")
    judul = ""
    if soup.find("h1"):
        judul = soup.find("h1").get_text(" ", strip=True)
    if not judul:
        meta = trafilatura.extract_metadata(r.text)
        judul = meta.title if meta and meta.title else ""
    teks = ambil_teks_berita(r.text)
    if len(teks) < 100:
        raise ValueError("Isi berita gagal diekstrak dari halaman itu. "
                         "Coba link artikel detik.com lainnya.")
    return {"judul": judul, "teks": teks}


def prediksi_teks(teks, bundle):
    wv = bundle["wv"]
    clf = bundle["clf"]
    token = tokenisasi_b(teks)
    dikenal = [w for w in token if w in wv]
    oov = [w for w in token if w not in wv]
    if not dikenal:
        raise ValueError("Tidak ada satu pun kata di berita ini yang dikenal model. "
                         "Isinya mungkin terlalu pendek atau bukan bahasa Indonesia.")
    vektor = np.mean([wv[w] for w in dikenal], axis=0).reshape(1, -1)
    proba = clf.predict_proba(vektor)[0]
    hasil = dict(zip(bundle["kelas"], [float(p) for p in proba]))
    urut = sorted(hasil.items(), key=lambda t: -t[1])
    return {
        "label": urut[0][0],
        "keyakinan": urut[0][1],
        "proba": hasil,
        "jumlah_token": len(token),
        "token_dikenal": len(dikenal),
        "token_oov": len(oov),
        "contoh_token": token[:15],
        "contoh_oov": oov[:10],
    }


def prediksi_url(url, bundle):
    url = url.strip()
    if not url:
        raise ValueError("Link beritanya masih kosong.")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Link harus diawali http:// atau https://.")
    artikel = crawl(url)
    hasil = prediksi_teks(artikel["teks"], bundle)
    hasil.update({"url": url, "judul": artikel["judul"], "teks": artikel["teks"]})
    return hasil


def persen(x):
    return f"{x * 100:.1f}".replace(".", ",")


def halaman():
    st.set_page_config(
        page_title="Klasifikasi Berita Detik",
        layout="centered",
    )

    st.title("Klasifikasi Berita Detik")
    st.caption("Tempel link berita detik.com, lalu tekan tombol Klasifikasi. "
               "Model: Skip-Gram (jalur B) + Gaussian Naive Bayes.")

    with st.sidebar:
        st.header("Tentang model")
        try:
            bundle = model_tercached()
        except Exception as e:
            st.error(f"Model gagal dimuat: {e}")
            bundle = None
        if bundle:
            m = bundle["metrik"]["testing"]
            st.write(f"Vocabulary: **{bundle['vocabulary']}** kata")
            st.write(f"Data latih: **{bundle['jumlah_data']}** berita "
                     f"({bundle['jumlah_token_training']} token training)")
            st.write(f"Akurasi data testing: **{persen(m['accuracy'])}%**")
            st.write(f"F1-score testing: **{persen(m['f1'])}%**")
        st.markdown(
            "**Cara kerja**\n"
            "1. Ambil isi berita dari link (scraping)\n"
            "2. Preprocessing jalur B (case folding, hapus emoji, "
            "normalisasi kata tidak baku, hapus tanda baca, hapus stopword)\n"
            "3. Rata-rata vektor skip-gram 100 dimensi\n"
            "4. Gaussian Naive Bayes → label + persentase"
        )

    if bundle is None:
        st.stop()

    url = st.text_input(
        "Link berita detik.com",
        placeholder="https://finance.detik.com/berita-ekonomi-bisnis/d-xxxxxxx/...",
    )
    tombol = st.button("Klasifikasi", type="primary", use_container_width=True)

    if tombol or st.session_state.get("hasil"):
        if tombol:
            try:
                with st.spinner("Mengambil berita dan mengklasifikasi..."):
                    hasil = prediksi_url(url, bundle)
                st.session_state["hasil"] = hasil
            except ValueError as e:
                st.session_state.pop("hasil", None)
                st.error(str(e))
            except Exception as e:
                st.session_state.pop("hasil", None)
                st.error(f"Terjadi kesalahan: {e}")

        hasil = st.session_state.get("hasil")
        if hasil:
            label = hasil["label"]
            st.success(f"**{label.upper()}** — kemungkinan {persen(hasil['keyakinan'])}%")
            st.markdown(f"### Hasil prediksi: **{label.upper()}**")
            for kelas in ["finance", "olahraga"]:
                nilai = hasil["proba"].get(kelas, 0.0)
                tanda = ">>" if kelas == label else "  "
                st.progress(nilai, text=f"{tanda} {kelas.upper()} — {persen(nilai)}%")

            st.divider()
            if hasil["judul"]:
                st.subheader(hasil["judul"])
            kolom1, kolom2, kolom3 = st.columns(3)
            kolom1.metric("Token", hasil["jumlah_token"])
            kolom2.metric("Dikenal model", hasil["token_dikenal"])
            kolom3.metric("Di luar vocab", hasil["token_oov"])

            with st.expander("Detail proses & isi berita"):
                st.write("**Token hasil preprocessing (jalur B):**")
                st.code(" ".join(hasil["contoh_token"]), language=None)
                if hasil["contoh_oov"]:
                    st.write("**Kata di luar vocabulary (diabaikan):**")
                    st.code(" ".join(hasil["contoh_oov"]), language=None)
                st.write("**Cuplikan berita:**")
                st.write(hasil["teks"][:600] + ("..." if len(hasil["teks"]) > 600 else ""))
                st.write(f"**Link sumber:** {hasil['url']}")

    st.divider()
    st.caption("Persentase adalah keyakinan model Gaussian Naive Bayes terhadap "
               "vektor berita, bukan jaminan kebenaran. Model dilatih dari 200 "
               "berita detik (100 finance, 100 olahraga).")


if __name__ == "__main__":
    halaman()
