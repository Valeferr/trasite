# Trasite 🕵️‍♂️🏨

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-3.10%2B-02569B?style=for-the-badge&logo=flutter&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)

**Trasite** è un'applicazione full-stack progettata per analizzare e rilevare recensioni ingannevoli (fake reviews) nel contesto degli affitti brevi (es. Airbnb). Il sistema utilizza tecniche avanzate di Natural Language Processing (NLP) e Machine Learning per classificare le recensioni come legittime o fraudolente, fornendo agli utenti una visualizzazione chiara e affidabile su mappa.

## ✨ Funzionalità Principali

*   **Rilevamento Recensioni False**: Utilizzo di modelli Transformer (RoBERTa, DeBERTa) e classificatori tradizionali (Random Forest) per identificare recensioni ingannevoli.
*   **Analisi del Sentiment**: Valutazione del sentiment delle recensioni per correlarlo con la veridicità.
*   **Analisi POS (Part-of-Speech)**: Estrazione di feature linguistiche per migliorare la precisione dei modelli.
*   **Visualizzazione su Mappa**: Frontend in Flutter che mostra gli alloggi su una mappa interattiva, evidenziando il rapporto tra recensioni vere e false.
*   **API RESTful**: Backend basato su FastAPI per servire i dati elaborati in tempo reale.

## 🧠 Modelli e AI

Il core del progetto si basa su diversi modelli di Machine Learning e Deep Learning:

*   **DistilRoBERTa-base**: Fine-tuned per la classificazione di testi ingannevoli.
*   **Microsoft DeBERTa-v3-xsmall**: Modello efficiente per la comprensione del linguaggio naturale.
*   **Random Forest**: Utilizzato come classificatore downstream basato su feature estratte (PCA, sentiment, POS).
*   **PCA (Principal Component Analysis)**: Riduzione della dimensionalità per l'analisi delle feature.

## 📊 Dataset Utilizzati

Il progetto è stato addestrato e validato su dataset di riferimento nel campo della deception detection:

*   **Deceptive Opinion Spam Corpus (Ott et al.)**: Dataset standard per la ricerca sulle recensioni false.
*   **Yelp Dataset**: Utilizzato per l'addestramento su larga scala e la generalizzazione.
*   **Listings Dataset**: Dati reali di alloggi (es. InsideAirbnb) per l'applicazione pratica.

## 🛠️ Tech Stack

### Backend
*   **Linguaggio**: Python
*   **Framework API**: FastAPI
*   **Database**: MongoDB (con PyMongo)
*   **ML & Data Science**: PyTorch, Transformers (Hugging Face), Scikit-learn, Pandas, NumPy, TextBlob.

### Frontend
*   **Framework**: Flutter
*   **Mappe**: `flutter_map`, `latlong2`
*   **Linguaggio**: Dart

## 🚀 Installazione e Configurazione

### Prerequisiti
*   Python 3.10+
*   Flutter SDK
*   MongoDB (locale o Atlas)

### Setup Backend

1.  Naviga nella cartella `backend`:
    ```bash
    cd backend
    ```
2.  Crea un ambiente virtuale e attivalo:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```
3.  Installa le dipendenze (assicurati di avere un file `requirements.txt` o installa manualmente le librerie principali):
    ```bash
    pip install fastapi uvicorn pymongo python-dotenv torch transformers scikit-learn pandas textblob
    ```
4.  Configura le variabili d'ambiente:
    Crea un file `.env` nella cartella `backend` e aggiungi la tua stringa di connessione MongoDB:
    ```env
    CONNECTION_STRING=mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority
    ```
5.  Avvia il server API:
    ```bash
    uvicorn api:app --reload
    ```

### Setup Frontend

1.  Naviga nella cartella `frontend/trasite`:
    ```bash
    cd frontend/trasite
    ```
2.  Ottieni le dipendenze Flutter:
    ```bash
    flutter pub get
    ```
3.  Avvia l'applicazione:
    ```bash
    flutter run
    ```

## 📂 Struttura del Progetto

```
trasite/
├── backend/
│   ├── api.py                 # Server FastAPI
│   ├── data/                  # Dataset CSV (Yelp, Deceptive, Listings)
│   ├── trained/               # Modelli salvati (Joblib, Safetensors)
│   ├── pipelines/             # Script di training e processing (PCA, RoBERTa)
│   └── notebooks/             # Jupyter Notebooks per analisi esplorativa
├── frontend/
│   └── trasite/               # Progetto Flutter
│       ├── lib/               # Codice sorgente Dart
│       └── pubspec.yaml       # Dipendenze Flutter
└── README.md
```

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi il file `LICENSE` per maggiori dettagli.