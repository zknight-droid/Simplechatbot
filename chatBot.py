import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords dari NLTK
nltk.download('stopwords')

# Load dataset
with open('dataset.txt', 'r', encoding='utf8') as f:
    dataset = f.read().splitlines()

# Inisialisasi Chatbot
print("Halo, saya Chatbot sederhana. Silakan tanyakan sesuatu atau ketik 'keluar' untuk mengakhiri percakapan.")
user_input = ''
chat_history = []

# Looping untuk percakapan
while user_input.lower() != 'keluar':
    # Meminta input dari user
    user_input = input("Anda: ")

    # Menambahkan input user ke history chat
    chat_history.append(user_input)

    # Preprocessing input user
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(chat_history)

    # Jika chat_history hanya memiliki satu baris, langsung tampilkan pesan default
    if tfidf_matrix.shape[0] < 2:
        print("Chatbot: Selamat datang!")
    else:
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        index = similarities.argmax()

        # Jika tidak ada jawaban yang cocok, ambil jawaban random dari dataset
        if similarities[0][index] == 0:
            response = random.choice(dataset)
            print("Chatbot:", response)
        else:
            response = chat_history[index]
            print("Chatbot:", response)
