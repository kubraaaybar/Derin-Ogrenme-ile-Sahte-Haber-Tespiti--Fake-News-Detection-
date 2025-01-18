import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords 
import stanza  # Stanza kütüphanesini import et
# Stanza'nın Türkçe modelini indir ve boru hattını başlat
stanza.download('tr')  # Türkçe modelini indir
nlp = stanza.Pipeline('tr', processors='tokenize,mwt,pos,lemma')  # Türkçe boru hattını başlat

stop_words = stopwords.words('turkish')  # NLTK'dan Türkçe stopwords listesini al.
stop_words.extend(['bir', 've', 'ile', 'diğer', 'olarak', 'gibi', 'ancak'])  # Ekstra stopwords ekle

# Eğitilmiş modelin ve tokenizer'ın yüklenmesi
model = tf.keras.models.load_model('fake_news_model.h5')  # Modeli yükle
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)  # Tokenizer'ı yükle

# Metin ön işleme fonksiyonu
def preprocess_and_lemmatize(text):
    if not isinstance(text, str):  # Eğer verilen metin bir string (metin) değilse, boş liste döndür.
        return []  # NaN veya yanlış tipteki verileri atmak için boş liste döner.
    result = []  # Sonuçları depolamak için boş bir liste oluştur.
    doc = nlp(text)  # Stanza boru hattını kullanarak metni analiz et (tokenization, POS etiketleme ve lemmatizasyon yap).
    for sentence in doc.sentences:  # Her bir cümleyi iterasyonla geç.
        for word in sentence.words:  # Cümledeki her kelimeyi iterasyonla geç.
            lemma = word.lemma  # Kelimenin kökünü (lemma) al.
            if lemma and lemma not in stop_words and len(lemma) > 2:  # Eğer kök var, stopwords (duru sözcükler) içinde değilse ve uzunluğu 2'den büyükse,
                result.append(lemma)  # Kökü sonuç listesine ekle.
    return result  # Temizlenmiş ve lemmatize edilmiş kelimelerle oluşan listeyi döndür.


# Sonsuz döngü ile kullanıcıdan sürekli haber metni al
while True:
    user_input = input("Lütfen haber metnini girin (çıkmak için 'exit' yazın): ")

    if user_input.lower() == 'exit':
        print("Çıkılıyor...")
        break

    # Metnin ön işleme (temizleme) aşaması
    cleaned_text = preprocess_and_lemmatize(user_input)

    # Metni sayısal verilere dönüştürme
    X_input = tokenizer.texts_to_sequences([cleaned_text])
    X_input = pad_sequences(X_input, maxlen=150, padding='post', truncating='post')

    # Model ile tahmin yapma
    y_pred = model.predict(X_input)
    prediction = (y_pred > 0.5).astype(int)  # Eşik değeri 0.5

    # Tahmin sonucu
    if prediction == 1:
        print("Bu haber DOĞRU.")
    else:
        print("Bu haber YANLIŞ.")