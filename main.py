#
# Krzysztof Matuszewski, nr indeksu: 160802
# AHE informatyka st. II, rok 1, sem. 2, sekcja wirtualni, grupa U1
# Projekt "Klasyfikator Spamu", Inteligentne przetwarzanie tekstu - Projekt
# 

# importowanie elementów biblioteki Natural Language Toolkit do przetwarzania tekstu
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# importowanie elementów biblioteki Scikit-learn do modelowania i ML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# importowanie biblioteki Pandas do analizy danych
import pandas as pd

# importowanie elementów biblioteki Tkinter do tworzenia GUI
import tkinter as tk
from tkinter import messagebox


# PRZYGOTOWANIE DANYCH (1)

# wczytywanie danych treningowych z pliku CSV (1.1)
training_data_path = './spam_NLP.csv'
training_data = pd.read_csv(training_data_path)

# wypisanie pierwszych 5 emaili
print(f'\nPoczątek pliku:\n')
print(training_data.head(5))

# wypisanie kolumn
print(f'\nZnalezione kolumny:\n')
print(training_data.columns)
print('\n')

# oczyszczanie danych: (1.2)
# sprawdzenie brakujących wartości
print('Liczba brakujących elementów:')
print(training_data.isnull().sum()) # (poprzez wypisanie ilość brakujących elementów po kolumnie)
print('\n')

# usuwanie duplikatów
training_data.drop_duplicates(inplace=True)

# tokenizacja wszystkich danych (1.3)
nltk.download('punkt')
training_data['tokens'] = training_data['MESSAGE'].apply(word_tokenize)

# lemantyzacja wszystkich danych (1.4)
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
training_data['lemmatized'] = training_data['tokens'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

# usuwanie słów z listy stopu (stopwords) jak przyimki, spójniki, zaimki itd (1.5)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
training_data['filtered'] = training_data['lemmatized'].apply(lambda tokens: [token for token in tokens if token.lower() not in stop_words])

# przekształcenie przetworzonych tokenów z powrotem na tekst (potrzebne do użycia CountVectorizer())
training_data['processed_message'] = training_data['filtered'].apply(lambda tokens: ' '.join(tokens))

# wektoryzacja (proces przetworzenia tekstów na wektory liczb) (1.6)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(training_data['MESSAGE'])
y = training_data['CATEGORY']


# BUDOWA MODELU KLASYFIKACJI (2)

# wybór algorytmu klasyfikacji (tutaj wielomianowy naiwny klasyfikator Bayesa) (2.1)
model = MultinomialNB()

# podział danych na zbiór treningowy i testowy (2.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# trenowanie modelu (2.3)
model.fit(x_train, y_train)


# OCENA MODELU (3)

# predykcje na zbiorze testowym
y_pred = model.predict(x_test)

# obliczanie metryk (3.1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# analiza macierzy pomyłek (3.2)
cm = confusion_matrix(y_test, y_pred) 

# wypisywanie wszystkich wyników
print('\n')
print(f'Metryki (zaokrąglone):\n')

print(f'Dokładność:\t {round(accuracy, 2)}%') # miara jaki procent wszystkich przewidywań modelu jest poprawny (stosunek liczby poprawnych przewidywań do liczby wszystkich przewidywań) 
print(f'Precyzja:\t {round(precision, 2)}%') # miara, która wskazuje, jaki procent przewidywań pozytywnych (spam) był rzeczywiście pozytywny. Jest obliczana jako stosunek liczby prawdziwie pozytywnych przewidywań do liczby wszystkich pozytywnych przewidywań (prawdziwie pozytywnych i fałszywie pozytywnych)
print(f'Czułość:\t {round(recall, 2)}%') # procent, ile pozytywnych prawdziwych przypadków zostało skutcznie przewidzianych 
print(f'F1-score:\t {round(f1, 2)}%') # średnia harmoniczna precyzji i czułośći

# wypisanie macierzy pomyłek
print('\n')
print(f"Macierz pomyłek:\n{cm}\n\n")


# IMPLEMENTACJA APLIKACJI GRAFICZNEJ (4)

# tworzenie okna aplikacji GUI
root = tk.Tk()

# nadanie tytułu oknu
root.title("Klasyfikator SPAMu - Krzysztof Matuszewski")

# tworzenie pola tekstowego do wprowadzenia emaila (4.1)
email_label = tk.Label(root, text="Wpisz lub wklej tekst email'a (tylko angielski!)")
email_label.pack()
email_entry = tk.Text(root, height=10)
email_entry.pack()

# funkcja do obsługi analizy uruchamiana po wciśnięciu przycisku "Analizuj" (4.2)
def classify_email():
    email_text = email_entry.get("1.0", 'end-1c')
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    result = "SPAM!" if prediction == 1 else "nie SPAM"
    # wyświetlenie wyniku klasyfikacji (4.3)
    messagebox.showinfo("Rezultat", f"Ten email to {result}")

# dodawanie przycisku
analyze_button = tk.Button(root, text="Analizuj", command=classify_email)
analyze_button.pack()

# uruchamianie GUI
root.mainloop()

# TESTOWANIE I WALIDACJA (5) - sprawdzono manualnie na przykładowych mailach, zrzuty ekranu w sprawozdaniu
# PREZENTACJA WYNIKÓW (6) - również w sprawozdaniu

# koniec :)
