
```markdown
# Sprawozdanie: Laboratorium 01 - Tworzenie i wersjonowanie modelu ML

## Zadanie 1: Przygotowanie środowiska i instalacja bibliotek

**Komendy w terminalu (tworzenie środowiska i instalacja pakietów):**
```bash
python -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn joblib

```

*(aktywacja środowiska: `venv\Scripts\activate`)*

---

## Zadanie 1, 2 i 3: Przygotowanie danych, trenowanie i zapis modelu

**Plik `train.py`:**

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(df.shape)
print(df.dtypes)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))

joblib.dump(model, 'model_v1.joblib')

```

---

## Zadanie 3: Ładowanie zapisanego modelu i predykcja

**Plik `load_model.py`:**

```python
import joblib

model = joblib.load('model_v1.joblib')
sample_data = [[5.1, 3.5, 1.4, 0.2]]
print(model.predict(sample_data))

```

---

## Zadanie 4: Wersjonowanie modelu w praktyce

**Komendy w terminalu (Git):**

```bash
echo "venv/" > .gitignore
git init
git add train.py load_model.py model_v1.joblib .gitignore
git commit -m "Wdrozenie modelu v1.0"
git tag v1.0

```

**Kiedy zmieniasz wersję modelu?**

Zmieniono ważne parametry – np. zmieniono ustawienia algorytmu (hiperparametry).
Model jest skuteczniejszy – widać realną poprawę w wynikach.
Odświeżono dane – nowe dane do nauki dla modeli.

**Deweloperka vs Produkcja**
Deweloperka: Tu się testuje wszystkie modele. Testujesz na starych, stałych danych. Nie musi być szybko, ma być wygodnie do sprawdzania pomysłów.
Produkcja: Tu model idzie "na żywioł". Musi być niezawodny, wytrzymywać obciążenie i radzić sobie z nowymi danymi w realnym środowisku.

**Jak nie wyłożyć się przy wdrożeniu?**
Kontenery (np. Docker): Pakujesz model ze wszystkim, czego potrzebuje, do jednego "pudełka". Dzięki temu masz pewność, że na serwerze ruszy tak samo jak u Ciebie.

Automaty (CI/CD): Zamiast wrzucać pliki ręcznie, masz automat, który sam wszystko testuje i publikuje. Mniej błędów, zero stresu.

Monitoring i douczanie: Cały czas patrzysz modelowi na ręce. Jak zaczyna podawać głupoty (bo np. świat się zmienił i stare dane nie pasują), od razu douczasz go na nowo.
