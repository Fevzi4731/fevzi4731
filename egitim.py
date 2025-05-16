from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Veriyi oku
df = pd.read_csv("veriseti.csv")
y = df["Etiket"]
X = df.drop("Etiket", axis=1)

# Eğitim ve test setlerine ayır (80% eğitim, 20% test)
Xegt, Xtst, Yegt, Ytst = train_test_split(X, y, test_size=0.2, random_state=42)

# Set boyutlarını yazdır
print(f"Eğitim veri sayısı: {len(Xegt)} ({(len(Xegt)/len(X))*100:.2f}%)")
print(f"Test veri sayısı: {len(Xtst)} ({(len(Xtst)/len(X))*100:.2f}%)")

# Pipeline oluştur: önce veriyi standartlaştır, sonra sınıflandır
pipeline = Pipeline([
    ("std", StandardScaler()), 
    ("sinif", KNeighborsClassifier(n_neighbors=5))
])

# Modeli eğit
pipeline.fit(Xegt, Yegt)

# Test verisi ile tahmin yap
Y_model = pipeline.predict(Xtst)

# Doğruluk oranını yazdır
dogruluk_orani = accuracy_score(Ytst, Y_model)
print(f"Modelin Test Setindeki Doğruluk Oranı: {dogruluk_orani:.4f}")

# Eğitilen modeli kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
