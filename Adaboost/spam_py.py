# ================================================================
# 📨 SMS Spam Classification using AdaBoost
# ================================================================

# ============================
# 📦 Import Libraries
# ============================
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# ================================================================
# PART A — Data Preprocessing & Exploration
# ================================================================

# 1️⃣ Load Dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'text']

# 2️⃣ Convert labels: spam → 1, ham → 0
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3️⃣ Text Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = [w for w in text.split() if w not in stop_words]  # remove stopwords
    return " ".join(words)

data['clean_text'] = data['text'].apply(preprocess_text)

# 4️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

# 5️⃣ Train-test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6️⃣ Show class distribution
print("Class distribution:")
print(data['label'].value_counts(normalize=True))

# ================================================================
# PART B — Weak Learner Baseline (Decision Stump)
# ================================================================
print("\n============================")
print("PART B — Decision Stump")
print("============================")

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)

y_pred_train = stump.predict(X_train)
y_pred_test = stump.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(cm).plot()
plt.title("Decision Stump — Confusion Matrix")
plt.show()

print("\nComment: A single-level decision tree (stump) is too simple to capture complex text patterns.\n"
      "It only splits based on one feature (word), missing multi-word relationships or nuanced context.")

# ================================================================
# PART C — Manual AdaBoost (T = 15 rounds)
# ================================================================
print("\n============================")
print("PART C — Manual AdaBoost (T = 15)")
print("============================")

T = 15
n = X_train.shape[0]
weights = np.ones(n) / n
alphas = []
weighted_errors = []

models = []

for t in range(1, T + 1):
    # Train weak learner
    stump_t = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump_t.fit(X_train, y_train, sample_weight=weights)
    y_pred = stump_t.predict(X_train)

    # Compute weighted error
    incorrect = (y_pred != y_train)
    error = np.dot(weights, incorrect) / np.sum(weights)

    # Compute alpha
    alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
    
    # Update weights
    weights *= np.exp(-alpha * y_train * (2 * y_pred - 1))
    weights /= np.sum(weights)

    alphas.append(alpha)
    weighted_errors.append(error)
    models.append(stump_t)

    # Print progress
    mis_idx = np.where(incorrect)[0]
    print(f"\nIteration {t}")
    print(f"Misclassified samples: {mis_idx[:10]}{'...' if len(mis_idx)>10 else ''}")
    print(f"Weighted Error: {error:.4f}")
    print(f"Alpha: {alpha:.4f}")

# ➕ Combine weak learners
def adaboost_predict(X, models, alphas):
    pred = np.zeros(X.shape[0])
    for model, alpha in zip(models, alphas):
        pred += alpha * (2 * model.predict(X) - 1)
    return (pred > 0).astype(int)

y_train_pred = adaboost_predict(X_train, models, alphas)
y_test_pred = adaboost_predict(X_test, models, alphas)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nFinal Train Accuracy: {train_acc:.3f}")
print(f"Final Test Accuracy: {test_acc:.3f}")

cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Manual AdaBoost — Confusion Matrix")
plt.show()

# 🔹 Plot weighted error and alpha vs iteration
plt.figure()
plt.plot(range(1, T+1), weighted_errors, marker='o')
plt.title("Iteration vs Weighted Error")
plt.xlabel("Iteration")
plt.ylabel("Weighted Error")
plt.show()

plt.figure()
plt.plot(range(1, T+1), alphas, marker='o')
plt.title("Iteration vs Alpha")
plt.xlabel("Iteration")
plt.ylabel("Alpha")
plt.show()

print("\nInterpretation: Initially, all samples have equal weight. "
      "Misclassified samples receive higher weights in later iterations, "
      "forcing new stumps to focus on harder examples.")

# ================================================================
# PART D — Sklearn AdaBoost
# ================================================================
print("\n============================")
print("PART D — Sklearn AdaBoost")
print("============================")

sk_adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.6,
    random_state=42
)
sk_adaboost.fit(X_train, y_train)

y_train_pred = sk_adaboost.predict(X_train)
y_test_pred = sk_adaboost.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Sklearn AdaBoost — Confusion Matrix")
plt.show()

print("\nComparison:")
print("Sklearn AdaBoost (with 100 estimators) achieves better performance "
      "than manual AdaBoost (15 iterations) since it combines more learners "
      "and tunes learning rate efficiently.")
