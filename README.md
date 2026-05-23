# 🎗️ Breast Cancer Classification — Machine Learning Pipeline

En komplett och reproducerbar ML-pipeline för klassificering av tumörer som benigna eller maligna.

---

## 📖 Om projektet

Detta projekt bygger en maskininlärningspipeline för att klassificera tumörer som benigna eller maligna med hjälp av Breast Cancer Wisconsin-datasetet från scikit-learn.

Projektet demonstrerar ett komplett arbetsflöde från rå data till utvärdering:
- Data exporteras till CSV och läses in från disk, precis som med ett lokalt dataset
- Irrelevanta kolumner identifieras och tas bort baserat på korrelationsanalys
- Fyra klassificeringsmodeller tränas, utvärderas och jämförs
- Hyperparametertuning med GridSearchCV för SVM
- Korsvalidering (k=5) för robust prestanda-uppskattning
- Tränade modeller och visualiseringar sparas för återanvändning

All kod körs i Jupyter Notebook via VS Code.

---

## 🧬 Om datasetet

| Egenskap        | Värde                                     |
|-----------------|-------------------------------------------|
| Källa           | scikit-learn (Breast Cancer Wisconsin)    |
| Antal samples   | 569                                       |
| Antal features  | 30 numeriska attribut (23 efter städning) |
| Klasser         | Benign (357) / Malign (212)               |
| Uppgift         | Binär klassificering                      |

Attributen beskriver egenskaper hos cellkärnor i tumörbiopsier, exempelvis radie, textur, perimeter och konkavitet.

---

## 🚀 Kom igång

### 1. Klona projektet
```bash
git clone https://github.com/<ditt-användarnamn>/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Skapa och aktivera virtuellt environment
```bash
# Skapa venv med Python 3.14
py -3.14 -m venv .venv

# Aktivera (Windows PowerShell)
.venv\Scripts\activate
```

### 3. Installera beroenden
```bash
pip install -r requirements.txt
```

### 4. Öppna notebooken i VS Code
```bash
code .
```
Öppna sedan `breast_cancer_classification.ipynb`, välj `.venv` som kernel och kör cellerna uppifrån och ned.

---

## 🧠 Pipeline-översikt

```
Exportera dataset → CSV
         │
         ▼
Läs in från disk (pd.read_csv)
         │
         ▼
Explorativ dataanalys (EDA)
  – Datastruktur och typer
  – Deskriptiv statistik
  – Klassfördelning
         │
         ▼
Datastädning
  – Korrelationsanalys mot target
  – Ta bort kolumner med |r| < 0.3
  – 30 → 23 features
         │
         ▼
Förbehandling
  – Train/test-split (80/20, random_state=42)
  – StandardScaler (normalisering)
         │
         ▼
Modellträning
  – Support Vector Machine (SVM)
  – Random Forest
  – Logistisk Regression
  – Beslutsträd
         │
         ▼
Hyperparametertuning
  – GridSearchCV för SVM
  – Optimerade parametrar: C=10, gamma=0.01
  – Scoring: recall_weighted
         │
         ▼
Korsvalidering (k=5)
  – Robust prestanda-uppskattning
  – Boxplot med medelvärde per modell
         │
         ▼
Spara modeller (joblib → models/)
         │
         ▼
Utvärdering per modell
  – Accuracy, Precision, Recall, F1-score
  – Classification Report
         │
         ▼
Modelljämförelse & visualisering
  – Jämförelsetabell
  – Confusion matrix heatmaps (2×2)
  – ROC-kurva & AUC
  – Sparas till outputs/
```

---

## 📊 Modeller och resultat

Tränat och utvärderat på 23 features efter datastädning, 80/20-split, StandardScaler.

### Testdata (114 prover)

| Modell               | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| SVM (optimerad)      | 97.4%    | 97.4%     | 97.4%  | 97.4%    |
| Logistisk Regression | 97.4%    | 97.4%     | 97.4%  | 97.4%    |
| Random Forest        | 96.5%    | 96.5%     | 96.5%  | 96.5%    |
| Beslutsträd          | 91.2%    | 91.4%     | 91.2%  | 91.3%    |

### Korsvalidering k=5 (mer robust uppskattning)

| Modell               | Medel  | Std dev |
|----------------------|--------|---------|
| SVM (optimerad)      | 97.4%  | 0.016   |
| Logistisk Regression | 97.1%  | 0.020   |
| Random Forest        | 95.8%  | 0.018   |
| Beslutsträd          | 93.2%  | 0.011   |

> **Notering om recall:** I ett medicinskt sammanhang är recall särskilt viktigt. Ett falskt negativt resultat innebär ett missat cancerfall, vilket är allvarligare än ett falskt positivt. GridSearchCV optimerade därför mot recall_weighted.

---

## 📁 Projektstruktur

```
breast-cancer-classification/
├── breast_cancer_classification.ipynb  # Huvudnotebook med hela pipeline
├── requirements.txt                    # Python-beroenden
├── README.md                           # Dokumentation
├── .gitignore                          # Filer som exkluderas från Git
├── data/
│   └── breast_cancer.csv               # Lokalt dataset (genereras vid körning)
├── models/
│   ├── svm_model_optimized.pkl         # Optimerad SVM-modell (GridSearchCV)
│   ├── rf_model.pkl                    # Tränad Random Forest-modell
│   ├── lr_model.pkl                    # Tränad logistisk regressionsmodell
│   ├── dt_model.pkl                    # Tränat beslutsträd
│   └── scaler.pkl                      # Sparad StandardScaler
└── outputs/
    ├── correlation_with_target.png     # Korrelationsdiagram
    ├── cross_validation.png            # Korsvalidering boxplot
    ├── confusion_matrices.png          # Confusion matrix heatmaps
    └── roc_curve.png                   # ROC-kurva med AUC
```

---

## 🛠️ Teknikstack

| Komponent            | Version  | Beskrivning                              |
|----------------------|----------|------------------------------------------|
| scikit-learn         | 1.8.0    | Dataset, modeller och utvärderingsmetrik |
| pandas               | 3.0.2    | Datahantering och analys                 |
| numpy                | 2.4.4    | Numeriska beräkningar                    |
| matplotlib           | 3.10.9   | Visualiseringar                          |
| seaborn              | 0.13.2   | Statistiska visualiseringar              |
| joblib               | 1.5.3    | Serialisering av tränade modeller        |
| ipykernel            | 7.2.0    | Jupyter kernel för virtuellt environment |
| Jupyter Notebook     | –        | Interaktiv utvecklingsmiljö              |

---

## 📄 Licens

Distribueras under MIT-licensen. Se `LICENSE` för mer information.
