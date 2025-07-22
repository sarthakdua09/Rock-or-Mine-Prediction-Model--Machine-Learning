
# 🪨 Rock or Mine Prediction Model

This project focuses on building a binary classification model that can distinguish between rocks and mines using sonar signals. The objective is to develop a machine learning pipeline that predicts the type of object based on 60 features extracted from sonar signals.

---

## 📌 Project Highlights

- 🔍 **Domain**: Signal Processing, Binary Classification
- 📊 **Algorithm**: Logistic Regression
- 🧠 **Libraries Used**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- 📁 **Dataset**: UCI Sonar Dataset - 60 features representing sonar signal strengths
- 🏁 **Output Classes**: `M` for Mine, `R` for Rock

---

## 📂 Dataset Description

The dataset consists of 208 instances, each with 60 numerical features and a label (`M` or `R`):

| Feature Count | Description                 |
|---------------|-----------------------------|
| 60            | Sonar signal attributes     |
| 1             | Target: Mine (M) or Rock (R)|

### Example:
```plaintext
0.02, 0.037, 0.042, ..., 0.003, M
```

---

## 🔧 Technologies Used

- `NumPy` – for numerical operations
- `Pandas` – for data manipulation
- `Matplotlib & Seaborn` – for data visualization
- `Scikit-learn` – for machine learning model development

---

## 🧪 Steps Followed

1. **Data Loading**  
   Loaded dataset into a Pandas DataFrame.

2. **Data Exploration & Visualization**  
   - Explored data shape, types, and value distribution.
   - Used seaborn/matplotlib for visual patterns.

3. **Preprocessing**  
   - Label encoding (`M` → 0, `R` → 1)
   - Feature-target split
   - Train-test split using `train_test_split`

4. **Model Training**  
   - Model: `LogisticRegression()`
   - Trained on training data.

5. **Evaluation**  
   - Evaluated using `accuracy_score`
   - Printed predictions and actual labels for comparison.

---

## 📈 Model Evaluation

- **Model Used**: Logistic Regression
- **Metric**: Accuracy Score
- **Performance**: ~85–90% (based on dataset and random split)

---

## ▶️ How to Run

1. Clone this repository.
2. Download the dataset and place it in the appropriate directory.
3. Open and run the Jupyter Notebook file:
```bash
jupyter notebook "Rock or Mine Prediction model.ipynb"
```

---

## 🚀 Future Improvements

- Experiment with other models: Random Forest, SVM, KNN, etc.
- Hyperparameter tuning and cross-validation
- Confusion matrix, precision, recall, and ROC-AUC evaluation
- GUI or web app using Streamlit or Flask

---

## 📚 References

- UCI ML Repository: [Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
- Scikit-learn: [Documentation](https://scikit-learn.org/stable/)

---

## 👤 Author

**Sarthak Dua**  
Aspiring Data Analyst | Skilled in Python, SQL, Power BI, and Excel  
Projects: Data Cleaning, EDA, Dashboarding, and Predictive Modeling

---

## ⭐ Acknowledgements

Thanks to the UCI Machine Learning Repository for providing the dataset and the open-source ML community for enabling accessible tools.

