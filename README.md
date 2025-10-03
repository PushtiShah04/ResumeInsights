# Resume Domain Classification using Machine Learning

## 1. Objective
The primary objective of this project is to build a machine learning-based application capable of scanning and analyzing resumes to determine the specific domain of Computer Science a candidate is most likely associated with.  

This system automates the classification of resumes by leveraging natural language processing (NLP) to extract meaningful content and applying trained models to identify patterns that align with various technical domains.  

**Goal:** Enhance the efficiency of resume screening by providing accurate, data-driven predictions that assist in talent identification and domain categorization.  

---

## 2. Dataset Used
**Dataset:** [Updated Resume Dataset (Kaggle)](https://www.kaggle.com/datasets)  

### Features:
- **Resume**: Raw textual content from resumes (education, experience, skills, projects, etc.)  
- **Category**: Labeled domain/functional area in Computer Science, including:  
  - Data Science  
  - Web Development  
  - Machine Learning  
  - DevOps  
  - Testing  
  - HR  
  - Operations  
  - Networking  
  - Mobile Development  
  - Cloud Computing  
  - Business Analyst  
  - Python Developer  
  - SAP Developer  
  - Java Developer  
  - Automation Testing  
  - Digital Marketing  
  - Others  

### Preprocessing Steps:
- **Text Cleaning**: Removed URLs, special characters, non-ASCII text, and extra spaces.  
- **Normalization**: Converted text to lowercase.  
- **Vectorization**: Applied TF-IDF to convert text into numerical vectors.  

---

## 3. Model Chosen
A comparative analysis of models was conducted to find the best fit for this text classification task.  

### Models Explored:
- Support Vector Classifier (SVC)  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  

All models were trained using **TF-IDF transformed resume text** and evaluated using **accuracy, confusion matrix, and classification report**.  

**Final Model:**  
- **SVC** was selected for deployment due to its superior performance.  
- Model, TF-IDF vectorizer, and label encoder were serialized using `pickle`.  
- Deployed using **Streamlit app** with features:  
  - Upload resume (PDF/DOCX/TXT)  
  - Instant prediction of the resume domain  

---

## 4. Performance Metrics
All models achieved very high accuracy across 21 classes:  

**a) RandomForestClassifier**  
- Accuracy: **100%**  
- Confusion Matrix: Perfect diagonal alignment  
- Classification Report: Precision, Recall, F1-Score = **1.00**  

**b) Support Vector Classifier (SVC)**  
- Accuracy: **100%**  
- Confusion Matrix: Identical to RandomForest  
- Classification Report: All metrics = **1.00**  

**c) KNeighborsClassifier**  
- Accuracy: **100%**  
- Confusion Matrix: Perfect alignment  
- Classification Report: Precision, Recall, F1-Score = **1.00**  

---

## 5. Challenges & Learnings
### Challenges:
- **Dataset Imbalance**: Resolved using augmentation and stratified sampling.  
- **Text Preprocessing**: Managed with NLP techniques and custom filtering.  
- **Multi-Class Precision**: Required fine-tuning and evaluation.  
- **Performance Validation**: Cross-validation and manual inspections used.  

### Learnings:
- **Model Selection**: RandomForest showed strong interpretability.  
- **Data Quality**: "Garbage in = garbage out".  
- **Feature Engineering**: Boosted model performance significantly.  
- **Advanced Evaluation**: Beyond accuracy (precision, recall, F1).  
- **Deployment**: Gained end-to-end knowledge from training to UI.  

---

## 6. Future Enhancements
- Suggest improvements and generate stronger resumes.  
- Save scanned resume history per user.  
- Extract structured data (name, email, skills, etc.).  
- Rank resumes using ML or cosine similarity-based scoring.  

---

## 7. References
- [Resume Dataset - Kaggle](https://www.kaggle.com/datasets)  
- [Automated Resume Classification Using Machine Learning](https://www.researchgate.net/publication/343306091_Automated_Resume_Classification_Using_Machine_Learning)  
- [Deep Learning for Resume Classification](https://ieeexplore.ieee.org/document/8614254)  
  
