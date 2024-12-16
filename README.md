# **Obesity Level Classification Using Random Forest**

![Project Overview](https://img.shields.io/badge/Project%20Status-Completed-brightgreen)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-blue)  
![Python](https://img.shields.io/badge/Language-Python%203.8%2B-blue)

## 🎯 **Project Overview**

This project focuses on predicting obesity levels in individuals using **Random Forest Classifier**, a powerful ensemble learning technique. The dataset used was collected from individuals in Mexico, Peru, and Colombia and includes various features such as eating habits, physical condition, and demographics.

The key goals of the project are:
- Build a robust classification model to predict obesity levels.
- Handle class imbalance using **SMOTE** oversampling technique.
- Fine-tune hyperparameters using **GridSearchCV** for performance optimization.
- Visualize and analyze feature importance to understand model decisions.

---

## 📊 **Dataset Information**

The dataset **"Estimation of Obesity Levels Based on Eating Habits and Physical Condition"** comes from the **UCI Machine Learning Repository**.

### **Features**:
- **Demographic**: Gender, Age, Height, Weight.
- **Behavioral**: Eating habits, water intake, physical activity, and technology usage.
- **Categorical**: Family history of overweight, alcohol consumption, transportation mode.

### **Target**:
- **ObesityLevel**: 7 classes:
   - Insufficient Weight  
   - Normal Weight  
   - Overweight Level I  
   - Overweight Level II  
   - Obesity Type I  
   - Obesity Type II  
   - Obesity Type III  

---

## 🛠️ **Tools and Libraries Used**

- **Programming Language**: Python 3.8+
- **Libraries**:
  - `pandas` and `numpy` for data manipulation.
  - `matplotlib` and `seaborn` for data visualization.
  - `scikit-learn` for building machine learning models.
  - `imbalanced-learn` for SMOTE oversampling.
  - `GridSearchCV` for hyperparameter tuning.

---

## 🚀 **Project Workflow**

1. **Data Preprocessing**
   - Encoding categorical features using **LabelEncoder**.
   - Normalization of numerical features.
   - Splitting the dataset into train and test sets (80:20).

2. **Class Imbalance Handling**
   - Oversampling minority classes using **SMOTE**.

3. **Model Development**
   - Baseline model using **RandomForestClassifier**.
   - Hyperparameter tuning using **GridSearchCV**.

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
   - Feature importance analysis.

5. **Visualization**
   - Class distribution (before and after oversampling).
   - Confusion Matrix and Feature Importance.

---

## 📈 **Key Results**

### **Baseline Model Results**
- **Accuracy**: 95.74%  
- **Weighted Precision, Recall, F1-Score**: ~96%

### **Final Model Results (After Hyperparameter Tuning)**
- Improved performance and stability across all classes.

### **Confusion Matrix**
![image](https://github.com/user-attachments/assets/79e52757-69bd-493f-bfe3-53fe807e05ff)


### **Feature Importance**
![image](https://github.com/user-attachments/assets/a0b9c823-6b3a-493b-ab5e-54a35aba25fe)


- **Weight**, **Height**, and **Age** were the most influential features in predicting obesity levels.

---

## 🧩 **How to Run the Project**

### **Prerequisites**
Make sure Python is installed (>=3.8). Install the required libraries using the command:

```bash
pip install -r requirements.txt
```

### **Run the Project**
1. Clone this repository:
   ```bash
   git clone https://github.com/username/obesity-classification.git
   cd obesity-classification
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `obesity_level_classification.ipynb` and execute each cell sequentially.

---

## 📂 **Project Structure**

```
├── data/
│   └── obesity_dataset.csv       # Original dataset
├── notebooks/
│   └── obesity_classification.ipynb   # Main project notebook
├── images/
│   ├── confusion_matrix.png      # Confusion Matrix
│   ├── feature_importance.png    # Feature Importance plot
│   └── class_distribution.png    # Class Distribution plot
├── README.md                     # Project description
├── requirements.txt              # Required libraries
└── LICENSE                       # License information
```

---

## 💡 **Insights and Takeaways**
- **Random Forest** effectively handles multi-class classification problems with high accuracy.
- Proper handling of **class imbalance** using SMOTE leads to improved model stability.
- Feature importance analysis reveals that weight, height, and age are strong predictors of obesity levels.

---

## 🤝 **Contributing**
Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

---

## 📜 **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 **Contact**
For any questions, feel free to reach out:

- **Name**: M. Nur Aqil Bahri  
- **Email**: aqilbahri1234@gmail.com  
- **GitHub**: [https://github.com/yourusername](https://github.com/Cill19)

---

## ⭐ **If you found this project helpful, don't forget to give it a star!** ⭐

---

