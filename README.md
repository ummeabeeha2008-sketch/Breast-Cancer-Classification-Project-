# Breast Cancer Classification using Machine Learning

## Overview
This project uses machine learning to classify breast tumors as malignant (cancerous) 
or benign (non-cancerous) based on cell measurements from biopsy data. It was built as 
part of my exploration into bioinformatics and the application of data science in 
disease diagnosis.

## Dataset
- **Source:** Breast Cancer Wisconsin Dataset (built into scikit-learn)
- **Samples:** 569 patients
- **Features:** 30 measurements per tumor (e.g., radius, texture, area, concavity)

## Approach
1. Loaded and explored the dataset
2. Split data into 80% training / 20% testing
3. Scaled features using StandardScaler
4. Trained two models:
   - Logistic Regression
   - Random Forest Classifier
5. Evaluated accuracy and classification reports
6. Identified the most important features for diagnosis

## Key Finding
The Random Forest model identified **worst area**, **worst concave points**, and 
**worst perimeter** as the most important features for distinguishing malignant from 
benign tumors — meaning larger, more irregular-shaped tumors were the strongest 
indicators of malignancy.

## Tools Used
- Python
- pandas
- scikit-learn

## How to Run
```bash
pip3 install pandas scikit-learn
python3 cancer_classification.py
```

## Future Improvements
- Add data visualizations (confusion matrix heatmap, feature importance chart)
- Try additional models (SVM, Neural Networks)
- Explore other medical datasets (diabetes, heart disease)

## Note# Breast Cancer Classification using Machine Learning

## Overview
This project uses machine learning to classify breast tumors as malignant (cancerous) 
or benign (non-cancerous) based on cell measurements from biopsy data. It was built as 
part of my exploration into bioinformatics and the application of data science in 
disease diagnosis.

## Dataset
- **Source:** Breast Cancer Wisconsin Dataset (built into scikit-learn)
- **Samples:** 569 patients
- **Features:** 30 measurements per tumor (e.g., radius, texture, area, concavity)

## Approach
1. Loaded and explored the dataset
2. Split data into 80% training / 20% testing
3. Scaled features using StandardScaler
4. Trained two models:
   - Logistic Regression
   - Random Forest Classifier
5. Evaluated accuracy and classification reports
6. Identified the most important features for diagnosis

## Key Finding
The Random Forest model identified **worst area**, **worst concave points**, and 
**worst perimeter** as the most important features for distinguishing malignant from 
benign tumors — meaning larger, more irregular-shaped tumors were the strongest 
indicators of malignancy.

## Tools Used
- Python
- pandas
- scikit-learn

## How to Run
```bash
pip3 install pandas scikit-learn
python3 cancer_classification.py
```

## Future Improvements
- Add data visualizations (confusion matrix heatmap, feature importance chart)
- Try additional models (SVM, Neural Networks)
- Explore other medical datasets (diabetes, heart disease)

## Note
This project was built with AI assistance for learning purposes — I used it to 
understand machine learning concepts and debug issues, while making sure I understood 
the logic behind every step.
