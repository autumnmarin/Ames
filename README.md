# 🏡 The Real Estate Crystal Ball: Predicting Prices with Smarter Data  

![Image](https://github.com/user-attachments/assets/1a2a7b22-77bd-44bf-8567-aa1d5722cd7c)

### **TL;DR**  
Grouped property features based on real estate logic, tested multiple models, and found that **feature engineering improved performance across models**, with **the largest gains in linear regression and gradient boosting**, while **CatBoost ultimately performed best**.  

---

Michael Lewis doesn't write for the masses—his words are written to one extraordinary, real individual.
RP, this project is for you.

---

## **📌 Project Overview**  
This project applies advanced **feature engineering and machine learning** techniques to predict home prices in Ames, Iowa. By leveraging **domain knowledge**, we structured features into meaningful categories and tested multiple models to assess performance improvements.  

### **Key Takeaways:**  
- **Feature engineering significantly improved model performance.**  
- **The largest gains were observed in linear regression and gradient boosting models.**  
- **CatBoost ultimately performed best, demonstrating minimal reliance on manual feature engineering.**  

---

## **🎯 Approach & Methodology**  

### **🔹 Feature Engineering Strategy**  
Rather than following a traditional approach of splitting features into categorical and numerical types, we grouped them into **real estate-relevant subcategories** first, then further classified them as numerical or categorical.  

#### **Feature Segments:**

<table>
<tr>
<td>

- **Basement**  
- **Bath**  
- **Exterior**  

 

</td>
<td>

- **Fireplace**  
- **Garage**  
- **General**  

</td>

<td>
  
- **Interior** 
- **Kitchen** 
- **Lot**  
 
</td>

<td>

- **Overall**  
- **Pool**  
- **Porch** 

</td>

<td>


- **Roof**  
- **Sale Info**


  
</td>
</tr>
</table>


This segmentation improved **interpretability** and **EDA efficiency**, allowing for more informed feature selection and engineering.  

---

## **📊 Exploratory Data Analysis (EDA)**  

EDA was crucial in identifying patterns, missing values, and outliers. Visualizations included:  

- **Sale Price by Decade** – Investigates long-term price trends.
- **Sale Price vs. GrLivArea** – Analyzes how above-ground living area correlates with price.  
- **Sale Price Distribution** – Highlights skewness in the target variable.  
- **Sale Price vs. Total Rooms** – Explores the impact of room count.  


- <table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1a77c594-da71-46cb-b89e-4fe876ddd06c" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/c109c5dc-5989-4e7f-b6c3-08c77cd22dab" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b22da2a0-6492-47e0-bcd6-a4940090f86a" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/09c1ea12-1349-452d-a032-cd4b95909a4b" width="400"></td>
  </tr>
</table>

---

## **📈 Model Performance**  

Feature engineering consistently improved RMSE across models.  

| Model | RMSE Before | RMSE After  |
|---------------|------------|------------|
| **Decision Tree**  | 37,033 | 36,282 |
| **Linear Regression** | 24,471 | 21,975 |
| **Gradient Boosting** | 22,612 | 21,077 |
| **XG Boost** | 21,995 | 20,250 |
| **CatBoost**   | 22,427 | **19,875** |

- **Linear regression saw the largest improvements**, reinforcing the value of structured feature engineering.  
- **Gradient Boosting models benefited as well**, indicating added predictive power from engineered features.  
- **CatBoost had the best overall RMSE**, proving its robustness even with minimal manual feature adjustments.  

---

## **🔎 Findings & Next Steps**  
1️⃣ **Feature engineering provided meaningful gains, especially for traditional models.**  
2️⃣ **Grouping features by real estate logic enhanced model interpretability.**  
3️⃣ **Future work includes model stacking, spatial analysis, and further refinement of temporal trends.**  

### **🏆 Kaggle Score: 0.12897** (Top 20%)  
Despite minimal hyperparameter tuning, the structured approach to feature engineering yielded strong performance, particularly in tree-based models.  

---

## **🏁 Conclusion**  
This project underscores the value of **domain-driven feature engineering** in improving predictive performance across multiple machine learning models.  

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=autumnmarin.Ames)

