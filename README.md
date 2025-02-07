# 🏡 The Real Estate Crystal Ball: Predicting Prices with Smarter Data  

![Image](https://github.com/user-attachments/assets/1a2a7b22-77bd-44bf-8567-aa1d5722cd7c)

### **TL;DR**  
Grouped property features based on real estate logic, tested multiple models, and found that **feature engineering improved performance across models**, with **the largest gains in linear regression and gradient boosting**, while **CatBoost ultimately performed best**.  

---

## **📌 The Non-Technical Summary**  
Michael Lewis writes to **one specific person**—someone real, someone who will challenge him.  
**RP, this project is for you.**  

Most property valuation models focus on **individual features** with high correlation, but real estate professionals think about **feature interactions**—location, condition, and land use **together**.  
This project tested whether **machine learning models could think more like an appraiser.**  

### **Key Takeaways:**  
- ✅ **Feature engineering improved traditional models** (Linear Regression RMSE ↓ **24,471 → 21,975**).  
- ✅ **Gradient Boosting and XGBoost also improved** (Gradient Boosting RMSE ↓ **22,612 → 21,077**).  
- ✅ **CatBoost performed best overall** (**RMSE 19,875**), showing minimal reliance on manual feature engineering.  

While this study focused on residential data, the approach applies to commercial real estate.  

Oh, and one more thing: **this is real-world, ugly data.** It’s full of missing values, inconsistencies, and weird edge cases—just like any dataset you’d find in commercial real estate. Cleaning it up and making it usable was half the battle.  

---

## **🎯 Project Approach**  

### **🔹 Feature Engineering with Real Estate Logic**  
Instead of just using high-correlation variables, features were **grouped logically**:  
- **Interior:** Square footage, HVAC, bedrooms  
- **Exterior & Land:** Lot size, frontage, porch, garage  
- **Basement & Storage:** Finished space, bathrooms  
- **Sales & Transactions:** Sale type, year  

This structure improved the EDA process and enhanced human interpretability—for example, combining all porch and basement related square footage. 

---

## **📌 Why Invest in EDA?**  

I chose this project as an opportunity to put **domain knowledge to work**. While feature engineering did not always produce dramatic gains, I learned in a previous project that **time spent on EDA—truly understanding the variables and their context to the desired outcome—is an investment that pays off significantly in the long run**.  

Many data science workflows **move through EDA quickly**, focusing on cleaning and preprocessing before diving into modeling. However, a deeper exploration of the dataset not only helps with **feature selection and engineering** but also improves **interpretability, error analysis, and long-term model stability**.  

The improvements seen in **linear regression and gradient boosting** suggest that structured feature engineering can enhance performance, especially for models that do not inherently detect complex interactions.  

---

## **📊 Visualizing Key Relationships**  

EDA is not just about cleaning data—it’s about understanding how features interact with the target variable. To explore these relationships, I included key visualizations:  

- **Sale Price Distribution** – Reveals the skewness of the target variable, highlighting the need for transformation.  
- **Sale Price vs. Total Rooms** – Examines whether more rooms consistently lead to higher prices or if other factors play a role.  
- **Sale Price vs. GrLivArea (Above-Ground Living Area)** – A crucial check, as square footage is often one of the strongest predictors of home value.  
- **Sale Price by Decade** – Investigates whether older homes follow different pricing trends, capturing potential temporal effects.  

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1a77c594-da71-46cb-b89e-4fe876ddd06c" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/c109c5dc-5989-4e7f-b6c3-08c77cd22dab" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b22da2a0-6492-47e0-bcd6-a4940090f86a" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/09c1ea12-1349-452d-a032-cd4b95909a4b" width="400"></td>
  </tr>
</table>

These charts help **validate assumptions**, **identify outliers**, and **inform feature engineering decisions**, ensuring that the model is built on meaningful, structured insights rather than just raw correlations.  

---

## **📊 Model Performance**  

Feature engineering **consistently improved RMSE across models**, with the strongest gains in **linear regression and gradient boosting models**.  

| Model | RMSE Before | RMSE After  |
|---------------|------------|------------|
| **Decision Tree**  | 37,033 | 36,282 |
| **Linear Regression** | 24,471 | 21,975 |
| **Gradient Boosting** | 22,612 | 21,077 |
| **XG Boost** | 21,995 | 20,250 |
| **CatBoost**   | 22,427 | **19,875** |

- **Linear regression saw significant improvements**, confirming that structured feature engineering is especially beneficial for traditional models.  
- **Gradient Boosting also improved**, showing that engineered features added some predictive power.  
- **Tree-based models, including CatBoost, saw minimal gains**, reinforcing their ability to extract feature relationships without extensive manual engineering.  

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/03f2ef38-badd-461a-8c5d-5bfc421e35aa" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/29b453b5-6a2d-47ae-afc3-da64e2ade0c3" width="400"></td>
  </tr>
</table>
---

## **🔎 Findings**  
1️⃣ **While feature engineering added value across models, its impact varied depending on how much the model relies on manual feature construction versus automated pattern recognition.**  

2️⃣ **Grouping related features based on meaning and context improved interpretability and will be a new best practice in my toolkit.**  
3️⃣ **Future work:** Model stacking, neighborhood and location based features, and exploring temporal trends.  

### **🏆 Kaggle Score: 0.12897** (Top 20%)  
CatBoost delivered the strongest performance, underscoring the power of boosting algorithms in real estate valuation. The **Kaggle score of 0.12897 (Top 20%)** is excellent, especially considering that **minimizing RMSE was not the primary goal**.  

This project focused on comparing **feature engineering methodologies rather than hyperparameter tuning**, reinforcing that advanced models like CatBoost inherently capture complex relationships with minimal manual intervention.

---

## **🏁 Conclusion**  
Domain-driven feature engineering is a valuable tool across all models, though its impact varies. 

