Customer Segmentation with K-Means Clustering

This project applies unsupervised machine learning to segment mall customers into meaningful groups. By clustering customers based on their Annual Income and Spending Score, the marketing team can design targeted campaigns that maximize engagement and sales.

🚀 Project Workflow

The project is structured into the following steps:

Data Loading & Cleaning

Import the Mall Customers dataset.

Handle missing values and prepare the data for analysis.

Exploratory Data Analysis (EDA)

Visualize distributions of age, income, and spending scores.

Generate pair plots and heatmaps to reveal feature relationships.

K-Means Clustering

Apply K-Means algorithm to group customers.

Use the Elbow Method and Silhouette Score to determine the optimal number of clusters.

Cluster Visualization

Plot customer clusters on a scatterplot.

Highlight distinct customer groups with clear separation.

Business Insights

Interpret each cluster.

Provide actionable strategies for targeted marketing campaigns.

📂 Files Included

Code & Data

customer_segmentation.py → Main Python script

Mall_Customers.csv → Dataset

Visualizations

genre_distribution.png → Genre distribution plot

age_distribution.png → Age distribution plot

annual_income_k_distribution.png → Annual income distribution

spending_score_1-100_distribution.png → Spending score distribution

pair_plot.png → Pair plot of numerical features

correlation_heatmap.png → Feature correlation heatmap

elbow_method.png → Elbow Method result

customer_clusters.png → Final customer clusters

⚙️ How to Run

Install Python and the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Run the script in your terminal:

python customer_segmentation.py


The program will:

Generate all plots.

Print insights from the clustering results.

🎯 Outcome

This project delivers clear customer segments that empower businesses to:

Identify high-value customers.

Personalize marketing campaigns.

Optimize product recommendations.

Drive data-driven decision-making.

✨ With this segmentation model, businesses can transform raw customer data into actionable marketing intelligence.