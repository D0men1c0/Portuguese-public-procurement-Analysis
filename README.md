# Analysis of Portuguese Public Construction Contracts

## Dataset Overview

- **Dataset Link**: [Portuguese Public Construction Contracts Dataset](https://www.sciencedirect.com/science/article/pii/S2352340923001816)
- **Description**:  
  This dataset contains information on 5,214 public construction contracts in Portugal, collected between 2015 and 2022. It comprises 37 different features including numerical, categorical, and textual variables. Key attributes include:
  - **Initial Price** and **Effective Price**
  - **Award Criteria** (e.g., lowest price, multifactorial criteria)
  - **Execution and Submission Deadlines**
  - **Number of Bidders**
  - Various other properties related to contract details and administrative aspects.
- **Objective**:  
  The main goals of this project are to:
  - **Cluster Analysis**: Identify natural groupings among the contracts based on contract amount, awarding entity type, assignment timings, and geographical region. This will help uncover behavioral and geographical patterns among contracting entities.
  - **Association Rule Mining**: Discover significant relationships between key variables, such as how different award criteria influence price variations and deadlines, to support informed decision-making in the bidding and contracting processes.

---

## Project Structure

- **preProcessData.ipynb**:  
  Contains the data preprocessing steps, including Exploratory Data Analysis (EDA), data cleaning, feature transformation and selection, and the construction of the final dataset for further analysis.

- **clusterAnalysis.ipynb**:  
  Focuses on clustering methods and association rule mining. It evaluates various clustering algorithms and integrates cluster labels with association rules to provide semantic insights into the discovered patterns.

---

## Importance of Proper Dataset Preparation

Proper dataset preparation is critical for the success of both clustering and association rule mining. Here’s why thorough preprocessing and careful handling of data are essential:

1. **Enhanced Data Quality**:  
   - **Cleaning**: Removing unnecessary columns, handling null values, and eliminating outliers ensure that the dataset is accurate and reliable.  
   - **Transformation**: Converting date columns, extracting meaningful features from text, and normalizing numerical data reduce noise and inconsistencies, leading to more robust analyses.

2. **Feature Selection and Engineering**:  
   - **Selection**: Identifying and retaining only the most informative features reduces dimensionality and focuses the analysis on variables that truly matter.  
   - **Engineering**: Creating new features (e.g., extracting year/month from dates) enriches the dataset, making it easier to uncover hidden patterns.

3. **Improved Clustering Performance**:  
   - **Data Preprocessing**: Techniques such as normalization and discretization help ensure that all features contribute equally to the clustering process, avoiding bias toward variables with larger scales.  
   - **Feature Reduction**: By eliminating redundant or less informative features, clustering algorithms can more effectively identify natural groupings in the data.
   - **Visualization**: Dimensionality reduction methods (like UMAP) allow for better visualization of high-dimensional data, which aids in interpreting and validating clusters.

4. **Effective Association Rule Mining**:  
   - **Consistency**: A clean, well-prepared dataset helps ensure that the calculated metrics (support, confidence, lift) accurately reflect the true relationships between variables.
   - **Discretization**: Converting continuous variables into categorical bins makes it possible to apply association rule mining techniques, revealing hidden relationships that might otherwise be obscured.
   - **Meaningful Patterns**: Proper preparation allows for the discovery of significant and actionable association rules, supporting strategic decisions.

In summary, thorough data preprocessing lays a strong foundation for advanced analytics. Whether it is for clustering or association rule mining, investing time in cleaning, transforming, and selecting the right features ultimately leads to more meaningful insights and more robust, interpretable models.

---

## Data Preprocessing and Analysis Process

Following the CRISP-DM methodology, the data preparation phase included:

1. **Exploratory Data Analysis (EDA)**
   - Visualized relationships between variables using histograms, boxplots, and barplots.
   - Examined distributions and correlations to understand the data structure.

2. **Data Cleaning and Initial Feature Selection**
   - Removed unnecessary columns, handled null values (using KNN imputation), and detected outliers via the IQR method.
   - Eliminated features with a high percentage of null values or low variance, ensuring a clean dataset for analysis.

3. **Feature Construction and Transformation**
   - Created new columns from textual and date variables (e.g., `Signing Year`, `Closing Date Month`).
   - Normalized numerical variables and discretized selected variables (e.g., prices, deadlines) to enhance interpretability.
   - Maintained original feature semantics by avoiding PCA for dimensionality reduction.

4. **Dataset Integration**
   - Combined all cleaning, transformation, and feature selection steps into an optimized DataFrame to serve as the basis for clustering and association analysis.

---

## Cluster Analysis

### Objectives:
- Identify natural groupings of contracts based on:
  - **Contract Amounts**: Low, medium, and high cost.
  - **Awarding Entities**: Differentiating between municipalities, regions, ministries, etc.
  - **Timelines**: Variation in assignment and execution times.
  - **Geographical Regions**: Assessing differences across locations.

### Methods Employed:
- **Clustering Algorithms**:  
  - K-Means  
  - Agglomerative Clustering (with various linkage methods)  
  - DBSCAN
- **Dimensionality Reduction**:  
  - Techniques like UMAP were used to visualize high-dimensional data.
- **Evaluation Metrics**:  
  - Silhouette Score, intra-cluster, and inter-cluster metrics, along with visualizations (scatter plots, heatmaps).

### Key Findings:
- **Cluster Label 0**:  
  Represents low-priced contracts, primarily influenced by the "lowest price" criterion, with shorter submission and execution deadlines.
- **Cluster Label 1**:  
  Corresponds to mid-range contracts, with a balanced mix of award criteria and intermediate timelines.
- **Cluster Label 2**:  
  Includes contracts with medium-high prices, influenced by multifactorial criteria, and associated with longer deadlines.

Integrating cluster labels with association rule mining enriched the overall interpretation of contract characteristics and revealed distinct behavioral patterns among contracting entities.

---

## Association Rule Mining

### Objectives:
- Uncover hidden relationships between key variables, such as:
  - **Price and Award Criteria**:  
    - Multifactorial criteria often correlate with a higher effective price relative to the initial price.
    - Contracts based solely on the "lowest price" show minimal differences between initial and effective prices.
  - **Environmental Criteria and EU Publications**:  
    - Contracts considering environmental criteria are more likely to be published in an EU journal and attract more bidders.
  - **Price Differences**:  
    - Significant differences (e.g., >20%) between the initial and effective price may indicate justified modifications.
  - **Bidder Participation and Electronic Auctions**:  
    - A higher number of bidders is correlated with electronic auctions and competitive final prices.
  - **Lot-Based Contracts**:  
    - Contracts executed in lots tend to have higher effective prices and are often associated with increased complexity.

### Metrics Used:
- **Support**: Frequency of an itemset within the dataset.
- **Confidence**: Likelihood that the presence of one variable implies the presence of another.
- **Lift**: Measures the strength of an association (lift > 1 indicates a non-random, significant association).

### Notable Observations:
- Some association rules demonstrated strong relationships (e.g., a rule with 100% confidence and a lift of 9.33).
- Even low-support rules, when combined with high confidence and lift, confirmed significant patterns.
- Integrating association rules with cluster labels further validated the discovered patterns.

---

## Conclusions

The analysis revealed significant patterns through both clustering and association rule mining:

- **Clustering**:  
  - Distinct groupings were identified that correspond to differences in contract value, timelines, and award criteria.
  - The clusters provided semantic insights into contract behaviors, such as low-priced contracts (Cluster 0), balanced mid-range contracts (Cluster 1), and high-value contracts (Cluster 2).

- **Association Rules**:  
  - Key relationships between price, award criteria, deadlines, and bidder participation were uncovered.
  - The integration of association rules with clustering results deepened the understanding of the underlying data patterns.

Together, these analyses enable informed decision-making in public procurement, offering strategic insights into bidding and contract management.

---

## Possible Improvements and Next Steps

1. **Clustering Optimization**:  
   - Further refine variable selection and test additional combinations to determine the optimal number of clusters.
2. **Alternative Feature Selection**:  
   - Explore methods based on variable importance or dimensionality reduction to enhance clustering quality.
3. **Alternative Preprocessing Techniques**:  
   - Investigate preprocessing methods that maintain original feature values without discretization.
4. **Advanced Clustering Algorithms**:  
   - Consider using algorithms like HDBSCAN to handle irregular cluster shapes and noise.
5. **Enhanced Model Evaluation**:  
   - Evaluate clustering models using external metrics (precision, recall, F1, purity index, Gini index) based on target variables.
6. **Data Enrichment**:  
   - Integrate additional open-source data to deepen the analysis and uncover new patterns.
7. **Advanced Text Analysis**:  
   - Replace TF-IDF with models like BERT for better semantic analysis of textual data and improve clustering with techniques such as BERTopic.
8. **Extended Association Rule Analysis**:  
   - Continue exploring association rules both with and without cluster labels to validate and expand on current findings.
9. **Search for Rare Patterns**:  
   - Utilize functions like **top_rules_high_confidence_low_support** to discover rare, significant patterns across multiple variables.
10. **Explainability of Cluster Results**:  
    - Enhance the interpretability of clusters by providing clearer semantic explanations of the observed associations.

---

## Final Remarks

This project demonstrates the effective use of clustering and association rule mining to uncover hidden patterns in Portuguese public construction contracts. The insights gained can support strategic decision-making in public procurement and offer a foundation for further exploration and model refinement.