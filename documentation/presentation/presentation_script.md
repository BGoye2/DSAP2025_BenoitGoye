# Presentation Script: Machine Learning for Income Inequality
**Duration:** 10-15 minutes
**Presenter:** Benoit Goye

---

## SLIDE 1: Title Slide (30 seconds)

Good morning/afternoon everyone. Today I'll be presenting my research on identifying the key socioeconomic determinants of income inequality using machine learning approaches with World Bank data.

This work was conducted as part of the Data Science and Advanced Programming course at the University of Lausanne.

---

## SLIDE 2: Outline (15 seconds)

I've structured this presentation into five parts: First, I'll motivate the research problem. Second, I'll describe our data and methodology. Third, I'll present the main results. Fourth, I'll discuss implementation and optimization. And finally, I'll conclude with policy implications and future directions.

---

## SLIDE 3: The Challenge of Income Inequality (1 minute)

Let me start by explaining why income inequality matters and what makes it challenging to study.

Income inequality is a major economic challenge. Rising inequality is associated with reduced social mobility, political polarization, and even slower economic growth. The GINI coefficient, which ranges from 0 to 100, provides a standardized way to measure and compare inequality across countries.

Despite extensive research, we still debate which policy levers are most effective for reducing inequality. Should we focus on education? Infrastructure? Labor market reforms? Tax policy?

Traditional econometric approaches face several challenges: First, relationships between socioeconomic factors and inequality are often non-linear with complex interactions. Second, we have high-dimensional data with many potential predictors, leading to multicollinearity. Third, many indicators have missing values, especially for developing countries.

This is where machine learning can help.

---

## SLIDE 4: Research Questions (45 seconds)

This study addresses three interrelated research questions:

First, which socioeconomic indicators are the strongest predictors of GINI coefficients? We have about 60 variables from the World Bank covering economic, demographic, labor market, infrastructure, and governance dimensions.

Second, are predictor rankings consistent across different machine learning algorithms? Or do different models identify different drivers?

Third, how well can we actually predict inequality, and does performance vary across different development contexts?

Our contribution is a comprehensive data-driven approach using five distinct ML algorithms with robust statistical validation, while implementing advanced optimization techniques that achieve 3 to 5 times speedup.

---

## SLIDE 5: Data Overview (1 minute)

Let me describe our data and preprocessing.

We use data from the World Bank Open Data API covering 2000 to 2023. Our target variable is the GINI coefficient, giving us approximately 1,800 country-year observations.

For predictors, we extracted about 60 socioeconomic indicators spanning six categories: economic indicators like GDP and trade, demographics like population and urbanization, human development measures like health and education expenditure, labor market conditions, infrastructure access including electricity and internet penetration, and governance quality indicators.

For preprocessing, we used KNN imputation with k=5 neighbors for missing values, but excluded features missing in more than 50% of observations. We used an 80-20 train-test split and 5-fold cross-validation for all models.

The diagram on the right shows our data partition strategy.

---

## SLIDE 6: Five Tree-Based Algorithms (1 minute)

We implement five tree-based machine learning algorithms ranging from simple to sophisticated.

Our baseline is a Decision Tree with recursive partitioning. Then we have Random Forest, which uses bootstrap aggregating—or bagging—combining 200 independent trees trained on random subsets of data and features. This reduces variance.

Gradient Boosting takes a different approach, building trees sequentially where each new tree fits the residuals from previous iterations. This reduces bias.

XGBoost enhances gradient boosting with a regularized objective function and second-order optimization, providing better accuracy and faster convergence.

Finally, LightGBM achieves superior computational efficiency through histogram-based split finding and gradient-based one-side sampling, or GOSS.

All ensemble methods use 200 estimators with a learning rate of 0.05 and various regularization parameters to prevent overfitting.

---

## SLIDE 7: Statistical Validation Framework (1 minute)

To ensure our findings are robust, we employ three complementary validation approaches.

First, bootstrap confidence intervals: we train each model 100 times on bootstrap samples and compute 95% confidence intervals for feature importance. We use parallel processing here, achieving a 6x speedup. Features whose confidence intervals exclude zero are considered statistically significant.

Second, permutation importance tests: we randomly permute each feature's values 50 times and measure the drop in R-squared. We then use a one-sample t-test to assess significance.

Third, cross-model consistency: we calculate Spearman rank correlations between feature importance rankings across all model pairs. High correlations above 0.7 indicate that rankings are robust across different modeling approaches.

---

## SLIDE 8: Segmentation Analysis (45 seconds)

An important aspect of our analysis is testing whether inequality drivers are universal or context-dependent.

We segment countries by income level using GDP per capita quartiles—creating low, lower-middle, upper-middle, and high-income groups. We also segment by geographic region following World Bank classifications.

For each segment, we train models separately and examine whether predictive performance and feature importance vary systematically.

The key question is: can we generalize policy prescriptions, or must they be tailored to specific development stages and regional characteristics?

---

## SLIDE 9: Model Performance (1 minute)

Now let's turn to the results. This table shows predictive performance across all five models.

The key finding is clear: ensemble methods substantially outperform the single Decision Tree. While the Decision Tree achieves an R-squared of only 0.787, all ensemble methods exceed 0.88.

Gradient Boosting achieves the best overall performance with R-squared of 0.905 and RMSE of 2.58 GINI points, followed closely by XGBoost at 0.902.

Importantly, the cross-validation scores align closely with test performance, which validates that our models generalize well beyond training data and aren't overfitting.

This suggests that income inequality is substantially more predictable from observable socioeconomic factors than traditional methods might imply.

---

## SLIDE 10: Top 10 Features (1.5 minutes)

This brings us to perhaps our most important finding: which features matter most?

Looking at the table on the left, rural electricity access completely dominates with an importance of 0.288—that's more than five times larger than any other feature.

This is followed by total electricity access at 0.067, trade openness at 0.034, and then renewable energy, forest area, and economic structure variables.

Why does rural electricity access matter so much? We believe it captures multiple dimensions simultaneously: it's a proxy for economic development, institutional capacity to deliver public services, geographic integration, and technological diffusion. It's essentially a composite measure of the multidimensional processes that shape inequality.

Notice the highly skewed distribution: the top 10 features account for approximately 60% of total importance, while the bottom 30 features contribute less than 15% collectively. This suggests inequality is driven by a small subset of critical factors rather than diffuse contributions across all indicators.

Economic structure variables also rank highly, supporting Kuznets curve dynamics where structural transformation affects inequality. And gender labor gap shows significant importance, indicating that labor market inclusiveness matters substantially.

---

## SLIDE 11: Statistical Significance (1 minute)

All these findings are statistically robust.

Bootstrap confidence intervals confirm that all top 10 features have 95% confidence intervals excluding zero, with narrow intervals indicating high stability across resampling.

Permutation tests show that all top features cause significant performance degradation when their relationship with the target is broken—all with p-values less than 0.001.

For cross-model consistency, we find moderate overall agreement with a mean Spearman correlation of 0.47. However, the highest consistency occurs between Gradient Boosting and XGBoost at 0.84, which makes sense given their similar algorithmic approaches.

The conclusion is clear: our top predictors are robust across data resampling, different algorithms, and various statistical tests.

---

## SLIDE 12: Context Matters (1 minute)

Now, does context matter? Absolutely.

Predictive performance varies substantially by income level. In high-income and upper-middle-income countries, we achieve R-squared values between 0.85 and 0.91—very strong performance. But in low-income countries, R-squared drops to 0.69-0.73. This reflects both data quality differences and more diverse inequality drivers in developing contexts.

More importantly, feature importance exhibits strong context-dependence. In low-income countries, rural electricity, freshwater access, and basic infrastructure rank highest—reflecting agrarian economies where access to basic services drives inequality.

In lower-middle-income countries, urbanization growth becomes dominant at 0.214, along with employment structure—consistent with Kuznets curve dynamics during industrialization.

In high-income countries, exports at 0.339, healthcare expenditure, and labor market dynamics dominate—reflecting post-industrial service economies where human capital and labor market institutions shape inequality.

The policy implication is clear: effective interventions must be tailored to country context. Developing economies should prioritize infrastructure investments, industrializing countries should emphasize education and skills training, and advanced economies should strengthen labor market institutions.

---

## SLIDE 13: Model Diagnostics (30 seconds)

Quick diagnostics confirm strong model performance.

The predicted versus actual plots show strong correlations above 0.90 for boosting methods, with tight clustering around the 45-degree line for GINI values between 20 and 45. However, models tend to underpredict extreme inequality above GINI 50, suggesting that extreme inequality involves factors we haven't fully captured.

Residual analysis shows approximately normal distributions centered near zero with symmetric tails, indicating no major specification issues, though some heteroskedasticity persists at mid-range values.

---

## SLIDE 14: Computational Performance (45 seconds)

Let me briefly highlight our implementation optimizations, which are important for reproducibility and scalability.

Through parallel processing strategies, we achieved substantial speedups. Data collection went from 12 minutes to 2 minutes using ThreadPoolExecutor with 6 workers. Bootstrap tests dropped from 60 seconds to 10 seconds using joblib parallelization. Model training saw a 5x speedup using scikit-learn's built-in parallelization.

Overall, the complete pipeline executes in 5-7 minutes compared to 25-35 minutes without optimizations—a 3 to 5 times overall speedup.

We also implemented intelligent caching with SHA256 hash validation, providing 60 to 100 times speedup on reruns while ensuring data-model consistency.

---

## SLIDE 15: Software Engineering (30 seconds)

The codebase follows best practices for reproducible research.

We have a modular architecture with 9 main scripts, each with a single well-defined responsibility. We provide four execution modes: quick, fast, optimized, and custom, accommodating different use cases.

We use comprehensive version control with Git, dependency management through conda and pip, type annotations, comprehensive docstrings, and data hash validation to prevent silent errors.

All code and data are publicly available for replication.

---

## SLIDE 16: Key Findings Summary (1 minute)

Let me summarize the key findings.

First, ensemble methods achieve R-squared above 0.90, demonstrating that inequality is highly predictable from socioeconomic factors. Gradient Boosting and XGBoost perform best.

Second, rural electricity access dominates with importance 0.304—five times more important than any other feature. It serves as a composite measure of development, institutions, and connectivity.

Third, we identify robust core drivers including trade openness, economic structure, and gender gaps. These are consistent across all five models and statistically significant in all tests.

Fourth, we find strong context-dependence: different income levels have different drivers. Low-income countries are driven by basic infrastructure, middle-income by urbanization and employment, and high-income by trade and labor markets.

---

## SLIDE 17: Policy Implications (1 minute)

These findings suggest three priority areas for reducing inequality.

First, expand infrastructure access, particularly rural electricity connectivity. This enables economic opportunities, facilitates education access, and signals institutional capacity for equitable service delivery.

Second, promote inclusive labor markets by reducing gender labor gaps, increasing labor force participation especially for women, and addressing youth unemployment.

Third, manage structural transformation as economies industrialize through progressive taxation, robust social insurance systems, education access that scales with industrial demands, and minimum wage floors to prevent excessive wage compression.

The key message is that these priorities should be calibrated to each country's development stage.

---

## SLIDE 18: Limitations & Future Directions (1 minute)

Of course, important limitations warrant caution.

Most importantly, our analysis is predictive, not causal. While we identify strong associations, establishing causal effects requires stronger identification strategies like instrumental variables, difference-in-differences, or regression discontinuity designs.

Missing data imputation introduces uncertainty we don't fully quantify. Our stationarity assumption may mask important dynamics as relationships change over time. And we lack some important variables like detailed institutional quality measures and tax progressivity.

Future research should combine machine learning with causal inference through techniques like double machine learning or causal forests. We should incorporate additional data sources like satellite imagery and institutional quality measures. And methodological advances through SHAP values could reveal feature interactions, while panel methods could capture how drivers evolve over time.

The bottom line is that ML complements traditional econometrics: it excels at prediction and pattern recognition, while econometrics provides causal identification and hypothesis testing.

---

## SLIDE 19: Contributions (45 seconds)

To conclude, this study makes both methodological and substantive contributions.

Methodologically, this is the first systematic application of five tree-based ML algorithms to inequality with robust statistical validation. We achieve 3 to 5 times speedup through advanced optimization and provide a reproducible pipeline with comprehensive version control.

Substantively, we show that infrastructure emerges as the dominant predictor, we demonstrate context-dependence of inequality drivers, and we achieve high predictability suggesting systematic patterns rather than random variation. Our policy recommendations are tailored to development stage.

All data and code are publicly available for replication.

---

## SLIDE 20: Thank You (15 seconds)

Thank you for your attention. I'm happy to take any questions.

[Pause for questions]

---

## ANTICIPATED QUESTIONS & ANSWERS

### Q1: "Why tree-based methods instead of neural networks or other ML approaches?"

**A:** Great question. We chose tree-based methods for three reasons. First, they provide interpretable feature importance rankings, which is crucial for policy insights. Second, they handle missing values and non-linear relationships naturally without extensive preprocessing. Third, they've proven highly effective for tabular data in similar economic applications. Neural networks would require much larger datasets and provide less interpretability. However, exploring deep learning with embeddings for country and time effects would be an interesting extension.

### Q2: "How do you address endogeneity concerns?"

**A:** That's the key limitation of our approach. We're explicitly doing prediction, not causal inference. The high feature importance of rural electricity doesn't mean building power plants will reduce inequality—it could be reverse causality where more equal societies invest more in infrastructure. To establish causality, we'd need instrumental variables, natural experiments, or quasi-experimental designs. Double machine learning, which combines ML for nuisance parameters with causal inference for treatment effects, is a promising direction we mention in future work.

### Q3: "Are your results sensitive to the imputation method?"

**A:** We use KNN imputation with k=5, which is standard practice and generally performs well. We also exclude features missing in more than 50% of observations. Sensitivity analysis with different imputation methods (mean imputation, multiple imputation, MissForest) would strengthen the results. The fact that our top features show narrow bootstrap confidence intervals suggests they're robust to data perturbations, but formal sensitivity analysis to imputation is worth pursuing.

### Q4: "Why does rural electricity matter so much more than other infrastructure?"

**A:** That's a fascinating finding. We believe rural electricity serves as a composite indicator capturing multiple dimensions: (1) economic development—wealthier countries can afford rural electrification, (2) institutional capacity—governments able to deliver services to remote areas, (3) geographic integration—connecting rural and urban economies, (4) technology diffusion—enabling modern economic activities. It's particularly salient for inequality because it represents whether development benefits reach beyond urban centers. Total electricity access matters too but has less variation since most countries have high urban access.

### Q5: "How would your recommendations change for a specific country?"

**A:** Great question. Let's take a concrete example. For a low-income Sub-Saharan African country, our analysis suggests prioritizing basic infrastructure—rural electricity, water access, road connectivity. For a middle-income Southeast Asian country undergoing industrialization, the focus should shift to managing urbanization, providing vocational training, and strengthening labor market institutions. For a high-income European country, emphasis should be on progressive taxation, addressing labor market dualities, and ensuring trade benefits are broadly shared. The segmentation analysis provides these context-specific insights.

### Q6: "What's the computational cost of running this analysis?"

**A:** With our optimizations, the full pipeline runs in 5-7 minutes on a modern workstation with parallel processing. Without optimizations, it would take 25-35 minutes. The bootstrap validation is the most expensive component, but we cache results so reruns are instant. For researchers wanting to replicate or extend this work, we provide four execution modes including a "fast" mode using 2015-2023 data that runs in about 3 minutes. All code is documented and publicly available.

### Q7: "How does this compare to previous work on inequality?"

**A:** Most previous work uses traditional econometric approaches—panel regressions with country and time fixed effects. These provide causal interpretation under strong assumptions but struggle with non-linearities and interactions. Our ML approach is complementary: we're optimized for prediction and pattern discovery, which can guide where to look for causal relationships. Some recent work uses random forests for GDP prediction or neural networks for poverty mapping from satellite data, but systematic ML application to inequality with robust validation has been limited. We're filling that methodological gap.

---

**TIMING BREAKDOWN:**
- Introduction & Motivation: 2.5 minutes
- Data & Methodology: 4 minutes
- Results: 4.5 minutes
- Implementation: 1.5 minutes
- Conclusions & Policy: 2.5 minutes
- **Total: ~15 minutes**

**TIPS FOR DELIVERY:**
1. Speak slowly and clearly—this is technical content
2. Make eye contact with audience, not just reading slides
3. Use the pointer to highlight specific numbers in tables
4. Pause after key findings to let them sink in
5. Show enthusiasm when presenting the main results (Slide 10)
6. If running short on time, compress Slides 14-15 (implementation)
7. If running over, skip detailed discussion of validation (Slide 11)
8. Save 2-3 minutes for questions
