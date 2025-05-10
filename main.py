import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV from project directory
df = pd.read_csv("sandwich.csv")

# Optional: preview the first few rows
print(df.head())

# Fit the ANOVA model
model = ols('antCount ~ C(bread) + C(topping) + C(butter)', data=df).fit()

# Run ANOVA
anova_result = sm.stats.anova_lm(model, typ=2)

# Print the result
print(anova_result)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey HSD for topping
tukey = pairwise_tukeyhsd(endog=df['antCount'], groups=df['topping'], alpha=0.05)
print(tukey)
print(df.groupby(['bread', 'topping', 'butter'])['antCount'].mean().sort_values())

# Set visual style
sns.set(style="whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot with individual data points
sns.boxplot(ax=axes[0], data=df, x="topping", y="antCount", palette="Set2")
sns.stripplot(ax=axes[0], data=df, x="topping", y="antCount", color="black", jitter=True, alpha=0.5)
axes[0].set_title("Boxplot of Ant Count by Topping")
axes[0].set_xlabel("Topping")
axes[0].set_ylabel("Ant Count")

# Violin plot
sns.violinplot(ax=axes[1], data=df, x="topping", y="antCount", palette="Set2", inner="box")
axes[1].set_title("Violin Plot of Ant Count by Topping")
axes[1].set_xlabel("Topping")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("boxplot_violinplot_antcount_topping.png", dpi=300)
plt.show()