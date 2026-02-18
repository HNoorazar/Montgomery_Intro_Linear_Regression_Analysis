# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3.13.11 (Conda)
#     language: python
#     name: py313
# ---

# %%
import sys
print(sys.version)
print(sys.executable)

# %%
import scipy
print(scipy.__version__)

# %%
import numpy as np
import pandas as pd

import scipy
from scipy import stats
from sklearn import preprocessing

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

from sklearn.linear_model import LinearRegression

# spatial regression package. 
# I like the way its output looks like,
# compared to statsmodels
# import spreg

import matplotlib
import matplotlib.pyplot as plt

# %%
database = "/Users/hn/Documents/01_research_data/Montgomery/"

# %% [markdown]
# # Question 1

# %%
B1_NFL1976 = pd.read_csv(database + "B1_NFL1976.csv")
# B1_NFL1976 = B1_NFL1976.rename(columns={
#     "y": "wins",
#     "x1": "rushing_yards",
#     "x2": "passing_yards",
#     "x3": "punting_avg",
#     "x4": "field_goal_pct",
#     "x5": "turnover_diff",
#     "x6": "penalty_yards",
#     "x7": "pct_rushing",
#     "x8": "opp_rushing_yard",
#     "x9": "opp_passing_yard"
# })
# B1_NFL1976.to_csv(database + 'B1_NFL1976.csv', index=False)
B1_NFL1976.head(2)

# %%

# %%
X = sm.add_constant(B1_NFL1976["opp_rushing_yard"])
model = sm.OLS(B1_NFL1976["wins"], X)
result = model.fit()
print(result.summary())

# %% [markdown]
# ### part b.

# %%
# this one includes intercept on its own
model_smf = smf.ols("wins ~ opp_rushing_yard", data=B1_NFL1976)
result_smf = model_smf.fit()
anova_table = sm.stats.anova_lm(result_smf, typ=2)
anova_table

# %%
result_smf.summary()

# %%
RSS = anova_table.loc["Residual", "sum_sq"]

# %% [markdown]
# # Lesson:
#
#
#  - ```sm.OLS``` is the array-based API --> it expects numeric matrices/arrays.
#
#  - ```ols``` (from ```statsmodels.formula.api```) is the formula-based API --> it expects a formula string, not arrays.
#
# And, ANOVA table uses the ols from ```statsmodels.formula.api```:
# https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
#
# ```result_smf.predict(X)``` from ```smf``` works just like ```result.predict(X)```. The same is true about ```conf_int()```
#
# ```get_prediction``` works like so:
#
# ```
# pred = result.get_prediction([1, x0])
# ci = pred.summary_frame(alpha=0.05)
# ```
#
# or from ```smf```:
#
# ```
# new_data = pd.DataFrame({"opp_rushing_yard": [x0]})
# pred = result_smf.get_prediction(new_data)
# ci = pred.summary_frame(alpha=0.05)
# ```

# %% [markdown]
# ## Lets stick to ```smf```
#
# ## Working with ```smf```
#
# Model like so
#
# ```
# mpg_model = smf.ols('mpg ~ engine_displacement', data = mpg_df)
# mpg_result = mpg_model.fit()
# ```
#
# - It automatically adds intercept.
# - ```mpg_result.summary()``` Produces summary that includes CI of coefficients at 95% sigfinicance level
# - ```sm.stats.anova_lm(mpg_result, typ=2)``` creates analysis-of-variance table
# - ```mpg_result.conf_int()``` shows the CI from the ```.summary()```
# - ```mpg_result.conf_int(alpha=0.01)``` shows CI at 99% significance level
# - RSS can be accessed in 2 ways:
#   - ```(mpg_result.resid ** 2).sum()```
#   - ```mpg_anova_tbl.loc["Residual", "sum_sq"]```
#   
# - To predict at new values of ```x``` we need a dataframe:
#    - new_data = pd.DataFrame({"engine_displacement": [x0]})
#    - predict_table = mpg_result.get_prediction(new_data)
#    - The above result is a table with predictions, CIs and PIs.
#    - ```yhat_tbl.summary_frame(alpha=0.01)```
#    - Predicted values are obtained by ```yhat_tbl.predicted_mean[0]```

# %% [markdown]
# ### Part C. done manually and using the results

# %%
result_smf.conf_int()

# %% [markdown]
# ### manual confidence interval.
#
# Confidence interval: $(\text{stat} - t_{\alpha/2, n-2}\text{SE}(\text{stat}), \text{stat} +  t_{\alpha/2, n-2} \text{SE}(\text{stat}))$ where stat here is the slope $\beta_1$.
#
# Variance of $\hat \beta_1 = \frac{\sigma^2}{s_{xx}}$
# where $s_{xx} = \sum(x_i - \bar x)^2$. 
#
# **Note** In other books $s_{xx}$ might be degined differently.
#
# Since we do not know $\sigma^2$ we estimate it by $\hat \sigma^2 = \frac{\text{RSS}}{df} = \frac{\text{RSS}}{n-2} = \text{MS}_{\text{Res}}$, and variance becomes SE and normal distribution becomes t-distribution and so on. and we end up with
#
# $$(\hat \beta_1 - t_{\alpha/2, n-2} \text{SE}(\hat \beta_1), \hat \beta_1 + t_{\alpha/2, n-2} \text{SE}(\hat \beta_1)),$$
# where $\text{SE}(\hat \beta_1) = \sqrt{\text{MS}_{Res} / S_{xx}} = \sqrt{\frac{RSS} {(n-2) \times S_{xx}}}$

# %%

# %%
degr_of_fr = len(B1_NFL1976)-2

x = "opp_rushing_yard"
x_mean = B1_NFL1976[x].mean()
S_xx = sum((B1_NFL1976[x] - x_mean)**2)

SE_B1 = np.sqrt(RSS / (degr_of_fr * S_xx))
slope = result.params[x]

alpha_95 = stats.t.ppf(0.975, df=degr_of_fr)
alpha_95

low = float(slope - alpha_95 * SE_B1)
hi = float(slope + alpha_95 * SE_B1)
[round(low, 4), round(hi, 4)]

# %%

# %% [markdown]
# ### Part (d)
#
# We see from above that $R^2 = 0.545$
#
# $$R^2 = \frac{\text{SS}_R}{\text{SS}_T} = \frac{\sum (\hat y_i - \bar y)^2}{\sum (y_i - \bar y) ^ 2} = 1 - \frac{\text{RSS}}{\text{SS}_T}$$

# %%
yhats = result.predict(X)

# %%
total_variability = sum((B1_NFL1976["wins"] - B1_NFL1976["wins"].mean())**2)
round(total_variability, 2)

# %%
remaining_variability = sum((yhats - B1_NFL1976["wins"].mean())**2)
round(remaining_variability, 2)

# %%
variability_explained = total_variability - remaining_variability
variability_explained_pct = 100 * (variability_explained / total_variability)
round(variability_explained_pct, 2)

# %%
## If we take R^2 route: R2 = 1 - RSS / SS_T:
## RSS: measure of variability after X has been considered
round(1 - result.ssr / total_variability, 4)

# %% [markdown]
# ### part 1e.
#
# We are looking for CI of a $y$ at $x_0$. From Eq. 2.43 on page 31 we have
#
# Let 
#
# $$\text{SE}(\widehat{E[Y | x_0]}) = \sqrt{\frac{\text{RSS}}{n-2} \left(\frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}} \right)}$$
#
# then we have:
#
# $$ E[y | x_0] \in \left(\hat \mu_{y | x_0} - t_{\alpha/2, n-2} \text{SE}, \hat \mu_{y | x_0} + t_{\alpha/2, n-2} \text{SE} \right) = \text{CI}$$
#
# Lets do it manually and using ```stats``` way

# %%
x0 = 2000
pred = result.get_prediction([1, x0])
ci = pred.summary_frame(alpha=0.05)
ci

# %%
yhat_2000 = result.predict([1, x0])[0]
SE = np.sqrt( (RSS /degr_of_fr) * (1/(degr_of_fr+2)) + (x0 - x_mean)**2 / (S_xx)  )
alpha_SE = alpha_95 * SE

# %%
lower_CI_y2000 = round(yhat_2000 - alpha_SE, 3)
upper_CI_y2000 = round(yhat_2000 + alpha_SE, 3)
[lower_CI_y2000, upper_CI_y2000]

# %% [markdown]
# # Question 2. Prediction interval. 
#
# They are asking at 90\% significance level.
#
# Again we do it in 2 ways. Recall:
#
# Let 
#
# $$\text{SE}(y_0 - \hat y_0) = \sqrt{\frac{\text{RSS}}{n-2} \left(1 + \frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}} \right)}$$
#
# then we have:
#
# $$ y_0 \in \left(\hat \mu_{y | x_0} - t_{\alpha/2, n-2} \text{SE}, \hat \mu_{y | x_0} + t_{\alpha/2, n-2} \text{SE} \right) = \text{PI}$$
#
# Lets do it manually and using ```stats``` way

# %%
x0 = 1800
new_data = pd.DataFrame({"opp_rushing_yard": [x0]})
pred = result_smf.get_prediction(new_data)
ci = pred.summary_frame(alpha=0.1)
ci

# %%
yhat_1800 = result.predict([1, x0])[0]

alpha_90 = stats.t.ppf(0.95, df=degr_of_fr)
SE = np.sqrt( (RSS /degr_of_fr) * (1 + 1/(degr_of_fr+2)) + (x0 - x_mean)**2 / (S_xx) )
alpha_SE = alpha_90 * SE

# %%
lower_PI_y1800 = round(yhat_1800 - alpha_SE, 3)
upper_PI_y1800 = round(yhat_1800 + alpha_SE, 3)
[lower_PI_y1800, upper_PI_y1800]

# %%
Rocket_propellant = pd.read_csv(database + "Rocket_propellant.csv")
Rocket_propellant.head(2)

# %%
y = Rocket_propellant["Shear strength"]
ybar = y.mean()

x = Rocket_propellant["Age of propellant"]
xbar= x.mean()


# %%
sum((y - ybar) * (x - xbar))

# %%
sum(y * (x - xbar))

# %%
sum(ybar * (xbar - x))

# %%
del(x)

# %% [markdown]
# # Question 2.3
#
# The text in question says $x_4$ is radial deflection of the deflected rays but the text under that table in Appendix says differently (position of focal point in north direction.)
#
# At any rate! Lets compute $\beta_0$ and $\beta_1$ manually as a review. It was in stackPile(?) interview question. Ridiculous.
#
# $$\beta_1 = \frac{s_{xy}}{s_{xx}} = \frac{\sum y_i (x_i - \bar x)}{\sum (x_i - \bar x)^2} =  \frac{\sum (y_i - \bar y) (x_i - \bar x)}{\sum (x_i - \bar x)^2}$$
#
# Last equality: It turns out $\sum \bar y (x_i - \bar x) = 0$.
#
#
# **Expanded version**
#
# $$S_{xy} = \sum x_i y_i - \frac{\sum x_i \sum y_i}{n}, S_{xx} = \sum x_i^2 - \frac{\left(\sum x_i \right)^2}{n} $$
#
# **Part a**

# %%
B2_SolarThEn = pd.read_csv(database + "B2_SolarThEn.csv")
B2_SolarThEn.head(2)

# %%
SolarTh_model_smf = smf.ols("total_heat_flux ~ focal_north", data=B2_SolarThEn)
SolarTh_result_smf = SolarTh_model_smf.fit()
SolarTh_result_smf.summary()

# %%
y = B2_SolarThEn['total_heat_flux']
x = B2_SolarThEn['focal_north']
s_xy = sum(y*x) - (sum(y) * sum(x) / len(y))
s_xx = sum(x*x) - sum(x) * sum(x) / len(y)

beta_1 = s_xy / s_xx
beta_0 = y.mean() - beta_1 * x.mean()

[round(beta_0, 3), round(beta_1, 3)]

# %% [markdown]
# **part b: ANOVA**

# %%
sm.stats.anova_lm(SolarTh_result_smf, typ=2)

# %% [markdown]
# **Part C: 99% CI on slope**

# %%
SolarTh_result_smf.conf_int()

# %%
df = len(y) - 2

# RSS: Reminder: RSS = sm.stats.anova_lm(SolarTh_result_smf, typ=2).loc["Residual", "sum_sq"]
rss = (SolarTh_result_smf.resid ** 2).sum()
MS_res = rss / df

SE = np.sqrt(MS_res / s_xx)
alpha_99 = stats.t.ppf(0.995, df=degr_of_fr)

[beta_1 - alpha_99 * SE, beta_1 + alpha_99 * SE]

# %%
# to double check with the package output.
df = len(y) - 2

# RSS: Reminder: RSS = sm.stats.anova_lm(SolarTh_result_smf, typ=2).loc["Residual", "sum_sq"]
rss = (SolarTh_result_smf.resid ** 2).sum()
MS_res = rss / df

SE = np.sqrt(MS_res / s_xx)
alpha_95 = stats.t.ppf(0.975, df=degr_of_fr)

[beta_1 - alpha_95 * SE, beta_1 + alpha_95 * SE]

# %% [markdown]
# **Part D: $R^2$**
#
# $$R^2 = \frac{\text{SS}_R}{\text{SS}_T} = \frac{\sum (\hat y_i - \bar y)}{\sum (y_i - \bar y)} = 1 - \frac{\text{RSS}}{\text{SS}_T}$$

# %%
print(f"Python R^2 = {SolarTh_result_smf.rsquared:.3f}")

SS_T = sum((y - y.mean())**2)
print(f"Manual R^2 = {1 - rss/SS_T:.3f}")

# %% [markdown]
# **Part E: 95% CI on mean heat flux when radial deflection is 16.5**
#
# Once again
#
# We are looking for CI of a $y$ at $x_0$. From Eq. 2.43 on page 31 we have
#
# Let 
#
# $$\text{SE} = \sqrt{\frac{\text{RSS}}{n-2} \left(\frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}} \right)}$$
#
# then we have:
#
# $$ E[y | x_0] \in \left(\hat \mu_{y | x_0} - t_{\alpha/2, n-2} \text{SE}, \hat \mu_{y | x_0} + t_{\alpha/2, n-2} \text{SE} \right) = \text{CI}$$
#
# Lets do it manually and using ```stats``` way

# %%
x0 = 16.5
new_data = pd.DataFrame({"focal_north": [x0]})
yhat_165 = SolarTh_result_smf.get_prediction(new_data)
ci = yhat_165.summary_frame(alpha=0.05)
ci

# %%
alpha_95 = stats.t.ppf(0.975, df=df)
SE = np.sqrt(MS_res * (1/df + (x0 - x.mean())/s_xx))

alpha_95_SE = alpha_95 * SE

yhat_165 = ci['mean'][0]

lower_CI_y165 = round(yhat_165 - alpha_95_SE, 3)
upper_CI_y165 = round(yhat_165 + alpha_95_SE, 3)
[lower_CI_y165, upper_CI_y165]

# %% [markdown]
# # Question 2.4

# %%
mpg_df = pd.read_csv(database + "B3_gasoline_mileage.csv")

mpg_df = mpg_df.rename(columns={"y" : "mpg", 
                                'x1': "engine_displacement",
                                'x2': "hp", 
                                'x3': "torque", 
                                'x4': "compression_ratio", 
                                'x5': "rear_axle_ratio",
                                'x6': "carburetor",
                                'x7': "num_trans_speeds",  
                                'x8': "length", 
                                'x9': "width", 
                                'x10': "weight", 
                                'x11': "transmission_type"
                               }
                      )

mpg_df.head(2)

# %% [markdown]
# ### Part a

# %%
mpg_model = smf.ols('mpg ~ engine_displacement', data = mpg_df)
mpg_result = mpg_model.fit()
mpg_result.summary()

# %% [markdown]
# ### Part b

# %%
mpg_anova_tbl = sm.stats.anova_lm(mpg_result, typ=2)
mpg_anova_tbl

# %%
# RSS two ways:
rss = (mpg_result.resid ** 2).sum()
print (round(rss, 2))

mpg_anova_tbl.loc["Residual", "sum_sq"].round(2)

# %% [markdown]
# #### 99% CI on slope
#
# Recall:
#
#  - $\text{SE}(\hat \beta_1) = \sqrt{\text{MS}_{Res} / S_{xx}} = \sqrt{\frac{RSS} {(n-2) \times S_{xx}}}$
#  - SE for mean of $y$ at a given $x_0$: $\text{SE}(\widehat{E[Y | x_0]}) = \sqrt{\frac{\text{RSS}}{n-2} \left(\frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}} \right)}$
#  - SE for prediction interval: $\text{SE}(y_0 - \hat y_0) = \sqrt{\frac{\text{RSS}}{n-2} \left(1 + \frac{1}{n} + \frac{(x_0 - \bar x)^2}{S_{xx}} \right)}$
#
#

# %%
# The followins is 95% CIs
mpg_result.conf_int()

# %%
mpg_result.conf_int(alpha=0.01)

# %%
####
#### Manual
####
degr_of_fr = len(mpg_df) - 2
alpha_99 = stats.t.ppf(0.995, df=degr_of_fr)

x = "engine_displacement"
S_xx = sum((mpg_df[x] - mpg_df[x].mean())**2)
SE_slope = np.sqrt(rss / (degr_of_fr * S_xx))
SE_slope

slope = mpg_result.params[x]
low_CI = slope - alpha_99 * SE_slope
hi_CI = slope + alpha_99 * SE_slope

[round(low_CI, 4), round(hi_CI, 4)]

# %% [markdown]
# ### Part C. 
#
# What percentage of total variability in MPG is accounted for by linear relatinship with engine displacement?
#
# I am not sure if this wording means they want $R^2$ or I need to compute total variability, the variability after regression as a percentage of total variability.

# %%
SS_T = sum((mpg_df["mpg"] - mpg_df["mpg"].mean())**2)
SS_R = sum((mpg_result.predict() - mpg_df["mpg"].mean()) ** 2)
print (round(SS_R/SS_T, 3))
print (mpg_result.rsquared.round(3))

# %% [markdown]
# ### Part d
#
# 95% confidence interval on prediction $x_0 = 275$

# %%
x0 = 275
new_data = pd.DataFrame({x: [x0]})
yhat_tbl = mpg_result.get_prediction(new_data)
ci = yhat_tbl.summary_frame(alpha=0.05)
ci

# %%
yhat = yhat_tbl.predicted_mean[0]

# %%
x_mean = mpg_df[x].mean()
alpha_95 = stats.t.ppf(0.975, df=degr_of_fr)

MS_res = rss / degr_of_fr
SE_CI_pred = np.sqrt(MS_res * (1/len(mpg_df) + ( (x0 - x_mean)**2/S_xx )))

low_CI = yhat - alpha_95*SE_CI_pred
hi_CI = yhat + alpha_95*SE_CI_pred

[round(low_CI, 3), round(hi_CI, 3)]

# %% [markdown]
# ## Recall: Summary of working with ```smf```
#
# Model like so
#
# ```python
# mpg_model = smf.ols('mpg ~ engine_displacement', data = mpg_df)
# mpg_result = mpg_model.fit()
# ```
#
# - It automatically adds intercept.
# - ```mpg_result.summary()``` Produces summary that includes CI of coefficients at 95% sigfinicance level
# - ```sm.stats.anova_lm(mpg_result, typ=2)``` creates analysis-of-variance table
# - ```mpg_result.conf_int()``` shows the CI from the ```.summary()```
# - ```mpg_result.conf_int(alpha=0.01)``` shows CI at 99% significance level
# - RSS can be accessed in 2 ways:
#   - ```(mpg_result.resid ** 2).sum()```
#   - ```mpg_anova_tbl.loc["Residual", "sum_sq"]```
#   
# - To predict at new values of ```x``` we need a dataframe:
#    - ```new_data = pd.DataFrame({"engine_displacement": [x0]})```
#    - ```predict_table = mpg_result.get_prediction(new_data)```
#    - The above result is a table with predictions, CIs and PIs.
#    - ```yhat_tbl.summary_frame(alpha=0.01)```
#    - Predicted values are obtained by ```yhat_tbl.predicted_mean[0]```

# %%
x0 = [275, 300]
new_data = pd.DataFrame({"engine_displacement": x0})
yhat_tbl = mpg_result.get_prediction(new_data)
print (yhat_tbl.predicted_mean)
yhat_tbl.summary_frame()

# %% [markdown]
# ### Part e and f 
#
# are similar to earlier problem. Do them when you like

# %%

# %% [markdown]
# # Question 2.5
#
# Repeat 2.4 but use $x_{10}$ (vehicle weight) as regression

# %%
mpg_df.head(2)

# %%
mpg_weight_model = smf.ols('mpg ~ weight', data = mpg_df)
mpg_weight_result = mpg_weight_model.fit()
mpg_weight_result.summary()

# %%
print (f"Displacement model has R2 of {float(mpg_result.rsquared.round(3))}")
print (f"Weight model has R2 of {float(mpg_weight_result.rsquared.round(3))}")

# %%
mpg_weight_anova_tbl = sm.stats.anova_lm(mpg_weight_result, typ=2)
mpg_weight_anova_tbl

# %%
mpg_anova_tbl

# %% [markdown]
# # Question 2.6
#
#   - **a.** Fit a simple linear regression model relating selling price of the house to
# the current taxes ($x_1$).
#   - **b.** Test for significance of regression.
#   - **c.** What percent of the total variability in selling price is explained by this model?
#   - **d.** Find a 95% CI on $\hat \beta_1$ .
#   - **e.** Find a 95% CI on the mean selling price of a house for which the current taxes are $750.

# %%
house_prices = pd.read_csv(database + "B4_house_prices.csv")
house_prices.head(2)

# %%
house_on_tax_model = smf.ols('sale_price_div1000 ~ taxes_div1000', data = house_prices)
house_on_tax_result = house_on_tax_model.fit()
house_on_tax_result.summary()

# %%
house_on_tax_anova = sm.stats.anova_lm(house_on_tax_result, typ=2)
house_on_tax_anova

# %% [markdown]
# **Recall** that in simple linear regression $t^2 = 8.518^2 = F = 72.55$
#
# - The t-statistic tests an individual regression coefficient. $t = \frac{\hat \beta_1}{\text{SE}(\hat \beta_1)}$
# - F-statistics tests the overall significance of the regression model. $F=\frac{\text{SSR}/1}{\text{SST}/(n-2)}$

# %%
print (fr"Variability explained by the model is R^2={float(house_on_tax_result.rsquared.round(3))}")

# %%
house_on_tax_result.bse # Beta Standard Error

# %% [markdown]
# Compute standard error manually and check.

# %%
SE_slope = float(house_on_tax_result.bse["taxes_div1000"].round(3))
SE_slope

# %%
RSS = float(house_on_tax_anova.sum_sq["Residual"].round(5))
RSS

# %%
dg_freedom = len(house_prices) - 2

x_ = "taxes_div1000"
x_mean = house_prices[x_].mean()
S_xx = sum((house_prices[x_] - x_mean)**2)

SE_B1 = np.sqrt(RSS / (dg_freedom * S_xx))
slope = house_on_tax_result.params[x_]

float(SE_B1.round(3))

# %%
alpha_95 = stats.t.ppf(0.975, df=dg_freedom)
low = float(slope - alpha_95 * SE_slope)
hi = float(slope + alpha_95 * SE_slope)
[round(low, 2), round(hi, 2)] 

# %%
# CI at x = 750
x0 = 750
new_data = pd.DataFrame({x_: [x0]})
pred = house_on_tax_result.get_prediction(new_data)
ci = pred.summary_frame(alpha=0.1)
ci

# %%
# Manual CI at x = 750
yhat0 = float(pred.summary_frame()["mean"][0])
yhat0

# %%
x_mean = house_prices[x_].mean()
SE = np.sqrt(RSS / dg_freedom * (1/(dg_freedom+2) + ((x0 - x_mean)**2/S_xx)))
SE = float(SE)
low = float(yhat0 - alpha_95 * SE)
hi = float(yhat0 + alpha_95 * SE)

[round(low, 2), round(low, 2)]

# %%
low

# %%
