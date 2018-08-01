#importing libraries 

#extracting data from csv
tips =pd.read_csv([the file containing csv])

#calculating tips 
tips['tip_pct'] = tips['tip'] / tips['total_bill']

#Pivoting the data based on smoker categories 
tips.pivot_table(['total_bill','size'], index =['time','day'], columns ='smoker')

#Treating the missing value 
tips.pivot_table(['total_bill','size'], index =['time','day'], columns ='smoker',fill_value=0)

#group means the pivot tables 
tips.pivot_table(['tip_pct', 'size'], index = ['time', 'day'], columns = 'smoker', margins=True, fill_value=0)

#importing visualising library 
import seaborn as sns

#visualising based on day & smokers vs non-smokers 
sns.countplot(x="day", hue="smoker", data=tips)

#importing equating matrix and regression libraries 
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

#converting the string to be 1 or 0 values 
tips.smoker.replace(('Yes', 'No'), (1, 0), inplace=True)

#Converting the array into equating matrix 
Y, X = patsy.dmatrices('total_bill~size+smoker+tip',tips)

#generating model
model=sm.OLS(Y,X)

#executing model 
model

#Displaying the regression estimation 
results =model.fit()
print(results.summary())
