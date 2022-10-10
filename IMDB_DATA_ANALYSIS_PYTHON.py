#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
pd.set_option('display.max_colwidth',None)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
from tqdm import tqdm

import scipy.stats as stats
import plotly.express as px
from plotly.offline import iplot
import plotly.figure_factory as ff


# In[2]:


df = pd.read_excel (r'250_IMDB_final.xlsx')


# In[3]:


print(df.shape)

print(df.columns)

display(df.head())


# # Cleaning and Feature Engineering

# In[4]:


#(Making Budget same scale), This should always be the first step before the null value imputation (we call this here as cleaning
#Feature Engineering:  Creating Country column


# In[5]:


# Budget is in Origin Currency, using column "Origin" to convrt budget to US currency.

df["Origin_Currency_Symbol"] = df["Origin_Currency_Symbol"].fillna("$")
#df["Origin_Currency_Symbol"] 


# In[6]:


symbol=[]
for i in range(df.shape[0]):
    a = re.findall(r'\D',df['Origin_Currency_Symbol'][i])
    
    
    if '$' in a :
        symbol.append(1)
        
    if 'M' in a and 'V' in a and 'R' in a :
        symbol.append(650000)
        
    if '£' in a :
        symbol.append(1.14)
        
    if '₹' in a :
        symbol.append(0.012)
        
    if '€' in a :
        symbol.append(0.99)
        
    if 'D' in a and 'E' in a and 'M' in a:
        symbol.append(0.5)
    
    if '¥' in a :
        symbol.append(0.0069)
    
    if 'F' in a and 'R' in a and 'F' in a :
        symbol.append(0.99)
        
    if '₩' in a :
        symbol.append(0.00071)
    

    


# In[7]:


#['$', 'M', '£', '₹', '€' ,'D', '¥', 'F']   FRF, DEM MVR   £=  1.14,  €= 0.99,  '₹'= 0.012, '¥'= 0.0069, FRP= 0.99, DEM-0.5067,mvr = 0.065 


# In[8]:


df.drop(['Origin_Currency_Symbol'],axis = 1 ,inplace = True)


# In[9]:


print(df.columns)


# In[10]:


df['symbol']=symbol


# In[11]:


country =[]
for x in symbol:
    if x ==1:
        country.append("USA")
     
    if x ==650000:
        country.append("Maldives")
        
    if x ==1.14:
        country.append("United Kingdom")
        
    if x ==0.012:
        country.append("India")
        
    if x ==0.99:
        country.append("France ")
        
    if x ==0.5:
        country.append("Germany")
        
    if x ==0.0069:
        country.append("Japan")
        
    if x ==0.00071:
        country.append("South Korea")


# In[12]:


df['Country'] = country


# In[13]:


b=list(df['Budget']) 
df.drop(['Budget', 'symbol'],axis = 1 ,inplace = True)

array1 = np.array(b)
array2 = np.array(symbol)

n = min(len(b), len(symbol))
result = array1[:n]*array2[:n] 

result = result.tolist()

df['Budget'] = result


# In[14]:


display(df.head())


# In[15]:


df["Country"].value_counts()


# In[16]:


df["Country"][(np.where(df["Name"]=='Downfall')[0])[0]] = "Germany"


# In[17]:


df["Country"].value_counts()


# In[18]:


print(df.shape)


# In[19]:


print(df.columns)


# In[20]:


display(df.describe())


# # Null Value Imputation

# In[21]:


print("Null values in each columns : \n")
print(df.isnull().sum())
print("\n\nTotal null values in dataset : ", df.isnull().sum().sum())


# In[22]:


# split df in to df_num and df_cat to fill out the missing values, later we again join them together.

df_num = df.select_dtypes(include='number')
df_cat=df.select_dtypes(exclude='number')


# In[23]:


display(df_num.head())


# In[24]:


display(df_cat.head())


# In[25]:


print(df_num.isnull().sum().sum())


# In[26]:


# missing value imputation for the numerical columns

from sklearn.impute import KNNImputer   # missing values imputation

imputer = KNNImputer(n_neighbors=2)  #impute missing values
df_num.iloc[:,0:]=imputer.fit_transform(df_num.iloc[:,0:])


# In[27]:


# verifying
print(df_num.isnull().sum().sum())


# In[28]:


print(df_cat.isnull().sum().sum())


# In[29]:


df= pd.concat([df_cat,df_num], axis=1)  # this df is without null values


# In[30]:


print(df.isnull().sum().sum())


# In[31]:


df.head(2)


# In[32]:


#df.duration=np.where(df.duration==2700,45, df.duration) 


# In[33]:


df.to_excel (r'C:\Users\Dell\Desktop\250_IMDB_final_clean.xlsx', index = False, header=True)


# # SUMMARIZING QUANTITATIVE DATA

# # YEAR

# # 1- OUTLIER DETECTION

# In[34]:


# Using Zscore( usually we use this method to find the outlier when the data is symmetric, skewness -0.67 is not  more, we can
#use this method to detect the outliers)

Year_zscore = list(np.round(stats.zscore(df['Year']), 2))
index=[]
outlier=[]
for i in range(len(Year_zscore)):
    if Year_zscore[i]<-3 or Year_zscore[i]>3:
        outlier.append(Year_zscore[i])
        index.append(i)

print(index)
print(outlier)


# According to Z- Score method, there are no outliers present in YEAR

# In[35]:


# Outlier Detection using IQR

index=[]
outlier=[]
percentile25 = df['Year'].quantile(0.25)
percentile75 = df['Year'].quantile(0.75)
iqr= percentile75-percentile25
upper = percentile75 + 1.5 * iqr
lower = percentile25 - 1.5 * iqr
for i in range(len(df['Year'])):
    if (df['Year'][i]>upper) or (df['Year'][i]<lower):
        outlier.append(Year_zscore[i])
        index.append(i)
        
print(index)
print(outlier)


# According to IQR method of detecting outliers, there are no outlier presents.

# In[36]:


# Outlier Detection using BOX PLOT (Outlier Detection using IQR graphically)

green_diamond = dict(markerfacecolor='g', marker='D')
data= df['Year'].values
plt.boxplot(data,vert=False, flierprops=green_diamond)
plt.show()


# We can see, that there are no points below lower limit and above upper limit.

# # 1- MEASURE OF DISTRIBUTION SHAPE

# In[37]:


# Graphical meausre of the distribution shape is Histogram
display(sns.displot(df['Year'], kind = "kde"))


# From above histogram, we can see that the YEAR variable is left skewed.

# In[38]:


# Numerical meausre of the distribution shape is : skewness
df['Year'].skew()


# Skewness value is : -0.679. Negative value implies that YEAR is left skewed. As skweness is not less than -1, we can say that YEAR is moderately skewed to Left.

# We can conclude that, there are very few movies among top 250 imdb movies which were released around 1920.

# # 3- MEASURE OF CENTRAL TENDENCY

# In[39]:


# mean
df['Year'].mean()


# In[40]:


#median
df['Year'].median()


# In[41]:


#mode
df['Year'].mode()


# Note: YEAR is asymmetric, so the better measure of the central tendency is median.

# The movie release years are clustered around 1994
# 50 percent of the top 250 imdb movies were released on or before 1994.
# 

# Mazority of the top 250 imdb movies were released in year 1995.

# In[42]:


# finding maximum movies released in a Year that make in top 250 imdb list.
m=0
for x in df['Year']:
    if x==1995:
        m+=1
print(m)


# In 1995, total 8 movies got their listing in imdb top 250 movie list.
# Producing 8 great movies in a single year is marvelous.

# # 4- PERCENTILES

# In[43]:


# percentiles

print(df['Year'].quantile(0.10))
print(df['Year'].quantile(0.25))
print(df['Year'].quantile(0.50))
print(df['Year'].quantile(0.75))
print(df['Year'].quantile(0.90))


# 10 percent of the top 250 imdb movies were released on or before 1950
# 
# 25 percent of the top 250 imdb movies were released on or before 1966
# 
# 50 percent of the top 250 imdb movies were released on or before 1994
# 
# 75 percent of the top 250 imdb movies were released on or before 2006

# # 5- MEASURE OF VARIABILITY

# In[44]:


df['Year'].var()


# In[45]:


df['Year'].std()


# # 6- FIVE NUMBER SUMMARY (BOX PLOT)

# In[46]:


fig = px.box(df, y="Year")
fig.show()


# # DURATION

# # 1- OUTLIER DETECTION

# In[47]:


# Using Zscore( usually we use this method to find the outlier when the data is symmetric, is not  more, we can
#use this method to detect the outliers)

Year_zscore = list(np.round(stats.zscore(df['duration']), 2))
index=[]
outlier=[]
for i in range(len(Year_zscore)):
    if Year_zscore[i]<-3 or Year_zscore[i]>3:
        outlier.append(Year_zscore[i])
        index.append(i)

print(index)
print(outlier)


# In[48]:


# Outlier Detection using IQR

index=[]
outlier=[]
percentile25 = df['duration'].quantile(0.25)
percentile75 = df['duration'].quantile(0.75)
iqr= percentile75-percentile25
upper = percentile75 + 1.5 * iqr
lower = percentile25 - 1.5 * iqr
for i in range(len(df['duration'])):
    if (df['duration'][i]>upper) or (df['duration'][i]<lower):
        outlier.append(df['duration'][i])
        index.append(i)
        
print(index)
print(outlier)

#note: these methods are giving us outliers, they are providing us the verhu high and very low values which have the potential
# to be an outlier. Outlier is an incorrect observation. For every potential outlier we verify if it is correct or not and based on this, we declare it as an outlier.


# In[49]:


# Outlier Detection using BOX PLOT (Outlier Detection using IQR graphically)

sns.boxplot(x=df["duration"])


# we can see that, only duration 2700.0 is not making any sense, a movie can not be that long. So, we have recorded 
# the false duration time for the movie at index 192. The movie is :  "Sherlock Jr."

# In[50]:


# removing outlier, we have to subsitute a correct value or a value near to it,otherwise we have to drop that observation(row) 

df.duration=np.where(df.duration==2700,45, df.duration) 


# we can find the correct movie duration from imdb website, above we can see that 45 minutes is a correct movie duration.

# # 2- MEASURE OF DISTRIBUTION SHAPE

# In[51]:


# Graphical meausre of the distribution shape is Histogram

display(sns.displot(df['duration'], kind = "kde"))



# From above histogram, we can see that the movie duration variable is symmetric.

# In[52]:


# Numerical meausre of the distribution shape is : skewness

print(df['duration'].skew())


# In[53]:


(a,b)= (df['duration'].mean() - df['duration'].std()),(df['duration'].mean() + df['duration'].std())
print(a,",",b)

(c,d)= (df['duration'].mean() - 2*df['duration'].std()),(df['duration'].mean() + 2*df['duration'].std())
print(c,",",d)

(e,f)= (df['duration'].mean() - 3*df['duration'].std()),(df['duration'].mean() + 3*df['duration'].std())
print(e,",",f)


# #68–95–99 rule ( not taken 99.7 as data is not fully symmetric)
# 
# Around 68 percent of the top 250 imdb movies have total duration between 99 and 159 minutes.
# 
# Around 95 percent of the top 250 imdb movies have total duration between 69 and 189 minutes.
# 
# Around 99 percent of the top 250 imdb movies have total duration between 39 and 220 minutes.

# # 3- MEASURE OF CENTRAL TENDENCY

# In[54]:


# mean
print(df['duration'].mean())


# In[55]:


#median
print(df['duration'].median())


# The movie duration is little asymmetric, so the better measure for central location is median.
# 
# The movie durations are clustered around 127 minutes. Median is a central vale. Central value divides the total observations in two halves.
# 
# let say, we decided to make a movie and our aimm to get our movie in top 250 list. Then we try to make movie of length around 
# 127 minutes.
# 
# 50 percent of the top 250 imdb movies having the running time less than 127 minutes and rest 50 percent have the running time more than 127 minutes.
# 

# # 4- PERCENTILES

# In[56]:


# percentiles

print(df['duration'].quantile(0.10))
print(df['duration'].quantile(0.25))
print(df['duration'].quantile(0.50))
print(df['duration'].quantile(0.75))
print(df['duration'].quantile(0.90))


# 10 percent of the top 250 imdb movies have their running time less than or equal to 94 minutes
# 
# 90 percent of the top 250 imdb movies have their running time more than 94 minutes. (imp)
# 
# 25 percent of the top 250 imdb movies have their running time less than or equal to 107 minutes
# 
# 50 percent of the top 250 imdb movies have their running time less than or equal to 127 minutes
# 
# 75 percent of the top 250 imdb movies have their running time less than or equal to 145 minutes
# 
# 10 percent of the top 250 imdb movies have their running time more than 170 minutes.(imp)
# 
# 50 percent of the top 250 imdb movies have their running time between 107 and 145 minutes (imp)
# 

# # 5- MEASURE OF VARIABILITY

# In[57]:


# numeric measures of variablity, it is always a good idea to compare the variation using numerical measure with histogram
print(df['Year'].var())
print(df['Year'].std())
print(df['duration'].quantile(0.75)- df['duration'].quantile(0.25))


# In[58]:


# breaking duration in 2 groups in order to compare the duration variation among these 2  groups.


duration_year_less2000= df.iloc[np.where(df['Year'] <= 2000)]['duration']
duration_year_more2000= df.iloc[np.where(df['Year'] > 2000)]['duration']


# In[59]:


print(duration_year_less2000.quantile(0.75)- duration_year_less2000.quantile(0.25))
print(duration_year_more2000.quantile(0.75)- duration_year_more2000.quantile(0.25))


# IQR FOR DURATION OF MOVIES WHICH WERE RELEASED ON AND BEFORE YEAR 2000 HAVE MORE IQR FOR DURATION OF MOVIES WHICH WERE RELEASED AFTER YEAR 2000.
# 
# SO, THE MOVIES WHICH WERE RELEASED ON AND BEFORE YEAR 2000 HAVE MORE VARIATION IN THEIR RUNNING TIME THEN THE VARIATION IN RUNNING TIME FOR THOSE MOVIES WHICH WERE RELEASED AFTER YEAR 2000.
# 
# Let's try to vizualize the variation:

# In[60]:


display(sns.displot(duration_year_less2000, kind = "kde"))


# In[61]:


display(sns.displot(duration_year_more2000, kind = "kde"))


# In[62]:


print(duration_year_more2000.describe())


# In[63]:


print(duration_year_less2000.describe())


# Movies which were released after year 2000, none of them have their running time more than 201 minutes, infact around 99 percent movies have their running time less than or equal to 3 hours.
# 
# Around 5 percent of the movies released after year 2000 are atleast 3 hours long, and around 9.5 percent of movies which were released before 2000 are atleast 3 hours long.
# After 2000, the trend of having movies longer than 3 hours gone down.
# 
# 

# In[64]:


movies_longer_3hours =0

for x in duration_year_more2000:
    if x>175:
        movies_longer_3hours+=1
        
print("percent of movies longer than 3 hours after year 2000 : ", (movies_longer_3hours/ len(duration_year_more2000))*100)

movies_longer_3hours =0

movies_longer_3hours =0

for x in duration_year_less2000:
    if x>175:
        movies_longer_3hours+=1

print("percent of movies longer than 3 hours before year 2000 : ",(movies_longer_3hours/len(duration_year_less2000))*100)


# # 6- FIVE NUMBER SUMMARY (BOX PLOT)

# In[65]:


fig = px.box(df, y="duration")
fig.show()


# # MONTH

# # 1- OUTLIER DETECTION

# In[66]:


# Using Zscore( usually we use this method to find the outlier when the data is symmetric, is not  more, we can
#use this method to detect the outliers)

Year_zscore = list(np.round(stats.zscore(df['Month']), 2))
index=[]
outlier=[]
for i in range(len(Year_zscore)):
    if Year_zscore[i]<-3 or Year_zscore[i]>3:
        outlier.append(Year_zscore[i])
        index.append(i)

print(index)
print(outlier)


# In[67]:


# Outlier Detection using IQR

index=[]
outlier=[]
percentile25 = df['Month'].quantile(0.25)
percentile75 = df['Month'].quantile(0.75)
iqr= percentile75-percentile25
upper = percentile75 + 1.5 * iqr
lower = percentile25 - 1.5 * iqr
for i in range(len(df['Month'])):
    if (df['Month'][i]>upper) or (df['Month'][i]<lower):
        outlier.append(df['Month'][i])
        index.append(i)
        
print(index)
print(outlier)

#note: these methods are giving us outliers, they are providing us the verhu high and very low values which have the potential
# to be an outlier. Outlier is an incorrect observation. For every potential outlier we verify if it is correct or not and based on this, we declare it as an outlier.


# In[68]:


# Outlier Detection using BOX PLOT (Outlier Detection using IQR graphically)

fig = px.box(df, y="Month")
fig.show()


# # 2- MEASURE OF DISTRIBUTION SHAPE

# In[69]:


# Graphical meausre of the distribution shape is Histogram



display(sns.displot(df['Month'], kind = "kde"))


# In[70]:


# Numerical meausre of the distribution shape is : skewness

print(df['Month'].skew())


# # 3- MEASURE OF CENTRAL TENDENCY

# In[71]:


# mean
print(df['Month'].mean())


# In[72]:


#median
print(df['Month'].median())


# In[73]:


#mode
print(df['Month'].mode())


# 50 percent of the movies were released on or before july
# 
# Most of the top 250 imdb movies were released in December.

# # USERS

# # 1- OUTLIER DETECTION

# In[75]:


# Using Zscore( usually we use this method to find the outlier when the data is symmetric, is not  more, we can
#use this method to detect the outliers)

Year_zscore = list(np.round(stats.zscore(df['Users']), 2))
index=[]
outlier=[]
for i in range(len(Year_zscore)):
    if Year_zscore[i]<-3 or Year_zscore[i]>3:
        outlier.append(Year_zscore[i])
        index.append(i)

print(index)
print(outlier)


# In[76]:


# Outlier Detection using IQR

index=[]
outlier=[]
percentile25 = df['Users'].quantile(0.25)
percentile75 = df['Users'].quantile(0.75)
iqr= percentile75-percentile25
upper = percentile75 + 1.5 * iqr
lower = percentile25 - 1.5 * iqr
for i in range(len(df['Users'])):
    if (df['Users'][i]>upper) or (df['Users'][i]<lower):
        outlier.append(df['Users'][i])
        index.append(i)
        
print(index)
print(outlier)

#note: these methods are giving us outliers, they are providing us the verhu high and very low values which have the potential
# to be an outlier. Outlier is an incorrect observation. For every potential outlier we verify if it is correct or not and based on this, we declare it as an outlier.


# They are the high values, but they are making sense,as ratings can be in millions. So we are not treating them as outliers

# In[77]:


# Outlier Detection using BOX PLOT (Outlier Detection using IQR graphically)

sns.boxplot(x=df["Users"])



# # 2- MEASURE OF DISTRIBUTION SHAPE

# In[78]:


# Graphical meausre of the distribution shape is Histogram


display(sns.displot(df['Users'], kind = "kde"))


# From above histogram, we cann see that the USERS is right skewed. So, we can see that the there are some movies which are rated by millions of users.

# In[79]:


# Numerical meausre of the distribution shape is : skewness

print(df['Users'].skew())


# # 3- MEASURE OF CENTRAL TENDENCY

# In[80]:


# mean
print(df['Users'].mean())


# In[81]:


#median
print(df['Users'].median())


# 488k is a median Users rated for a movie.

# # 4- PERCENTILES

# In[82]:


# percentiles

print(df['Users'].quantile(0.10))
print(df['Users'].quantile(0.25))
print(df['Users'].quantile(0.50))
print(df['Users'].quantile(0.75))
print(df['Users'].quantile(0.90))
print(df['Users'].quantile(1))


# 25 percent of the top 250 imdb movies got users between 215k and 488k 
# 10 percent of the top 250 imdb movies got users more than 1.2 million.
# There are around 31 movies which where rated by more than 1.2 million users.

# In[83]:


c=0
for x in df['Users']:
    if x>1200000:
        c+=1


# In[84]:


c


# # 5- MEASURE OF VARIABILITY

# In[85]:


# numeric measures of variablity, it is always a good idea to compare the variation using numerical measure with histogram
print(df['Users'].var())
print(df['Users'].std())
print(df['Users'].quantile(0.75)- df['duration'].quantile(0.25))


# In[86]:


# breaking duration in 2 groups in order to compare the duration variation among these 2  groups.


users_imdb_more_9= df.iloc[np.where(df['Rating'] >=9.0)]['Users']
users_imdb_more_8_9= df.iloc[np.where(df['Rating'] <9.0) and np.where(df['Rating']>=8.0)] ['Users']


# In[87]:


print(users_imdb_more_9.quantile(0.75)- users_imdb_more_9.quantile(0.25))
print(users_imdb_more_8_9.quantile(0.75)- users_imdb_more_8_9.quantile(0.25))


# 
# The total ratings for a movie with imdb 9 or more showing high variation then for movies with imdb rating between 8 and 9.

# # 6- FIVE NUMBER SUMMARY (BOX PLOT)

# In[88]:


fig = px.box(df, y="Users")
fig.show()


# # ANALYSIS (Part - 2)

# # Genre

# In[89]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x=df["Genre"])


# We can see that drama movies followed by action movies are in majority among the other genre movies.
# 
# More important thing to notice here is that, Horror movies count is similar to the Mystery movies count. 
# 
# Also, we can see a tie between Animation and Comedy genre movies.
# 

# # Director

# In[90]:


fig = px.histogram(df, x="Director")
fig.show()


# Christopher Nolan,  Steven Spielberg, Akira Kurosawa, Martin Scorsese, Stanley Kubrick are the only directors who have directed more than 5 movies that are the part of top 250.
# 
# All of these directed 7 movies in total. Out of 250 movies, 35 movies are alone directed by these directors in total, i.e 14 percent of the movies were directed by these 5 gems only.

# # Production

# In[91]:


fig = px.histogram(df, x="Production")
fig.show()


# 8.4 percent of the total top 250 are come from the production house of Warner Bros followed by Paramount Pictures and Walt Disney Pictures.
# 
# 8.4 percent is a great number, Warner Bros did awesome job.

# # CAST

# In[92]:


Actor_1 =[]
Actor_2=[]
for x in df['Cast']:
    Actor_1.append(x.split(",")[0].lower())
    Actor_2.append(x.split(",")[1].lower().strip())
 


# In[93]:


act_1= Actor_1
act_2=Actor_2

Actor_1.extend(Actor_2)


# In[94]:


fig = px.histogram(Actor_1)
fig.show()


# Lets find out, which actors are the part of most of the movies.
# 
# Robert De Niro : He acted in 9 movies, i.e 3.4 percent of movies had him. 9 is a great number, for a reson is among gretest
#                  actors.
# 
# Tom Hanks: If you have watched "Forest Gump", then this name will not surprise you. Apart from Forest Gump, He was a part of 5            more movies. 
#     

# In[95]:


actor_combo=[]
for i in range(0,250):
    x=act_1[i]
    y=act_2[i]
    if x < y:
        actor_combo.append(x+" "+y)
    else:
        actor_combo.append(y+" "+x)


# In[96]:


fig = px.histogram(actor_combo)
fig.show()


# Now, lets see that which combination of actors contributed to this list most.
# 
# First name in the combo comes of: Harrison Ford and Mark Hamill, both have acted in 3 movies together.
#     
# The pair which have acted together in exactly 2 movies are below:
#     
# 1- Al Pacino and Robert de Niro
# 
# 2- Elijah Wood and Ian Mckellen
# 
# 3- Arnold Schwarzenegger and Linda Hamilton
# 
# 4- Charles Cahplin and Paulette Goddard
# 
# 5- Tim Allen and Tom Hanks
# 
# 6- Joseph Cotten and Orson Welles
# 
# 7- Graham Chaman and John Cleese
# 
# 8- Ethan Hawke and Julie Delpy

# In[128]:


# list all the movies in which Robert De Niero acted

for i in range(df.shape[0]):
    if "Robert De Niro" in df['Cast'][i]:
        print(df["Name"][i])


# Lets see, in which movies Robert De Niro acted. Above is the list of those movies.
# 
# Godfather made him a superstar and how one can forget "Murry" from "Joker". 

# In[130]:


# list all the movies which include both the actors : Mark Hamill, Harrison Ford

for i in range(df.shape[0]):
    if "Mark Hamill" in df['Cast'][i] and "Harrison Ford"  in df['Cast'][i]:
        print(df["Name"][i])


# In[99]:


# list all the movies which include both the actors : Arnold Schwarzenegger, Linda Hamilton

for i in range(df.shape[0]):
    if "Arnold Schwarzenegger" in df['Cast'][i] and "Linda Hamilton"  in df['Cast'][i]:
        print(df["Name"][i])


# In[100]:


print(df.columns)


# # BUDGET

# In[101]:


# Budget with Genre

fig = px.histogram(x=df['Genre'], y = df['Budget'])
fig.update_xaxes(type='category')
fig.show()


# The total budget for the Drama movies are less as comapared to action movies as comapred to the action movies. Action genre is followed by Animation.
# 
# In total, the total budget for action movies was around 3.5 Billion dollars and for Animation movies, around 1.9 Billion dollars.

# In[102]:


# Budget with Production

fig = px.histogram(x=df['Production'], y = df['Budget'])
fig.update_xaxes(type='category')
fig.show()


# We have seen that Warner Bros has produced 21 movies and Walt disney has produced 8 movies. This made up their total budget of 1.39 billion dollars and 768 million dollars respectively.
# 
# But, Paramount pictures have also produced 8 movies in budget of 623 million dollars, which is approx. 150 million dollar less than Disney's budget. The reason behind this can be : that walt disney's favourite genre is Animation.
# 
# Lets dive deep for these 3 productions and find out taht in what kind of movies does they spend most.

# In[103]:


np.where(df['Production']=='Walt Disney Pictures')


# In[104]:


np.where(df['Production']=='Warner Bros.')


# In[105]:


Warner_Bros_budget=0
for i in range(df.shape[0]):
    if df['Production'][i] == 'Warner Bros.':
        Warner_Bros_budget =Warner_Bros_budget+df['Budget'][i]
        
print(Warner_Bros_budget)
        


# In[106]:


walt_disney_budget=0
for i in range(df.shape[0]):
    if df['Production'][i] == 'Walt Disney Pictures':
        walt_disney_budget = walt_disney_budget+df['Budget'][i]
        
print(walt_disney_budget)
        


# In[107]:


for i in range(df.shape[0]):
    if df['Production'][i] == 'Walt Disney Pictures':
        print(df['Name'][i])
        print(df['Genre'][i])
        print(df['Budget'][i])
        print(df['Year'][i])
        print("\n")
        


# Disney produced 75 percent Animation movies.

# In[108]:


for i in range(df.shape[0]):
    if df['Production'][i] == 'Warner Bros.':
        print(df['Name'][i])
        print(df['Genre'][i])
        print(df['Budget'][i])
        print(df['Year'][i])
        print("\n")
        


# The interesting thing to notice about the Warner Bros is that, this production house didn't limit itself to only particular type of genre. It has produced Animation, Horror, Adventure, Drama, Crime, Action, Biography.
# 
# Warner Bros. has covered all the genres.
# 

# In[109]:


Y = df['Budget'].loc[df['Genre'] == 'Animation']
X = df['Year'].loc[df['Genre'] == 'Animation']


# In[110]:


# Budget with Year for animated movies

fig = px.scatter(df, x=X, y=Y)
fig.show()


# The most expensive animated movies was produced in year 2010, lets find about the details of that movie.

# In[136]:


df.iloc[np.where((df['Genre']=="Animation") & (df['Year']== 2010 ))]


# The movie is : "Toy Story 3", with budget of 200 million dollars. 
# For other details of this movie, please refer above table.

# # Distribution By Rating

# In[141]:


# Bar graph for Ratings
fig = px.histogram(df, x="Rating")
fig.update_xaxes(type='category')
fig.show()


# Only 2 movies among top 250 imdb movies, got the imdb rating of 9.2. No movie till now, got more than 9.2 rating.

# # Distribution By Month

# In[118]:


# Bar graph for Year
fig = px.histogram(df, x="Month")
fig.update_xaxes(type='category')
fig.show()


# Most of the top movies released in December. 33 out of 250, i.e around 13 percent of the movies released in month of december.

# # Budget and Worldwide Collection for Drama Movies

# In[119]:


#  Worldwide_Collection for all the genres


fig = px.histogram(x=df['Genre'], y = df['Worlwide_Collection'])
fig.show()


# In[120]:


# Budget of all the genres.

fig = px.histogram(x=df['Genre'], y = df['Budget'])
fig.update_xaxes(type='category')
fig.show()


# 1-In total among top 250 imdb movies, 68 were drama movies produced with the total budget of around 906 millions dollars and  the world wide collection for these drama movies in genral is around 6.5 billions dollar.
# 
# 
# 

# let's find that which drama movies did not earn and floped commercially.

# In[143]:


df_new= df.loc[np.where(df['Genre']=='Drama')]
df_new =df_new[["Name", "Budget", "Worlwide_Collection" ]]


# The drama movies below were not commercially successful

# In[144]:


df_new =df_new.iloc[np.where(df_new['Worlwide_Collection'] < df_new['Budget'])]
df_new


# In[147]:


# loss 

df_new['Budget']-df_new['Worlwide_Collection']


# The drama movie which faced the highest loss among all the drama movies is "The BOAT", with a loss of 4.5 million dollars.

# In[ ]:




