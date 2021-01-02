### Let's import all necessery libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(100)

## Let's Import the dataframe and manipulate it so that we can plot graph on top of it
## It will chnage the data type of variable
def change_dtype(x):
    try:
        return float(x)
    except:
        try:
            return float(x.strip())
        except:
            return None

## it will create a arranged dta frame
def create_dataframe(df):
    rows=[]
    for row in df["ir_level;red_level;temperature"].str.split(";"):
        rows.append(row)
    columns=df.columns[0].split(";")
    df=pd.DataFrame(rows,columns=columns)
    for col in df.columns:
        df[col]=df[col].apply(lambda x:change_dtype(x))
    return df

df1=pd.read_csv("alper.csv")
df1=create_dataframe(df1)

## Let's define a function which can plot red_level and ir_level
def plot_ir_red(df):
   plt.figure(figsize=(8,5))
   plt.plot(df["ir_level"],color="purple",label="ir_level")
   plt.plot(df["red_level"],color="green",label="red_level")
   plt.xlabel("Timestamp",fontsize=15)
   plt.title("Ir and Red level Plot")
   plt.legend()
## Let's define a function which can plot temperature values
def plot_temp(df):
    plt.figure(figsize=(8,5))
    plt.plot(df["temperature"],color="y")
    plt.title("Temperature Plot")
    plt.show()

## Now plot ir_level and red_level in a plot
plot_ir_red(df1)
plt.show()

## Now see the temperature plot
plot_temp(df1)
plt.show()

## Similary lets do the same thing for all three dataframes

df2=pd.read_csv("nilufer.csv")
df2=create_dataframe(df2)
## Now plot ir_level and red_level in a plot
plot_ir_red(df2)
plt.show()

## Now see the temperature plot
plot_temp(df2)
plt.show()

df3=pd.read_csv("person1.csv")
df3=create_dataframe(df3)

## Now plot ir_level and red_level in a plot
plot_ir_red(df3)
plt.show()

## Now see the temperature plot
plot_temp(df3)
plt.show()

df4=pd.read_csv("person2.csv")
df4=create_dataframe(df4)

## Now plot ir_level and red_level in a plot
plot_ir_red(df4)
plt.show()

## Now see the temperature plot
plot_temp(df4)
plt.show()

df5=pd.read_csv("person3.csv")
df5=create_dataframe(df5)

## Now plot ir_level and red_level in a plot
plot_ir_red(df5)
plt.show()

## Now see the temperature plot
plot_temp(df5)
plt.show()


df6=pd.read_csv("person4.csv")
df6=create_dataframe(df6)
df7=pd.read_csv("person5.csv")
df7=create_dataframe(df7)
df8=pd.read_csv("person6.csv")
df8=create_dataframe(df8)
df9=pd.read_csv("person7.csv")
df9=create_dataframe(df9)
df10=pd.read_csv("person8.csv")
df10=create_dataframe(df10)
df11=pd.read_csv("person9.csv")
df11=create_dataframe(df11)
df12=pd.read_csv("person10.csv")
df12=create_dataframe(df12)

## Now Model Building

#### The person having heart risk will be labeled as 1
####  The person not having heart risk will be labeled as 0

## Let's create lables for our datasets
def create_label(x):
    val=x["ir_level"]/x["red_level"]
    temp=x["temperature"]
    if (temp>36.5)and (temp<38.6)and val>0.99 and val<1.01:
        return 0
    else:
        return 1

## Let's cocat all dataframes

df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12])
df["Risk"]=df.apply(lambda x: create_label(x),axis=1)
#As there are  nan values , we need to treat this values
df["temperature"].fillna(df["temperature"].mean(),inplace=True)

##creating Dependent and Independent variable
X=df.iloc[:,:-1]
y=df.iloc[:,-1]


## Now let's split our dataset into train and testing datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3,stratify=y)


### Let's Scale our dataset
sc=StandardScaler()
train_X=sc.fit_transform(train_X)
test_X=sc.transform(test_X)

##Now Let's Apply Logistic Regression

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=100)
lr.fit(train_X,train_y)
## Now test it on testing datasets
pred_y=lr.predict(test_X)

## Now let's define a function to evalute our classifier performance
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
def check_scores(test_y,pred_y):
    print(confusion_matrix(test_y,pred_y))
    print(classification_report(test_y,pred_y))
    print(accuracy_score(test_y,pred_y))



check_scores(test_y,pred_y)

##Now let's Apply SVM with linear kernel
from sklearn.svm import SVC
svc_linear=SVC(kernel="linear",random_state=100)
svc_linear.fit(train_X,train_y)
pred_y=svc_linear.predict(test_X)
check_scores(test_y,pred_y)

## Now let's plot decisiion boundry for linear kernel
from mlxtend.plotting import plot_decision_regions
plt.figure(figsize=(10,6))
width=0.75
value=1.5
plot_decision_regions(X=train_X,y=train_y.values,clf=svc_linear,legend=2,feature_index=[1,2],filler_feature_ranges={0:width},
                     filler_feature_values={0:value})
plt.title("SVM with linear kernel Plotting",fontsize=15)
plt.show()

#Let's apply svm with rbf kernel¶
svc_rbf=SVC(kernel="rbf",random_state=100)
svc_rbf.fit(train_X,train_y)
pred_y=svc_rbf.predict(test_X)
check_scores(test_y,pred_y)
print("The accuracy score with SVM with rbf kernel is",accuracy_score(test_y,pred_y))

## Let's plot decision boundry for rbf kernel
plt.figure(figsize=(10,6))
width=0.75
value=1.5
plot_decision_regions(X=train_X,y=train_y.values,clf=svc_rbf,legend=2,feature_index=[0,2],filler_feature_ranges={1:width},
                     filler_feature_values={1:value})
plt.title("SVM with rbf kernel Plotting",fontsize=15)
plt.show()