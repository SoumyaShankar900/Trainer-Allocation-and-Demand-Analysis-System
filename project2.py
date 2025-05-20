import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data from Excel
trainers_df = pd.read_excel('trainers.xlsx', sheet_name='Trainer')
companies_df = pd.read_excel('trainers.xlsx', sheet_name='Companies')
allocations_df = pd.read_excel('trainers.xlsx', sheet_name='TrainerAllocation')

# Check for validity and integrity of the data
print(trainers_df.isnull().sum())
print(companies_df.isnull().sum())
print(allocations_df.isnull().sum())

# Check for missing values and fill with empty string if any
trainers_df.fillna('', inplace=True)
companies_df.fillna('', inplace=True)
allocations_df.fillna('', inplace=True)

# Convert date columns to datetime format
companies_df['Expected_Start_Date'] = pd.to_datetime(companies_df['Expected_Start_Date'])
allocations_df['Start_Date'] = pd.to_datetime(allocations_df['Start_Date'])

# Apply mathematical formula
average_work_experience = trainers_df['Work_Experience'].mean()
max_experience = trainers_df['Work_Experience'].max()
min_experience = trainers_df['Work_Experience'].min()

print(f'Maximum Work Experience: {max_experience} years')
print(f'Minimum Work Experience: {min_experience} years')
print(f'Average Work Experience: {average_work_experience}')

# Aggregate the data to count the number of trainers for each level of work experience
experience_count = trainers_df['Work_Experience'].value_counts().sort_index()
print(experience_count)

# Descriptive statistics
print(trainers_df.describe())
print(companies_df.describe())
print(allocations_df.describe())

# Distribution analysis
print("Mean of Work Experience:", trainers_df['Work_Experience'].mean())
print("Standard Deviation of Work Experience:", trainers_df['Work_Experience'].std())
print("Mode of Work Experience:", trainers_df['Work_Experience'].mode()[0])
print("Mean of Trainers Count:", companies_df['Trainers_Count'].mean())
print("Standard Deviation of Trainers Count:", companies_df['Trainers_Count'].std())
print("Mode of Trainers Count:", companies_df['Trainers_Count'].mode()[0])



#Trainers Background
data = {
    'Trainer_Id': range(1, 26),
    'Work_Experience':[5,8,6,10,7,4,9,5,6,8,7,10,4,5,6,7,8,9,5,6,7,8,9,2,6]
    }
df = pd.DataFrame(data)

# Work Experience of trainers
plt.figure(figsize=(10, 6))
sns.histplot(trainers_df['Work_Experience'], kde=True)
plt.title('Distribution of Work Experience')
plt.show()


# Plotting the bar plot
plt.figure(figsize=(25, 10))
sns.barplot(x=experience_count.index, y=experience_count.values, palette='viridis')
plt.title('Number of Trainers by Work Experience')
plt.xlabel('Work Experience (Years)')
plt.ylabel('Number of Trainers')
plt.xticks(rotation=0)
plt.show()

# Demand for trainers on specific technologies
technology_demand = companies_df.groupby('Technology').size()
plt.figure(figsize=(12, 8))
technology_demand.plot(kind='bar')
plt.title('Demand for Trainers on Specific Technologies')
plt.xlabel('Technology')
plt.ylabel('Number of Companies')
plt.show()

# Matching skill sets of trainers
skill_sets = trainers_df['Technologies'].value_counts()
plt.figure(figsize=(12, 8))
skill_sets.plot(kind='bar')
plt.title('Skill Sets of Trainers')
plt.xlabel('Technologies')
plt.ylabel('Number of Trainers')
plt.show()

# Requirements fulfilled by in-house & consultant trainers
inhouse_fulfillment = allocations_df[allocations_df['Status'] == 'Fulfilled'][allocations_df['Trainer_Id'].isin(trainers_df[trainers_df['Employee_Type'] == 'In-house']['Trainer_Id'])]
consultant_fulfillment = allocations_df[allocations_df['Status'] == 'Fulfilled'][allocations_df['Trainer_Id'].isin(trainers_df[trainers_df['Employee_Type'] == 'Consultant']['Trainer_Id'])]

# Upskilling needed for trainers based on unfulfilled requirements
unfulfilled_requirements = allocations_df[allocations_df['Status'] != 'Fulfilled']
upskilling_needed = unfulfilled_requirements.groupby('Technology').size()
plt.figure(figsize=(12, 8))
upskilling_needed.plot(kind='bar')
plt.title('Upskilling Needed Based on Unfulfilled Requirements')
plt.xlabel('Technology')
plt.ylabel('Number of Unfulfilled Requirements')
plt.show()












