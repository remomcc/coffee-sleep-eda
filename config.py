'''
## **Configuration**
Settings for Coffee, Lifestyle & Sleep EDA notebook and Streamlit application.
'''

import kagglehub
from kagglehub import KaggleDatasetAdapter

"""#### **Kaggle dataset**"""

# Download Kaggle dataset
COFFEE_PATH = 'uom190346a/global-coffee-health-dataset'
K_PATH = kagglehub.dataset_download(COFFEE_PATH)
FILE_PATH = 'synthetic_coffee_health_10000.csv'

"""#### **Convert features**"""

# Binary
BINARY_FEAT = ['Smoking', 'Alcohol Consumption']

# Nominal
NOMINAL_FEAT = ['Gender', 'Country', 'Occupation']

# Ordinal
ORDINAL_FEAT = ['Ages', 'Sleep Quality', 'Stress Level', 'Health Issues', 'Lifestyle']

# Ages and Lifestyle are already categorical variables
ORDINAL_DICT = {'Sleep Quality': ['Poor', 'Fair', 'Good', 'Excellent'],
                'Stress Level': ['Low', 'Medium', 'High'],
                'Health Issues': ['None', 'Mild', 'Moderate', 'Severe']}

# Categorical
CATEGORICAL_FEAT = BINARY_FEAT + NOMINAL_FEAT + ORDINAL_FEAT

"""### **Notebook**

#### **Grouped features**
"""

# Demograghics & Health
DEMO_HEALTH_FEAT = ['Age', 'BMI', 'Heart Rate', 'Gender']

# Lifestyle
LIFESTYLE_FEAT = ['Smoking', 'Alcohol Consumption', 'Physical Activity Hours', 'Coffee Intake', 'Coffee']

"""#### **Plot arguments**"""

# Hyperparameters
HIST_KWARGS = {'hue': 'Sleep Quality',
               'stat': 'density',
               'common_norm': False,
               'alpha': 0.4}
BOX_KWARGS = {'hue': 'Sleep Quality'}

"""### **Streamlit application**"""

# Features of interest
FEATS = ['Age', 'BMI', 'Heart Rate', 'Gender',
         'Smoking', 'Alcohol Consumption', 'Physical Activity Hours',
         'Coffee Intake', 'Coffee']

sleep_color = {'Poor': '#D62728',
               'Fair': '#EDC001',
               'Good': '#2CA02C',
               'Excellent': '#1F77B4'}

# Average values for graphs
COFFEE_FEAT= {
    'Stress Level': {
        'Column': 'Stress Level',
        'Rotation': 0
    },
    'Health Issues': {
        'Column': 'Health Issues',
        'Rotation': 0
    },
    'Physical Activity Level': {
        'Column': 'Lifestyle',
        'Rotation': 0
    },
    'Age' :{
        'Column': 'Ages',
        'Rotation': 0
    },
    'Country': {
        'Column': 'Country',
        'Rotation': 90
    },
    'Occupation': {
        'Column': 'Occupation',
        'Rotation': 0
    }
}
