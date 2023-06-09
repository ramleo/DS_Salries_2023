def import_file(path):
  '''
  imports relevant dataset from the given path
  Output: Data Frame.
  path: Path where file is located
  '''

  import pandas as pd

  ds_sal = pd.read_csv(path, encoding='cp1252')

  return ds_sal



def jp_classi(x):

  '''
  Classifies 93 different job profiles into 13 broad categories.
  Categories: 'Data Scientist', 'Data Engineer', 'Data Analyst',
              'Data Science', 'Machine Learning', 'Artificial Intelligence'
              'Business Intelligence', 'Computer Vision', 'Analytics',
              'Data Related', 'Research Scientist', 'Applied Scientist',
              'Autonomous Vehicle Technician'.

  '''

  for i in x:
    if 'Data Scientist' in i:
      return 'Data Scientist'
    elif 'Data Engineer' in i:
      return 'Data Engineer'
    elif 'ETL' in i:
      return 'Data Engineer'
    elif 'Analyst' in i:
      return 'Data Analyst'
    elif 'Data Science' in i:
      return 'Data Science'
    elif 'Machine Learning' in i:
      return 'Machine Learning'
    elif 'AI' in i:
      return 'Artificial Intelligence'
    elif 'BI' in i:
      return 'Business Intelligence'
    elif 'Business Intelligence' in i:
      return 'Business Intelligence'
    elif 'Computer Vision' in i:
      return 'Computer Vision'
    elif 'Analytics' in i:
      return 'Analytics'
    elif 'Data' in i:
      return 'Data Related'
    elif 'Research' in i:
      return 'Research Scientist'
    elif 'Applied Scientist' in i:
      return 'Applied Scientist'
    elif 'Engineer' in i:
      return 'Machine Learning'
    elif 'Engineer' in i:
      return 'Machine Learning'
    elif 'Autonomous Vehicle Technician' in i:
      return 'Autonomous Vehicle Technician'
    else:
      return 'null'  



def jobprofile_classification(df):

  '''
  Classifies 93 different job profiles into 13 broad categories.
  Categories: 'Data Scientist', 'Data Engineer', 'Data Analyst',
              'Data Science', 'Machine Learning', 'Artificial Intelligence'
              'Business Intelligence', 'Computer Vision', 'Analytics',
              'Data Related', 'Research Scientist', 'Applied Scientist',
              'Autonomous Vehicle Technician'.
  Output: Converted data frame.
  df: Data frame to be converted.
  '''
  
  # import jp_classi

  df['job_role'] = df[['job_title']].apply(jp_classi, axis=1)

  df.drop(['salary', 'salary_currency', 'job_title', 'work_year'], axis=1, inplace=True)

  return df



def target_conv_discrete(df):

  '''
  This function converts the target variable into 3 classes.
  Classes: 'Low Income', 'Medium Income' and 'High Income'.
  Output: Converted data frame
  df: Data frame which contains target variable that needs to be converted.
  '''
  import pandas as pd

  bins = [5000, 75000, 175000, 450000]
  labels = ['Low Income', 'Medium Income', 'High Income']
  df['income_group'] = pd.cut(df['salary_in_usd'], bins=bins, labels=labels)
  df.drop(['salary_in_usd'], axis=1, inplace=True)

  return df



def continent_converter(code):

  '''
  Converts countries column into continents.
  '''
  
  # !pip install pycountry-convert
  import pycountry_convert as pc

  country_continent_code = pc.country_alpha2_to_continent_code(code)
  country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)

  return country_continent_name



def country_to_continent(df):

  '''
  Converts countries column into continents.
  Output: Converted data frame
  df: Data frame which contains target variable that needs to be converted.
  '''  

  # import continent_converter

  df['company_continent'] = df['company_location'].apply(continent_converter)
  df['employee_continent'] = df['employee_residence'].apply(continent_converter)
  df.drop(['company_location', 'employee_residence'], axis=1, inplace=True)

  return df



def train_test_split(df):

  from sklearn.model_selection import train_test_split
  import joblib

  X = df.drop(['income_group'], axis=1)
  y = df['income_group']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
  joblib.dump(y_train, 'target_traindata')
  X_test.to_csv('datascience_testdata', index=False)
  joblib.dump(y_test, 'target_testdata')
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

  return X_train


def required_columns():

  ohe_names = ['job_role_Analytics','job_role_Applied Scientist',
                'job_role_Artificial Intelligence','job_role_Autonomous Vehicle Technician','job_role_Business Intelligence','job_role_Computer Vision',
                'job_role_Data Analyst','job_role_Data Engineer','job_role_Data Related','job_role_Data Science','job_role_Data Scientist','job_role_Machine Learning','job_role_Research Scientist','company_continent_Africa','company_continent_Asia','company_continent_Europe',
                'company_continent_North America','company_continent_Oceania','company_continent_South America','employee_continent_Africa','employee_continent_Asia','employee_continent_Europe',
                'employee_continent_North America','employee_continent_Oceania','employee_continent_South America']

  not_ohe_names = ['experience_level', 'employment_type', 'company_size', 'remote_ratio']

  total_cols = ohe_names + not_ohe_names

  rename_cols = not_ohe_names + ohe_names

  rfe_names = ['experience_level','employment_type','remote_ratio','job_role_Data Analyst', 
              'company_continent_North America','employee_continent_Europe','employee_continent_North America']
  
  return total_cols, rename_cols, rfe_names


def ohe_ord_to_df_conv(df):

  '''
  Transforms dataset using one hot encoder and returns data frame instead of ndarray.
  Output: one hot encoded Dataframe
  encoder: OneHotEncoder.
  df: DataFrame.
  vars: List of variables to be encoded.
  '''
  from Library.user_defined import required_columns
  import pandas as pd

  total_cols,_,_ = required_columns()

  df_train = pd.DataFrame(df, columns=total_cols)

  return df_train


def rename_filter(data):

  from Library.user_defined import required_columns
  import pandas as pd

  _,rename_cols,_ = required_columns()

  df = pd.DataFrame(data, columns=rename_cols)

  return df


def rfe_to_df_conv(df):

  from Library.user_defined import required_columns
  import pandas as pd

  _,_,rfe_names = required_columns()

  df = pd.DataFrame(df, columns=rfe_names)

  return df


def target_label_encode(encoder=None,y=None):
  '''
  This function encode the target variable using label encoder.
  Output: Label encoded target variable.
  encoder: LabelEncoder
  y: Target variable
  '''
  le = encoder()
  target = le.fit_transform(y)

  return target, le
