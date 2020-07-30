import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_info = pd.read_csv('data/lending_club_info.csv', index_col='LoanStatNew')

def feat_info(feature_name: str):
    print(df_info.loc[feature_name]['Description'])


def plot(plot_func, *args, rotation = 0, align = 'center', **kwargs):
    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    plt.xticks(rotation=rotation, horizontalalignment=align)
    plot_func(*args, **kwargs)


df = pd.read_csv('data/lending-club.csv')
df.info()

plot(sns.countplot, df['loan_status'])
plot(sns.distplot, df['loan_amnt'], kde=False)

plot(sns.heatmap, df.corr(), annot=True, cmap='coolwarm')

feat_info('installment')
feat_info('loan_amnt')
plot(sns.scatterplot, x='loan_amnt', y='installment', data=df)

plot(sns.boxplot, x='loan_status', y='loan_amnt', data=df)

df.groupby('loan_status')['loan_amnt'].describe()

grade_order = sorted(df['grade'].unique())
sub_grade_order = sorted(df['sub_grade'].unique())

plot(
    sns.countplot,
    df['grade'],
    hue='loan_status',
    data=df,
    order=grade_order,
    palette='coolwarm'
)

plot(
    sns.countplot,
    df['sub_grade'],
    hue='loan_status',
    data=df,
    order=sub_grade_order,
    palette='coolwarm'
)

plot(
    sns.countplot,
    df['sub_grade'],
    data=df,
    order=sub_grade_order,
    palette='plasma'
)

grade_f_g = df[(df['grade'] == 'F') | (df['grade'] == 'G')]['sub_grade']

plot(
    sns.countplot,
    grade_f_g,
    data=df,
    order=sub_grade_order[-10:],
    palette='plasma'
)

plot(
    sns.countplot,
    grade_f_g,
    data=df,
    hue='loan_status',
    order=sub_grade_order[-10:],
    palette='coolwarm'
)

df['loan_repaid'] = df['loan_status'].apply(
    lambda x: 1 if x == 'Fully Paid' else 0
)


loan_corr = df.corr()['loan_repaid'].drop('loan_repaid').sort_values()

plot(
    sns.barplot,
    x=loan_corr.index,
    y=loan_corr.values,
    palette='plasma',
    rotation=45,
    align='right'
)
loan_corr.max()
loan_corr.min()

# Data Pre-processing
def get_missing_data():
    df_len = df.shape[0]
    missing_data = df.isnull().sum()

    return missing_data.apply(lambda x: 100*x/df_len)


df = df.drop('emp_title', axis=1)

emp_order = sorted(df['emp_length'].dropna().unique())
emp_order[0], emp_order[1], emp_order[-1] = (
    emp_order[-1], emp_order[0], emp_order[1]
)

plot(
    sns.countplot,
    x='emp_length',
    data=df,
    order=emp_order,
    hue='loan_status'
)

unpaid = df[df['loan_repaid'] == 0].groupby('emp_length')['loan_repaid'].count()
total = df.groupby('emp_length')['loan_repaid'].count()
perc = 100 * unpaid / total

plot(
    sns.barplot,
    x=perc.index,
    y=perc.values,
    order=emp_order
)

df = df.drop('emp_length', axis=1)

get_missing_data()

df['title'].head()
df['title'].nunique()

df['purpose'].head()

df = df.drop('title', axis=1)

get_missing_data()

feat_info('mort_acc')
df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values(ascending=False)

feat_info('total_acc')
df['total_acc'].value_counts()


total_acc_groups = df.groupby('total_acc').mean()['mort_acc']

def fill_mort_acc(row):
    mort_acc = row['mort_acc']
    total_acc = row['total_acc']

    if pd.isna(mort_acc):
        mort_acc = round(total_acc_groups.loc[total_acc])

    return pd.Series(
        [total_acc, mort_acc],
        index=['total_acc', 'mort_acc']
    )


mort_acc_fill = df[['total_acc', 'mort_acc']].apply(fill_mort_acc, axis=1)

df['mort_acc'] = mort_acc_fill['mort_acc']

get_missing_data()

df = df.drop(df[pd.isna(df['revol_util'])].index)

df = df.drop(df[pd.isna(df['pub_rec_bankruptcies'])].index)


# Categorical Variables and Dummy Variables
def get_categorical_cols(data: pd.DataFrame):
    return data.dtypes[data.dtypes == 'O'].index

# Convert 'term' to int64
df['term'] = pd.Series(df['term'].apply(lambda x: x.split()[0]), dtype='int64')
df.info()

# Drop grade: redundant with sub_grade
df = df.drop('grade', axis=1)
df.info()

# Create dummies
cols_dummies = [
    'sub_grade',
    'verification_status',
    'application_type',
    'initial_list_status',
    'purpose'
]
dummies = pd.get_dummies(df[cols_dummies], drop_first=True)
df = pd.concat([df, dummies], axis=1).drop(cols_dummies, axis=1)
df.info()

feat_info('home_ownership')

df['home_ownership'].value_counts()

# Convert home ownership 'NONE' and 'ANY' to 'OTHER'
def convert_ho(label):
    if label in ['ANY', 'NONE']:
        return 'OTHER'

    return label


df['home_ownership'] = df['home_ownership'].apply(convert_ho)
home_ownership = pd.get_dummies(df['home_ownership'], drop_first=True)

df = pd.concat([df, home_ownership], axis=1).drop('home_ownership', axis=1)
df.info()

# Extract zipcode
df['zipcode'] = df['address'].map(lambda x: x[-5:])
df['zipcode'].value_counts()

zipcode = pd.get_dummies(df['zipcode'], drop_first=True)
zipcode.head()

df = pd.concat([df, zipcode], axis=1).drop(['zipcode', 'address'], axis=1)
df.info()

# Drop issue_d
df = df.drop('issue_d', axis=1)

feat_info('earliest_cr_line')

df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: int(x[-4:]))
df = df.drop('earliest_cr_line', axis=1)
df.info()

get_categorical_cols(df)

df.to_csv('data/cleansed_data.csv', index=False)
