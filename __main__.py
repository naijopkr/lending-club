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
