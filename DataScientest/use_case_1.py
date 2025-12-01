"""
A complete data science project!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def mk_pages(pages: list[str]) -> str:
    """Make three pages"""
    st.title("Titanic: binary classification project!")
    st.sidebar.title("Table of contents")
    page = st.sidebar.radio("Go to", pages)
    return page


def des_page_zero(df: pd.DataFrame) -> None:
    """Do the first page"""
    st.write("### Presentatio of data")
    st.dataframe(df.head(10))
    st.write(f"Shape of the df: {df.shape}")
    st.write("Description of the df:")
    st.dataframe(df.describe())
    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())


def des_page_vizuls(df: pd.DataFrame) -> None:
    """Make the visualization page"""
    st.write("### DataVizualization")
    fig = plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)

    fig = sns.displot(x='Age', data=df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)

    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)

    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), ax=ax)
    st.pyplot(fig)


def prediction(classifier, x_train, y_train):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        return clf
    elif classifier == 'SVC':
        clf = SVC()
        clf.fit(x_train, y_train)
        return clf
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        return clf
    else:
        print('Wrong inputs')


def scores(clf, choice, x_test, y_test):
    if choice == 'Accuracy':
        return clf.score(x_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(x_test))


def des_page_modeling(df: pd.DataFrame) -> None:
    """Do somthing in the modeling"""
    st.write("### Modelling")
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = df['Survived']
    x_cat = df[['Pclass', 'Sex',  'Embarked']]
    x_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
    for col in x_cat.columns:
        x_cat[col] = x_cat[col].fillna(x_cat[col].mode()[0])
    for col in x_num.columns:
        x_num[col] = x_num[col].fillna(x_num[col].median())
        x_cat_scaled = pd.get_dummies(x_cat, columns=x_cat.columns)
        x = pd.concat([x_cat_scaled, x_num], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=123)
        scaler = StandardScaler()
        x_train[x_num.columns] = scaler.fit_transform(x_train[x_num.columns])
        x_test[x_num.columns] = scaler.transform(x_test[x_num.columns])
        choice = ['Random Forest', 'SVC', 'Logistic Regression']
        option = st.selectbox('Choice of the model', choice)
        st.write('The chosen model is :', option)

def make_page(fname: str) -> None:
    """Make a page in Streamlit"""
    df = pd.read_csv(fname)
    pages = ["Exploration", "DataVizualization", "Modeling"]

    page: str = mk_pages(pages)

    if page == pages[0]:
        des_page_zero(df)

    if page == pages[1]:
        des_page_vizuls(df)

    if page == pages[2]:
        des_page_modeling(df)


if __name__ == '__main__':
    make_page(fname="data/titanic/train.csv")