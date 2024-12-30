import streamlit as st
from streamlit_option_menu import option_menu
import sqlite3
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def create_user():
    conn = sqlite3.connect("expenses.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    password TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    date TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

def main():
    create_user()
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            width: 300px;
        }
        </style>
        """,
        unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <h1 style='font-size: 2em;'>
            <img src="https://img.icons8.com/doodle/48/000000/money.png" alt="Swift Icon" style="vertical-align: middle;"> SWIFT
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.header("NAVIGATION")
    with st.sidebar.expander("Menu", expanded=True):
        page = option_menu(menu_title="Navigation", options=["Lander", "Budget Tracking", "Predictions", "Visualizations"],
                           icons=["house", "list", "bar-chart", "pie-chart"], menu_icon="cast", default_index=0)

    if page == "Lander":
        st.title("SWIFT")
        st.header("Budgeting Made Effortless: Control Your Money, Transform Your Life.")
        st.write("Welcome to SWIFT – the ultimate solution for mastering your finances effortlessly!")
        st.write("With SWIFT, budgeting becomes a breeze. Say goodbye to guesswork and hello to clarity!")
        st.header("KEY FEATURES")
        option_menu(menu_title="KEY FEATURES", options=[
            "Simplified Budgeting: Take the complexity out of budgeting with our user-friendly interface.",
            "Real-Time Tracking: Stay up-to-date on your finances with real-time expense tracking.",
            "Insightful Reports: Gain valuable insights into your spending patterns."
        ], icons=["diamond", "diamond", "diamond"], default_index=0)
        st.header("Ready to take control of your finances?")
        st.header("Get started now!")

    elif page == "Budget Tracking":
        selected = option_menu(menu_title=None, options=["Add expense", "View expenses", "Visualise expenses"])

        if selected == "Add expense":
            st.title("Add your expenses")
            category = st.radio("Categories", ["Housing", "Transportation", "Foodandgroceries", "Healthcare", "PersonalandLifestyle", "DebtandSavings"])
            expense_name = st.text_input("Expense Name", "")
            amount = st.number_input("Amount", min_value=0.01, step=0.01)
            description = st.text_area("Description", "")

            if st.button("Add"):
                add_expense(expense_name, category, amount)
                st.success("Transaction added successfully!")

        elif selected == "View expenses":
            st.title("View your expenses")
            df = load_transactions()
            st.write("Expense Data:")
            st.write(df)
            if df.empty:
                st.write("No transactions recorded yet")
            else:
                st.download_button(label="Download as CSV", data=convert_df_to_csv(df), file_name='expense_data.csv', mime='text/csv')

        elif selected == "Visualise expenses":
            transactions = load_transactions()
            if transactions.empty:
                st.error("No transactions recorded yet.")
            else:
                transactions_df = load_transactions()
                category_totals = transactions_df.groupby("category")["amount"].sum()

                st.subheader("Expense Distribution by Category")
                fig_pie = px.pie(category_totals, values=category_totals.values, names=category_totals.index)
                st.plotly_chart(fig_pie, use_container_width=True)

                transactions_df["Day"] = pd.to_datetime(transactions_df["date"]).dt.date
                daily_totals = transactions_df.groupby("Day")["amount"].sum()

                st.subheader("Daily Expenses")
                daily_df = daily_totals.reset_index()
                daily_df["Day"] = daily_df["Day"].astype(str)
                fig_bar = px.bar(daily_df, x="Day", y="amount", labels={'Amount': 'Total Expense'})
                fig_bar.update_xaxes(title_text='Day')
                fig_bar.update_yaxes(title_text='Total Expense')
                st.plotly_chart(fig_bar, use_container_width=True)

    elif page == "Predictions":
        transactions_df = load_transactions()
        st.header("Future Predictions")
        st.write("Enter the category and amount for future prediction:")
        category_input = st.selectbox("category", ["Housing", "Transportation", "Foodandgroceries", "Healthcare", "PersonalandLifestyle", "DebtandSavings"])

        if st.button("Predict"):
            existing_dates, preds_e, future_dates, preds_f = predict_expenses(category_input)
            fig = go.Figure()
            existing_daily_totals = transactions_df[transactions_df['category'] == category_input].groupby(pd.to_datetime(transactions_df["date"]).dt.date)["amount"].sum()
            fig.add_trace(go.Scatter(x=existing_dates, y=existing_daily_totals, mode='markers', name='Existing Expenses'))
            fig.add_trace(go.Scatter(x=np.concatenate((existing_dates, future_dates)), y=np.concatenate((preds_e, preds_f)), mode='lines', name='Predicted Expenses'))

            fig.update_layout(title='Expense Prediction', xaxis_title='Date', yaxis_title='Expense Amount', showlegend=True)
            future_dates1 = [pd.Timestamp(dt).date() for dt in future_dates]
            future_data = {'Date': future_dates1, 'Predicted Expense': preds_f}
            future_df = pd.DataFrame(future_data)

            st.write("Future Predicted Expenses:")
            st.write(future_df)
            st.plotly_chart(fig)

    elif page == "Visualizations":
        st.header("Visualizations")
        expenses_df = load_transactions()
        opt = st.selectbox("Cluster by", ["Categories", "Dates", "Amount"])
        expenses_df['l'] = LabelEncoder().fit_transform(expenses_df['category'])
        expenses_df['dl'] = LabelEncoder().fit_transform(expenses_df['date'])

        if opt == "Categories":
            features = ['l']

        elif opt == "Dates":
            features = ['dl']

        elif opt == "Amount":
            features = ['amount']

        X = expenses_df[features]

        kmeans = KMeans(n_clusters=6)
        clusters = kmeans.fit_predict(X)
        expenses_df['cluster'] = clusters

        custom_color_scale = px.colors.qualitative.Plotly
        if opt == "Categories":
            fig = px.scatter(x=expenses_df['id'], y=expenses_df['category'], color=clusters, color_continuous_scale=custom_color_scale, title="Clustering by Categories")
            fig.update_yaxes(title=opt)

        elif opt == "Dates":
            fig = px.scatter(x=expenses_df['id'], y=expenses_df['date'], color=clusters, color_continuous_scale=custom_color_scale, title="Clustering by Dates")
            fig.update_yaxes(title=opt)

        elif opt == "Amount":
            fig = px.scatter(x=expenses_df['id'], y=expenses_df['amount'], color=clusters, color_continuous_scale=custom_color_scale, title="Clustering by Amount")
            fig.update_yaxes(title=opt)

        fig.update_xaxes(title="Entry ID")
        fig.update_layout(showlegend=False)

        st.write(expenses_df[['id', 'name', 'category', 'amount', 'date', 'cluster']])
        st.plotly_chart(fig)

def add_expense(expense_name, category, amount):
    conn = sqlite3.connect("expenses.db")
    c = conn.cursor()
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    c.execute("INSERT INTO expenses(name,category,amount,date) VALUES(?,?,?,?)", (expense_name, category, amount, date))
    conn.commit()
    conn.close()

def load_transactions():
    conn = sqlite3.connect("expenses.db")
    c = conn.cursor()
    c.execute("SELECT * FROM expenses")
    transactions = c.fetchall()
    conn.close()
    columns = ["id", "name", "category", "amount", "date"]
    transactions_df = pd.DataFrame(transactions, columns=columns)
    return transactions_df

def convert_df_to_csv(df):
    return df.to_csv(index=False)

def predict_expenses(category):
    # Placeholder for the prediction logic
    # Replace with real implementation
    existing_dates = np.array(pd.date_range(start='2023-09-01', periods=5))
    preds_e = np.random.rand(5) * 100
    future_dates = np.array(pd.date_range(start='2023-09-06', periods=5))
    preds_f = np.random.rand(5) * 100
    return existing_dates, preds_e, future_dates, preds_f

if __name__ == "__main__":
    main()
