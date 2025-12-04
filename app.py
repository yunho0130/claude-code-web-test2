"""
Boston Housing Price Visualization Web Application
Using Streamlit and PandasAI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandasai import Agent
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Page configuration
st.set_page_config(
    page_title="Boston Housing Analysis",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 Boston Housing Price Analysis")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("boston_house_prices.csv", skiprows=1)
    return df

df = load_data()

# Column descriptions
column_descriptions = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
    "NOX": "Nitric oxides concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to five Boston employment centres",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
    "LSTAT": "% lower status of the population",
    "MEDV": "Median value of owner-occupied homes in $1000's (Target)"
}

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Data Overview", "Visualizations", "Linear Regression", "Correlation Analysis", "PandasAI Chat"]
)

if page == "Data Overview":
    st.header("📊 Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Average Price ($1000)", f"{df['MEDV'].mean():.2f}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Descriptions")
    desc_df = pd.DataFrame({
        "Column": column_descriptions.keys(),
        "Description": column_descriptions.values()
    })
    st.table(desc_df)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "Visualizations":
    st.header("📈 Data Visualizations")

    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Price Distribution", "Feature vs Price", "Box Plots", "Scatter Matrix"]
    )

    if viz_type == "Price Distribution":
        st.subheader("Distribution of Housing Prices (MEDV)")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(df['MEDV'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Price ($1000)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Histogram of Housing Prices')

        # Box plot
        axes[1].boxplot(df['MEDV'], vert=True)
        axes[1].set_ylabel('Price ($1000)')
        axes[1].set_title('Box Plot of Housing Prices')

        plt.tight_layout()
        st.pyplot(fig)

    elif viz_type == "Feature vs Price":
        st.subheader("Feature vs Housing Price")
        feature = st.selectbox(
            "Select Feature",
            [col for col in df.columns if col != 'MEDV']
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[feature], df['MEDV'], alpha=0.6, color='steelblue')
        ax.set_xlabel(f'{feature}: {column_descriptions.get(feature, "")}')
        ax.set_ylabel('MEDV (Price in $1000)')
        ax.set_title(f'{feature} vs Housing Price')

        # Add trend line
        z = np.polyfit(df[feature], df['MEDV'], 1)
        p = np.poly1d(z)
        ax.plot(df[feature].sort_values(), p(df[feature].sort_values()),
                "r--", alpha=0.8, label='Trend line')
        ax.legend()

        st.pyplot(fig)

        # Show correlation
        corr = df[feature].corr(df['MEDV'])
        st.info(f"Correlation between {feature} and MEDV: **{corr:.4f}**")

    elif viz_type == "Box Plots":
        st.subheader("Box Plots by Feature")
        feature = st.selectbox(
            "Select Feature for Box Plot",
            df.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(df[feature].dropna())
        ax.set_ylabel(feature)
        ax.set_title(f'Box Plot of {feature}')
        st.pyplot(fig)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[feature].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[feature].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df[feature].std():.2f}")
        with col4:
            st.metric("IQR", f"{df[feature].quantile(0.75) - df[feature].quantile(0.25):.2f}")

    elif viz_type == "Scatter Matrix":
        st.subheader("Scatter Matrix (Selected Features)")

        selected_features = st.multiselect(
            "Select Features (2-5 recommended)",
            df.columns.tolist(),
            default=['RM', 'LSTAT', 'PTRATIO', 'MEDV']
        )

        if len(selected_features) >= 2:
            fig = plt.figure(figsize=(12, 12))
            pd.plotting.scatter_matrix(df[selected_features], figsize=(12, 12),
                                       diagonal='hist', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Please select at least 2 features.")

elif page == "Linear Regression":
    st.header("🤖 Linear Regression Analysis")
    st.markdown("Build and evaluate a linear regression model to predict housing prices.")

    # Prepare data
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Sidebar for model parameters
    st.sidebar.subheader("Model Parameters")
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    # Train model
    with st.spinner("Training model..."):
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    st.success("✅ Model trained successfully!")

    # Display data split info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Testing Samples", len(X_test))
    with col3:
        st.metric("Total Features", X.shape[1])

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "📈 Predictions", "🔍 Residual Analysis", "⚙️ Model Coefficients"])

    with tab1:
        st.subheader("Model Performance Metrics")

        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)

        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Display metrics in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Set Performance**")
            st.metric("R² Score", f"{train_r2:.4f}")
            st.metric("RMSE", f"{train_rmse:.4f}")
            st.metric("MAE", f"{train_mae:.4f}")

        with col2:
            st.markdown("**Test Set Performance**")
            st.metric("R² Score", f"{test_r2:.4f}")
            st.metric("RMSE", f"{test_rmse:.4f}")
            st.metric("MAE", f"{test_mae:.4f}")

        # Interpretation
        st.markdown("---")
        st.subheader("Model Interpretation")
        st.markdown(f"""
        - The model explains **{test_r2*100:.1f}%** of the variance in housing prices
        - Average prediction error: **${test_mae:.2f}k**
        - Model shows **{'good' if abs(train_r2 - test_r2) < 0.05 else 'some'}** generalization
        """)

        if abs(train_r2 - test_r2) < 0.05:
            st.success("✅ The model generalizes well to unseen data!")
        else:
            st.warning("⚠️ There's a gap between training and test performance. Consider feature engineering or regularization.")

    with tab2:
        st.subheader("Actual vs Predicted Prices")

        # Create prediction plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Training set
        axes[0].scatter(y_train, y_train_pred, alpha=0.5, color='blue')
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price ($1000)')
        axes[0].set_ylabel('Predicted Price ($1000)')
        axes[0].set_title(f'Training Set\nR² = {train_r2:.4f}')
        axes[0].grid(True, alpha=0.3)

        # Test set
        axes[1].scatter(y_test, y_test_pred, alpha=0.5, color='green')
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Price ($1000)')
        axes[1].set_ylabel('Predicted Price ($1000)')
        axes[1].set_title(f'Test Set\nR² = {test_r2:.4f}')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        st.info("Points closer to the red diagonal line indicate better predictions.")

    with tab3:
        st.subheader("Residual Analysis")

        # Calculate residuals
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred

        # Create residual plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Residual plot - Training
        axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.5, color='blue')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 0].set_xlabel('Predicted Price')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Training Set: Residual Plot')
        axes[0, 0].grid(True, alpha=0.3)

        # Residual plot - Test
        axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.5, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Price')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Test Set: Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Residual distribution - Training
        axes[1, 0].hist(train_residuals, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Training Set: Residual Distribution')

        # Residual distribution - Test
        axes[1, 1].hist(test_residuals, bins=30, edgecolor='black', color='lightgreen', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Test Set: Residual Distribution')

        plt.tight_layout()
        st.pyplot(fig)

        st.info("Residuals should be randomly scattered around zero with no clear pattern.")

    with tab4:
        st.subheader("Model Coefficients")

        # Create coefficient dataframe
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)

        st.markdown(f"**Model Intercept:** {model.intercept_:.4f}")

        # Display coefficients table
        st.dataframe(
            coef_df[['Feature', 'Coefficient']].style.background_gradient(
                subset=['Coefficient'], cmap='RdYlGn'
            ),
            use_container_width=True
        )

        # Visualize coefficients
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Coefficients (Impact on Price)')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **Interpretation:**
        - **Positive coefficients** (green): Increase in feature leads to increase in price
        - **Negative coefficients** (red): Increase in feature leads to decrease in price
        - **Magnitude**: Larger absolute values indicate stronger influence
        """)

elif page == "PandasAI Chat":
    st.header("🤖 PandasAI Chat")
    st.markdown("Ask questions about the Boston Housing data in natural language!")

    # API Key input
    api_key = st.text_input("Enter your PandasAI API Key:", type="password")

    if api_key:
        os.environ["PANDASAI_API_KEY"] = api_key

        try:
            agent = Agent(df)

            # Example questions
            st.markdown("**Example Questions:**")
            st.markdown("""
            - What is the average price of houses?
            - Which features have the highest correlation with price?
            - Show me the top 10 most expensive houses
            - What is the distribution of rooms per dwelling?
            - How does crime rate affect housing prices?
            """)

            # User question input
            user_question = st.text_area("Enter your question:", height=100)

            if st.button("Ask PandasAI"):
                if user_question:
                    with st.spinner("Analyzing..."):
                        try:
                            response = agent.chat(user_question)

                            st.subheader("Answer:")

                            # Check if response is a plot
                            if isinstance(response, plt.Figure):
                                st.pyplot(response)
                            elif isinstance(response, pd.DataFrame):
                                st.dataframe(response)
                            else:
                                st.write(response)

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a question.")
        except Exception as e:
            st.error(f"Error initializing PandasAI: {str(e)}")
    else:
        st.info("Please enter your PandasAI API key to use PandasAI chat feature. Get your API key at https://pandabi.ai")

        st.markdown("---")
        st.subheader("Without API Key - Basic Data Query")

        query_type = st.selectbox(
            "Select Query Type",
            ["Top N Records", "Filter by Column", "Group Statistics"]
        )

        if query_type == "Top N Records":
            n = st.slider("Number of records", 5, 50, 10)
            sort_col = st.selectbox("Sort by", df.columns.tolist())
            ascending = st.checkbox("Ascending order")

            result = df.sort_values(by=sort_col, ascending=ascending).head(n)
            st.dataframe(result)

        elif query_type == "Filter by Column":
            col = st.selectbox("Select column", df.columns.tolist())
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            range_vals = st.slider(
                f"Select {col} range",
                min_val, max_val, (min_val, max_val)
            )

            filtered = df[(df[col] >= range_vals[0]) & (df[col] <= range_vals[1])]
            st.write(f"Found {len(filtered)} records")
            st.dataframe(filtered)

        elif query_type == "Group Statistics":
            col = st.selectbox("Group by (binned)",
                             ["CHAS", "RAD"])

            grouped = df.groupby(col)['MEDV'].agg(['mean', 'median', 'std', 'count'])
            st.dataframe(grouped)

elif page == "Correlation Analysis":
    st.header("🔗 Correlation Analysis")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, ax=ax)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)

    # Top correlations with target
    st.subheader("Top Correlations with Housing Price (MEDV)")

    correlations = df.corr()['MEDV'].drop('MEDV').sort_values(key=abs, ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Positive Correlations:**")
        positive = correlations[correlations > 0].sort_values(ascending=False)
        for feat, corr in positive.items():
            st.write(f"- {feat}: {corr:.4f}")

    with col2:
        st.markdown("**Negative Correlations:**")
        negative = correlations[correlations < 0].sort_values()
        for feat, corr in negative.items():
            st.write(f"- {feat}: {corr:.4f}")

    # Bar chart of correlations
    st.subheader("Correlation Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in correlations.values]
    correlations.plot(kind='barh', color=colors, ax=ax)
    ax.set_xlabel('Correlation with MEDV')
    ax.set_title('Feature Correlations with Housing Price')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Boston Housing Price Analysis | Built with Streamlit & PandasAI</p>
    </div>
    """,
    unsafe_allow_html=True
)

