import pandas as pd
import numpy as np
import plotly.express as px

def survival_demographics():
    """Creating a function to analyze survival patterns based
    on passenger class, sex, and age group."""
    
    # Loading and copying the dataset
    url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv'
    df = pd.read_csv(url)
    df1 = df.copy()
    
    # Creating bins for age groups and assigning labels
    bins = [0, 12, 19, 59, float('inf')]
    labels = ['Child', 'Teenager', 'Adult', 'Senior']
    df1['AgeGroup'] = pd.cut(df1['Age'], bins=bins, labels=labels, right=True)
    
    # Group by class, sex, and age group
    grouped = df1.groupby(['Pclass', 'Sex', 'AgeGroup'])
    
    # Calculate the number of passengers and survivors
    results = grouped.agg(
        TotalPassengers=('Survived', 'count'),
        Survivors=('Survived', 'sum')
    ).reset_index()
    
    # Calculate survival rate
    results['SurvivalRate'] = (results['Survivors'] / results['TotalPassengers'])
    
    # Sort the results for better readability
    results = results.sort_values(by=['Pclass', 'Sex', 'AgeGroup'])
    
    # Display the results
    return results

st.write("Did women have a higher survival rate than men?")

def visualize_demographic():
    """Visualizing survival rates by passenger class,
    sex, and age group using a bar chart."""
    
    # Get the survival demographics data
    results = survival_demographics()
    
    # Aggregate the data to get the overall survival rate by gender
    gender_data = results.groupby('Sex', as_index=False).agg(
        TotalPassengers=('TotalPassengers', 'sum'),
        Survivors=('Survivors', 'sum')
    )
    gender_data['SurvivalRate'] = gender_data['Survivors'] / gender_data['TotalPassengers']
    
    # Create a bar chart using Plotly Express
    fig = px.bar(
        gender_data,
        x='Sex',
        y='SurvivalRate',
        title='Survival Rate by Gender',
        color='Sex',
        labels={'SurvivalRate': 'Survival Rate', 'Sex': 'Gender'},
        color_discrete_map={'male': 'blue', 'female': 'pink'},
    )
    
    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title='Gender',
        yaxis_title='Survival Rate',
        yaxis_tickformat='.0%',
        yaxis=dict(range=[0, 1]),
        title_x=0.5,
        template='plotly_white'
    )
    
    return fig

def family_groups():
    """Creating a function to explore the relationship
    between family size, passenger class, and ticket fare."""
    
    # Loading and copying the dataset
    url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv'
    df = pd.read_csv(url)
    df1 = df.copy()
    
    # Calculate family size
    df1['family_size'] = df1['SibSp'] + df1['Parch'] + 1
    
    # Group by family size and passenger class
    grouped = df1.groupby(['family_size', 'Pclass'])
    
    # Calculate average fare and number of passengers
    results = grouped.agg(
        avg_fare=('Fare', 'mean'),
        num_passengers=('PassengerId', 'count'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()
    
    # Sorting the results by Pclass and family_size in ascending order
    results = results.sort_values(by=['Pclass', 'family_size'], ascending=[True, True])
    
    # Display the results
    return results

def last_names():
    """Creating a function to extract and analyze last names
    from the passenger names in the Titanic dataset."""
    
    # Loading and copying the dataset
    url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv'
    df = pd.read_csv(url)
    df1 = df.copy()
    
    # Extract last names from the 'Name' column
    df1['LastName'] = df1['Name'].apply(lambda x: x.split(',')[0].strip())
    
    # Group by last name to count occurrences
    last_name_counts = df1['LastName'].value_counts().reset_index()
    last_name_counts.columns = ['LastName', 'Count']
    
    # Display the results
    return last_name_counts

st.write("The results neither agree nor disagree with the data table 
         provided in family_groups() since that table did not have
         last names inside of it. It would appear from my below
         visualization that the last names with the highest
         ticket fares are associated with Pclass 1 though.")

st.write("What last names had the highest average ticket fare and what class were they in?")

def visualize_families():
    """Creating a visualization to explore the relationship
    between family last name and average fare paid."""
    
    # Get the last names data counts
    last_name_counts = last_names()
    last_name_counts.columns = ['LastName', 'Count']
    
    # Load the original dataset to get fare information
    url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv'
    df = pd.read_csv(url)
    
    # Extract last names from the 'Name' column
    df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    
    # Filter for valid Pclass values (1, 2, 3)
    df = df[df['Pclass'].isin([1, 2, 3])]
    
    # Convert Pclass to a categorical variable
    df['Pclass'] = df['Pclass'].astype('category')
    
    # Merge last name counts with fare information
    merged = pd.merge(last_name_counts, df[['LastName', 'Fare', 'Pclass']], on='LastName', how='left')
    
    # Group by last name to calculate average fare
    fare_data = merged.groupby(['LastName', 'Pclass']).agg(
        AverageFare=('Fare', 'mean'),
        Count=('Count', 'first')
    ).reset_index()
    
    # Filter to include only last names with more than one occurrence
    fare_data = fare_data[fare_data['Count'] > 1]
    
    # Sort by AverageFare in descending order and select the top 10
    top_10_families = fare_data.nlargest(10, 'AverageFare')
    
    # Create a bar chart using Plotly Express
    fig = px.bar(
        top_10_families,
        x='LastName',
        y='AverageFare',
        color='Pclass',
        title='Top 10 Families by Average Fare Paid',
        labels={'AverageFare': 'Average Fare', 'LastName': 'Family Last Name', 'Pclass': 'Passenger Class'},
        barmode='group',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title='Family Last Name',
        yaxis_title='Average Fare',
        title_x=0.5,
        template='plotly_white'
    )
    
    # Show the figure
    return fig