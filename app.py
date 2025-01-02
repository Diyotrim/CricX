from flask import Flask, render_template, request, redirect
import pandas as pd
import os
from urllib.parse import quote  # For URL encoding
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import train_test_split

# Initialize the Flask app
app = Flask(__name__)

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'player_data.csv')
FULL_NAMES_FILE = 'Full Names.csv'
DATA_FOLDER = 'data'

# Mapping cohorts to dataset filenames
COHORT_MAPPING = {
    "2008-2010": "Combined_Year_Wise_2008_2010.csv",
    "2011-2013": "Combined_Year_Wise_2011_2013.csv",
    "2014-2017": "Combined_Year_Wise_2014_2017.csv",
    "2018-2021": "Combined_Year_Wise_2018_2021.csv",
    "2022-2024": "Combined_Year_Wise_2022_2024.csv",
}

# Mapping of seasons to cohorts
cohort_mapping = {
    "2008-10": (2008, 2010),
    "2011-13": (2011, 2013),
    "2014-17": (2014, 2017),
    "2018-21": (2018, 2021),
    "2022-24": (2022, 2024),
}

# Load player data
try:
    player_data = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    player_data = None
    print(f"Error: CSV file not found at {CSV_FILE}")

# Add cohort column to player data
if player_data is not None:
    def map_cohort(season):
        for cohort, (start, end) in cohort_mapping.items():
            if start <= season <= end:
                return cohort
        return None

    player_data['cohort'] = player_data['season'].apply(map_cohort)

# Function to load Full Names mapping
def load_full_names():
    full_names_df = pd.read_csv(FULL_NAMES_FILE)
    return dict(zip(full_names_df['full_name'], full_names_df['player_name']))

# Function to load and preprocess the dataset
def load_dataset(cohort):
    if cohort not in COHORT_MAPPING:
        return None, None, None

    filepath = os.path.join(DATA_FOLDER, COHORT_MAPPING[cohort])
    data = pd.read_csv(filepath)
    player_names = data['player_name']
    data = data.drop(columns=['Unnamed: 0', 'full_name', 'team', 'Day', 'Month', 'Year', 'age_at_auction'])
    data = pd.get_dummies(data, columns=['age_category'], drop_first=True)
    data = data.fillna(0)
    X = data.drop(columns=['retention', 'player_name'])
    y = data['retention']
    return X, y, player_names

# Function to train the EasyEnsembleClassifier
def train_model(X, y):
    model = EasyEnsembleClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route to display the Dashboard page
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if player_data is None:
        return "<h1>Data not available. Please upload the CSV file.</h1>", 500

    if request.method == 'GET':
        player_names = player_data['player_name'].unique()
        cohort_list = player_data['cohort'].unique()
        return render_template('index.html', player_names=player_names, cohort_list=cohort_list)

    full_name = request.form.get('full_name')
    cohort = request.form.get('cohort')

    if not full_name or not cohort:
        return render_template('error.html', message="Both full name and cohort must be provided.")

    matched_player = player_data[player_data['full_name'].str.lower() == full_name.lower()]
    if matched_player.empty:
        return render_template('error.html', message=f"Player with full name '{full_name}' not found.")

    player_name = matched_player.iloc[0]['player_name']
    filtered_data = player_data[(player_data['player_name'] == player_name) & (player_data['cohort'] == cohort)]

    if filtered_data.empty:
        return render_template('error.html', message=f"No data found for {full_name} in cohort {cohort}.")

    player_name_encoded = quote(player_name)
    cohort_encoded = quote(cohort)
    tableau_embed_url = f"https://public.tableau.com/app/profile/aayush.tiwari1243/viz/ipl_17341821228710/Dashboard1?publish=yes&PlayerName={player_name_encoded}&Cohort={cohort_encoded}"
    player_info = filtered_data.to_dict(orient='records')[0]

    return render_template('dashboard.html', player=player_info, tableau_embed_url=tableau_embed_url, cohort=cohort)

# Route to display the Retention Model page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index1.html')

    full_name = request.form['full_name']
    cohort = request.form['cohort']

    full_name_to_player_name = load_full_names()
    player_name = full_name_to_player_name.get(full_name)

    if not player_name:
        return render_template('index1.html', error="Full name not found in the dataset.")

    X, y, player_names = load_dataset(cohort)
    if X is None or y is None:
        return render_template('index1.html', error="Cohort data not found.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_model(X_train, y_train)

    player_row = player_names == player_name
    if player_row.any():
        player_features = X[player_row].iloc[0:1]
        prediction = model.predict(player_features)[0]
        prediction_result = "Retained" if prediction == 1 else "Not Retained"
    else:
        prediction_result = "Player not found in the selected cohort!"

    return render_template('index1.html', full_name=full_name, cohort=cohort, result=prediction_result)

@app.route('/metrics_overview')
def metrics_overview():
    return render_template('metrics_overview.html')

@app.route('/redirect_to_page', methods=['GET'])
def redirect_to_page():
    # Get the selected data format from the dropdown
    data_format = request.args.get('data_format')
    
    # Redirect the user to the respective HTML page based on selection
    if data_format:
        return redirect(f"/{data_format}")
    else:
        # If no data format is selected, redirect back to metrics_overview
        return redirect('/metrics_overview')

# Routes for the different data format pages
@app.route('/quantitative.html')
def quantitative():
    return render_template('quantitative.html')

@app.route('/qualitative.html')
def qualitative():
    return render_template('qualitative.html')

@app.route('/combined.html')
def combined():
    return render_template('combined.html')


if __name__ == '__main__':
    app.run(debug=True)
