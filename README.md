<h1>  Cricket Score Prediction Project</h1> <br>
<h3> Project Description</h3>
This repository contains the Cricket Score Prediction project, focusing on predicting T20 cricket match scores using machine learning techniques. The goal is to build a model that can accurately predict the first innings score in a T20 match at the end of the 30th over.
<h3>How to Use</h3><br>
<h5>Clone the Repository:</h5><br>
git clone https://github.com/yourusername/cricket-score-prediction.git
 <br>
<h5>Download the Dataset:</h5><br>
Download the T20I_ball_by_ball_updated.csv file from this repository and ensure it is placed in the same directory as the script files.
<br>
<h5>Run the Score Prediction Model:</h5> <br>
this is first file, that have to run first Execute the "score_prediction_model.py" script to train and save the model.
<br>
<h5>Deploy the Model Using Streamlit:</h5><br>
Launch the Streamlit app to start making predictions: "streamlit run streamlit_score_prediction.py"
<br>

<h3>Project Workflow</h3><br>
<h5>Data Preparation:</h5><br>
Loading and cleaning the dataset.<br>
Handling missing values and outliers.<br>
Feature Engineering:<br>
Creating relevant features such as current wickets, runs in the last few overs, etc.<br>
<h5>Model Training:</h5><br>
Preprocessing data using one-hot encoding and scaling.<br>
Training regression models and selecting the best one based on performance metrics.<br>
Saving the trained model for deployment.<br>
<h5>Deployment:</h5><br>
Building and running a Streamlit app to provide an interactive interface for score predictions.<br>
<h4>Conclusion</h4><br>
This project demonstrates a complete workflow from data preparation and model training to deployment using Streamlit. It serves as a practical example of applying machine learning techniques to predict cricket scores, specifically for T20 matches.

Feel free to explore the repository and use the provided resources to build and improve your own cricket score prediction models. If you have any questions or suggestions, please open an issue or submit a pull request.
