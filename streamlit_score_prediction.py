import streamlit as st
import math
import numpy as np
import pickle
import base64
st.set_page_config(page_title='T20_Score_Predictor',layout="centered"  )
model = pickle.load(open('forest_score_predi_pickel','rb'))

st.markdown("<h1 style='text-align: center; #333;'> T20 Score Predictor </h1>", unsafe_allow_html=True)

image_path = "C:/Users/rk272/OneDrive - Indian Institute of Technology Bombay/Desktop/score_prediction_model/score_prediction_model/96_removebg_preview.jpg"

# Read the image file
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Display the image using HTML
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{encoded_image});
        background-attachment: fixed;
        # background-color: #F0F0F0;
        opacity: 1;
        # z-index: -1; 
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .label-text {
        color: red; /* Change this to the desired color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.markdown('<p class="label-text">This is a red label</p>', unsafe_allow_html=True)


options=('Australia','New Zealand','England','South Africa','West Indies','Sri Lanka','Pakistan','India','Bangladesh','Scotland','Zimbabwe','Netherlands','Ireland','Afghanistan','United Arab Emirates','Namibia')
batting_team= st.selectbox('Select the Batting Team ',options, index=options.index('India')) 
prediction_array = []
if batting_team == 'Australia':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
elif batting_team == 'Bangladesh':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
elif batting_team == 'England':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
elif batting_team == 'India':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
elif batting_team == 'Ireland':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
elif batting_team == 'Namibia':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
elif batting_team == 'Netherlands':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
elif batting_team == 'New Zealand':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
elif batting_team == 'Pakistan':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
elif batting_team == 'Scotland':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
elif batting_team == 'South Africa':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
elif batting_team == 'Sri Lanka':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
elif batting_team == 'United Arab Emirates':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
elif batting_team == 'West Indies':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
elif batting_team == 'Zimbabwe':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

bowling_team= st.selectbox('bowling_team ',('Australia','New Zealand','England','South Africa','West Indies','Sri Lanka','Pakistan','India','Bangladesh','Scotland','Zimbabwe','Netherlands','Ireland','Afghanistan','United Arab Emirates','Namibia'),index=options.index('Australia'))
if bowling_team==batting_team:
    st.error('Bowling and Batting teams should be different')

if bowling_team == 'Australia':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
elif bowling_team == 'Bangladesh':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
elif bowling_team == 'England':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
elif bowling_team == 'India':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
elif bowling_team == 'Ireland':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
elif bowling_team == 'Namibia':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
elif bowling_team == 'Netherlands':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
elif bowling_team == 'New Zealand':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
elif bowling_team == 'Pakistan':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
elif bowling_team == 'Scotland':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
elif bowling_team == 'South Africa':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
elif bowling_team == 'Sri Lanka':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
elif bowling_team == 'United Arab Emirates':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
elif bowling_team == 'West Indies':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
elif bowling_team == 'Zimbabwe':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]


col1, col2,col3 = st.columns(3)


with col1:
    Overs = st.number_input('Enter the Current Over',min_value=5.1,max_value=19.5,value=5.1,step=0.1)
    if Overs-math.floor(Overs)>0.5:
        st.error('Please enter valid over input as one over only contains 6 balls')
with col2:
    runs = st.number_input('Enter Current runs',min_value=0,max_value=354,step=1,format='%i')

wickets =st.slider('Enter Wickets fallen till now',0,9,value=1)
wickets=int(wickets)


with col3:
    runs_last_5_over = st.number_input('Runs scored in the last 5 overs'  ,min_value=0,max_value=runs,step=1,format='%i')
 
         
         
col4, col5,col6 = st.columns(3)
with col4:
    if wickets==0:
        wickets_last_5_over=0
        st.success(wickets_last_5_over)
    else:
     wickets_last_5_over=st.slider('Wickets taken in the last 5 overs',0,wickets)
     


with col5:
    boundaries_in_last_5_over = st.number_input('boundaries in the last 5 overs',min_value=0,max_value=30,step=1,format='%i')

with col6:
    dot_ball_in_last_5_over = st.number_input('dot balls in the last 5 overs',min_value=0,max_value=30, step=1,format='%i')


prediction_array = prediction_array + [Overs, runs, wickets_last_5_over, runs_last_5_over, boundaries_in_last_5_over, dot_ball_in_last_5_over, wickets]
prediction_array = np.array([prediction_array])
predict = model.predict(prediction_array)

if st.button('Predict Score'):
    my_prediction = int(round(predict[0]))
    x=f'PREDICTED MATCH SCORE : {my_prediction-5} to {my_prediction+5}' 
    st.success(x)