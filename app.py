import streamlit as st
import pickle
import pandas as pd
teams= ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']
cities= ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
       'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
       'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
       'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah',
       'Mohali', 'Bengaluru']
pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictiors')
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('select the batting team',teams)
with col2:
    balling_team = st.selectbox('select the balling team',teams)
selected_city = st.selectbox('select the selected city',sorted(cities))
targets = st.number_input('Target')

col3,col4,col5 = st.columns(3)
with col3:
    score =st.number_input('Score')
with col4:
    overs = st.number_input('overs')
with col5:
    wickets = st.number_input('Wickets')
if st.button('Predict probabilities'):
    runs_left = targets - score
    balls_left = 120-(overs*6)
    wicket= 10-wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[balling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[targets],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team+"-"+str(round(win*100))+"%")
    st.header(balling_team+"-"+str(round(loss*100))+"%")
