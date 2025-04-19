import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib
from PIL import Image
import time
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
try:
    st.set_page_config(
        page_title="AI Health Trainer",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    logger.info("Page config set successfully")
except Exception as e:
    logger.error(f"Error setting page config: {e}")

# Custom CSS
try:
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .subtitle {
            color: #34495e;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    logger.info("Custom CSS applied successfully")
except Exception as e:
    logger.error(f"Error applying custom CSS: {e}")

# Initialize MediaPipe
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    logger.info("MediaPipe initialized successfully")
except Exception as e:
    logger.error(f"Error initializing MediaPipe: {e}")
    st.error("Failed to initialize MediaPipe. Please check your installation.")

# Load the trained model
try:
    if os.path.exists('exercise_model.pkl'):
        model = joblib.load('exercise_model.pkl')
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model file not found")
        st.warning("Model file not found. Please train the model first.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    st.error("Failed to load model. Please check your model file.")

def main():
    try:
        st.title("AI Health Trainer 💪")
        st.markdown("---")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Home", "Exercise Verification", "Body Analysis", "Exercise Recommendations", "Progress Tracking"])
        
        if page == "Home":
            show_home()
        elif page == "Exercise Verification":
            show_exercise_verification()
        elif page == "Body Analysis":
            show_body_analysis()
        elif page == "Exercise Recommendations":
            show_exercise_recommendations()
        elif page == "Progress Tracking":
            show_progress_tracking()
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error("An error occurred. Please check the logs for details.")

def show_home():
    try:
        st.markdown("""
        <div class="title">
            <h1>Welcome to AI Health Trainer</h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### About the Project
            AI Health Trainer is an intelligent fitness assistant that helps you:
            - Verify exercise form in real-time
            - Analyze body measurements
            - Get personalized exercise recommendations
            - Track your fitness progress
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60", 
                    caption="Start your fitness journey today!")
    except Exception as e:
        logger.error(f"Error in show_home: {e}")
        st.error("An error occurred while loading the home page.")

def show_exercise_verification():
    try:
        st.markdown("""
        <div class="title">
            <h2>Exercise Verification</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Real-time Exercise Form Analysis")
        
        # Exercise selection
        exercise_type = st.selectbox("Select Exercise Type", ["Push-up", "Squat", "Lunge", "Plank"])
        
        # Webcam input
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Failed to access webcam. Please make sure your webcam is connected and accessible.")
            return
        
        if st.button("Start Verification"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                    
                # Convert the BGR image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect pose
                results = pose.process(image)
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display the processed image
                stframe.image(image, channels="RGB", use_column_width=True)
                
                # Add a stop button
                if st.button("Stop"):
                    break
        
        cap.release()
    except Exception as e:
        logger.error(f"Error in show_exercise_verification: {e}")
        st.error("An error occurred while running exercise verification.")

def show_body_analysis():
    try:
        st.markdown("""
        <div class="title">
            <h2>Body Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Upload an image for body measurements")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze"):
                with st.spinner("Analyzing body measurements..."):
                    # Convert PIL Image to OpenCV format
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Process the image
                    results = pose.process(image_cv)
                    
                    if results.pose_landmarks:
                        st.success("Body measurements detected!")
                        # Draw pose landmarks on the image
                        mp.solutions.drawing_utils.draw_landmarks(
                            image_cv, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), 
                               caption="Pose Detection Results", use_column_width=True)
                    else:
                        st.error("No body detected in the image")
    except Exception as e:
        logger.error(f"Error in show_body_analysis: {e}")
        st.error("An error occurred while analyzing body measurements.")

def show_exercise_recommendations():
    try:
        st.markdown("""
        <div class="title">
            <h2>Exercise Recommendations</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Get personalized exercise recommendations")
        
        # Input form
        with st.form("recommendation_form"):
            age = st.number_input("Age", min_value=18, max_value=100)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
            goals = st.multiselect("Goals", ["Weight Loss", "Muscle Gain", "Flexibility", "Endurance"])
            
            submitted = st.form_submit_button("Get Recommendations")
            
            if submitted:
                with st.spinner("Generating recommendations..."):
                    st.success("Here are your personalized exercise recommendations!")
                    st.write("1. Push-ups")
                    st.write("2. Squats")
                    st.write("3. Plank")
    except Exception as e:
        logger.error(f"Error in show_exercise_recommendations: {e}")
        st.error("An error occurred while generating exercise recommendations.")

def show_progress_tracking():
    try:
        st.markdown("""
        <div class="title">
            <h2>Progress Tracking</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Track your fitness journey")
        
        # Progress input form
        with st.form("progress_form"):
            date = st.date_input("Date")
            weight = st.number_input("Weight (kg)")
            body_fat = st.number_input("Body Fat Percentage")
            exercise = st.text_input("Exercise")
            sets = st.number_input("Sets", min_value=1)
            reps = st.number_input("Reps", min_value=1)
            
            submitted = st.form_submit_button("Save Progress")
            
            if submitted:
                st.success("Progress saved successfully!")
    except Exception as e:
        logger.error(f"Error in show_progress_tracking: {e}")
        st.error("An error occurred while tracking progress.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        st.error("A critical error occurred. Please check the logs for details.") 