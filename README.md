# HERO-Glove-SVM-Detection

Introduction: 

This code was developed to be used with the HERO Glove (https://toronto.ctvnews.ca/hero-glove-helps-restore-basic-hand-movements-to-stroke-patients-1.4483785), an orthotic assistive and rehabilitative device created to assist users with stroke and spinal cord injuries perform activities of daily living (ADL).

One of the main design considerations for the glove has to do with its control modes. Currently the glove switches states between flexion and extension by sensing sudden jerk using the glove’s IMU (tinyTILE - Intel Curie Dev Board). The glove also has a manual mode in the form of a button. However, the main goal is to use a control mode which can allow the glove to simply feel like an extension of the user’s body through an intent-based control mode. 

This code allows users to train a set of gestures upon first setup that feels the most natural to them. From that point onward the user can use their trained gestures to trigger the glove. 

Another potential use of this code is to track how many times a user has completed a set gesture (e.g. assigned therapy exercises like stretches, reaches, and ADLs) and use that number to notify the user, their occupational therapist, and their physiotherapist about their progress. 

Usage:

*** Setup ***

Python Version: 3.7

Arduino Version: 1.8.9

- Open the Arduino Code
- Plug in the tinyTILE and install via Board Manager
- Add curiepme.cpp and curiepmh.h into the same folder as the .ino code: https://github.com/intel/Intel-Pattern-Matching-Technology
- Then go to Tools -> Manage Libraries -> Install SerialFlash
- When the code compiles, ignore the warnings: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing] int f = *(int *)val;
- Plug in the tinyTILE
- Upload the code to the tinyTILE
- Open the Python Code
- Import libraries as needed (listed at the top of the code)

*** First Time Launch ***

- Add the Good Training Set CSVs to the same place where the Python Code is
- Ensure the bool FORCE_RETRAIN = true in the Arduino Code (towards the end of the variable initialization section)
- Ensure the boolean RELOAD_GOOD_PARAMETERS = True in the Python Code (towards the end of the variable initialization section)
- Upload the Code to the tinyTILE
- Run the Python Code and wait for it to completely finish running
- Open the Arduino Serial Monitor and you can now see the results of the classification of real-time data that you now send it
- The “Good Training Set” parameters are now saved onto the tinyTILE
- Switch bool FORCE_RETRAIN = false to ensure you don’t retrain the data accidently

Understanding the Prediction Results of the “Good Training Set”
  - The tinyTILE at rest will return a NEGATIVE
  - Pronating the tinyTILE will return a GLOVE TRIGGER ON 0
  - Performing a side to side shake with the tinyTILE will return a GLOVE TRIGGER ON 1 (because the tinyTILE’s accelerometer is almost always shaking, or has residual noise; this is the most common prediction)
  - Performing a slow, forward reach will return a GLOVE TRIGGER ON 2

*** Using the Trained Parameters to Classify Real-Time Gestures (no training) ***

- Ensure the bool FORCE_RETRAIN = false in the Arduino Code (towards the end of the variable initialization section)
- Upload the Code to the tinyTILE
- Open Serial Monitor to see the results of gesture classification

*** Training for the First Time ***

- Ensure the bool FORCE_RETRAIN = true in the Arduino Code (towards the end of the variable initialization section)
- Ensure the boolean RELOAD_GOOD_PARAMETERS = False in the Python Code (towards the end of the variable initialization section)
- Upload the Arduino Code to the tinyTILE
- Run the Python Code (ensure Arduino’s Serial Monitor is not open)
- Press enter and make the training motion in approximately 3 seconds or less (counter on the left side of the data being printed goes up by 100 entries per second)
- Wait for the Python Code to finish running (this ensure all the parameters from training are sent over with no interruptions)
- Open the Arduino Serial Monitor and you can now see the results of the classification of real-time data that you now send it

*** Reloading the “Good” Training Set ***

- Ensure the bool FORCE_RETRAIN = true in the Arduino Code (towards the end of the variable initialization section)
- Ensure the boolean RELOAD_GOOD_PARAMETERS = True in the Python Code (towards the end of the variable initialization section)
- Upload the Code to the tinyTILE
- Run the Python Code and wait for it to completely finish running
- Open the Arduino Serial Monitor and you can now see the results of the classification of real-time data that you now send it
- Switch bool FORCE_RETRAIN = false to ensure you don’t retrain the data accidently
