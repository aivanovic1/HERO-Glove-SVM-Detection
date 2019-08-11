//Test of learning using neural network and computer link
//Will save or load a set of training examples and counter examples
//Uses Curie.PME library for access to neural network https://github.com/intel/Intel-Pattern-Matching-Technology
//Uses SerialFlash library for access to on-board flash memory chip https://github.com/PaulStoffregen/SerialFlash
#include "CurieIMU.h"
#include <Servo.h>
#include "CuriePME.h"
#include <SerialFlash.h>
#include <SPI.h>
#include <cmath>

//******************* Variables ****************
// Motors
Servo myservoflex;                //create servo object to control flexion
Servo myservoextend;              //create servo object to control extension
const int extendMotor = 150;      //150 = fully squeeze bottle; 80 = partially squeeze bottle
const int retractMotor = 50;      //50 = fully extend fingers; 120 = partially extend fingers
const int extend = 1;             //fingers extended
const int flex = 2;               //fingers flexed
const int relax = 3;              //fingers relaxed
int fingerposition = relax;       //finger position tracker
// Input Timeout
unsigned long currenttime = 0;    //current time
const int shortinterval = 1000;   //short interval between accepting inputs (ms)
const int longinterval = 1000;    //long interval (ms)
unsigned long timeoutend = 0;     //time when inputs are accepted again
bool timeout = false;             //flag for input timeout
// Lock Button
bool button = false;              //button position
bool lock = false;                //lock
volatile bool motionflag = false; //motion flag

//constants for readability
const int IMULow = -32768;    //IMU min and max at 4g range 
const int IMUHigh = 32767;
const int repeats = 5;        //number of times to repeat during training
const int patterns = 3;       //number of different movement patterns trained
const int axis = 3;           //number of axis - 3(xyz) or 6(xyz accel and gyro)

const int neuronsize = 128;   //size of neurons in bytes

const int x = 0;
const int y = 1;
const int z = 2;

int readAx = 0;
int readAy = 0;
int readAz = 0;

float inbuffA[axis];          //buffer for incoming imu data
float inbuffJerk[axis];       //buffer for integrated imu data (integrating acceleration)
float inbuffJounce[axis];     //buffer for integrated velocity data (position)

const float accel_dt = 0.01;  //inverse of accelerometer freq
const float IMU_scale = 9.8;  //9.8 m/s^2 per G

const int window = 300;                   // window for classifying an action
const int nr_boundaries = 5;              // amount of boundaries created to parse the data             
const int slice = window/nr_boundaries;   // the size of the "slices" made by the boundary number     
int window_counter = 0;                   // progress of the window
bool inference_phase = false;             // boolean of whether or not online prediction has begun
bool firstFillDone = false;               // boolean of whether or not the buffer has been filled for the first time
const int predFreq = 50;                  // how much data outputs pass between each prediction 
float wdata[axis][window];                // the online buffer

int total[axis];                          //running sum for moving average

//***************** NOT USED *********************

// Small buffer handling - only how much windowing is required
byte sbuff[axis][window];     //small buffers
bool sbff = false;            //small buffer filled flag
int sbend = 0;                //pointer for end of small buffer (it will start from 0 index)
int sbstart = 1;              //pointer for start of small buffer

// Large buffer used to model neurons
byte lbuff[axis][neuronsize]; //large buffers for last 128 filtered points
bool lbff = false;            //large buffer filled flag
int lbend = 0;                //pointer for end of large buffer
int lbstart = 1;              //pointer for start of large buffer

//************************************************

float prevInbuffA[axis];      //used for trapezoidal rule - stores previous input
float prevInbuffJerk[axis];   //used for trapezoidal rule - stores previous input

int count = 1;
bool trainingPhase = true;              // boolean of whether or not training has begun 
bool waitingForTrainingData = false;    // boolean of whether or not the chip is waiting for Python to send over its results from training  

int context[axis] = {1, 2, 3};  //because context 0 is global, xyz has to be 1, 2, 3
int answer;                     //classifier catagory

byte singledatafeed[126];     //vector with every 3rd value of xyz interlaced
byte printdata[axis];         //for diagnostic printing
int timer = 0;                //to check for disconnects in main loop

const int nr_class = 4;                                             // number of classes for multi-class classification
int nr_sv;                                                          // size of support vectors
const int features_per_axis = 2;                                    // number of features per axis
const int n_features = features_per_axis * axis * (nr_boundaries);  // number of features for the feature vector
const int nr_intercepts = (nr_class*(nr_class-1))/2;

float* min_feat_vals;         // minimum features - was used for normalization, data not normalized anymore
float* peak_to_peak_vals;     // peak to peak values - was used for normalization, data not normalized anymore
float* intercepts;            // intercepts (fed into the prediction/voting function)
float** support_vectors;      // support vectors (used as an imput into the SVM kernel for classication)
float** dual_coefs;           // vector of the dual_coefs (classification parameter)
int* n_supports;              // number of support vectors

//NN data arrays
byte trainingexamples[patterns][repeats][axis][neuronsize];
byte counterexamples[repeats][axis][neuronsize];

//file names
const char *TP_filename = "TrainingParameters.dat";     // file names to save and load data under
const char *CE_filename = "CounterExamples.dat";        // file names to save and load data under
const char *TE_filename = "TrainingExamples.dat";       // file names to save and load data under
byte *slpointer;                                        //save & load iteration pointer    

// at the START wait for Python to say that it is ready
bool dont_send_on_serial = true;

bool FORCE_RETRAIN = false; //CHANGE THIS TO TRUE IF YOU WANT TO RETRAIN

//******************** Setup *******************
void setup() {
  // Set up servos and I/O pins
  myservoflex.attach(A1);         // attaches the servo
  myservoextend.attach(A3);       // attaches the servo
  pinMode(2, INPUT);
  pinMode(3, OUTPUT);
  digitalWrite(3, LOW);
  pinMode(4, INPUT);
  pinMode(6, INPUT);
  pinMode(7, OUTPUT);
  digitalWrite(7, LOW);
  pinMode(8, INPUT);
  digitalWrite(13, LOW);  
  button = digitalRead(2);        //initialize lock button to starting position
  
  // Set up IMU
  CurieIMU.begin();
  // Set Accelerometer sample rate in Hz (12.5, 25, 50, 100, 200, 400, 800, 1600)
  CurieIMU.setAccelerometerRate(100);
  // Set Accelerometer range (2g, 4g, 8g, 16g)
  CurieIMU.setAccelerometerRange(4);

  // Set Gyro sample rate in Hz (25, 50, 100, 200, 400, 800, 1600, 3200)
  // CurieIMU.setGyroRate(100);
  // Set Gyro range (2000, 1000, 500, 250, 125)
  // CurieIMU.setGyroRange(250);

  // Relax motors when glove turns on and wait before accepting inputs
  movemotors(relax);
  settimeout(longinterval);

  // Set up serial communication
  Serial.begin(9600);
  while(!Serial);

  // Init. SPI Flash chip
  if (!SerialFlash.begin(ONBOARD_FLASH_SPI_PORT, ONBOARD_FLASH_CS_PIN)) {
    Serial.println("Unable to access SPI Flash chip");
  }

  CuriePME.begin();

  //Set previous values to 0 when taking first derivative of dataset

  prevInbuffA[x] = 0.0;
  prevInbuffA[y] = 0.0;
  prevInbuffA[z] = 0.0;

  prevInbuffJerk[x] = 0.0;
  prevInbuffJerk[y] = 0.0;
  prevInbuffJerk[z] = 0.0;

  if (SerialFlash.exists(TP_filename) && FORCE_RETRAIN == false){
    Serial.println("LOADING");
    loadParameters(TP_filename);
    
    waitingForTrainingData = false;
    inference_phase = true;
    trainingPhase = true;
  }
  
  //Serial.println("Setup Complete");
}

//***************************** Main Loop ***********************************
void loop() {

  //diagnostic check for disconnection
  timer++;
  if (timer == 20000){
    //Serial.println ("Looping");
    timer = 0;
  }
  
  //use moving average filter to stream incoming imu data into buffers
  if (CurieIMU.dataReady()){
    bufferdata();  
  }
}

//*************** Movement function for finger motors *******************
void movemotors (int fp){
  if (fp == extend){
    myservoflex.write(retractMotor);       //extend fingers
    myservoextend.write(extendMotor);
  }
  else if (fp == flex){
    myservoflex.write(extendMotor);        //flex fingers
    myservoextend.write(retractMotor);
  }
  else if (fp == relax){
    myservoflex.write(retractMotor);       //relax fingers
    myservoextend.write(retractMotor);
  }
}

//************** Input Timeout function ******************************
void settimeout(int howlong){
  currenttime = millis();                 //get current time
  timeout = true;                         //set timeout flag
  timeoutend = currenttime+howlong;       //set end of input timeout
}

//************** Buffer accelerometer data function ******************************
void bufferdata(){ 
  //read data from imu and map into single byte size
  //read accelerometer x,y,z
  //unsigned long t = millis();
  
  CurieIMU.readAccelerometerScaled(inbuffA[x], inbuffA[y], inbuffA[z]);
  CurieIMU.readAccelerometer(readAx, readAy, readAz);
  
  inbuffA[x] *= IMU_scale;
  inbuffA[y] *= IMU_scale;
  inbuffA[z] *= IMU_scale;

  //find jerk by taking acceleration's derivative
  
  inbuffJerk[x] = derivative(inbuffA[x], prevInbuffA[x]);
  inbuffJerk[y] = derivative(inbuffA[y], prevInbuffA[y]);
  inbuffJerk[z] = derivative(inbuffA[z], prevInbuffA[z]);

  //find jounce by taking jerk's derivative

  inbuffJounce[x] = derivative(inbuffJerk[x], prevInbuffJerk[x]);
  inbuffJounce[y] = derivative(inbuffJerk[y], prevInbuffJerk[y]);
  inbuffJounce[z] = derivative(inbuffJerk[z], prevInbuffJerk[z]);

  sbuff[x][sbend] = (byte) map(inbuffA[x], IMULow, IMUHigh, 0, 255); 
  sbuff[y][sbend] = (byte) map(inbuffA[y], IMULow, IMUHigh, 0, 255);
  sbuff[z][sbend] = (byte) map(inbuffA[z], IMULow, IMUHigh, 0, 255);

  //update previous values to current values for future derivation

  prevInbuffA[x] = inbuffA[x];
  prevInbuffA[y] = inbuffA[y];
  prevInbuffA[z] = inbuffA[z];

  prevInbuffJerk[x] = inbuffJerk[x];
  prevInbuffJerk[y] = inbuffJerk[y];
  prevInbuffJerk[z] = inbuffJerk[z];

  // this is REGULAR mode when we send accelerator measurement for TRAINING (Python)
  if (trainingPhase) {

    // if there is a message from Python
    if (Serial.available() > 0) {

        String s = Serial.readString();

        //Serial.println("SYS: ********* GOT " + s);
        
        if (String("START_WAIT").equalsIgnoreCase(s)) {
          dont_send_on_serial = true;
          Serial.println("SYS:START_WAIT ... Arduino received the START_WAIT signal");
        }
        else if (String("END_WAIT").equalsIgnoreCase(s)) {
          dont_send_on_serial = false;
          Serial.println("SYS:END_WAIT ... Arduino received the END_WAIT signal");
        }   
        else if (String("END_TRAINING").equalsIgnoreCase(s)) {
          dont_send_on_serial = true;
          waitingForTrainingData = true;
          trainingPhase = false;
          Serial.println("SYS:TRAINING_END_ACK ... Arduino received the End Of Training signal");
        }
    }

    // Python is not letting us to send any data as yet
    if (!dont_send_on_serial) {
      
        if (Serial.availableForWrite()) {
          //Writing data piece by piece because sending it all at once would cause an issue
        
          //Serial.write(":H:");
          Serial.print(count, DEC);
          Serial.write(",");
          Serial.print(millis(), DEC);
          Serial.flush();
      
          while (!Serial.availableForWrite());
          Serial.write(",");
          Serial.print(inbuffA[x], DEC);
          Serial.write(",");
          Serial.print(inbuffJerk[x], DEC);
          Serial.write(",");
          Serial.print(inbuffJounce[x], DEC);
          Serial.flush();
      
          while (!Serial.availableForWrite());
          Serial.write(",");
          Serial.print(inbuffA[y], DEC);
          Serial.write(",");
          Serial.print(inbuffJerk[y], DEC);
          Serial.write(",");
          Serial.print(inbuffJounce[y], DEC);
          Serial.flush();
      
          while (!Serial.availableForWrite());
          Serial.write(",");
          Serial.print(inbuffA[z], DEC);
          Serial.write(",");
          Serial.print(inbuffJerk[z], DEC);
          Serial.write(",");
          Serial.println(inbuffJounce[z], DEC);
          //Serial.println(":F:")
          Serial.flush();    
        }
        else {
          Serial.println("LOST CONNECTION... RECONNECTING");
          Serial.end(); 
          
          Serial.begin(9600);
          while(!Serial);
        
          // Init. SPI Flash chip
          if (!SerialFlash.begin(ONBOARD_FLASH_SPI_PORT, ONBOARD_FLASH_CS_PIN)) {
            Serial.println("Unable to access SPI Flash chip");
          }
        }
    }
  }

  // once the training is completed, wait for the Python to send it over
  if (waitingForTrainingData && Serial.available() > 0) {
     
    Serial.print("*******************START NR_SV**********************\n");
    while (Serial.available() > 0){
      byte val[4];
      Serial.readBytes(val, 4);
      int f = *(int *)val;
      nr_sv = f;
      Serial.println(f);
      break;
    }
    Serial.print("*******************END NR_SV**********************\n");

    Serial.print("*******************START MIN_FEAT_VALS**********************\n");
    while (Serial.available() > 0){
      min_feat_vals = (float*)malloc(n_features*sizeof(float)); 
      for(int i = 0; i < n_features; i++){
        byte val[4];
        Serial.readBytes(val, 4);
        float f = *(float *)val;
        min_feat_vals[i] = f;
        Serial.println(f,6);
      }
      break;
    }
    Serial.print("**************END MIN_FEAT_VALS******************************\n");

    Serial.print("*******************START PEAK_TO_PEAK_VALS**********************\n");
    while (Serial.available() > 0){
      peak_to_peak_vals = (float*)malloc(n_features*sizeof(float)); 
      for(int i = 0; i < n_features; i++){
        byte val[4];
        Serial.readBytes(val, 4);
        float f = *(float *)val;
        peak_to_peak_vals[i] = f;
        Serial.println(f,6);
      }
      break;
    }
    Serial.print("**************END PEAK_TO_PEAK_VALS******************************\n");
  
    Serial.print("*******************START INTERCEPT**********************\n");
    while (Serial.available() > 0){
      intercepts = (float*)malloc(nr_intercepts*sizeof(float)); 
      for(int i = 0; i < nr_intercepts; i++){
        byte val[4];
        Serial.readBytes(val, 4);
        float f = *(float *)val;
        intercepts[i] = f;
        Serial.println(f,6);
      }
      break;
    }
    Serial.print("**************END INTERCEPT******************************\n");
      
    Serial.print("*******************START SUPPORT_VECTORS**********************\n");
    while (Serial.available() > 0){
      support_vectors = (float **)malloc(nr_sv * sizeof(float*)); 
      for (int i=0; i<nr_sv; i++) 
         support_vectors[i] = (float *)malloc(n_features * sizeof(float)); 
         
      for(int i = 0; i < nr_sv; i++){
        for(int j = 0; j < n_features; j++){
          byte val[4];
          Serial.readBytes(val, 4);
          float f = *(float *)val;
          support_vectors[i][j] = f;
          Serial.print("Support Vector [" + String(i) + "][" + String(j) + "] = ");
          Serial.println(f,6);
        }
      }
      break;
    }
    Serial.print("**************END SUPPORT_VECTORS******************************\n");
  
    Serial.print("*******************START DUAL_COEF**********************\n");
    while (Serial.available() > 0){
      dual_coefs = (float **)malloc((nr_class-1) * sizeof(float*));
      if (dual_coefs == NULL)
        Serial.println("DUAL IS NULL"); 
      for (int i=0; i<nr_class-1; i++){ 
         dual_coefs[i] = (float *)malloc(nr_sv * sizeof(float));
         if (dual_coefs[i] == NULL)
          Serial.println("DUAL[i] IS NULL"); 
      }
         
      for(int i = 0; i < nr_class-1; i++){
        for(int j = 0; j < nr_sv; j++){
          byte val[4];
          Serial.readBytes(val, 4);
          float f = *(float *)val;
          dual_coefs[i][j] = f;
          Serial.print("Dual Coef [" + String(i) + "][" + String(j) + "] = ");
          Serial.println(f,6);
        }
      }
      break;
    }
    Serial.print("**************END DUAL_COEF******************************\n");
    
    Serial.print("*******************START N_SUPPORT**********************\n");
    while (Serial.available() > 0){ 
      n_supports = (int*)malloc(nr_class*sizeof(int));
      for(int i = 0; i < nr_class; i++){
        byte val[4];
        Serial.readBytes(val, 4);
        int f = *(int *)val;
        n_supports[i] = f;
        Serial.println(f);
      }
      break;
    }
    Serial.print("**************END N_SUPPORT******************************\n");

    saveParameters(TP_filename);

    inference_phase = true;
    waitingForTrainingData = false;
  }   

  if (inference_phase) {
    //Serial.println("STARTING INFERENCE");
    //Serial.println(String(window_counter));
    wdata[x][window_counter] = inbuffA[x];
    wdata[y][window_counter] = inbuffA[y];
    wdata[z][window_counter] = inbuffA[z];

    if (window_counter == window){
      firstFillDone = true;
      window_counter = 0;
    }
      
    if (firstFillDone && window_counter % predFreq == 0) {
      float* feature_vector = (float*)malloc(n_features*sizeof(float));
      for(int j = 0; j < window; j+= slice){
        for(int i = 0; i < axis; i++){
          float* a_dot = derivativeArray(&(wdata[i][0]), slice, j+window_counter, window);
          //float* a_ddot = derivativeArray(a_dot, slice-1, 0, slice-1);

          float mean_a = meanOfArray(&(wdata[i][0]), slice, j+window_counter, window);
          float mean_a_dot = meanOfArray(a_dot, slice-1, 0, slice-1);
          //float mean_a_ddot = meanOfArray(a_ddot, slice-2, 0, slice-2);

          free(a_dot);
          //free(a_ddot);

          int idx = (j/slice)*axis*features_per_axis+i*features_per_axis;
          
          feature_vector[idx] = mean_a;
          feature_vector[idx + 1] = mean_a_dot;
          //feature_vector[idx + 2] = mean_a_ddot; 
        }
      }

     // scaling
//      for(int i = 0; i < n_features; i++){
//        feature_vector[i] = (feature_vector[i] - min_feat_vals[i])/peak_to_peak_vals[i];
//      }
      
//      for(int i = 0; i < 45; i++){
//        Serial.println("feature_vector[" + String(i) + "] = " + feature_vector[i]);
//      }

//      Serial.println("****SUPPORT****");
//      for (int i = 0; i < n_features; i++)  
//        Serial.println(support_vectors[7][i]);
//      Serial.println("****FEATURE****");
//      for (int i = 0; i < n_features; i++)
//        Serial.println(feature_vector[i]);

      int prediction = svm_predict(feature_vector); // should pass in feature_vector as an argument
      free(feature_vector);

//      Serial.println("INF:" + String(prediction));
      
      if(prediction == 3){ 
        Serial.println(String(count) + ": NEGATIVE"); 
      }
      else {
        Serial.println(String(count) + ": GLOVE TRIGGER ON " + String(prediction));
      }  
    }
    ++window_counter;
  } 

  count++;
}

//************* Take the derivative *****************
inline float derivative(float curr, float prev){
  return (curr-prev)/accel_dt;
}

inline float* derivativeArray(float* arr, int len, int startIdx, int arrSize){
  float* derivativeArray = (float*)malloc((len-1)*sizeof(float));
  if (derivativeArray == NULL){
    Serial.print("derivativeArray is null!");
  }
  for(int i = 1; i < len; i++){
    derivativeArray[i-1] = derivative(arr[(startIdx+i)%arrSize], arr[(startIdx+i-1)%arrSize]);
  }
  return derivativeArray;
}

inline float meanOfArray(float* arr, int len, int startIdx, int arrSize){
  float sum = 0;
  for (int i = 0; i < len; i++){
    sum += arr[(startIdx+i)%arrSize];
  }
  return sum/len;
}

inline float rbf(float* a, float* b, float gamma, int n_features){
  float sub = 0;
  for (int i = 0; i < n_features; i++){
    sub += (a[i]-b[i])*(a[i]-b[i]); 
  }
  return exp(-gamma*sub);
} 
      
int svm_predict(float* input){
  
  double *kvalue = (double*)malloc(nr_sv*sizeof(double));
  double gamma = 1./n_features;

  for (int i = 0; i < nr_sv; i++){
    kvalue[i] = rbf(input, support_vectors[i], gamma, n_features);
  }

  int *start = (int*)malloc(nr_class*sizeof(int));
  start[0] = 0;
  for (int i = 1; i < nr_class; i++){
    start[i] = start[i-1] + n_supports[i-1];
  }

  int *vote = (int*)calloc(nr_class,sizeof(nr_class));

  int p = 0;
  for (int i = 0; i < nr_class; i++){
    for (int j = i+1; j < nr_class; j++){
      float sum = 0;
      int si = start[i];
      int sj = start[j];
      int ci = n_supports[i];
      int cj = n_supports[j];

      int k;
      float *coef1 = dual_coefs[j-1];
      float *coef2 = dual_coefs[i];
      
      for(k=0;k<ci;k++)
        sum += coef1[si+k] * kvalue[si+k];
      for(k=0;k<cj;k++)
        sum += coef2[sj+k] * kvalue[sj+k];
      sum += intercepts[p];

      if(sum > 0)
        ++vote[i];
      else
        ++vote[j];
      p++;
    }
  }
  
  int vote_max_idx = 0;
  for(int i = 1; i < nr_class; i++)
      if(vote[i] > vote[vote_max_idx])
        vote_max_idx = i;

  free(kvalue);
  free(start);
  free(vote);
  return vote_max_idx;
}

void saveParameters(const char *filename){
  Serial.println("Saving...");
  int fileSize = sizeof(int) + (nr_intercepts)*sizeof(float) + (nr_sv * n_features)*sizeof(float) + 
                  ((nr_class-1) * nr_sv)*sizeof(float) + (nr_class)*sizeof(int) + 
                  (n_features)*sizeof(float) + (n_features)*sizeof(float);
  Serial.println(String(fileSize));
  SerialFlashFile file;
  create_if_not_exists( filename, fileSize );
  // Open the file and save data
  file = SerialFlash.open(filename);
  file.erase();

  slpointer = (byte*) &nr_sv;
  file.write(slpointer, sizeof(int));

  slpointer = (byte*) &(min_feat_vals[0]);
  file.write(slpointer, (n_features)*sizeof(float));

  slpointer = (byte*) &(peak_to_peak_vals[0]);
  file.write(slpointer, (n_features)*sizeof(float));
  
  slpointer = (byte*) &(intercepts[0]);
  file.write(slpointer, (nr_intercepts)*sizeof(float));
  
  for(int i = 0; i < nr_sv; i++){
    slpointer = (byte*) &(support_vectors[i][0]);
    file.write(slpointer, (n_features)*sizeof(float)); 
  }

  for(int i = 0; i < nr_class-1; i++){
    slpointer = (byte*) &(dual_coefs[i][0]);
    file.write(slpointer, (nr_sv)*sizeof(float));
  }

  slpointer = (byte*) &(n_supports[0]);
  file.write(slpointer, (nr_class)*sizeof(int));

  file.close();
  Serial.println("Saving complete.");
}

void loadParameters(const char *filename){
  SerialFlashFile file;
  if (!SerialFlash.exists(filename)){
    Serial.println("No Training Parameters File Exists!");
  }
  else {
    Serial.println("Loading Training Parameters...");
    file = SerialFlash.open(filename);

    slpointer = (byte*) &nr_sv;
    file.read(slpointer, sizeof(int));

    Serial.println("Loading nr_sv complete.");

    min_feat_vals = (float*)malloc(n_features*sizeof(float));
    slpointer = (byte*) &(min_feat_vals[0]);   
    file.read(slpointer, (n_features)*sizeof(float));

    Serial.println("Loading min_feat_vals complete.");

    peak_to_peak_vals = (float*)malloc(n_features*sizeof(float));
    slpointer = (byte*) &(peak_to_peak_vals[0]);   
    file.read(slpointer, (n_features)*sizeof(float));

    Serial.println("Loading peak_to_peak_vals complete.");

    intercepts = (float*)malloc(nr_intercepts*sizeof(float));
    slpointer = (byte*) &(intercepts[0]);   
    file.read(slpointer, (nr_intercepts)*sizeof(float));

    Serial.println("Loading intercepts complete.");

    support_vectors = (float **)malloc(nr_sv * sizeof(float*)); 
    for (int i=0; i<nr_sv; i++)
      support_vectors[i] = (float *)malloc(n_features * sizeof(float)); 
    
    for(int i = 0; i < nr_sv; i++){
      slpointer = (byte*) &(support_vectors[i][0]);
      file.read(slpointer, (n_features)*sizeof(float));
    }

    Serial.println("Loading support_vectors complete.");
    
    dual_coefs = (float **)malloc((nr_class-1) * sizeof(float*)); 
      for (int i=0; i<nr_class-1; i++) 
         dual_coefs[i] = (float *)malloc(nr_sv * sizeof(float)); 
  
    for(int i = 0; i < nr_class-1; i++){
      slpointer = (byte*) &(dual_coefs[i][0]);
      file.read(slpointer, (nr_sv)*sizeof(float));
    }

    Serial.println("Loading dual_coefs complete.");

    n_supports = (int*)malloc(nr_class*sizeof(int));
    slpointer = (byte*) &(n_supports[0]);
    file.read(slpointer, (nr_class)*sizeof(int));   

    Serial.println("Loading n_supports complete.");
  }
  file.close();
  Serial.println("Loading complete."); 
}

//******************* Create if not exists ***************
//Creates a new file if it does not already exist.
bool create_if_not_exists (const char *filename, uint32_t fileSize) {
  if (!SerialFlash.exists(filename)) {
    Serial.println("Creating file " + String(filename));
    return SerialFlash.createErasable(filename, fileSize);
  }

  Serial.println("File " + String(filename) + " already exists");
  return true;
}
