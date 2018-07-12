/*
 * opencvtest.cpp
 *
 *  Created on: Mar 15, 2017
 *      Author: aspa1
 */

#include <ros/ros.h>

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;




/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);



// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>

#ifdef PYLON_WIN_BUILD
#    include <pylon/PylonGUI.h>
#endif

// Namespace for using pylon objects.
using namespace Pylon;

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 1;

// The exit code of the sample application.
int exitCode = 0;

Pylon::PylonAutoInitTerm autoInitTerm;

// Before using any pylon methods, the pylon runtime must be initialized.
//PylonInitialize();

// Create an instant camera object.
CInstantCamera camera;

// Create pointers to access the camera Width and Height Parameters
GenApi::CIntegerPtr width;
GenApi::CIntegerPtr height;

// image column and row
int img_col;
int img_row;

// Create pylon image format converter and pylon image
CImageFormatConverter formatConverter;

// Create a Pylon Image which will be used to creat OpenCV images later
CPylonImage pylonImage;

// This smart pointer will receive the grab result data.
CGrabResultPtr ptrGrabResult;

// Create a OpenCV image
Mat openCVImage;
Mat gray_image;
Mat draw_h, edge_h;
Mat hough_line_h;
Mat det_line_h;
Mat gray_image_h;
Mat dst_blur;
Mat gray_image_blur;
Mat edge_h_blur;
Mat roi;
Mat det_line;

Mat draw_v, edge_v;
Mat hough_line_v;
Mat det_line_v;
Mat gray_image_v;

// Parameters for edge detection
int lowThreshold = 340;
int const max_lowThreshold = 500;
int ratio = 3;
int kernel_size = 1;

int threshold_value = 256;
int threshold_value_max =255;

/// Variables for dilation and erosion
int erosion_elem = 0;
int erosion_size = 35;
int dilation_elem = 0;
int dilation_size = 35;
int const max_elem = 2;
int const max_kernel_size = 50;
int erosion_type = 0;
int dilation_type = 0;

/// Variables for blur and resize
int resize_val = 25;
int blur_val1 = 5;
int blur_val2 = 3;
int blur_val3 = 2;

int blur_de = 3;
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 100;

/// Variables for find the contours
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
Mat contourImage;

//The resolution of the parameter r in pixels. We use 1 pixel as default.
int rho = 1;
int max_linPix = 1;
//The minimum number of intersections to “detect” a line
int lin_thr = 97;
//The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
int minlinlen = 122;
//The maximum gap between two points to be considered in the same line.
int maxlingap = 100;
// the maximum value for all line generation parameters
int max_linThreshold = 500;
//The resolution of the parameter \theta in radians
int rad_theta = 1;

// Initilaize Basler camera
int caminit (){
	// attach the object of camera to the camera device found first
	camera.Attach(CTlFactory::GetInstance().CreateFirstDevice());

	// Specify the output pixel format
	formatConverter.OutputPixelFormat = PixelType_BGR8packed;

	// The parameter MaxNumBuffer can be used to control the count of buffers
    // allocated for grabbing. The default value of this parameter is 10.
    camera.MaxNumBuffer = 1;

    try
	    {
		// Print the model name of the camera.
		cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

		// Open the camera before accessing any parameters.
		camera.Open();

		// Start the grabbing of c_countOfImagesToGrab images.
		// The camera device is parameterized with a default configuration which
		// sets up free-running continuous acquisition.
		camera.StartGrabbing( c_countOfImagesToGrab);

		// Camera.StopGrabbing() is called automatically by the RetrieveResult() method
		// when c_countOfImagesToGrab images have been retrieved.
		while ( camera.IsGrabbing())
		{
			// Wait for an image and then retrieve it. A timeout of 5000 ms is used.
			camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);

			// Image grabbed successfully?
			if (ptrGrabResult->GrabSucceeded())
			{
				ROS_INFO("Connected to the target camera successfully!!!");
			}
			else
			{
				cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
			}
		}
		}
		catch (const GenericException &e)
		{
			// Error handling.
			cerr << "An exception occurred." << endl
			<< e.GetDescription() << endl;
			exitCode = 1;
		}
    // Comment the following two lines to disable waiting on exit.
	cerr << endl << "Press Enter to exit." << endl;
	return exitCode;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
     }
  }
  //-- Show what you got
  imshow( window_name, frame );
 }

// Face Detection function
int facedete(){
	 int face_n = 0;
     Mat frame;
	  //-- 1. Load the cascades
	   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	   while( true )
	       {
	     frame = dst_blur;
	     //-- 2. Apply the classifier to the frame
	         if( !frame.empty() )
	         { detectAndDisplay( frame ); }
	         else
	         { printf(" --(!) No captured frame -- Break!"); break; }

	         int c = waitKey(10);
	         if( (char)c == 'c' ) { break; }
	        }
    return (face_n);
}




int main( int argc, char** argv)
{
    ros::init(argc, argv, "cvimage_node");
    ros::NodeHandle n;
    // Ros sleep rate
    ros::Rate rs(100);

	// Initialize the camera
	if(caminit() == 1) { ROS_INFO("Initialization failed!!!"); }
	else
	{
		ROS_INFO("Capturing Images!!!");
		while(ros::ok())
			{
			  camera.StartGrabbing(1,GrabStrategy_OneByOne);
			  camera.RetrieveResult( 1000, ptrGrabResult, TimeoutHandling_ThrowException);
			  // Image grabbed successfully?
			  if (ptrGrabResult->GrabSucceeded())
			  {
				img_col = ptrGrabResult->GetWidth();
				img_row = ptrGrabResult->GetHeight();
				// Convert the grabbed buffer to a pylon image
				formatConverter.Convert(pylonImage, ptrGrabResult);
				openCVImage = cv::Mat(img_row, img_col, CV_8UC3, (uint8_t *) pylonImage.GetBuffer());
				//imwrite("test.jpg", openCVImage);
				//ROS_INFO("Image saved!!!");
			  }
			  else
			  cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
			  ROS_INFO(" Next image!!!");
		     //imshow("real time image",openCVImage);
		     //waitKey(1);
			img_col = openCVImage.cols;
			img_row = openCVImage.rows;
			// select the interesting region
			Rect roi_rec (img_col/6,img_row/4, 2*img_col/3,3*img_row/5);
			roi = openCVImage(roi_rec);
			Mat control_bar;
			control_bar= Mat::zeros(1,100, roi.type());
			control_bar = Scalar::all(255);
			imshow("control bar",control_bar);
			waitKey(1);
			/// Reduce noise with blur function
			createTrackbar( "Blur d: ", "control bar", &blur_val1, MAX_KERNEL_LENGTH);
			createTrackbar( "Blur delta color: ", "control bar", &blur_val2, MAX_KERNEL_LENGTH);
			createTrackbar( "Blur delta space: ", "control bar", &blur_val3, MAX_KERNEL_LENGTH);
			//GaussianBlur(gray_image, gray_image, Size(1,1), 0, 0);
/*			 for ( int i = 1; i < blur_val; i = i + 2 )
			     { bilateralFilter( roi, dst_blur, i, i*2, i/2);}*/
			 bilateralFilter( roi, dst_blur, blur_val1, blur_val2, blur_val3);
			// blur( roi, roi, Size(1,1), Point(-1,-1));
			// GaussianBlur( roi, roi, Size(1,1), 0, 0 );

			 // convert rgb image to gray image
			 cvtColor(dst_blur, gray_image, CV_BGR2GRAY );
			 //imshow("roi image to gray image",gray_image);
			 //waitKey(1);
		     facedete();
		     rs.sleep();
			};
	}
    // Releases all pylon resources.
    //PylonTerminate();
	//destroy the attached device
    camera.DestroyDevice();
    return exitCode;
}
