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

using namespace std;
using namespace cv;

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

// Image processing to get the target position (x,y) under the coordinate of the 2-D plane,
// i.e. the value of Z direction is not generated by imgage processing
float img2pos(){
	 float x = 0, y =0 , z = 0;
    /// Resize the input image
    createTrackbar( "resize coefficient: ", "control bar", &resize_val,100);
    double percent = resize_val*0.01;
    ROS_INFO("percent of shrinkage :%f ", percent);
    if (resize_val != 0){
    resize(gray_image, gray_image, Size(), percent, percent);
    resize(roi, roi, Size(), percent, percent);
    }

    // Threshold the gray image
    createTrackbar( "threshold_value:", "control bar", &threshold_value,256);
    createTrackbar( "threshold_value_max:", "control bar", &threshold_value_max,255);
   if( threshold_value != 256 && threshold_value < threshold_value_max ){
    threshold( gray_image, gray_image, threshold_value, threshold_value_max,0);
    ROS_INFO("threshold_value :%d ", threshold_value);
   } else
	ROS_INFO("threshold does not apply! ");

    /// Create Erosion Trackbar
    createTrackbar( "Erosion Element", "control bar", &erosion_elem, max_elem );
    createTrackbar( "Erosion Kernel size", "control bar",&erosion_size, max_kernel_size);
    /// Apply the erosion operation
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    ROS_INFO("erosion_type :%d ", erosion_type);
    Mat element_ero = getStructuringElement( erosion_type,
                                         Size(1, 2*erosion_size + 1 ),
                                         Point( -1, -1 ) );
    ROS_INFO("erosion_size :%d ", erosion_size);
    erode( gray_image, gray_image_h, element_ero );

    /// Create Dilation Trackbar
    createTrackbar( "Dilation Element", "control bar", &dilation_elem, max_elem);
    createTrackbar( "Dilation Kernel size", "control bar", &dilation_size, max_kernel_size);
    ROS_INFO("dilation_size :%d ", dilation_size);
    /// Apply the dilation operation
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
      else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
      else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    ROS_INFO("dilation_type :%d ", dilation_type);
    Mat element_dil = getStructuringElement( dilation_type,Size( 1, 2*dilation_size + 1),Point( -1, -1 ) );
    dilate( gray_image_h, gray_image_h, element_dil );

    //GaussianBlur( gray_image_h, gray_image_h, Size(1,1), 0,0);
    //imshow("gray image",gray_image);
    //waitKey(1);

    // edge detection
    /// Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", "control bar", &lowThreshold, max_lowThreshold);
    /// Canny detector
    /// Create a Trackbar for user to enter the kernel_size
     createTrackbar( "Kernel size:", "control bar", &kernel_size, 2);
     Canny( gray_image_h, edge_h, lowThreshold, lowThreshold*ratio, kernel_size*2+3 );
     ROS_INFO("lowThreshold :%d ", lowThreshold);

/*     // Erode and dilate the edge image
     Mat kernel_edgdil = Mat::ones(3, 3, CV_8UC1);
	 dilate(edge_h, edge_h, kernel_edgdil);
	 erode(edge_h, edge_h, kernel_edgdil);*/

     blur( edge_h, edge_h_blur, Size(3,3), Point(-1,-1));
     //GaussianBlur( edge_h, edge_h_blur, Size(1,1), 0,0);
	 // Bilateral filtering
/*	 for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
	 { bilateralFilter(edge_h, edge_h_blur, i, i*2, i/2);
	   //waitKey(100);
	 }*/
	 imshow("edge detection", edge_h_blur);
     waitKey(1);

     /// Create a Trackbar for user to control line generation
     createTrackbar( "rho:", "control bar", &rho, max_linPix);
	 createTrackbar( "lin_thr:", "control bar", &lin_thr, max_linThreshold);
	 createTrackbar( "rad_theta:", "control bar", &rad_theta, 10);
	 createTrackbar( "linminlen:", "control bar", &minlinlen, max_linThreshold);
	 createTrackbar( "maxlingap:", "control bar", &maxlingap, max_linThreshold);

	 Mat smooth_h;
	 det_line_h = Mat::zeros(edge_h_blur.size(), edge_h_blur.type());
	 det_line_h = Scalar::all(0);
	 cvtColor(det_line_h, det_line_h, CV_GRAY2BGR );

     vector<Vec4i> lines, lines_h, lines_v;
	 HoughLinesP(edge_h_blur, lines, (double)rho, CV_PI/180, (double)lin_thr, (double)minlinlen, (double)maxlingap);
	 ROS_INFO("line number is :%d ", (int)lines.size());

	 for( size_t i = 0; i < lines.size(); i++ )
		 {
		   Vec4i l = lines[i];
		   line( gray_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 4);
	 }
	 imshow("gray image",gray_image);
	 waitKey(1);

     int r = 0, g = 0, b = 255, v_n = 0, v_n_d = 0, h_n = 0, h_n_d = 0;
     int reso_rec = 100;
	 int l_v[100] = {}; // store the vertical line numbers
	 int l_v_s[100] = {}; // store the vertical line numbers which are the same line
	 int l_h[100] = {}; // store the horizontal line number
	 int l_h_s[100] = {}; // store the horizontal line numbers which are the same line

	 // find the vertical line and store them
	 for( size_t i = 0; i < lines.size(); i++ )
	 {
	   Vec4i l = lines[i];
      if (abs((l[3]- l[1]))!=0 && abs((l[2]- l[0]))/(abs((l[3]- l[1]))) < 0.00000001  )
	   {
		 l_v[v_n] = i;
		 v_n++;
	   }
	 }
	 ROS_INFO(" Vertical line number is :%d ", v_n);

	 // draw one vertical line if they are at the same level by using rectangle shapes to enclose
  	 for( size_t i = 0; i < lines.size(); i++ ) {
		   Vec4i l = lines[l_v[i]];
		   if ( abs((l[2]- l[0])) <20 && -180/3.14159 < atan(abs((l[2]- l[0]))/img_row) < 180/3.14159 )
			   {
			   	 l_v_s[v_n] = l_v[i];
			   	 //l_v[v_n] = i;
				 v_n++;
				 line(roi, Point(l[0], 0), Point(l[2], img_row), Scalar(0,0,255), 2, 4);
			   }
	 }
/*  	if (v_n > 0){
  		for( int i = 0; i < v_n; i++ )
  		  {
  			Vec4i l = lines[l_v_s[i]];
  			for( int j = i+1; j < v_n-1; j++ ){
  				Vec4i l = lines[l_v_s[j]];
  				if ( );
  			}
  		  }
  	}*/

	 // draw one vertical line if they are at the same level by using rectangle shapes to enclose
/*    if (v_n > 0){
		   int find_in_val_v = false;
		   for( int j = 0; j < img_col/reso_rec; j++ ){
			 for( size_t i = 0; i < v_n-1; i++ )
			  {
				 Vec4i l = lines[l_v[i]];
				 if (  (j*reso_rec < l[1] < j*reso_rec+reso_rec) && (j*reso_rec < l[3] < j*reso_rec+reso_rec)){
					 l_v_s[v_n_d] = l_v[i];
					 find_in_val_v = true;
				 } else
				 {
					 l_v_s[v_n_d] = l_v[i];
				 }
			  }
			   v_n_d++;
			   find_in_val_v = false;
			}
		   ROS_INFO("v_n_d :%d ", v_n_d);
			 for( size_t i = 0; i < v_n_d; i++ )
			 {
			   Vec4i l = lines[l_v_s[i]];
			   line(roi , Point(l[0], 0), Point(l[2], img_row), Scalar(0,0,255), 2, 4);
			 }
		}

	 if (v_n > 0){
	 		   int find_in_val_v = false;
	 		   for( int j = 0; j < img_col/reso_rec; j++ ){
	 			 for( size_t i = 0; i < v_n-1; i++ )
	 			  {
	 				 Vec4i l = lines[l_v[i]];
	 				 if (  (j*reso_rec < l[1] < j*reso_rec+reso_rec) && (j*reso_rec < l[3] < j*reso_rec+reso_rec)){
	 					 l_v_s[v_n_d] = l_v[i];
	 					 //find_in_val_v = true;
		 				 v_n_d++;

	 				 } else
	 				 {
	 					 l_v_s[v_n_d] = l_v[i];
	 				 }

	 			  }

	 			}
	 		   ROS_INFO("v_n_d :%d ", v_n_d);
	 			 for( size_t i = 0; i < v_n_d; i++ )
	 			 {
	 			   Vec4i l = lines[l_v_s[i]];
	 			   line(roi , Point(l[0], 0), Point(l[2], img_row), Scalar(0,0,255), 2, 4);
	 			 }
	 		}
*/


		 // find the horizontal line and store them
		 for( size_t i = 0; i < lines.size(); i++ )
		 {
		   Vec4i l = lines[i];
		   if ( abs((l[2]- l[0]))!=0 && abs((l[3]- l[1]))/(abs((l[2]- l[0]))) < 0.00000001 )
		   {
			   l_h[h_n] = i;
			   h_n++;
		   }
	 }
	 ROS_INFO(" Horizontal line number is :%d ", h_n);

	 for( size_t i = 0; i < lines.size(); i++ )
		 {
		   Vec4i l = lines[l_h[i]];
		   if ( abs((l[3]- l[1])) <10)
			   {
				 line(roi, Point(0, l[1]), Point(img_col, l[3]), Scalar(0,255,0), 2, 4);
			   }
	 }

 /*
	 if (h_n > 0){
		 // draw one horizontal line if they are at the same level by using rectangle shapes to enclose
		 int find_in_val_h = false;
		 for( int j = 0; j < img_row/reso_rec; j++ ){
		   for( size_t i = 0; i < h_n-1; i++ )
			{
			 Vec4i l = lines[l_h[i]];
			 if (  (j*reso_rec < l[0] < j*reso_rec+reso_rec) && (j*reso_rec < l[2] < j*reso_rec+reso_rec)){
				 l_h_s[h_n_d] = l_h[i];
				 find_in_val_h = true;
			 }
		  }
		   if (find_in_val_h == true)
		   h_n_d++;
		   find_in_val_h = false;
		}
		 for( size_t i = 0; i < h_n_d; i++ )
		 {
		   Vec4i l = lines[l_h_s[i]];
		   line(roi , Point(0, l[1]), Point(img_col, l[3]), Scalar(0,255,0), 2, 4);
		 }
	 }
*/

	 cvtColor(det_line_h, det_line_h, CV_BGR2GRAY );
	 GaussianBlur( det_line_h, det_line_h, Size(1,1), 0,0);
	 imshow("ROI original image",roi);
	 waitKey(1);

    // find contour
/*    findContours(det_line_h, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0,0) );
     ROS_INFO("the size of contours is: %lu",contours.size());
     //contourImage = roi;
     if (contours.size() > 0){
			 for (unsigned int i=0; i<contours.size(); i++)
				 //if (hierarchy[i][3] >= 0)   //has parent, inner (hole) contour of a closed edge (looks good)
					 drawContours(roi, contours, i, Scalar(255, 0, 0), 3);
			 imshow("ROI original image",roi);
			 waitKey(1);
     }
*/
    return (x,y,z);
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
		     img2pos();
		     rs.sleep();
			};
	}
    // Releases all pylon resources.
    //PylonTerminate();
	//destroy the attached device
    camera.DestroyDevice();
    return exitCode;
}
