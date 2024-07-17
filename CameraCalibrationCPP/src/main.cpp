#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
using namespace std;

int import_original_images(std::string path,vector<cv::Mat> &images,cv::Size  PatternSize,
							vector<vector<cv::Point2f>> &all_corners,bool im_show=false);
vector<vector<cv::Point3f>> get_control_points_world_coordinate_system (cv::Size checker_size,float length,int image_num);


int main(int argc, char *argv[]) {
	std::string image_folder_path = "/home/vahid/cpp/square";
	std::vector<cv::Mat> original_images;
	cv::Size patternsize(14,10); //interior number of corners
	float square_length = 0.0250f;// in m
	vector<vector<cv::Point2f>> Checker_bord_2D_points;
	vector<vector<cv::Point3f>> checker_board_3D_points;
	vector< vector<cv::Point3f>> control_points;
	int image_num = import_original_images(image_folder_path,original_images,
							patternsize,Checker_bord_2D_points); 
	control_points = get_control_points_world_coordinate_system (patternsize,square_length,image_num);
	
	// Compute intrinsic and extrinsic
	cv::Mat kmatrix,distcoeffs,R,T;

	cv::Size a = cv::Size(original_images[0].rows,original_images[0].cols);
	double reprojection_err = cv::calibrateCamera(control_points, Checker_bord_2D_points, a
						, kmatrix, distcoeffs, R, T);
	
	std::cout << "Process Finished" << std::endl;
	std::cout << "cameraMatrix : " << kmatrix << std::endl;
    std::cout << "distCoeffs : " << distcoeffs << std::endl;
	std::cout << "re_projection_error : " << reprojection_err << std::endl;
    //std::cout << "Rotation vector : " << R << std::endl;
    //std::cout << "Translation vector : " << T << std::endl;
}

vector <vector<cv::Point3f>> get_control_points_world_coordinate_system (cv::Size checker_size,float length,int image_num)
{
	/* Descrption:
		Defining the world coordinates for 3D points
	*/
	vector<cv::Point3f> object_points;	
	for (int i=0;i<checker_size.height;i++)
	{
		for (int j = 0; j < checker_size.width; j++)
		{
			object_points.push_back(cv::Point3f(j*length,i*length,0));
		}
		
	}
	vector<vector<cv::Point3f>> all_points;
	for (int i = 0; i < image_num; i++)
	{
		all_points.push_back(object_points);
	}
	
	return all_points;

}

int import_original_images(std::string path,vector<cv::Mat> &images,cv::Size  PatternSize,
							vector<vector<cv::Point2f>> &all_corners,bool im_show)
{
	/* Descrption:
		Read all images from the path and store them in image vector
	*/
	// find all images in the folder
	std::vector <cv::String> image_names;
	cv::glob (path+"/*.bmp",image_names,true);//recursive
	int image_num = 0;
	// save the found images to the vector of images
	cv::Mat Temp;
	vector <cv::Point2f> corners;
	for (size_t i=0;i<image_names.size();++i)
	{
		Temp = cv::imread(image_names[i]);
		if (Temp.empty()) continue; //only proceed if sucsessful
		// ........ Do the processing on image .......

		//CALIB_CB_FAST_CHECK saves a lot of time on images		
		bool patternfound = cv::findChessboardCorners(Temp, PatternSize, corners,
        					cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
        					+ cv::CALIB_CB_FAST_CHECK);

		if(patternfound)
		{
			    image_num++; 
				cv::Mat Tempgray;
				cv::cvtColor( Temp, Tempgray, cv:: COLOR_RGB2GRAY); 
  				cv::cornerSubPix(Tempgray, corners, cv::Size(11, 11), cv::Size(-1, -1),
				 			 cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
				cv::drawChessboardCorners(Temp, PatternSize, cv::Mat(corners), patternfound);
				images.push_back(Temp);	
				all_corners.push_back(corners);
				// End of process on image
				
				if (im_show)
				{
					cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    				cv::imshow( "Display window", Temp );                // Show our image inside it.
					cv::waitKey(0);             // Wait for a keystroke in the window
				}
				
		}	

	}
	return image_num;
}