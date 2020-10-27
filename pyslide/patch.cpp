#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	string image_path=samples::findFile("fangfang.png");
	Mat img = imread(image_path, IMREAD_COLOR);

	if (img.empty())
	{
		cout << "could not find image";  
		return 1;
	}

	imshow("Display", img);
	int k  = waitKey(0);
	
	return 0;
}


