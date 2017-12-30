#include <opencv2\opencv.hpp>
#include <opencv\cv.h>
#include <queue>

using namespace cv;
using namespace std;

Mat original_image, binimage;
int thresh = 100, N = 5;

class Player
{
private:
	vector<Point> triangle;
	Point center; //center of triangle
	vector<vector<Point>> boxes; 
	vector<Point> box_centers;
	char team; //team
	short number; //player's number
	Mat image;

	void setTeam()
	{
		Mat labels = Mat::zeros(image.size(), CV_8UC1);
		vector<vector<Point>> contour;
		contour.push_back(triangle);
		drawContours(labels, contour, -1, Scalar(255), CV_FILLED);

		Rect roi;
		Scalar area;

		roi = boundingRect(triangle);
		area = mean(image(roi), labels(roi));
		team = area[0] < 200 ? 'a' : 'b';

		labels.release();
		contour.clear();
	}

	Point getSquareCenter(vector<Point> box)
	{
		int x = 0;
		int y = 0;
		
		for (size_t i = 0; i < box.size(); i++)
		{
			x += box[i].x;
			y += box[i].y;
		}
		return(Point(x / box.size(), y / box.size()));
		
	}

	bool getColor(int idx)
	{
		Mat labels = Mat::zeros(image.size(), CV_8UC1);
		drawContours(labels, boxes, -1, Scalar(255), CV_FILLED);

		Rect roi;
		Scalar area;

		roi = boundingRect(boxes[idx]);
		area = mean(image(roi), labels(roi));

		labels.release();
		return area[0] < 200 ? false : true;
	}

	/*
	void setNumber()
	{
		Mat labels = Mat::zeros(image.size(), CV_8UC1);
		drawContours(labels, boxes, -1, Scalar(255), CV_FILLED);

		Rect roi;
		Scalar area;

		number = 0;
		short fill;
		for (size_t i = 0; i < boxes.size(); i++)
		{
			roi = boundingRect(boxes[i]);
			area = mean(image(roi), labels(roi));

			fill = area[0] < 200 ? 0 : 1;
			number += fill;
			number << 1;
		}
		labels.release();
	}
	*/

	bool orientation(Point pstart, Point pend, Point point)
	{
		double d = (pend.x - pstart.x)*(point.y - pstart.y) - (pend.y - pstart.y)*(point.x - pstart.x);

		if (d < 0)
			return false;

		return true;
	}

	double eucleadDistance(Point& a, Point& b)
	{
		return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
	}

	void calcNumber()
	{
		int backIdx1 = 10;
		int backIdx2 = 10;
		double dist;
		double minDist = image.size().height*image.size().width;

		for (int i = 0; i < boxes.size(); i++)
			box_centers.push_back(getSquareCenter(boxes[i]));

		for (int i = 0; i < boxes.size(); i++)
			for (int j = i + 1; j < boxes.size(); j++)
			{
				dist = eucleadDistance(box_centers[i], box_centers[j]);
				if (dist < minDist)
				{
					minDist = dist;
					backIdx1 = i;
					backIdx2 = j; 
				}
			}

		// индексы передних квадратов
		int frontIdx1 = 0;
		int frontIdx2 = 0;
		while (frontIdx1 == backIdx1 || frontIdx1 == backIdx2)
			frontIdx1++;
		while (frontIdx2 == backIdx1 || frontIdx2 == backIdx2 || frontIdx2 == frontIdx1)
			frontIdx2++;

		// определ€ем ближайший квадрат из передних
		int startbox = backIdx1;
		double dist1 = eucleadDistance(box_centers[startbox], box_centers[frontIdx1]);
		double dist2 = eucleadDistance(box_centers[startbox], box_centers[frontIdx2]);

		int next = dist1 < dist2 ? 1 : 2;

		// ориентаци€ игрока
		if (orientation(box_centers[backIdx1], box_centers[backIdx2], box_centers[frontIdx1]))
		{
			startbox = backIdx2;
			next = next == 1 ? 2 : 1;
		}

		// получение номера
		number = 0;
		if (getColor(startbox))
			number++;
		number <<= 1;
		if (getColor(next == 1 ? frontIdx1 : frontIdx2))
			number++;
		number <<= 1;
		if (getColor(next == 1 ? frontIdx2 : frontIdx1))
			number++;
		number <<= 1;
		if (getColor(startbox == backIdx1 ? backIdx2 : backIdx1))
			number++;
	}

public:
	Player(vector<Point>& tri, Mat& img)
	{
		image = img.clone();
		for (int i = 0; i < tri.size(); i++)
			triangle.push_back(tri[i]);

		setTeam();
		number = -1;
	}

	~Player()
	{
		triangle.clear();
		boxes.clear();
		box_centers.clear();
		image.release();
	}

	void addBoxes(vector<Point>& box)
	{
		boxes.push_back(box);

		//if (boxes.size() == 4)
			//setNumber();
	}

	int getNumber()
	{
		return number;
	}

	char getTeam()
	{
		return team;
	}

	Point getCenter()
	{
		return center;
	}

	void updatePlayer()
	{
		if (boxes.size() == 4)
			calcNumber();
	}

	void drawPlayer(Mat& img)
	{
		Rect roi;
		Scalar area, color;

		// Draw Triangle
		const Point* pp = &triangle[0];
		int n = (int)triangle.size();
		color = team == 'a' ? Scalar(0, 255, 255) : Scalar(0, 0, 255);

		polylines(img, &pp, &n, 1, true, color, 1, LINE_AA);

		// Draw Squares
		Mat labels = Mat::zeros(img.size(), CV_8UC1);
		drawContours(labels, boxes, -1, Scalar(255), CV_FILLED);

		for (size_t i = 0; i < boxes.size(); i++)
		{
			pp = &boxes[i][0];
			n = (int)boxes[i].size();

			// is filled
			roi = boundingRect(boxes[i]);
			area = mean(image(roi), labels(roi));
			color = area[0] < 200 ? Scalar(0, 255, 0) : Scalar(255, 0, 0);

			polylines(img, &pp, &n, 1, true, color, 1, LINE_AA);
		}
		labels.release();

		// Draw Enclosing Circle
		Point2f center;
		float radius;
		vector<Point> points;
		for (int i = 0; i < boxes.size(); i++)
			for (int j = 0; j < boxes[i].size(); j++)
				points.push_back(boxes[i][j]);
		if (points.size() > 0)
		{
			minEnclosingCircle(points, center, radius);
			circle(img, center, radius, Scalar(0, 255, 255), 1, 8);
		}
		points.clear();
	
		// Draw Number
		//setNumber();
		putText(img, to_string(number), center, FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 255), 1, CV_FILLED);

		imshow("image", img);
	}	
};

// angle (for squares)
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

static void getAllCenters(vector<Point>& centers, vector<vector<Point>>& shape)
{
	centers.clear();

	for (int i = 0; i < shape.size(); i++)
	{
		int x = 0;
		int y = 0;
		for (int j = 0; j < shape[i].size(); j++)
		{
			x += shape[i][j].x;
			y += shape[i][j].y;
		}
		centers.push_back(Point(x / shape[i].size(), y / shape[i].size()));
	}
}

static double eucleadDistance(Point& a, Point& b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

void findShapes(const Mat& image, vector<vector<Point>>& squares, vector<vector<Point>>& triangles)
{
	squares.clear();
	triangles.clear();

	Mat gray, pyr, timg, gray0(image.size(), CV_8U);
	vector<vector<Point>> contours;

	cvtColor(image, gray, CV_BGR2GRAY);

	pyrDown(gray, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, gray0, image.size());

	Canny(gray0, gray, 50, thresh, 3);
	dilate(gray, gray, Mat(), Point(-1, -1));
	erode(gray, gray, Mat(), Point(-1, -1));

	findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	
	Mat labels = Mat::zeros(image.size(), CV_8UC1);
	drawContours(labels, contours, -1, Scalar(255), 1);
	
	imshow("contours", labels);
	labels.release();

	vector<Point> approx;

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

		if (approx.size() == 4 &&
			fabs(contourArea(Mat(approx))) > 100 &&
			isContourConvex(Mat(approx)))
		{
			
			//double maxCosine = 0;		
			//for (int j = 2; j < 5; j++)
			//{
			//	double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
			//	maxCosine = MAX(maxCosine, cosine);
			//}

			//// если убрать проверку, то иногда принимает треугольники за четырЄхугольники
			//if (maxCosine < 0.3)
				squares.push_back(approx);
		}

		if (approx.size() == 3 &&
			fabs(contourArea(Mat(approx))) > 100 && 
			isContourConvex(Mat(approx)))
				triangles.push_back(approx);

		approx.clear();
	}
}

static void getPlayers(Mat& image, vector<Player>& players, vector<vector<Point>>& triangles, vector<vector<Point>>& squares, vector<Point>& tri_centers, vector<Point>& squ_centers)
{
	getAllCenters(tri_centers, triangles);
	getAllCenters(squ_centers, squares);

	multimap <double, vector<Point>> dist;
	for (size_t i = 0; i < triangles.size(); i++)
	{
		players.push_back(Player(triangles[i], image));

		dist.clear();
		for (size_t j = 0; j < squares.size(); j++)
			dist.insert(make_pair(eucleadDistance(tri_centers[i], squ_centers[j]), squares[j]));

		int k = 0;
		auto it = dist.begin();
		while (it != dist.end() && k < 4)
		{			
			players[i].addBoxes((*it).second);
			k++;
			it++;
		}
	}
}


void clearPlayers(vector<Player>& players)
{
	for (int i = 0; i < players.size(); i++)
		players[i].~Player();
	players.clear();
}


int setNumber(Mat& image, vector<vector<Point>> boxes)
{
	Mat labels = Mat::zeros(image.size(), CV_8UC1);
	Rect roi;
	Scalar area;

	int number = 0;
	int fill;
	for (int i = 0; i < boxes.size(); i++)
	{
		roi = boundingRect(boxes[i]);
		area = mean(image(roi), labels(roi));

		fill = area[0] < 200 ? 0 : 1;
		number += fill;
		number << 1;
	}
	labels.release();
	return number;
}

int main()
{
	

	//ofstream fout("log.txt");

	vector<vector<Point>> squares;
	vector<vector<Point>> triangles;
	vector<Player> players;
	vector<Point> tri_centers;
	vector<Point> squ_centers;

	cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("contours", CV_WINDOW_AUTOSIZE);

	// —татический режим
	//-----------------------------------------------
	/*original_image = imread("test.png");

	cvtColor(original_image, binimage, CV_BGR2GRAY);
	adaptiveThreshold(binimage, binimage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 2);
	dilate(binimage, binimage, Mat(), Point(-1, -1));
	erode(binimage, binimage, Mat(), Point(-1, -1));

	clearPlayers(players);

	findShapes(original_image, squares, triangles);
	getPlayers(binimage, players, triangles, squares, tri_centers, squ_centers);

	for (int i = 0; i < players.size(); i++)
	{
		players[i].updatePlayer();
		players[i].drawPlayer(original_image);
	}*/
	//-----------------------------------------------

	// захват видео с камеры
	// VideoCapture cap = VideoCapture(0);

	// захват видео их файла
	// VideoCapture cap = VideoCapture("test1.mpg");
	VideoCapture cap = VideoCapture("test1.mp4");

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	for (;;)
	{
		cap >> original_image; // get a new frame from camera

		if (!cap.grab())
			break;

		cvtColor(original_image, binimage, CV_BGR2GRAY);
		adaptiveThreshold(binimage, binimage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 2);
		dilate(binimage, binimage, Mat(), Point(-1, -1));
		erode(binimage, binimage, Mat(), Point(-1, -1));

		clearPlayers(players);

		findShapes(original_image, squares, triangles);
		getPlayers(binimage, players, triangles, squares, tri_centers, squ_centers);

		for (int i = 0; i < players.size(); i++)
		{
			players[i].updatePlayer();
			players[i].drawPlayer(original_image);
		}

		if (waitKey(30) >= 0) break;
	}	

	waitKey(0);	
	destroyAllWindows(); 

	// освобождаем ресурсы	
	cap.release();
	original_image.release();
	binimage.release();

	return 0;
}