#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "jpeglib.h"
#include "SLIC.h"
#include "opencv_lbp.h"
#include "cnpy.h"
#include "densecrf.h"
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include "time.h"
#include "math.h"
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

#define NumberOfSuperpixel 2000
#define LBP_VECTORS 16
#define CLASS 21	//label number
#define IterativeNumber 10

// 21 color for labels
int colors[21] = {0, 128, 32768, 32896, 8388608, 8388736,8421376, 8421504, 64, 192, 32832, 32960,8388672, 8388800, 8421440, 8421568,8192, 16512, 49152, 49280, 8404992};

unsigned int    getColor( const unsigned char * c );
unsigned char * colorize( int * map, int W, int H, unsigned char * r );
int             putColor( unsigned char * c, unsigned int cc );
void            writePPM( const char* filename, int W, int H, unsigned char* data );
void            writePNG( const char* filename, int W, int H, unsigned char* data );
void            elbp(Mat& src, Mat &dst, int radius, int neighbors);
float           calcDistance(int first, int second, vector<vector<float> > & lbp_superpixel, vector<vector<float> > & labxy_superpixel, int h, int w);


int main(int argc, char* argv[])
{
	clock_t start, finish;
	printf("main paras = %d.\n", argc);
	const char * img_name = argv[1];
	const char * img_fcn  = argv[2];
	
	//-----------------------------------
	// 01: read jpg to buffer
	//-----------------------------------
	start = clock();
	Mat raw_image = imread(img_name , 1);
	if(raw_image.empty())
		printf("imread failed!\n");
	int h = raw_image.rows;
	int w = raw_image.cols;
	int i(0), j(0);
	int x(0), y(0);	//pixel:x,y
	unsigned char * image_buffer = new unsigned char[ w * h * 4 ];	// BGRA char[4] 
    unsigned int  * pbuff        = new unsigned int [ w * h ];		// BGRA int[]
	for(i = 0; i < h; i++)
	{	
		for(j = 0; j < w; j++)
		{	
			*(image_buffer + i * w * 4 + j*4+0 ) = raw_image.at<Vec3b>(i,j).val[0];		//B
			*(image_buffer + i * w * 4 + j*4+1 ) = raw_image.at<Vec3b>(i,j).val[1];		//G
			*(image_buffer + i * w * 4 + j*4+2 ) = raw_image.at<Vec3b>(i,j).val[2];		//R
			*(image_buffer + i * w * 4 + j*4+3 ) = 0;									//A=0
		}
	}	
	finish = clock();
	printf("01 Jpeg to buffer :\t\t%.2lfs.\n", (double)(finish-start)/CLOCKS_PER_SEC);

	// char[4] transfor to int[1]
    // unsigned int (32 bits) to hold a pixel in ARGB format as follows:
    for(i = j = 0; i < w*h; i++,j+=4 )
    {
        *(pbuff + i) = *(image_buffer+j+3) + \
                      (*(image_buffer+j+2)) * 256 + \
                      (*(image_buffer+j+1)) * 256 * 256 + \
                      (*(image_buffer+j+0)) * 256 * 256 * 256;
    }
	memset( image_buffer, 0, w*h*4 );
		
	//----------------------------------------------------
    // 02: SLIC Initialize params & Perform on buffer
    //----------------------------------------------------
	start = clock();
	int    k_real  = 0;	 	//Truth number of superpixel, k_real>k generally.
    double m       = 10;		//Compactness factor. use a value from 10 to 40. Default is 10.
    int *  klabels = new int[ w * h ];		//lable map
    int    numlabels(0);
    SLIC segment;
    segment.PerformSLICO_ForGivenK( pbuff, w, h, klabels, numlabels, NumberOfSuperpixel, m );
    for(i = 0; i < w * h; i++)
        if( klabels[i] > k_real )
            k_real = klabels[i];
	k_real += 1;
    printf("Get %d superpixels.i\n",k_real);
	finish = clock();
	printf("02 SLIC on buffer :\t\t%.2lfs.\n", (double)(finish-start) / CLOCKS_PER_SEC );

	//------------------------------------------	
	// 03: put pixel cluster into vector
	//------------------------------------------
	start = clock();
	vector<vector<int> > superpixel;	// 0 ~ n superpixel
	vector<int> i_superpixel;			// the pixel:x,y in the ith superpixel 
	int j_temp(0);

	for(i = 0; i < k_real; i++)
	{	
		for(j = 0; j < w * h; j++)
		{
			if( klabels[j] == i )
			{
				y = j % w ;
				x = (j - y ) / w;
				i_superpixel.push_back(x);
				i_superpixel.push_back(y);
			}
		}
		if( i_superpixel.size() != 0 )
			superpixel.push_back(i_superpixel);	
		i_superpixel.clear();
	}
//	printf("superpixel.size=%d.\n", superpixel.size());
	finish = clock();
	printf("03 Put s_pixel_vector :\t\t%.2lfs.\n", (double)(finish-start)/CLOCKS_PER_SEC);

	//-------------------------------------
	// 04: Extract Feature 
	//-------------------------------------
	printf("Feature Extracting ...\n");
	// LBP feature[16]  LABXY[5]
	start = clock();
	vector<vector<float> > lbp_superpixel;
	vector<vector<float> > labxy_superpixel;
	vector<float> temp;
	int max_x(0), min_x(65536), max_y(0), min_y(65536);
	int color_l(0), color_a(0), color_b(0); 

    Mat image_lbp = imread(img_name, 0);		// 0:grayscale model
//	printf("image_lbp H: %d, W: %d\n", image_lbp.rows, image_lbp.cols);
	if(image_lbp.empty())
		printf("imread fail.\n");

	Mat image_lab( raw_image );		// BGR color model
//	printf("image_lab H: %d, W: %d\n", image_lab.rows, image_lab.cols);
	if(image_lab.empty())
		printf("imread fail.\n");

	Mat img_lbp; 
	Mat img_lab;
	int radius(1), neighbors(8);
	Mat lbp_feature, lab_feature;	
	MatND hist;
	int bins = LBP_VECTORS;     //default = 16
	int hist_size[] = {bins};   
	float range[]   = {0, 256};
	const float * ranges[] = {range};
	int channels[]  = {0};
	int sizeofrect;
	
	for(vector<vector<int> >::size_type ix = 0; ix < superpixel.size(); ix++) 	//superpixel label number
	{
		max_x = max_y = 0;
		min_x = min_y = 65536;
		color_l = color_a = color_b = 0;
		for(vector<int>::size_type iy = 0; iy < superpixel[ix].size(); iy += 2)	//get all x,y pixel in ith super
		{	
			if(superpixel[ix][iy  ] > max_x)	max_x = superpixel[ix][iy];
			if(superpixel[ix][iy+1] > max_y)	max_y = superpixel[ix][iy+1];
			if(superpixel[ix][iy]   < min_x)	min_x = superpixel[ix][iy];
			if(superpixel[ix][iy+1] < min_y)	min_y = superpixel[ix][iy+1];
		}
		Rect rect(min_y, min_x, max_y-min_y+1, max_x-min_x+1);	//(1,3)>rect(3,1)
		sizeofrect = (max_y-min_y+1) * (max_x-min_x+1);

		// lbp feature
   	 	img_lbp = image_lbp(rect);
    	lbp_feature = Mat(img_lbp.rows-2*radius, img_lbp.cols-2*radius,CV_8UC1, Scalar(0));
   	 	elbp(img_lbp, lbp_feature, 1, 8);
 
	   	calcHist(&lbp_feature, 1, channels, Mat(), hist, 1, hist_size, ranges, true, false);
		for(i=0; i < bins; i++)
			temp.push_back( hist.at<float>(i) );
		lbp_superpixel.push_back(temp);
		temp.clear();    	
		
		// LAB,xy feature	
		img_lab = image_lab(rect);    	//imshow("image",img);
		cvtColor(img_lab, lab_feature, COLOR_BGR2Lab);
		for(i=0; i<(max_x-min_x+1); i++)
		{	
			for(j=0; j<(max_y-min_y+1); j++)
			{
				color_l += (int)lab_feature.at<Vec3b>(i,j).val[0];				
				color_a += (int)lab_feature.at<Vec3b>(i,j).val[1];				
				color_b += (int)lab_feature.at<Vec3b>(i,j).val[2];				
			}
		}
		temp.push_back( (float)color_l / sizeofrect );
		temp.push_back( (float)color_a / sizeofrect );
		temp.push_back( (float)color_b / sizeofrect );
		temp.push_back( (max_x-min_x+1)/2 + min_x );
		temp.push_back( (max_y-min_y+1)/2 + min_y );
		labxy_superpixel.push_back(temp);
		temp.clear();    	
	}//for 0~2016
	finish = clock();	
	printf("04 : Extract feature :\t\t%.2lfs.\n", (double)(finish-start)/CLOCKS_PER_SEC);

	//----------------------------------
	// 05: Unary from file.npy test
	//----------------------------------
	start = clock();
	cnpy::NpyArray arr = cnpy::npy_load(img_fcn);	
   	float * raw_unary  = new float[ w * h * CLASS ];
   	float * unary      = new float[ k_real * CLASS ];
	memset(unary, 0.0, k_real * CLASS * sizeof(float));
	float temp_score(0);
	if(arr.shape[1] != h || arr.shape[2] != w || arr.shape[0] != CLASS)
		printf("\nimage not match npy.\n");	
	float min_score(0), max_score(0);

    // Put into unary[0-21][1-21][2-21]...
    for(i=0; i<w*h; i++)
	{	
		min_score = INT_MAX;
		max_score = INT_MIN;
		for(j=0; j<CLASS; j++)
        {	
			temp_score = ((const float *)(arr.data))[i + j*w*h];
			raw_unary[i * CLASS + j] = temp_score;
			if( temp_score < min_score )    min_score = temp_score ;
			if( temp_score > max_score )    max_score = temp_score ;
		}
		for(j=0; j<CLASS; j++)
			raw_unary[i * CLASS + j] = -log((raw_unary[i * CLASS + j] - min_score) / (max_score - min_score));
	}
	arr.destruct();	
	// pixel pros -> s_pixel pros	
	x = y = 0; 
	for(vector<vector<int> >::size_type ix = 0; ix < superpixel.size(); ix++)
	{
		for(vector<int>::size_type iy = 0; iy < superpixel[ix].size(); iy += 2) 
		{
			x = superpixel[ix][iy];
			y = superpixel[ix][iy+1];
			for(i=0; i<CLASS; i++)
				unary[ int(ix) * CLASS + i] += raw_unary[ ( x * w + y ) * CLASS + i];	
		}	
		for(i=0; i<CLASS; i++)
			unary[ int(ix) * CLASS + i] /= (( int( superpixel[ix].size() ) ) / 2 ); 
	}
	finish = clock();
	printf("05 : Read unary npy :\t\t%.2lfs.\n", (double)(finish-start)/CLOCKS_PER_SEC);

	//-------------------------------------
	// 06 : CRF model
	//-------------------------------------
#if 1
	start = clock();
	printf("Start CRF modeling...\n");
	short * s_map = new short[k_real];
	int   * map   = new int[ w * h ];
	int w1        = atoi(argv[5]);
	int std_xy    = atoi(argv[6]);
	int std_color = atoi(argv[7]);
	int w3        = atoi(argv[8]);
	int std_lbp   = atoi(argv[9]);

	DenseCRF2D crf(k_real, 1, CLASS);
	if( atoi(argv[4]) == 0 )
	{
		// 把像素点概率映射到块上
		crf.setUnaryEnergy( unary );	//块的分属21类的概率
		crf.addPairwiseGaussian_lh ( labxy_superpixel, 2,  3,  3,  3, NULL, true);	//块的坐标
		crf.addPairwiseBilateral_lh( labxy_superpixel, 5, std_xy, std_xy, std_color, std_color, std_color, w1, NULL);	//块的颜色
		crf.addPairwiseGaussian_lh ( lbp_superpixel, 16,  std_lbp,  0,  w3, NULL, false);	//块的纹理
	}
	else
	{
		//TODO 
		//crf.setUnaryEnergy( unary );	//块的分属21类的概率
		//crf.addPairwiseGaussian_lh ( labxy_superpixel, 2,  3,  3,  5, NULL, true);	//块的坐标
		//crf.addPairwiseBilateral_lh_2( labxy_superpixel, 5, 30, 30, 10, 10, 10, 5, NULL);	//块的纹理
	}
	//迭代求每个块的最大概率	
	crf.map(IterativeNumber, s_map);
	// 将块标记转为点标记，做图
    for(vector<vector<int> >::size_type ix = 0, i=0; ix < superpixel.size(); ix++,i++)
    {
        for(vector<int>::size_type iy = 0; iy < superpixel[ix].size(); iy += 2)
        {
            int x_temp = superpixel[ix][iy];
            int y_temp = superpixel[ix][iy+1];
			map[x_temp * w + y_temp] = s_map[i];
    	}
	}
	colorize( map, w, h, image_buffer );
	writePNG( argv[3], w, h, image_buffer );
	finish = clock();
	printf("06 : CRF inference :\t\t%.2lfs.\n", (double)(finish-start)/CLOCKS_PER_SEC);
	printf("Segment image save done !\n");
#endif	

    //----------------------------------
    // Clean up
    //----------------------------------
	delete[] klabels;
	delete[] image_buffer;
	delete[] pbuff;
	delete[] raw_unary;
	delete[] unary;
	delete[] map;	
	delete[] s_map;	

	return 0;
}


//----------------------
// 若干辅助函数
//----------------------

// 将RGB值存为一个整型变量值
unsigned int getColor( const unsigned char * c )
{
    return c[0] + 256*c[1] + 256*256*c[2];
}

// 从整型变量中得到RGB值
int putColor( unsigned char * c, unsigned int cc )
{
    c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
	return 0;
}

// Produce a color_image from Map Labels
unsigned char * colorize( int * map, int W, int H, unsigned char * r )
{
    for( int k=0; k<W*H; k++ )
	{
        int c = colors[ map[k] ];
		putColor( r + 3*k, c);
    }
    return r;
}

// 将数组存为PPM图
void writePPM ( const char* filename, int W, int H, unsigned char* data )
{
    FILE* fp = fopen ( filename, "wb" );
    if ( !fp )
    {
        printf ( "Failed to open file '%s'!\n", filename );
    }
    fprintf ( fp, "P6\n%d %d\n%d\n", W, H, 255 );
    fwrite  ( data, 1, W*H*3, fp );
    fclose  ( fp );
}

void writePNG ( const char* filename, int W, int H, unsigned char* data )
{
	writePPM( "temp.ppm", W, H, data);
	Mat temp = imread("temp.ppm", 1);

	FILE * fp = fopen( filename, "w");
	if(fp == NULL)
	{
        char buf[256], path[256];
        int i;
        strcpy(buf, filename);
        for(i = strlen(filename) - 1; i > 0; i--)
        {
            if(buf[i] != '/')
                buf[i] = NULL;
            else
                break;
        }
        buf[i] == NULL;
        strcpy(path, buf);
        int isCreate = mkdir(path, S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        //printf("isCreate = %d.\n", isCreate);
    }
	else
	{ 
		fclose(fp);
		remove(filename);			
	}

	try
	{
		imwrite( filename, temp);	
	}
	catch (runtime_error & ex)
	{
		printf("Exception converting image to PNG image:%s\n", ex.what());
	}

	remove("temp.ppm");
}


// 计算两个21-dimen特征的距离
float calcDistance(int first, int second, vector<vector<float> > & lbp_superpixel, vector<vector<float> > & labxy_superpixel, int h, int w)
{
	float a[CLASS], b[CLASS];
	int i(0), histcount(0);

	printf("[");
    for(vector<int>::size_type ix = 0, i=0; ix < lbp_superpixel[first].size(), i<16; ix++,i++)  //label number
    {
        a[i] = lbp_superpixel[first][ix];
        histcount +=  a[i];
        printf("%3d", int(lbp_superpixel[first][ix]) );
    }
    for(i=0; i<16; i++)
        a[i] /= histcount;  // 16个LBP特征归一化 -> a[]
    histcount = 0;

    printf("\t");
    for(vector<int>::size_type ix = 0, i=16; ix < labxy_superpixel[first].size(), i<21; ix++,i++) //label number
    {
        a[i] = labxy_superpixel[first][ix];
        printf("%5d", int(labxy_superpixel[first][ix]) );
    }
    printf("]\n");
    for(i=16; i<19; i++)
        a[i] /= 255;    // LAB归一化
    a[19] /= h;         // XY归一化
    a[20] /= w;

	printf("\n[");
    for(vector<int>::size_type ix = 0, i=0; ix < lbp_superpixel[second].size(), i<16; ix++,i++)  //label number
    {
        b[i] = lbp_superpixel[second][ix];
        histcount +=  b[i];
        printf("%3d", int(lbp_superpixel[second][ix]) );
    }
    for(i=0; i<16; i++)
        b[i] /= histcount;
    histcount = 0;

    printf("\t");
    for(vector<int>::size_type ix = 0, i=16; ix < labxy_superpixel[second].size(), i<21; ix++,i++)   //label number
    {
        b[i] = labxy_superpixel[second][ix];
        printf("%5d", int(labxy_superpixel[second][ix]) );
    }
    printf("]\n");
    for(i=16; i<19; i++)    b[i] /= 255;
    b[19] /= h;
    b[20] /= w;

    printf("\nAfter normalized.\n");
    for(i=0; i<21; i++)     printf("%f,", a[i]);
    printf("\n");
    for(i=0; i<21; i++)     printf("%f,", b[i]);
    printf("\n");

	float sum1(0), sum2(0), sum3(0);
    for(i=0; i<16; i++)
        sum1 += pow( (a[i]-b[i]), 2);
    for(i=16; i<19; i++)
        sum2 += pow( (a[i]-b[i]), 2);
    for(i=19; i<21; i++)
        sum3 += pow( (a[i]-b[i]), 2);
    return (0.7 * sqrt(sum1) + 0.2 * sqrt(sum2) + 0.1 * sqrt(sum3));
}
