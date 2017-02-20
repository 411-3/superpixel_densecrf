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

using namespace cv;
using namespace std;

#define LBP_VECTORS 16
#define CLASS 21	//label number
#define IterativeNumber 10

// 21 color for labels
int colors[21] = {0, 128, 32768, 32896, 8388608, 8388736,8421376, 8421504, 64, 192, 32832, 32960,8388672, 8388800, 8421440, 8421568,8192, 16512, 49152, 49280, 8404992};
unsigned int getColor( const unsigned char * c );
int putColor( unsigned char * c, unsigned int cc );
unsigned char * colorize( int * map, int W, int H, unsigned char * r );
void writePPM( const char* filename, int W, int H, unsigned char* data );
void elbp(Mat& src, Mat &dst, int radius, int neighbors);
float calcDistance(int first, int second, vector<vector<float> > & lbp_superpixel, vector<vector<float> > & labxy_superpixel, int h, int w);

int main(int argc, char* argv[])
{
	clock_t start, finish;
	printf("main paras = %d.\n", argc);
	const char * img_name = argv[1];
	const char * img_fcn  = argv[2];
	
	//-----------------------------------
	// 01: read jpg to buffer[char]
	//-----------------------------------
	start = clock();
	Mat raw_image = imread(img_name , 1);
	if(raw_image.empty())
		printf("imread failed!\n");
	
	int h = raw_image.rows;
	int w = raw_image.cols;
	int i(0), j(0);
	int x(0), y(0);	//pixel:x,y

//	unsigned char * image_buffer = (unsigned char *)malloc(sizeof(char) * w * h * 4)  ; // BGRA  
//	unsigned int  * pbuff        = (unsigned int  *)malloc(sizeof(int) * w * h);
	unsigned char * image_buffer = new unsigned char[w * h * 4]; 		// BGRA  
    unsigned int  * pbuff        = new unsigned int[w * h];
	
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
	printf("01 read jpeg to buffer[char]: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);
		
	//----------------------------------
    // 02: SLIC Initialize parameters
    //----------------------------------
    int k = 2000;		//Desired number of superpixels.
	int k_real = 0;	 	//Truth number of superpixel, k_real>k generally.
    double m = 10;		//Compactness factor. use a value from 10 to 40. Default is 10.
    int* klabels = new int[w * h];		//lable map
    int numlabels(0);
    string filename = "temp.jpg";		//seg image
    string savepath = "/home/xduser/LiHuan/SLICO/example/result/";

	//---------------------------------------------
	// 03: transform buffer[char] to buffer[int]
	//---------------------------------------------	
    // unsigned int (32 bits) to hold a pixel in ARGB format as follows:
    // from left to right,
    // the first 8 bits are for the alpha channel (and are ignored)
    // the next 8 bits are for the red channel
    // the next 8 bits are for the green channel
    // the last 8 bits are for the blue channel
	start = clock();
	for(i = j = 0; i < w*h; i++,j+=4 )
	{	
		*(pbuff + i) = *(image_buffer+j+3) + \
					  (*(image_buffer+j+2))	* 256 + \
					  (*(image_buffer+j+1)) * 256 * 256 + \
				      (*(image_buffer+j+0)) * 256 * 256 * 256;
	}
	for(i=0; i<w*h*4; i++)
	{
		*(image_buffer+i)=0;
	}
    finish = clock();
	printf("02 transform buffer[char] to [int]: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);
	
	//-----------------------------------------
    // 04: Perform SLIC on the image buffer
    //-----------------------------------------
	start = clock();
    SLIC segment;
    segment.PerformSLICO_ForGivenK(pbuff, w, h, klabels, numlabels, k, m);
    // Alternately one can also use the function PerformSLICO_ForGivenStepSize() for a desired superpixel size

    //---------------------------------------
    // 05: Save the labels to a text file
    //---------------------------------------
	//segment.SaveSuperpixelLabels(klabels, w, h, filename, savepath);
	
	//--------------------------------------------
	// 06: find the number of real superpixels
	//--------------------------------------------
    for(i = 0; i < w*h; i++)
    {
        if(klabels[i] > k_real)
            k_real = klabels[i];
    }
    printf("k_real = %d\n",k_real);
	finish = clock();
	printf("06 SLIC on buffer: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);

	//------------------------------------------	
	// 07: put pixel cluster into vector
	//------------------------------------------
	start = clock();
	vector<vector<int> > superpixel;	// 0 ~ n superpixel
	vector<int> i_superpixel;			// the pixel:x,y in the ith superpixel 

	for(i = 0; i < k_real + 1; i++)
	{	
		for(j = 0; j < w * h; j++)
		{
			if(klabels[j] == i)
			{	
				y = j % w ;
				x = (j - y ) / w;
				i_superpixel.push_back(x);
				i_superpixel.push_back(y);
			}
		}
		superpixel.push_back(i_superpixel);	
		i_superpixel.clear();
	}
	printf("Read into vector ok!\n");
	finish = clock();
	printf("07 put into vector: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);

#if 0
	for(vector<vector<int> >::size_type ix = 0; ix < superpixel.size(); ix++) 	//superpixel label number
		for(vector<int>::size_type iy = 0; iy < superpixel[1000].size(); iy += 2)	//get all x,y pixel in ith super
			printf("(%d, %d)",superpixel[1000][iy], superpixel[1000][iy+1]);
	printf("\n");	
#endif
#if 0	
	int n20,n30,n40,n50,n60,n70,n80,n90,n100,n110,n120,n130,n140;
	n20 = n30 = n40 = n50 = n60 = n70 = n80 = n90 = n100 = n110 = n120 = n130 = n140 = 0;
	for(vector<vector<int> >::size_type ix = 0; ix < superpixel.size(); ix++) 	//superpixel label number
	{
		int p = int(superpixel[ix].size()) / 2;
		if( p>=20 && p<30 ) n20++;
		if( p>=30 && p<40 ) n30++;
		if( p>=40 && p<50 ) n40++;
		if( p>=50 && p<60 ) n50++;
		if( p>=60 && p<70 ) n60++;
		if( p>=70 && p<80 ) n70++;
		if( p>=80 && p<90 ) n80++;
		if( p>=90 && p<100 ) n90++;
		if( p>=100 && p<110 ) n100++;
		if( p>=110 && p<120 ) n110++;
		if( p>=120 && p<130 ) n120++;
		if( p>=130 && p<140 ) n130++;
		if( p>=140 && p<200 ) n140++;
	}
	printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", n20,n30,n40,n50,n60,n70,n80,n90,n100,n110,n120,n130,n140);
#endif

	//-------------------------------------
	// 08: Extract Feature 
	//-------------------------------------
	printf("Feature Extracting ...\n");
	// LBP feature[16]  LABXY[5]
	vector<vector<float> > lbp_superpixel;
	vector<vector<float> > labxy_superpixel;
	vector<float> temp;
	int max_x(0), min_x(65536), max_y(0), min_y(65536);
	int color_l(0), color_a(0), color_b(0); 

    Mat image_lbp = imread(img_name, 0);		// 0:grayscale model
	printf("image_lbp H: %d, W: %d\n", image_lbp.rows, image_lbp.cols);
	if(image_lbp.empty())
		printf("imread fail.\n");

	Mat image_lab = imread(img_name, 1);		// BGR color model
	printf("image_lab H: %d, W: %d\n", image_lab.rows, image_lab.cols);
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
	
	start = clock();
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
	printf("Feature get ! \n");
	finish = clock();	
	printf("for loop needs: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);

	//----------------------------------------------------------------------------------
	// 09:该处对两个块特征归一化，计算相似性 (越小越好)
	// @params : first_s_pixel_No. sencond_s_pixel_No. lbpvector, labxyvector, h, w  
	//----------------------------------------------------------------------------------
#if 0
	float Distance = calcDistance(840, 1100, lbp_superpixel, labxy_superpixel, h, w);
	printf("\nDistance = %f\n", Distance);
#endif

	//----------------------------------
	// 10 : Unary from file.npy test
	//----------------------------------
	cnpy::NpyArray arr = cnpy::npy_load(img_fcn);	
   	float * raw_unary  = new float[w * h * CLASS];
   	float * unary      = new float[k_real * 1 * CLASS];
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
	arr.destruct();		//释放掉读取的标注文件内存	
	printf("raw_unary is ok ! \n");
	// pixel pros -> s_pixel pros	
	for(vector<vector<int> >::size_type ix = 0; ix < superpixel.size(); ix++)
	{
		for(vector<int>::size_type iy = 0; iy < superpixel[ix].size(); iy += 2) 
		{
			int x_temp = superpixel[ix][iy];
			int y_temp = superpixel[ix][iy+1];
			for(i=0; i<CLASS; i++)
			{
				unary[ int(ix) + i] += raw_unary[ ( x_temp*w+y )*CLASS + i];	
			}
		}	
		for(i=0; i<CLASS; i++)
			unary[ int(ix) * CLASS + i] /= (( int( superpixel[ix].size() ) ) / 2 ); 
	}
	printf("unary is ok ! \n");   


	//-------------------------------------
	// 11 : CRF model
	//-------------------------------------
#if 1
	printf("Start CRF modeling...\n");
	short * s_map = new short[k_real];
	int * map   = new int[w * h];

	//DenseCRF2D crf(w, h, CLASS);
	DenseCRF2D crf(k_real, 1, CLASS);

	// 把像素点概率映射到块上
	//crf.setUnaryEnergy( raw_unary );
	crf.setUnaryEnergy( unary );	//块的分属21类的概率

	// 把块的特征值加到模型中
	//crf.addPairwiseGaussian( 3, 3, 10);
	//crf.addPairwiseBilateral( 60, 60, 20, 20, 20, im, 10 );
	crf.addPairwiseGaussian_lh( labxy_superpixel, 2, 3, 3, 3, NULL, true);	//块的坐标
	crf.addPairwiseBilateral_lh( lbp_superpixel, 5, 60, 60, 20, 20, 20, 10, NULL);	//块的颜色
	printf("feature into crf ok ! \n");
	 
	//迭代求每个块的最大概率	
	//short * map = new short[w*h];
	//crf.map(IterativeNumber, map);
	//unsigned char *res = colorize( map, w, h );
	//writePPM( argv[3], w, h, res );
	crf.map(IterativeNumber, s_map);
	printf("iterative crf ok ! \n");
	printf("k_real = %d.\n", k_real);
	int numone(0), numtwo(0);
    for(i=0; i<k_real; i++)
    {
        if( s_map[i]<0 || s_map[i]>21 )
            printf("error!!!!! : %d, s_map[%d]=%d \n",i,i,s_map[i]);
		if( s_map[i] == 0 ) numone++;
		if( s_map[i] == 17 ) numtwo++;
    }   
	printf("class0 = %d. class1= %d.\n", numone,numtwo);
	

	// 将块标记转为点标记，做图

    for(vector<vector<int> >::size_type ix = 0, i=0; ix < superpixel.size(); ix++,i++)
    {
        for(vector<int>::size_type iy = 0; iy < superpixel[ix].size(); iy += 2)
        {
            int x_temp = superpixel[ix][iy];
            int y_temp = superpixel[ix][iy+1];
			map[x_temp * w + y_temp] = s_map[i];
			if( (x_temp *w + y_temp) == 164710 )
				printf("164710, map = %d. ss_map=%d\n",map[164710],s_map[i]);
    	}
	}
	printf("S_label trans to pixel_label ok !\n");

	for(i=0; i<w*h; i++)
	{
		if( map[i]<0 || map[i]>21 )
		{
			printf("error!!!!! : %d, map[%d]=%d \n",i,i,map[i]);
			map[i] = 17;
		}
	}	

//	printf("map[0] : %d\n", map[0]);
	colorize( map, w, h , image_buffer);
	printf("map to colorize ok !\n");

	writePPM( argv[3], w, h, image_buffer);

	printf("Segment image save done !\n");
	printf("CRF finish successful !\n");
#endif	
	
#if 0
	//----------------------------------
    // Draw boundaries around segments
    //----------------------------------
	printf("\n06. draw contour to buffer.\n");
    segment.DrawContoursAroundSegments(pbuff, klabels, w, h, 0xff0000);
	printf("06 ok.\n");

    //----------------------------------
    // Save the image with segment boundaries.
    //----------------------------------
	for(i = j = 0; i < w*h; i++,j+=3 )
    {
        *(image_buffer+j+2) = ( (*(pbuff + i)) & 0xff);
        *(image_buffer+j+1) = ( ( (*(pbuff + i)) >> 8) & 0xff);
        *(image_buffer+j+0) = ( ( (*(pbuff + i)) >> 16) & 0xff);
    } 
    write_jpeg_file("/home/xduser/LiHuan/SLICO/example/result/temp_seg.jpg", image_buffer, w, h, JPEG_QUALITY);
    printf("savejpeg ok!\n");
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
unsigned char * colorize(int * map, int W, int H , unsigned char * r)
{
	printf("entet into colorize ! \n");
	printf("w=%d ,h=%d \n", W, H);
	printf("new ok!\n");
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
