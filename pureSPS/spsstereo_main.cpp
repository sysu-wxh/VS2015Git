/*
    Copyright (C) 2014  Koichiro Yamaguchi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv.hpp>
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include "SPSStereo.h"
#include "defParameter.h"


void makeSegmentBoundaryImage(const IplImage* inputImage,
	const IplImage* segmentImage,
	std::vector< std::vector<int> >& boundaryLabels,
	IplImage* segmentBoundaryImage);
void writeDisparityPlaneFile(const std::vector< std::vector<double> >& disparityPlaneParameters, const std::string outputDisparityPlaneFilename);
void writeBoundaryLabelFile(const std::vector< std::vector<int> >& boundaryLabels, const std::string outputBoundaryLabelFilename);


int main(int argc, char* argv[]) {
    
	std::string leftImageFileStr = "I://Master//dataset//08-2018-03-15-LW//image_2//";
	std::string rightImageFileStr = "I://Master//dataset//08-2018-03-15-LW//image_3//";
	std::string sgmDisparityImageFileStr = "I://Master//dataset//08-2018-03-15-LW//SGM//";
	std::string disparityImageFileStr = "I://Master//dataset//08-2018-03-15-LW//SPS//";
	char baseName[64];
	for (int imgNum = 0; imgNum < 200; ++imgNum){
		sprintf_s(baseName, "%06d.png", imgNum);
		std::string leftImageFilename = leftImageFileStr + baseName;   //得到左图的路径
		std::cout << leftImageFilename << std::endl;     //输出左图路径
		std::string rightImageFilename = rightImageFileStr + baseName;  //得到右图的路径
		std::string disparityFilename = disparityImageFileStr + baseName;    //得到输出的视差图的路径名称
		std::string sgmDisparityFilename = sgmDisparityImageFileStr + baseName;   //得到segment视差图的路径名称
		IplImage *leftImage = cvLoadImage(leftImageFilename.c_str(), 1);     //读取左图
		IplImage *rightImage = cvLoadImage(rightImageFilename.c_str(), 1);    //读取右图

		SPSStereo sps;         //定义SPSStereo变量
		sps.setIterationTotal(outerIterationTotal, innerIterationTotal);  //设置迭代总数（包括外层迭代总数，内层迭代总数）
		sps.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);  //设置权重参数(位置权重，视差权重，边界长度权重，平滑权重）
		sps.setInlierThreshold(lambda_d);   //设置内点的阈值
		sps.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen); //设置惩罚参数

		CvSize imgSize;   //读入图片的大小
		imgSize.width = leftImage->width, imgSize.height = leftImage->height;   //分别记录下宽与高
		IplImage *segmentImage, *disparityImage;         //定义了seg图变量，视差图变量
		segmentImage = cvCreateImage(imgSize, 16, 1);   //创建seg图像首地址并分配空间，这里创建了一个大小为imgsize，深度为16，通道为1的图
		disparityImage = cvCreateImage(imgSize, 8, 1);   //创建视差图像首地址并分配空间，这里创建了一个大小为imgsize，深度为8，通道为1的图
		std::vector< std::vector<double> > disparityPlaneParameters;   //定义视差平面参数变量，是一个存放了类型为浮点型向量的向量
		std::vector< std::vector<int> > boundaryLabels;    //定义边界标记变量，是一个存放了类型为整型向量的向量
		IplImage* sgmImage = cvCreateImage(cvSize(leftImage->width, leftImage->height), IPL_DEPTH_8U, 3);  //创建一个大小为imgsize，深度为无符号8bit，三通道的图像首地址
		//执行计算函数。并代入相关的参数（超像素总数，左视图，右视图，读取图像的seg变量1，单通道的seg变量2，视差平面参数，边界标识）
		sps.compute(superpixelTotal, leftImage, rightImage, sgmImage, segmentImage, disparityImage, disparityPlaneParameters, boundaryLabels);
		//将视差图与seg图保存的变量存入文件）
		cvSaveImage(disparityFilename.c_str(), disparityImage);
		cvSaveImage(sgmDisparityFilename.c_str(), sgmImage);
		
		//IplImage* segmentBoundaryImage;
		//segmentBoundaryImage = cvCreateImage(imgSize, 8, 3);
		//makeSegmentBoundaryImage(leftImage, segmentImage, boundaryLabels, segmentBoundaryImage);
		//std::string segmentImageStr = "F://dataset//data_scene_flow//training//SPSSegment//";
		//cvNamedWindow("disparityImage");
		//cvShowImage("disparityImage", segmentBoundaryImage);
		//cvSaveImage((segmentImageStr + baseName).c_str(), segmentBoundaryImage);
		//cvWaitKey(0);
	}

	//cvNamedWindow("disparityImage");
	//cvShowImage("disparityImage", disparityImage);
	//cvWaitKey(0);
	system("pause");
	return 0;
}


void makeSegmentBoundaryImage(const IplImage* inputImage,
	const IplImage* segmentImage,
	std::vector< std::vector<int> >& boundaryLabels,
	IplImage* segmentBoundaryImage)
{
	int width = static_cast<int>(inputImage->width);
	int height = static_cast<int>(inputImage->height);
	int boundaryTotal = static_cast<int>(boundaryLabels.size());

	//segmentBoundaryImage.resize(width, height);
	/*for (int y = 0; y < height; ++y) {
	for (int x = 0; x < width; ++x) {
	//segmentBoundaryImage.set_pixel(x, y, inputImage.get_pixel(x, y));
	cvSet2D(segmentBoundaryImage, y, x, cvGet2D(inputImage, y, x));
	}
	}*/
	cvCopy(inputImage, segmentBoundaryImage, NULL);

	/*cvNamedWindow("Original Segment Boundary Image");
	cvShowImage("Original Segment Boundary Image",segmentBoundaryImage);
	cvWaitKey(0);
	printf("Original Segment Boundary Image已显示\n");*/

	int boundaryWidth = 2;
	CvScalar temp_pixel;
	for (int y = 0; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			CvScalar pixelLabelIndex = cvGet2D(segmentImage, y, x);
			temp_pixel.val[0] = 128, temp_pixel.val[1] = 128, temp_pixel.val[2] = 128;

			if (cvGet2D(segmentImage, y, x + 1).val[0] != pixelLabelIndex.val[0]) {
				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (x - w >= 0) {
						cvSet2D(segmentBoundaryImage, y, x - w, temp_pixel);
					}
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (x + w < width) {
						cvSet2D(segmentBoundaryImage, y, x + w, temp_pixel);
					}
				}
			}
			if (cvGet2D(segmentImage, y + 1, x).val[0] != pixelLabelIndex.val[0]) {
				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (y - w >= 0) {
						cvSet2D(segmentBoundaryImage, y - w, x, temp_pixel);
					}
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (y + w < height) {
						cvSet2D(segmentBoundaryImage, y + w, x, temp_pixel);
					}
				}
			}
		}
	}

	boundaryWidth = 0;
	for (int y = 0; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			//int pixelLabelIndex = segmentImage.get_pixel(x, y);
			CvScalar pixelLabelIndex = cvGet2D(segmentImage, y, x);

			if (cvGet2D(segmentImage, y, x + 1).val[0] != pixelLabelIndex.val[0]) {
				//png::rgb_pixel negativeSideColor, positiveSideColor;
				CvScalar negativeSideColor, positiveSideColor;
				int pixelBoundaryIndex = -1;
				for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {  //为什么可以换一下位置
					if ((boundaryLabels[boundaryIndex][0] == pixelLabelIndex.val[0] && boundaryLabels[boundaryIndex][1] == cvGet2D(segmentImage, y, x + 1).val[0])
						|| (boundaryLabels[boundaryIndex][0] == cvGet2D(segmentImage, y, x + 1).val[0] && boundaryLabels[boundaryIndex][1] == pixelLabelIndex.val[0]))
					{
						pixelBoundaryIndex = boundaryIndex;
						break;
					}
				}
				if (boundaryLabels[pixelBoundaryIndex][2] == 3) continue;
				else if (boundaryLabels[pixelBoundaryIndex][2] == 2) {   //B G R
					negativeSideColor.val[0] = 0;  negativeSideColor.val[1] = 225;  negativeSideColor.val[2] = 0;
					positiveSideColor.val[0] = 0;  positiveSideColor.val[1] = 225;  positiveSideColor.val[2] = 0;
				}
				else if (pixelLabelIndex.val[0] == boundaryLabels[pixelBoundaryIndex][boundaryLabels[pixelBoundaryIndex][2]]) {
					//negativeSideColor.red = 225;  negativeSideColor.green = 0;  negativeSideColor.blue = 0;
					//positiveSideColor.red = 0;  positiveSideColor.green = 0;  positiveSideColor.blue = 225;
					negativeSideColor.val[0] = 0;  negativeSideColor.val[1] = 0;  negativeSideColor.val[2] = 225;
					positiveSideColor.val[0] = 225;  positiveSideColor.val[1] = 0;  positiveSideColor.val[2] = 0;
				}
				else {
					//negativeSideColor.red = 0;  negativeSideColor.green = 0;  negativeSideColor.blue = 225;
					//positiveSideColor.red = 225;  positiveSideColor.green = 0;  positiveSideColor.blue = 0;
					negativeSideColor.val[0] = 225;  negativeSideColor.val[1] = 0;  negativeSideColor.val[2] = 0;
					positiveSideColor.val[0] = 0;  positiveSideColor.val[1] = 0;  positiveSideColor.val[2] = 225;
				}

				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (x - w >= 0) cvSet2D(segmentBoundaryImage, y, x - w, negativeSideColor);
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (x + w < width) cvSet2D(segmentBoundaryImage, y, x + w, positiveSideColor);
				}
			}
			if (cvGet2D(segmentImage, y + 1, x).val[0] != pixelLabelIndex.val[0]) {
				CvScalar negativeSideColor, positiveSideColor;
				int pixelBoundaryIndex = -1;
				for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
					if ((boundaryLabels[boundaryIndex][0] == pixelLabelIndex.val[0] && boundaryLabels[boundaryIndex][1] == cvGet2D(segmentImage, y + 1, x).val[0])
						|| (boundaryLabels[boundaryIndex][0] == cvGet2D(segmentImage, y + 1, x).val[0] && boundaryLabels[boundaryIndex][1] == pixelLabelIndex.val[0]))
					{
						pixelBoundaryIndex = boundaryIndex;
						break;
					}
				}
				if (boundaryLabels[pixelBoundaryIndex][2] == 3) continue;
				else if (boundaryLabels[pixelBoundaryIndex][2] == 2) {
					negativeSideColor.val[0] = 0;  negativeSideColor.val[1] = 225;  negativeSideColor.val[2] = 0;
					positiveSideColor.val[0] = 0;  positiveSideColor.val[1] = 225;  positiveSideColor.val[2] = 0;
				}
				else if (pixelLabelIndex.val[0] == boundaryLabels[pixelBoundaryIndex][boundaryLabels[pixelBoundaryIndex][2]]) {
					negativeSideColor.val[0] = 0;  negativeSideColor.val[1] = 0;  negativeSideColor.val[2] = 225;
					positiveSideColor.val[0] = 225;  positiveSideColor.val[1] = 0;  positiveSideColor.val[2] = 0;
				}
				else {
					negativeSideColor.val[0] = 225;  negativeSideColor.val[1] = 0;  negativeSideColor.val[2] = 0;
					positiveSideColor.val[0] = 0;  positiveSideColor.val[1] = 0;  positiveSideColor.val[2] = 225;
				}

				for (int w = 0; w < boundaryWidth - 1; ++w) {
					if (y - w >= 0) cvSet2D(segmentBoundaryImage, y - w, x, negativeSideColor);
				}
				for (int w = 1; w < boundaryWidth; ++w) {
					if (y + w < height) cvSet2D(segmentBoundaryImage, y + w, x, positiveSideColor);
				}
			}
		}
	}
}

void writeDisparityPlaneFile(const std::vector< std::vector<double> >& disparityPlaneParameters, const std::string outputDisparityPlaneFilename) {
	std::ofstream outputFileStream(outputDisparityPlaneFilename.c_str(), std::ios_base::out);
	if (outputFileStream.fail()) {
		std::cerr << "error: can't open file (" << outputDisparityPlaneFilename << ")" << std::endl;
		exit(0);
	}

	int segmentTotal = static_cast<int>(disparityPlaneParameters.size());
	for (int segmentIndex = 0; segmentIndex < segmentTotal; ++segmentIndex) {
		outputFileStream << disparityPlaneParameters[segmentIndex][0] << " ";
		outputFileStream << disparityPlaneParameters[segmentIndex][1] << " ";
		outputFileStream << disparityPlaneParameters[segmentIndex][2] << std::endl;
	}

	outputFileStream.close();
}

void writeBoundaryLabelFile(const std::vector< std::vector<int> >& boundaryLabels, const std::string outputBoundaryLabelFilename) {
	std::ofstream outputFileStream(outputBoundaryLabelFilename.c_str(), std::ios_base::out);
	if (outputFileStream.fail()) {
		std::cerr << "error: can't open output file (" << outputBoundaryLabelFilename << ")" << std::endl;
		exit(1);
	}

	int boundaryTotal = static_cast<int>(boundaryLabels.size());
	for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
		outputFileStream << boundaryLabels[boundaryIndex][0] << " ";
		outputFileStream << boundaryLabels[boundaryIndex][1] << " ";
		outputFileStream << boundaryLabels[boundaryIndex][2] << std::endl;
	}
	outputFileStream.close();
}
