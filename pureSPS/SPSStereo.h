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

#pragma once

#include <vector>
#include <stack>
#include <opencv.hpp>

//SPSStereo类
class SPSStereo {
public:
	SPSStereo();
	
	//设置输出视差因子
	void setOutputDisparityFactor(const double outputDisparityFactor);
	//设置迭代总数（包括外层迭代总数，内层迭代总数）
	void setIterationTotal(const int outerIterationTotal, const int innerIterationTotal);
	//设置权重参数
	void setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight);
	//设置内点的阈值
	void setInlierThreshold(const double inlierThreshold);
	//设置惩罚参数
	void setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty);
	//*****计算函数*****
	void compute(const int superpixelTotal,
		const IplImage* leftImage,
		const IplImage* rightImage,
		IplImage* sgmImage,
		IplImage* segmentImage,
		IplImage* disparityImage,
		std::vector< std::vector<double> >& disparityPlaneParameters,
		std::vector< std::vector<int> >& boundaryLabels);
	void SPSStereo::getSegmentAroundIndex(std::vector< std::vector<int> >& segmentAroundIndex, int segmentNums, int outputNum);

private:
	//segment类（目测为分割相关的类）
	class Segment {
	public:
		Segment() {
			pixelTotal_ = 0;
			colorSum_[0] = 0;  colorSum_[1] = 0;  colorSum_[2] = 0;
			positionSum_[0] = 0;  positionSum_[1] = 0;
			disparityPlane_[0] = 0;  disparityPlane_[1] = 0;  disparityPlane_[2] = -1;
		}
		//增加像素
		void addPixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {   
			pixelTotal_ += 1;   //像素数目加1
			colorSum_[0] += colorL;  colorSum_[1] += colorA;  colorSum_[2] += colorB;   //LAB颜色空间和进行更新，加入新的值
			positionSum_[0] += x;  positionSum_[1] += y;     //位置和进行更新，加入新坐标
		}
		//移除像素
		void removePixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {
			pixelTotal_ -= 1;    //像素数目减1
			colorSum_[0] -= colorL;  colorSum_[1] -= colorA;  colorSum_[2] -= colorB;   //LAB颜色空间和进行更新，减去新的值
			positionSum_[0] -= x;  positionSum_[1] -= y;    //位置和进行更新，减去新坐标
		}

		//设置视差平面
		void setDisparityPlane(const double planeGradientX, const double planeGradientY, const double planeConstant) {
			disparityPlane_[0] = planeGradientX;   //x的视差梯度
			disparityPlane_[1] = planeGradientY;    //y的视差梯度
			disparityPlane_[2] = planeConstant;    //平面常量
		}

		//返回seg的像素总数
		int pixelTotal() const { return pixelTotal_; }   
		//返回一个seg中某一通道的平均颜色值
		double color(const int colorIndex) const { return colorSum_[colorIndex]/pixelTotal_; }   
		//返回一个seg中x或者y的平均坐标
		double position(const int coordinateIndex) const { return positionSum_[coordinateIndex]/pixelTotal_; }    
		//返回相对于x，y点估计的视差值（也就是深度）z =（ax+by+c)
		double estimatedDisparity(const double x, const double y) const { return disparityPlane_[0]*x + disparityPlane_[1]*y + disparityPlane_[2]; }  
		//检查是否有估计的平面。只要a,b,c中参数存在，说明存在平面
		bool hasDisparityPlane() const { if (disparityPlane_[0] != 0.0 || disparityPlane_[1] != 0.0 || disparityPlane_[2] != -1.0) return true; else return false; }

		//清除配置
		void clearConfiguration() {
			neighborSegmentIndices_.clear();  //清除相邻seg的索引
			boundaryIndices_.clear();    //清除边界索引
			for (int i = 0; i < 9; ++i) polynomialCoefficients_[i] = 0;  //清除多项式系数
			for (int i = 0; i < 6; ++i) polynomialCoefficientsAll_[i] = 0;   //清除所有多项式系数
		}
		//扩大边界索引
		void appendBoundaryIndex(const int boundaryIndex) { boundaryIndices_.push_back(boundaryIndex); }
		//扩大seg的像素
		void appendSegmentPixel(const int x, const int y) {
			polynomialCoefficientsAll_[0] += x*x;    //a = a + x*x
			polynomialCoefficientsAll_[1] += y*y;    //b = b + y*y
			polynomialCoefficientsAll_[2] += x*y;    //c = c + x*y
			polynomialCoefficientsAll_[3] += x;      //d = d + x
			polynomialCoefficientsAll_[4] += y;      //e = e + y
			polynomialCoefficientsAll_[5] += 1;      //f = f + 1
		}
		//扩大带有深度信息的seg的像素
		void appendSegmentPixelWithDisparity(const int x, const int y, const double d) {
			polynomialCoefficients_[0] += x*x;     //a = a + x*x
			polynomialCoefficients_[1] += y*y;     //b = b + y*y
			polynomialCoefficients_[2] += x*y;     //c = c + x*y
			polynomialCoefficients_[3] += x;       //d = d + x
			polynomialCoefficients_[4] += y;       //e = e + y
			polynomialCoefficients_[5] += x*d;     //f = f + x*d
			polynomialCoefficients_[6] += y*d;     //g = g + y*d
			polynomialCoefficients_[7] += d;       //h = h + d
			polynomialCoefficients_[8] += 1;       //i = i + 1
		}

		//返回相邻的seg的总数
		int neighborTotal() const { return static_cast<int>(neighborSegmentIndices_.size()); }
		//返回相邻的索引为index的seg的值
		int neighborIndex(const int index) const { return neighborSegmentIndices_[index]; }

		//返回相邻的边界的总数
		int boundaryTotal() const { return static_cast<int>(boundaryIndices_.size()); }
		//返回相邻的边界为index的seg的值
		int boundaryIndex(const int index) const { return boundaryIndices_[index]; }

		//返回不包含深度信息的第index个多项式系数
		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }
		//返回包含深度信息的第index个多项式系数
		double polynomialCoefficientAll(const int index) const { return polynomialCoefficientsAll_[index]; }

		//返回平面参数（也就是确定平面的a,b,c）
		double planeParameter(const int index) const { return disparityPlane_[index]; }

	private:
		int pixelTotal_;
		double colorSum_[3];
		double positionSum_[2];
		double disparityPlane_[3];

		std::vector<int> neighborSegmentIndices_;
		std::vector<int> boundaryIndices_;
		double polynomialCoefficients_[9];
		double polynomialCoefficientsAll_[6];
	};
	//Boundary类（目测与边界相关）
	class Boundary {
	public:
		//构造函数 将segment索引的两个元素都置为-1，并清除系数
		Boundary() { segmentIndices_[0] = -1; segmentIndices_[1] = -1; clearCoefficients(); }
		//带参的构造函数，传入边界左右的第一、第二个相邻seg的索引
		Boundary(const int firstSegmentIndex, const int secondSegmentIndex) {
			if (firstSegmentIndex < secondSegmentIndex) {   //如果第一个小于第二个，那么0是第一个，1是第二个
				segmentIndices_[0] = firstSegmentIndex; segmentIndices_[1] = secondSegmentIndex;
			} else {    //否则0是第二个，1是第一个
				segmentIndices_[0] = secondSegmentIndex; segmentIndices_[1] = firstSegmentIndex;
			} 
			clearCoefficients();//清除多项式系数
		}

		//清除不不含深度信息的多项式系数
		void clearCoefficients() {
			for (int i = 0; i < 6; ++i) polynomialCoefficients_[i] = 0;
		}
		
		//设置类型
		void setType(const int typeIndex) { type_ = typeIndex; }

		//扩大不带深度信息的边界像素
		void appendBoundaryPixel(const double x, const double y) {
			boundaryPixelXs_.push_back(x);   //压入边界像素的x
			boundaryPixelYs_.push_back(y);   //压入边界像素的y
			polynomialCoefficients_[0] += x*x;   //a = a + x*x
			polynomialCoefficients_[1] += y*y;   //b = b + y*y
			polynomialCoefficients_[2] += x*y;   //c = c + x*y
			polynomialCoefficients_[3] += x;     //d = d + x
			polynomialCoefficients_[4] += y;     //e = e + y
			polynomialCoefficients_[5] += 1;     //f = f + 1
		}

		//返回类型
		int type() const { return type_; }
		//返回挨着seg的索引，index = 0是左边的，index = 1是右边的
		int segmentIndex(const int index) const { return segmentIndices_[index]; }

		//检查两个seg是否是一条边界的两侧（也就是他们是否相邻），是的话返回true
		bool consistOf(const int firstSegmentIndex, const int secondSegmentIndex) const {
			if ((firstSegmentIndex == segmentIndices_[0] && secondSegmentIndex == segmentIndices_[1])
				|| (firstSegmentIndex == segmentIndices_[1] && secondSegmentIndex == segmentIndices_[0]))
			{
				return true;
			}
			return false;
		}
		int include(const int segmentIndex) const {
			if (segmentIndex == segmentIndices_[0]) return 0;
			else if (segmentIndex == segmentIndices_[1]) return 1;
			else return -1;
		}

		//返回这条边界上有的像素个数
		int boundaryPixelTotal() const { return static_cast<int>(boundaryPixelXs_.size()); }
		//返回边界上制定index的像素的x坐标
		double boundaryPixelX(const int index) const { return boundaryPixelXs_[index]; }
		//返回边界上制定index的像素的y坐标
		double boundaryPixelY(const int index) const { return boundaryPixelYs_[index]; }

		//返回制定的第index个多项式系数（带深度有9个，不带深度有6个）
		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }

	private:
		int type_;
		int segmentIndices_[2];  //seg的索引，数组表示其记录了边界两个方向的。
		std::vector<double> boundaryPixelXs_;
		std::vector<double> boundaryPixelYs_;

		double polynomialCoefficients_[6];
	};


	void allocateBuffer();   //分配缓存
	void freeBuffer();    //释放缓存
	void setInputData(const IplImage* leftImage, const IplImage* rightImage, IplImage* sgmImage);
	void setLabImage(const IplImage* leftImage);
	void computeInitialDisparityImage(const IplImage* leftImage, const IplImage* rightImage, IplImage* sgmImage);
	void initializeSegment(const int superpixelTotal);
	void makeGridSegment(const int superpixelTotal);
	void assignLabel();
	void extractBoundaryPixel(std::stack<int>& boundaryPixelIndices);
	bool isBoundaryPixel(const int x, const int y) const;
	bool isUnchangeable(const int x, const int y) const;
	int findBestSegmentLabel(const int x, const int y) const;
	std::vector<int> getNeighborSegmentIndices(const int x, const int y) const;
	double computePixelEnergy(const int x, const int y, const int segmentIndex) const;
	double computeBoundaryLengthEnergy(const int x, const int y, const int segmentIndex) const;
	void changeSegmentLabel(const int x, const int y, const int newSegmentIndex);
	void addNeighborBoundaryPixel(const int x, const int y, std::stack<int>& boundaryPixelIndices) const;
	void initialFitDisparityPlane();
	void estimateDisparityPlaneRANSAC(const float* disparityImage);
	void solvePlaneEquations(const double x1, const double y1, const double z1, const double d1,
							 const double x2, const double y2, const double z2, const double d2,
							 const double x3, const double y3, const double z3, const double d3,
							 std::vector<double>& planeParameter) const;
	int computeRequiredSamplingTotal(const int drawTotal, const int inlierTotal, const int pointTotal, const int currentSamplingTotal, const double confidenceLevel) const;
	void interpolateDisparityImage(float* interpolatedDisparityImage) const;
	void initializeOutlierFlagImage();
	void performSmoothingSegmentation();
	void buildSegmentConfiguration();
	bool isHorizontalBoundary(const int x, const int y) const;
	bool isVerticalBoundary(const int x, const int y) const;
	int appendBoundary(const int firstSegmentIndex, const int secondSegmentIndex);
	void planeSmoothing();
	void estimateBoundaryLabel();
	void estimateSmoothFitting();
	void makeOutputImage(IplImage* segmentImage, IplImage* segmentDisparityImage) const;
	void makeSegmentBoundaryData(std::vector< std::vector<double> >& disparityPlaneParameters, std::vector< std::vector<int> >& boundaryLabels) const;


	// Parameter
	double outputDisparityFactor_;
	int outerIterationTotal_;
	int innerIterationTotal_;
	double positionWeight_;
	double disparityWeight_;
	double boundaryLengthWeight_;
	double smoothRelativeWeight_;
	double inlierThreshold_;
	double hingePenalty_;
	double occlusionPenalty_;
	double impossiblePenalty_;

	// Input data
	int width_;
	int height_;
	float* inputLabImage_;
	float* initialDisparityImage_;

	// Superpixel segments
	int segmentTotal_;
	std::vector<Segment> segments_;
	int stepSize_;
	int* labelImage_;
	unsigned char* outlierFlagImage_;
	unsigned char* boundaryFlagImage_;
	std::vector<Boundary> boundaries_;
	std::vector< std::vector<int> > boundaryIndexMatrix_;
};
