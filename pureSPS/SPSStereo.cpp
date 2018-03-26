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

#include "SPSStereo.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <float.h>
#include "SGMStereo.h"

// Default parameters
const double SPSSTEREO_DEFAULT_OUTPUT_DISPARITY_FACTOR = 256.0;
const int SPSSTEREO_DEFAULT_OUTER_ITERATION_COUNT = 10;
const int SPSSTEREO_DEFAULT_INNER_ITERATION_COUNT = 10;
const double SPSSTEREO_DEFAULT_POSITION_WEIGHT = 500.0;
const double SPSSTEREO_DEFAULT_DISPARITY_WEIGHT = 2000.0;
const double SPSSTEREO_DEFAULT_BOUNDARY_LENGTH_WEIGHT = 1500.0;
const double SPSSTEREO_DEFAULT_SMOOTHNESS_WEIGHT = 400.0;
const double SPSSTEREO_DEFAULT_INLIER_THRESHOLD = 3.0;
const double SPSSTEREO_DEFAULT_HINGE_PENALTY = 5.0;
const double SPSSTEREO_DEFAULT_OCCLUSION_PENALTY = 15.0;
const double SPSSTEREO_DEFAULT_IMPOSSIBLE_PENALTY = 30.0;

// Pixel offsets of 4- and 8-neighbors
const int fourNeighborTotal = 4;
const int fourNeighborOffsetX[4] = { -1, 0, 1, 0 };
const int fourNeighborOffsetY[4] = { 0, -1, 0, 1 };
const int eightNeighborTotal = 8;
const int eightNeighborOffsetX[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
const int eightNeighborOffsetY[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };


SPSStereo::SPSStereo() : outputDisparityFactor_(SPSSTEREO_DEFAULT_OUTPUT_DISPARITY_FACTOR),
						 outerIterationTotal_(SPSSTEREO_DEFAULT_OUTER_ITERATION_COUNT),
						 innerIterationTotal_(SPSSTEREO_DEFAULT_INNER_ITERATION_COUNT),
						 positionWeight_(SPSSTEREO_DEFAULT_POSITION_WEIGHT),
						 disparityWeight_(SPSSTEREO_DEFAULT_DISPARITY_WEIGHT),
						 boundaryLengthWeight_(SPSSTEREO_DEFAULT_BOUNDARY_LENGTH_WEIGHT),
						 inlierThreshold_(SPSSTEREO_DEFAULT_INLIER_THRESHOLD),
						 hingePenalty_(SPSSTEREO_DEFAULT_HINGE_PENALTY),
						 occlusionPenalty_(SPSSTEREO_DEFAULT_OCCLUSION_PENALTY),
						 impossiblePenalty_(SPSSTEREO_DEFAULT_IMPOSSIBLE_PENALTY)
{
	smoothRelativeWeight_ = SPSSTEREO_DEFAULT_SMOOTHNESS_WEIGHT/SPSSTEREO_DEFAULT_DISPARITY_WEIGHT;
}

void SPSStereo::setOutputDisparityFactor(const double outputDisparityFactor) {
	if (outputDisparityFactor < 1) {
		throw std::invalid_argument("[SPSStereo::setOutputDisparityFactor] disparity factor is less than 1");
	}

	outputDisparityFactor_ = outputDisparityFactor;
}

//设置迭代总数（包括外层迭代总数，内层迭代总数）
void SPSStereo::setIterationTotal(const int outerIterationTotal, const int innerIterationTotal) {
	if (outerIterationTotal < 1 || innerIterationTotal < 1) {
		throw std::invalid_argument("[SPSStereo::setIterationTotal] the number of iterations is less than 1");
	}

	//以下对相关变量进行了赋值
	outerIterationTotal_ = outerIterationTotal;   
	innerIterationTotal_ = innerIterationTotal;
}
//设置权重参数(位置权重，视差权重，边界长度权重，平滑权重）
void SPSStereo::setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight) {
	if (positionWeight < 0 || disparityWeight < 0 || boundaryLengthWeight < 0 || smoothnessWeight < 0) {
		throw std::invalid_argument("[SPSStereo::setWeightParameter] weight value is nagative");
	}

	positionWeight_ = positionWeight;
	disparityWeight_ = disparityWeight;
	boundaryLengthWeight_ = boundaryLengthWeight;
	smoothRelativeWeight_ = smoothnessWeight/disparityWeight;
}
//设置内点的阈值
void SPSStereo::setInlierThreshold(const double inlierThreshold) {
	if (inlierThreshold <= 0) {
		throw std::invalid_argument("[SPSStereo::setInlierThreshold] threshold of inlier is less than zero");
	}

	inlierThreshold_ = inlierThreshold;
}
//设置惩罚参数
void SPSStereo::setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty) {
	if (hingePenalty <= 0 || occlusionPenalty <= 0 || impossiblePenalty < 0) {
		throw std::invalid_argument("[SPSStereo::setPenaltyParameter] penalty value is less than zero");
	}
	if (hingePenalty >= occlusionPenalty) {
		throw std::invalid_argument("[SPSStereo::setPenaltyParameter] hinge penalty is larger than occlusion penalty");
	}

	hingePenalty_ = hingePenalty;
	occlusionPenalty_ = occlusionPenalty;
	impossiblePenalty_ = impossiblePenalty;
}

void SPSStereo::compute(const int superpixelTotal,
	const IplImage* leftImage,
	const IplImage* rightImage,
	IplImage* sgmImage,
	IplImage* segmentImage,
	IplImage* disparityImage,
	std::vector< std::vector<double> >& disparityPlaneParameters,
	std::vector< std::vector<int> >& boundaryLabels){

	if (superpixelTotal < 2) {
		throw std::invalid_argument("[SPSStereo::compute] the number of superpixels is less than 2");
	}
	width_ = static_cast<int>(leftImage->width);    //左图的宽
	height_ = static_cast<int>(leftImage->height);   //左图的高
	if (rightImage->width != width_ || rightImage->height != height_) {  //左右图大小不匹配
		throw std::invalid_argument("[SPSStereo::setInputData] sizes of left and right images are different");
	}

	allocateBuffer();   //分配内存

	setInputData(leftImage, rightImage, sgmImage);
	initializeSegment(superpixelTotal);
	performSmoothingSegmentation();

	makeOutputImage(segmentImage, disparityImage);
	makeSegmentBoundaryData(disparityPlaneParameters, boundaryLabels);

	freeBuffer();
}


void SPSStereo::allocateBuffer() {
	inputLabImage_ = reinterpret_cast<float*>(malloc(width_*height_*3*sizeof(float)));   //为 输入的图像 创建width * height * 3通道大小的空间
	initialDisparityImage_ = reinterpret_cast<float*>(malloc(width_*height_*sizeof(float)));   //为 初始化视差图创建 width * height大小的空间
	labelImage_ = reinterpret_cast<int*>(malloc(width_*height_*sizeof(int)));   //标识（label）图分配内存
	outlierFlagImage_ = reinterpret_cast<unsigned char*>(malloc(width_*height_*sizeof(unsigned char)));  //为外点标志图分配内存
	boundaryFlagImage_ = reinterpret_cast<unsigned char*>(malloc(width_*height_*sizeof(unsigned char)));  //为边界标志图分配内存
}

void SPSStereo::freeBuffer() {
	free(inputLabImage_);
	free(initialDisparityImage_);
	free(labelImage_);
	free(outlierFlagImage_);
	free(boundaryFlagImage_);
}

//设置输入数据，传入了左图，右图，空的seg图
void SPSStereo::setInputData(const IplImage* leftImage, const IplImage* rightImage, IplImage* sgmImage) {
	setLabImage(leftImage);  //设置LAB图（也就是传入了左图）
	computeInitialDisparityImage(leftImage, rightImage, sgmImage);  //计算初始的视差图
}

void SPSStereo::setLabImage(const IplImage* leftImage) {    //CIE LAB三原色
	std::vector<float> sRGBGammaCorrections(256);
	for (int pixelValue = 0; pixelValue < 256; ++pixelValue) {
		double normalizedValue = pixelValue / 255.0;  //对应于每一个像素颜色值进行归一化。赋值给标准化值变量
		//标准化的值是否小于0.04045，也就是pixelValue是否小于10.31475，小于则将转换值记为标准值/12.92（范围为0~0.0031308），否则记为另一个式子，范围为（0.0031308-1）
		double transformedValue = (normalizedValue <= 0.04045) ? normalizedValue / 12.92 : pow((normalizedValue + 0.055) / 1.055, 2.4);

		sRGBGammaCorrections[pixelValue] = static_cast<float>(transformedValue);  //对应于每一个颜色的伽马sRGB校正值
	}

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			CvScalar rgbPixel = cvGet2D(leftImage, y, x);  //获取左图的rgb值

			//得到对应的gamma矫正值
			float correctedR = sRGBGammaCorrections[rgbPixel.val[2]];
			float correctedG = sRGBGammaCorrections[rgbPixel.val[1]];
			float correctedB = sRGBGammaCorrections[rgbPixel.val[0]];

			//得到对应于gamma矫正后的RGB值
			float xyzColor[3];
			xyzColor[0] = correctedR*0.412453f + correctedG*0.357580f + correctedB*0.180423f;
			xyzColor[1] = correctedR*0.212671f + correctedG*0.715160f + correctedB*0.072169f;
			xyzColor[2] = correctedR*0.019334f + correctedG*0.119193f + correctedB*0.950227f;

			const double epsilon = 0.008856;
			const double kappa = 903.3;
			const double referenceWhite[3] = { 0.950456, 1.0, 1.088754 };

			float normalizedX = static_cast<float>(xyzColor[0] / referenceWhite[0]);
			float normalizedY = static_cast<float>(xyzColor[1] / referenceWhite[1]);
			float normalizedZ = static_cast<float>(xyzColor[2] / referenceWhite[2]);
			float fX = (normalizedX > epsilon) ? static_cast<float>(pow(normalizedX*1.0, 1.0 / 3.0)) : static_cast<float>((kappa*normalizedX + 16.0) / 116.0);
			float fY = (normalizedY > epsilon) ? static_cast<float>(pow(normalizedY*1.0, 1.0 / 3.0)) : static_cast<float>((kappa*normalizedY + 16.0) / 116.0);
			float fZ = (normalizedZ > epsilon) ? static_cast<float>(pow(normalizedZ*1.0, 1.0 / 3.0)) : static_cast<float>((kappa*normalizedZ + 16.0) / 116.0);
			//float fX = 1.0, fY = 1.0, fZ = 1.0;

			//存入对应x,y坐标的LAB空间中的值。位置分别为 width*x*3 + 3*x + (0,1,2)
			inputLabImage_[width_ * 3 * y + 3 * x] = static_cast<float>(116.0*fY - 16.0);
			inputLabImage_[width_ * 3 * y + 3 * x + 1] = static_cast<float>(500.0*(fX - fY));
			inputLabImage_[width_ * 3 * y + 3 * x + 2] = static_cast<float>(200.0*(fY - fZ));
		}
	}
}

void SPSStereo::computeInitialDisparityImage(const IplImage* leftImage, const IplImage* rightImage, IplImage* sgmImage) {
	SGMStereo sgm;
	sgm.compute(leftImage, rightImage, initialDisparityImage_);
	//此处应该尝试输出initialDisparityImage_

	uchar* pValue;
	int idx = 0;
	for (int y = 0; y < sgmImage->height; ++y) {
		for (int x = 0; x < sgmImage->width; ++x) {
			pValue = &((uchar*)(sgmImage->imageData + sgmImage->widthStep*(y)))[(x)*sgmImage->nChannels];
			pValue[0] = static_cast<uint>(initialDisparityImage_[idx]) & 0xff;  //the sgm output is single channel
			pValue[1] = pValue[0];
			pValue[2] = pValue[0];
			//pValue[1] = (static_cast<uint>(initialDisparityImage_[idx]) >> 8) & 0xff;
			//pValue[2] = (static_cast<uint>(initialDisparityImage_[idx]) >> 16) & 0xff;
			idx++;
		}
	}
	/*cv::Mat depthImage(leftImage->height, leftImage->width, CV_16UC1);
	std::cout << depthImage.rows << " " << depthImage.cols << std::endl;
	for (int x = 0; x < depthImage.rows; ++x) {
		for (int y = 0; y < depthImage.cols; ++y) {
			depthImage.at<unsigned char>(x, y) = initialDisparityImage_[x * width_ + y];
		}
	}
	cvNamedWindow("SGM");
	cvShowImage("SGM", sgmImage);
	cvWaitKey(0);*/
}

void SPSStereo::initializeSegment(const int superpixelTotal) {
	makeGridSegment(superpixelTotal);
	assignLabel();
	initialFitDisparityPlane();
}

//创建格形的分割
void SPSStereo::makeGridSegment(const int superpixelTotal) {
	int imageSize = width_*height_;
	double gridSize = sqrt(static_cast<double>(imageSize)/superpixelTotal);   //每个格子的边长
	stepSize_ = static_cast<int>(gridSize + 2.0);   //步长，是每个格子加2

	int segmentTotalX = static_cast<int>(ceil(width_/gridSize));   //宽向上的seg数
	int segmentTotalY = static_cast<int>(ceil(height_/gridSize));   //高向上的seg数

	segmentTotal_ = segmentTotalX*segmentTotalY;   //总的seg数
	segments_.resize(segmentTotal_);   //重置segment_变量的大小为总的seg数
	for (int y = 0; y < height_; ++y) {
		int segmentIndexY = static_cast<int>(y/gridSize);
		for (int x = 0; x < width_; ++x) {
			int segmentIndexX = static_cast<int>(x/gridSize);   //确定是横向第几个seg
			int segmentIndex = segmentTotalX*segmentIndexY + segmentIndexX;   //确定是整体第几个seg

			labelImage_[width_*y + x] = segmentIndex;   //记录下标注的序号
			//第segmentIndex个seg执行增加像素操作。传入x，y以及LAB的值。相当于对每一个像素都划定了一块区域
			segments_[segmentIndex].addPixel(x, y, inputLabImage_[width_*3*y + 3*x], inputLabImage_[width_*3*y + 3*x + 1], inputLabImage_[width_*3*y + 3*x + 2]);
		}
	}

	memset(outlierFlagImage_, 0, width_*height_);   //初始化outlierFlagImage_变量中width_height个字节为0
}

void SPSStereo::assignLabel() {
	std::stack<int> boundaryPixelIndices;     //边界像素索引
	extractBoundaryPixel(boundaryPixelIndices);  //提取边界像素

	while (!boundaryPixelIndices.empty()) {  //非空
		int pixelIndex = boundaryPixelIndices.top();  //最后一个是边界的像素
		boundaryPixelIndices.pop();  //弹出
		int pixelX = pixelIndex%width_;   //得到第几列
		int pixelY = pixelIndex/width_;   //得到第几行
		boundaryFlagImage_[width_*pixelY + pixelX] = 0;   //标志位记为0

		if (isUnchangeable(pixelX, pixelY)) continue;   //是否为不可变更的，也就是该xy是否应该属于这个seg

		int bestSegmentIndex = findBestSegmentLabel(pixelX, pixelY);//找最适合的label
		if (bestSegmentIndex == labelImage_[width_*pixelY + pixelX]) continue;  //最好的还是原来的，那么直接进入下一次循环

		changeSegmentLabel(pixelX, pixelY, bestSegmentIndex); // 将pixelX与pixelY的信息更新到bestSegment中
		addNeighborBoundaryPixel(pixelX, pixelY, boundaryPixelIndices);   //添加邻近的边界像素
	}
}

void SPSStereo::extractBoundaryPixel(std::stack<int>& boundaryPixelIndices) {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (isBoundaryPixel(x, y)) {   //是的话。将索引（也就是第几个像素位置）压入栈中并，并记录下来
				boundaryPixelIndices.push(width_*y + x);
				boundaryFlagImage_[width_*y + x] = 1;
			} else {
				boundaryFlagImage_[width_*y + x] = 0;
			}
		}
	}
}

bool SPSStereo::isBoundaryPixel(const int x, const int y) const {
	int pixelSegmentIndex = labelImage_[width_*y + x];   //该像素属于第几个seg
	for (int neighborIndex = 0; neighborIndex < fourNeighborTotal; ++neighborIndex) {
		int neighborX = x + fourNeighborOffsetX[neighborIndex];
		if (neighborX < 0 || neighborX >= width_) continue;
		int neighborY = y + fourNeighborOffsetY[neighborIndex];
		if (neighborY < 0 || neighborY >= height_) continue;

		if (labelImage_[width_*neighborY + neighborX] != pixelSegmentIndex) return true;
	}

	return false;
}

bool SPSStereo::isUnchangeable(const int x, const int y) const {
	int pixelSegmentIndex = labelImage_[width_*y + x];  //拿到对应xy的seg的序号

	int outCount = 0;
	for (int neighborIndex = 0; neighborIndex < eightNeighborTotal; ++neighborIndex) {
		int neighborX = x + eightNeighborOffsetX[neighborIndex];
		if (neighborX < 0 || neighborX >= width_) continue;
		int neighborY = y + eightNeighborOffsetY[neighborIndex];
		if (neighborY < 0 || neighborY >= height_) continue;

		if (labelImage_[width_*neighborY + neighborX] != pixelSegmentIndex) continue;

		int nextNeighborIndex = neighborIndex + 1;
		if (nextNeighborIndex >= eightNeighborTotal) nextNeighborIndex = 0;

		int nextNeighborX = x + eightNeighborOffsetX[nextNeighborIndex];
		if (nextNeighborX < 0 || nextNeighborX >= width_) { ++outCount; continue; }
		int nextNeighborY = y + eightNeighborOffsetY[nextNeighborIndex];
		if (nextNeighborY < 0 || nextNeighborY >= height_) { ++outCount; continue; }

		if (labelImage_[width_*nextNeighborY + nextNeighborX] != pixelSegmentIndex) ++outCount;
	}
	if (outCount > 1) return true;

	return false;
}

int SPSStereo::findBestSegmentLabel(const int x, const int y) const {
	int bestSegmentIndex = -1;   //初始化应有的最好的seg序号为不存在（-1）
	double bestEnergy = DBL_MAX;  //最好的能量值为DBL_MAX

	std::vector<int> neighborSegmentIndices = getNeighborSegmentIndices(x, y);  //得到xy相邻的的seg索引
	for (int neighborIndex = 0; neighborIndex < static_cast<int>(neighborSegmentIndices.size()); ++neighborIndex) {
		int segmentIndex = neighborSegmentIndices[neighborIndex];

		double pixelEnergy = computePixelEnergy(x, y, segmentIndex);  //计算在segmentIndex下对应xy的像素的能量值
		double boundaryLengthEnergy = computeBoundaryLengthEnergy(x, y, segmentIndex);  //计算在segmentIndex下对于的xy的边界能量值

		double totalEnergy = pixelEnergy + boundaryLengthEnergy;   //总能量为二者求和
		if (totalEnergy < bestEnergy) {   //小于最佳的
			bestSegmentIndex = segmentIndex;   //则更改
			bestEnergy = totalEnergy;  //并记录下最佳的方便下次对比
		}
	}

	return bestSegmentIndex;
}

std::vector<int> SPSStereo::getNeighborSegmentIndices(const int x, const int y) const {
	std::vector<int> neighborSegmentIndices(1);
	neighborSegmentIndices[0] = labelImage_[width_*y + x];
	for (int neighborIndex = 0; neighborIndex < fourNeighborTotal; ++neighborIndex) {
		int neighborX = x + fourNeighborOffsetX[neighborIndex];
		if (neighborX < 0 || neighborX >= width_) continue;
		int neighborY = y + fourNeighborOffsetY[neighborIndex];
		if (neighborY < 0 || neighborY >= height_) continue;

		bool newSegmentFlag = true;
		for (int listIndex = 0; listIndex < static_cast<int>(neighborSegmentIndices.size()); ++listIndex) {
			if (labelImage_[width_*neighborY + neighborX] == neighborSegmentIndices[listIndex]) {
				newSegmentFlag = false;
				break;
			}
		}

		if (newSegmentFlag) neighborSegmentIndices.push_back(labelImage_[width_*neighborY + neighborX]);
	}

	return neighborSegmentIndices;
}

double SPSStereo::computePixelEnergy(const int x, const int y, const int segmentIndex) const {
	const double normalizedPositionWeight = positionWeight_/(stepSize_*stepSize_);
	const double inlierThresholdSquare = inlierThreshold_*inlierThreshold_;

	double segmentL = segments_[segmentIndex].color(0);
	double segmentA = segments_[segmentIndex].color(1);
	double segmentB = segments_[segmentIndex].color(2);
	double distanceColor = (inputLabImage_[width_*3*y + 3*x] - segmentL)*(inputLabImage_[width_*3*y + 3*x] - segmentL)
		+ (inputLabImage_[width_*3*y + 3*x + 1] - segmentA)*(inputLabImage_[width_*3*y + 3*x + 1] - segmentA)
		+ (inputLabImage_[width_*3*y + 3*x + 2] - segmentB)*(inputLabImage_[width_*3*y + 3*x + 2] - segmentB);

	double segmentX = segments_[segmentIndex].position(0);
	double segmentY = segments_[segmentIndex].position(1);
	double distancePosition = (x - segmentX)*(x - segmentX) + (y - segmentY)*(y - segmentY);

	double distanceDisparity = inlierThresholdSquare;
	double estimatedDisparity = segments_[segmentIndex].estimatedDisparity(x, y);
	if (estimatedDisparity > 0) {
		distanceDisparity = (initialDisparityImage_[width_*y + x] - estimatedDisparity)*(initialDisparityImage_[width_*y + x] - estimatedDisparity);
		if (distanceDisparity > inlierThresholdSquare) distanceDisparity = inlierThresholdSquare;
	}

	double pixelEnergy = distanceColor + normalizedPositionWeight*distancePosition + disparityWeight_*distanceDisparity;

	return pixelEnergy;
}

double SPSStereo::computeBoundaryLengthEnergy(const int x, const int y, const int segmentIndex) const {
	int boundaryCount = 0;
	for (int neighborIndex = 0; neighborIndex < eightNeighborTotal; ++neighborIndex) {
		int neighborX = x + eightNeighborOffsetX[neighborIndex];
		if (neighborX < 0 || neighborX >= width_) continue;
		int neighborY = y + eightNeighborOffsetY[neighborIndex];
		if (neighborY < 0 || neighborY >= height_) continue;

		if (labelImage_[width_*neighborY + neighborX] != segmentIndex) ++boundaryCount;
	}

	return boundaryLengthWeight_*boundaryCount;
}

void SPSStereo::changeSegmentLabel(const int x, const int y, const int newSegmentIndex) {
	int previousSegmentIndex = labelImage_[width_*y + x];
	labelImage_[width_*y + x] = newSegmentIndex;

	double estimatedDisparity = segments_[newSegmentIndex].estimatedDisparity(x, y);
	double disparityError = fabs(initialDisparityImage_[width_*y + x] - estimatedDisparity);
	if (disparityError > inlierThreshold_) outlierFlagImage_[width_*y + x] = 255;
	else outlierFlagImage_[width_*y + x] = 0;

	float pixelL = inputLabImage_[width_*y + x];
	float pixelA = inputLabImage_[width_*y + x + 1];
	float pixelB = inputLabImage_[width_*y + x + 2];

	segments_[previousSegmentIndex].removePixel(x, y, pixelL, pixelA, pixelB);
	segments_[newSegmentIndex].addPixel(x, y, pixelL, pixelA, pixelB);
}

void SPSStereo::addNeighborBoundaryPixel(const int x, const int y, std::stack<int>& boundaryPixelIndices) const {
	for (int neighorPixelIndex = 0; neighorPixelIndex < fourNeighborTotal; ++neighorPixelIndex) {
		int neighborX = x + fourNeighborOffsetX[neighorPixelIndex];
		if (neighborX < 0 || neighborX >= width_) continue;
		int neighborY = y + fourNeighborOffsetY[neighorPixelIndex];
		if (neighborY < 0 || neighborY >= height_) continue;

		if (boundaryFlagImage_[width_*neighborY + neighborX] > 0) continue;

		if (isBoundaryPixel(neighborX, neighborY)) {
			boundaryPixelIndices.push(width_*neighborY + neighborX);
			boundaryFlagImage_[width_*neighborY + neighborX] = 1;
		}
	}
}

void SPSStereo::initialFitDisparityPlane() {
	estimateDisparityPlaneRANSAC(initialDisparityImage_);  //得到对应seg的平面方程的参数
	float* interpolatedDisparityImage = reinterpret_cast<float*>(malloc(width_*height_*sizeof(float)));
	interpolateDisparityImage(interpolatedDisparityImage);
	estimateDisparityPlaneRANSAC(interpolatedDisparityImage);  //得到对应内插视图的平面方程的参数
	free(interpolatedDisparityImage);

	initializeOutlierFlagImage();
}

void SPSStereo::estimateDisparityPlaneRANSAC(const float* disparityImage) {
	const double confidenceLevel = 0.99;
	const double initialInlierThreshold = 1.0;

	std::vector< std::vector<int> > segmentPixelXs(segmentTotal_);
	std::vector< std::vector<int> > segmentPixelYs(segmentTotal_);
	std::vector< std::vector<float> > segmentPixelDisparities(segmentTotal_);
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (disparityImage[width_*y + x] == 0) continue;

			int pixelSegmentIndex = labelImage_[width_*y + x];
			segmentPixelXs[pixelSegmentIndex].push_back(x);
			segmentPixelYs[pixelSegmentIndex].push_back(y);
			segmentPixelDisparities[pixelSegmentIndex].push_back(disparityImage[width_*y + x]);
		}
	}

	for (int segmentIndex = 0; segmentIndex < segmentTotal_; ++segmentIndex) {
		if (segments_[segmentIndex].hasDisparityPlane()) continue;

		int segmentPixelTotal = static_cast<int>(segmentPixelXs[segmentIndex].size());
		if (segmentPixelTotal < 3) continue;

		int bestInlierTotal = 0;
		std::vector<bool> bestInlierFlags(segmentPixelTotal);
		int samplingTotal = segmentPixelTotal;
		int samplingCount = 0;
		while (samplingCount < samplingTotal) {
			int drawIndices[3];
			drawIndices[0] = rand()%segmentPixelTotal;
			drawIndices[1] = rand()%segmentPixelTotal;
			while (drawIndices[1] == drawIndices[0]) drawIndices[1] = rand()%segmentPixelTotal;
			drawIndices[2] = rand()%segmentPixelTotal;
			while (drawIndices[2] == drawIndices[0] || drawIndices[2] == drawIndices[1]) drawIndices[2] = rand()%segmentPixelTotal;

			std::vector<double> planeParameter;
			solvePlaneEquations(segmentPixelXs[segmentIndex][drawIndices[0]], segmentPixelYs[segmentIndex][drawIndices[0]], 1, segmentPixelDisparities[segmentIndex][drawIndices[0]],
								segmentPixelXs[segmentIndex][drawIndices[1]], segmentPixelYs[segmentIndex][drawIndices[1]], 1, segmentPixelDisparities[segmentIndex][drawIndices[1]],
								segmentPixelXs[segmentIndex][drawIndices[2]], segmentPixelYs[segmentIndex][drawIndices[2]], 1, segmentPixelDisparities[segmentIndex][drawIndices[2]],
								planeParameter);

			// Count the number of inliers
			int inlierTotal = 0;
			std::vector<bool> inlierFlags(segmentPixelTotal);
			for (int pixelIndex = 0; pixelIndex < segmentPixelTotal; ++pixelIndex) {
				double estimateDisparity = planeParameter[0]*segmentPixelXs[segmentIndex][pixelIndex]
					+ planeParameter[1]*segmentPixelYs[segmentIndex][pixelIndex]
					+ planeParameter[2];
				if (fabs(estimateDisparity - segmentPixelDisparities[segmentIndex][pixelIndex]) <= initialInlierThreshold) {
					++inlierTotal;
					inlierFlags[pixelIndex] = true;
				} else {
					inlierFlags[pixelIndex] = false;
				}
			}

			if (inlierTotal > bestInlierTotal) {
				bestInlierTotal = inlierTotal;
				bestInlierFlags = inlierFlags;

				samplingTotal = computeRequiredSamplingTotal(3, bestInlierTotal, segmentPixelTotal, samplingTotal, confidenceLevel);
			}

			++samplingCount;
		}

		double sumXSqr = 0, sumYSqr = 0, sumXY = 0, sumX = 0, sumY = 0;
		double sumXD = 0, sumYD = 0, sumD = 0;
		int inlierIndex = 0;
		for (int pixelIndex = 0; pixelIndex < segmentPixelTotal; ++pixelIndex) {
			if (bestInlierFlags[pixelIndex]) {
				int x = segmentPixelXs[segmentIndex][pixelIndex];
				int y = segmentPixelYs[segmentIndex][pixelIndex];
				float d = segmentPixelDisparities[segmentIndex][pixelIndex];

				sumXSqr += x*x;
				sumYSqr += y*y;
				sumXY += x*y;
				sumX += x;
				sumY += y;
				sumXD += x*d;
				sumYD += y*d;
				sumD += d;
				++inlierIndex;
			}
		}
		std::vector<double> planeParameter(3);
		solvePlaneEquations(sumXSqr, sumXY, sumX, sumXD,
							sumXY, sumYSqr, sumY, sumYD,
							sumX, sumY, inlierIndex, sumD,
							planeParameter);

		segments_[segmentIndex].setDisparityPlane(planeParameter[0], planeParameter[1], planeParameter[2]);
	}
}

void SPSStereo::solvePlaneEquations(const double x1, const double y1, const double z1, const double d1,
									const double x2, const double y2, const double z2, const double d2,
									const double x3, const double y3, const double z3, const double d3,
									std::vector<double>& planeParameter) const
{
	const double epsilonValue = 1e-10;

	planeParameter.resize(3);

	double denominatorA = (x1*z2 - x2*z1)*(y2*z3 - y3*z2) - (x2*z3 - x3*z2)*(y1*z2 - y2*z1);
	if (denominatorA < epsilonValue) {
		planeParameter[0] = 0.0;
		planeParameter[1] = 0.0;
		planeParameter[2] = -1.0;
		return;
	}

	planeParameter[0] = ((z2*d1 - z1*d2)*(y2*z3 - y3*z2) - (z3*d2 - z2*d3)*(y1*z2 - y2*z1))/denominatorA;

	double denominatorB = y1*z2 - y2*z1;
	if (denominatorB > epsilonValue) {
		planeParameter[1] = (z2*d1 - z1*d2 - planeParameter[0]*(x1*z2 - x2*z1))/denominatorB;
	} else {
		denominatorB = y2*z3 - y3*z2;
		planeParameter[1] = (z3*d2 - z2*d3 - planeParameter[0]*(x2*z3 - x3*z2))/denominatorB;
	}
	if (z1 > epsilonValue) {
		planeParameter[2] = (d1 - planeParameter[0]*x1 - planeParameter[1]*y1)/z1;
	} else if (z2 > epsilonValue) {
		planeParameter[2] = (d2 - planeParameter[0]*x2 - planeParameter[1]*y2)/z2;
	} else {
		planeParameter[2] = (d3 - planeParameter[0]*x3 - planeParameter[1]*y3)/z3;
	}
}

int SPSStereo::computeRequiredSamplingTotal(const int drawTotal, const int inlierTotal, const int pointTotal, const int currentSamplingTotal, const double confidenceLevel) const {
	double ep = 1 - static_cast<double>(inlierTotal)/static_cast<double>(pointTotal);
	if (ep == 1.0) {
		ep = 0.5;
	}

	int newSamplingTotal = static_cast<int>(log(1 - confidenceLevel)/log(1 - pow(1 - ep, drawTotal)) + 0.5);
	if (newSamplingTotal < currentSamplingTotal) {
		return newSamplingTotal;
	} else {
		return currentSamplingTotal;
	}
}

void SPSStereo::interpolateDisparityImage(float* interpolatedDisparityImage) const {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			interpolatedDisparityImage[width_*y + x] = initialDisparityImage_[width_*y + x];
		}
	}

	for (int y = 0; y < height_; ++y) {
		int count = 0;
		for (int x = 0; x < width_; ++x) {
			if (interpolatedDisparityImage[width_*y + x] > 0) {
				if (count >= 1) {
					int startX = x - count;
					int endX = x - 1;

					if (startX > 0 && endX < width_ - 1) {
						float interpolationDisparity = std::min(interpolatedDisparityImage[width_*y + startX - 1], interpolatedDisparityImage[width_*y + endX + 1]);
						for (int interpolateX = startX; interpolateX <= endX; ++interpolateX) {
							interpolatedDisparityImage[width_*y + interpolateX] = interpolationDisparity;
						}
					}
				}

				count = 0;
			} else {
				++count;
			}
		}

		for (int x = 0; x < width_; ++x) {
			if (interpolatedDisparityImage[width_*y + x] > 0) {
				for (int interpolateX = 0; interpolateX < x; ++interpolateX) {
					interpolatedDisparityImage[width_*y + interpolateX] = interpolatedDisparityImage[width_*y + x];
				}
				break;
			}
		}

		for (int x = width_ - 1; x >= 0; --x) {
			if (interpolatedDisparityImage[width_*y + x] > 0) {
				for (int interpolateX = x + 1; interpolateX < width_; ++interpolateX) {
					interpolatedDisparityImage[width_*y + interpolateX] = interpolatedDisparityImage[width_*y + x];
				}
				break;
			}
		}
	}

	for (int x = 0; x < width_; ++x) {
		for (int y = 0; y < height_; ++y) {
			if (interpolatedDisparityImage[width_*y + x] > 0) {
				for (int interpolateY = 0; interpolateY < y; ++interpolateY) {
					interpolatedDisparityImage[width_*interpolateY + x] = interpolatedDisparityImage[width_*y + x];
				}
				break;
			}
		}

		// extrapolate to the bottom
		for (int y = height_ - 1; y >= 0; --y) {
			if (interpolatedDisparityImage[width_*y + x] > 0) {
				for (int interpolateY = y+1; interpolateY < height_; ++interpolateY) {
					interpolatedDisparityImage[width_*interpolateY + x] = interpolatedDisparityImage[width_*y + x];
				}
				break;
			}
		}
	}
}

void SPSStereo::initializeOutlierFlagImage() {
	memset(outlierFlagImage_, 0, width_*height_);
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (initialDisparityImage_[width_*y + x] == 0) {
				outlierFlagImage_[width_*y + x] = 255;
				continue;
			}

			int pixelSegmentIndex = labelImage_[width_*y + x];
			double estimatedDisparity = segments_[pixelSegmentIndex].estimatedDisparity(x, y);
			if (fabs(initialDisparityImage_[width_*y + x] - estimatedDisparity) > inlierThreshold_) {
				outlierFlagImage_[width_*y + x] = 255;
			}
		}
	}
}

void SPSStereo::performSmoothingSegmentation() {
	for (int outerIterationCount = 0; outerIterationCount < outerIterationTotal_; ++outerIterationCount) {
		assignLabel();
		buildSegmentConfiguration();
		planeSmoothing();
		//estimateDisparityPlaneRANSAC(initialDisparityImage_);
	}
}

void SPSStereo::buildSegmentConfiguration() {
	for (int segmentIndex = 0; segmentIndex < segmentTotal_; ++segmentIndex) {
		segments_[segmentIndex].clearConfiguration();
	}
	boundaries_.clear();
	boundaryIndexMatrix_.resize(segmentTotal_);
	for (int i = 0; i < segmentTotal_; ++i) {
		boundaryIndexMatrix_[i].resize(segmentTotal_);
		for (int j = 0; j < segmentTotal_; ++j) boundaryIndexMatrix_[i][j] = -1;
	}

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			int pixelSegmentIndex = labelImage_[width_*y + x];
			segments_[pixelSegmentIndex].appendSegmentPixel(x, y);
			if (initialDisparityImage_[width_*y + x] > 0 && outlierFlagImage_[width_*y + x] == 0) {
				segments_[pixelSegmentIndex].appendSegmentPixelWithDisparity(x, y, initialDisparityImage_[width_*y + x]);
			}

			if (isHorizontalBoundary(x, y)) {
				int neighborSegmentIndex = labelImage_[width_*y + x + 1];
				int boundaryIndex = appendBoundary(pixelSegmentIndex, neighborSegmentIndex);
				boundaries_[boundaryIndex].appendBoundaryPixel(x + 0.5, y);
			}
			if (isVerticalBoundary(x, y)) {
				int neighborSegmentIndex = labelImage_[width_*(y + 1) + x];
				int boundaryIndex = appendBoundary(pixelSegmentIndex, neighborSegmentIndex);
				boundaries_[boundaryIndex].appendBoundaryPixel(x, y + 0.5);
			}
		}
	}
}

bool SPSStereo::isHorizontalBoundary(const int x, const int y) const {
	if (x >= width_ - 1) return false;

	if (labelImage_[width_*y + x] != labelImage_[width_*y + x + 1]) return true;
	return false;
}

bool SPSStereo::isVerticalBoundary(const int x, const int y) const {
	if (y >= height_ - 1) return false;

	if (labelImage_[width_*y + x] != labelImage_[width_*(y + 1) + x]) return true;
	return false;
}

int SPSStereo::appendBoundary(const int firstSegmentIndex, const int secondSegmentIndex) {
	if (boundaryIndexMatrix_[firstSegmentIndex][secondSegmentIndex] >= 0) return boundaryIndexMatrix_[firstSegmentIndex][secondSegmentIndex];

	boundaries_.push_back(Boundary(firstSegmentIndex, secondSegmentIndex));
	int newBoundaryIndex = static_cast<int>(boundaries_.size()) - 1;
	boundaryIndexMatrix_[firstSegmentIndex][secondSegmentIndex] = newBoundaryIndex;
	boundaryIndexMatrix_[secondSegmentIndex][firstSegmentIndex] = newBoundaryIndex;

	segments_[firstSegmentIndex].appendBoundaryIndex(newBoundaryIndex);
	segments_[secondSegmentIndex].appendBoundaryIndex(newBoundaryIndex);

	return newBoundaryIndex;
}

void SPSStereo::planeSmoothing() {
	for (int innerIterationCount = 0; innerIterationCount < innerIterationTotal_; ++innerIterationCount) {
		estimateBoundaryLabel();
		estimateSmoothFitting();
	}
}

void SPSStereo::estimateBoundaryLabel() {
	int boundaryTotal = static_cast<int>(boundaries_.size());
	for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
		Boundary& currentBoundary = boundaries_[boundaryIndex];
		int firstSegmentIndex = currentBoundary.segmentIndex(0);
		int secondSegmentIndex = currentBoundary.segmentIndex(1);
		Segment& firstSegment = segments_[firstSegmentIndex];
		Segment& secondSegment = segments_[secondSegmentIndex];

		double ai = firstSegment.planeParameter(0);
		double bi = firstSegment.planeParameter(1);
		double ci = firstSegment.planeParameter(2);
		double aj = secondSegment.planeParameter(0);
		double bj = secondSegment.planeParameter(1);
		double cj = secondSegment.planeParameter(2);

		std::vector<double> boundaryEnergies(4);

		// Hinge
		double hingeSquaredError = 0;
		double hingeError = 0;
		hingeError = (ai - aj)*currentBoundary.polynomialCoefficient(3)
			+ (bi - bj)*currentBoundary.polynomialCoefficient(4)
			+ (ci - cj)*currentBoundary.polynomialCoefficient(5);
		hingeSquaredError = currentBoundary.polynomialCoefficient(0)*(ai*ai + aj*aj - 2*ai*aj)
			+ currentBoundary.polynomialCoefficient(1)*(bi*bi + bj*bj - 2*bi*bj)
			+ currentBoundary.polynomialCoefficient(2)*(2*ai*bi + 2*aj*bj - 2*ai*bj -2*aj*bi)
			+ currentBoundary.polynomialCoefficient(3)*(2*ai*ci + 2*aj*cj - 2*ai*cj -2*aj*ci)
			+ currentBoundary.polynomialCoefficient(4)*(2*bi*ci + 2*bj*cj - 2*bi*cj -2*bj*ci)
			+ currentBoundary.polynomialCoefficient(5)*(ci*ci + cj*cj - 2*ci*cj);
		hingeSquaredError /= currentBoundary.boundaryPixelTotal();
		boundaryEnergies[2] = hingePenalty_ + hingeSquaredError;

		// Occlusion
		if (hingeError > 0) {
			boundaryEnergies[0] = occlusionPenalty_;
			boundaryEnergies[1] = occlusionPenalty_ + impossiblePenalty_;
		} else {
			boundaryEnergies[0] = occlusionPenalty_ + impossiblePenalty_;
			boundaryEnergies[1] = occlusionPenalty_;
		}

		// Coplanar
		double coplanarSquaredError = 0;
		coplanarSquaredError = firstSegment.polynomialCoefficientAll(0)*(ai*ai + aj*aj - 2*ai*aj)
			+ firstSegment.polynomialCoefficientAll(1)*(bi*bi + bj*bj - 2*bi*bj)
			+ firstSegment.polynomialCoefficientAll(2)*(2*ai*bi + 2*aj*bj - 2*ai*bj -2*aj*bi)
			+ firstSegment.polynomialCoefficientAll(3)*(2*ai*ci + 2*aj*cj - 2*ai*cj -2*aj*ci)
			+ firstSegment.polynomialCoefficientAll(4)*(2*bi*ci + 2*bj*cj - 2*bi*cj -2*bj*ci)
			+ firstSegment.polynomialCoefficientAll(5)*(ci*ci + cj*cj - 2*ci*cj);
		coplanarSquaredError += secondSegment.polynomialCoefficientAll(0)*(ai*ai + aj*aj - 2*ai*aj)
			+ secondSegment.polynomialCoefficientAll(1)*(bi*bi + bj*bj - 2*bi*bj)
			+ secondSegment.polynomialCoefficientAll(2)*(2*ai*bi + 2*aj*bj - 2*ai*bj -2*aj*bi)
			+ secondSegment.polynomialCoefficientAll(3)*(2*ai*ci + 2*aj*cj - 2*ai*cj -2*aj*ci)
			+ secondSegment.polynomialCoefficientAll(4)*(2*bi*ci + 2*bj*cj - 2*bi*cj -2*bj*ci)
			+ secondSegment.polynomialCoefficientAll(5)*(ci*ci + cj*cj - 2*ci*cj);
		coplanarSquaredError /= (firstSegment.pixelTotal() + secondSegment.pixelTotal());
		boundaryEnergies[3] = coplanarSquaredError;

		int minBoundaryLabel = 0;
		if (boundaryEnergies[1] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 1;
		if (boundaryEnergies[2] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 2;
		if (boundaryEnergies[3] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 3;

		boundaries_[boundaryIndex].setType(minBoundaryLabel);
	}
}

void SPSStereo::estimateSmoothFitting() {
	for (int segmentIndex = 0; segmentIndex < segmentTotal_; ++segmentIndex) {
		Segment currentSegment = segments_[segmentIndex];
		int segmentPixelTotal = currentSegment.pixelTotal();
		int disparityPixelTotal = 0;

		double sumXSqr = 0, sumYSqr = 0, sumXY = 0, sumX = 0, sumY = 0;
		double sumXD = 0, sumYD = 0, sumD = 0;
		double pointTotal = 0;

		sumXSqr += currentSegment.polynomialCoefficient(0);
		sumYSqr += currentSegment.polynomialCoefficient(1);
		sumXY += currentSegment.polynomialCoefficient(2);
		sumX += currentSegment.polynomialCoefficient(3);
		sumY += currentSegment.polynomialCoefficient(4);
		sumXD += currentSegment.polynomialCoefficient(5);
		sumYD += currentSegment.polynomialCoefficient(6);
		sumD += currentSegment.polynomialCoefficient(7);
		pointTotal += currentSegment.polynomialCoefficient(8);

		disparityPixelTotal += static_cast<int>(currentSegment.polynomialCoefficient(8));

		for (int neighborIndex = 0; neighborIndex < currentSegment.boundaryTotal(); ++neighborIndex) {
			int boundaryIndex = currentSegment.boundaryIndex(neighborIndex);
			int boundaryLabel = boundaries_[boundaryIndex].type();
			if (boundaryLabel < 2) continue;

			Boundary& currentBoundary = boundaries_[boundaryIndex];
			int neighborSegmentIndex = currentBoundary.segmentIndex(0);
			if (neighborSegmentIndex == segmentIndex) neighborSegmentIndex = currentBoundary.segmentIndex(1);
			Segment& neighborSegment = segments_[neighborSegmentIndex];

			if (boundaryLabel == 2) {
				// Hinge
				int boundaryPixelTotal = currentBoundary.boundaryPixelTotal();
				double weightValue = smoothRelativeWeight_/boundaryPixelTotal*stepSize_*stepSize_;

				sumXSqr += weightValue*currentBoundary.polynomialCoefficient(0);
				sumYSqr += weightValue*currentBoundary.polynomialCoefficient(1);
				sumXY += weightValue*currentBoundary.polynomialCoefficient(2);
				sumX += weightValue*currentBoundary.polynomialCoefficient(3);
				sumY += weightValue*currentBoundary.polynomialCoefficient(4);
				pointTotal += weightValue*currentBoundary.polynomialCoefficient(5);

				sumXD += weightValue*(neighborSegment.planeParameter(0)*currentBoundary.polynomialCoefficient(0)
									  + neighborSegment.planeParameter(1)*currentBoundary.polynomialCoefficient(2)
									  + neighborSegment.planeParameter(2)*currentBoundary.polynomialCoefficient(3));
				sumYD += weightValue*(neighborSegment.planeParameter(0)*currentBoundary.polynomialCoefficient(2)
									  + neighborSegment.planeParameter(1)*currentBoundary.polynomialCoefficient(1)
									  + neighborSegment.planeParameter(2)*currentBoundary.polynomialCoefficient(4));
				sumD += weightValue*(neighborSegment.planeParameter(0)*currentBoundary.polynomialCoefficient(3)
									 + neighborSegment.planeParameter(1)*currentBoundary.polynomialCoefficient(4)
									 + neighborSegment.planeParameter(2)*currentBoundary.polynomialCoefficient(5));

				disparityPixelTotal += static_cast<int>(currentBoundary.polynomialCoefficient(5));

			} else {
				// Coplanar
				int neighborSegmentPixelTotal = neighborSegment.pixelTotal();
				double weightValue = smoothRelativeWeight_/(segmentPixelTotal + neighborSegmentPixelTotal)*stepSize_*stepSize_;

				sumXSqr += weightValue*currentSegment.polynomialCoefficientAll(0);
				sumYSqr += weightValue*currentSegment.polynomialCoefficientAll(1);
				sumXY += weightValue*currentSegment.polynomialCoefficientAll(2);
				sumX += weightValue*currentSegment.polynomialCoefficientAll(3);
				sumY += weightValue*currentSegment.polynomialCoefficientAll(4);
				pointTotal += weightValue*currentSegment.polynomialCoefficientAll(5);

				sumXD += weightValue*(neighborSegment.planeParameter(0)*currentSegment.polynomialCoefficientAll(0)
									  + neighborSegment.planeParameter(1)*currentSegment.polynomialCoefficientAll(2)
									  + neighborSegment.planeParameter(2)*currentSegment.polynomialCoefficientAll(3));
				sumYD += weightValue*(neighborSegment.planeParameter(0)*currentSegment.polynomialCoefficientAll(2)
									  + neighborSegment.planeParameter(1)*currentSegment.polynomialCoefficientAll(1)
									  + neighborSegment.planeParameter(2)*currentSegment.polynomialCoefficientAll(4));
				sumD += weightValue*(neighborSegment.planeParameter(0)*currentSegment.polynomialCoefficientAll(3)
									 + neighborSegment.planeParameter(1)*currentSegment.polynomialCoefficientAll(4)
									 + neighborSegment.planeParameter(2)*currentSegment.polynomialCoefficientAll(5));

				disparityPixelTotal += static_cast<int>(currentSegment.polynomialCoefficientAll(5));

				sumXSqr += weightValue*neighborSegment.polynomialCoefficientAll(0);
				sumYSqr += weightValue*neighborSegment.polynomialCoefficientAll(1);
				sumXY += weightValue*neighborSegment.polynomialCoefficientAll(2);
				sumX += weightValue*neighborSegment.polynomialCoefficientAll(3);
				sumY += weightValue*neighborSegment.polynomialCoefficientAll(4);
				pointTotal += weightValue*neighborSegment.polynomialCoefficientAll(5);

				sumXD += weightValue*(neighborSegment.planeParameter(0)*neighborSegment.polynomialCoefficientAll(0)
									  + neighborSegment.planeParameter(1)*neighborSegment.polynomialCoefficientAll(2)
									  + neighborSegment.planeParameter(2)*neighborSegment.polynomialCoefficientAll(3));
				sumYD += weightValue*(neighborSegment.planeParameter(0)*neighborSegment.polynomialCoefficientAll(2)
									  + neighborSegment.planeParameter(1)*neighborSegment.polynomialCoefficientAll(1)
									  + neighborSegment.planeParameter(2)*neighborSegment.polynomialCoefficientAll(4));
				sumD += weightValue*(neighborSegment.planeParameter(0)*neighborSegment.polynomialCoefficientAll(3)
									 + neighborSegment.planeParameter(1)*neighborSegment.polynomialCoefficientAll(4)
									 + neighborSegment.planeParameter(2)*neighborSegment.polynomialCoefficientAll(5));

				disparityPixelTotal += static_cast<int>(neighborSegment.polynomialCoefficientAll(5));

			}
		}

		if (disparityPixelTotal >= 3) {
			std::vector<double> planeParameter(3);
			solvePlaneEquations(sumXSqr, sumXY, sumX, sumXD,
								sumXY, sumYSqr, sumY, sumYD,
								sumX, sumY, pointTotal, sumD,
								planeParameter);
			segments_[segmentIndex].setDisparityPlane(planeParameter[0], planeParameter[1], planeParameter[2]);
		}
	}
}

void SPSStereo::makeOutputImage(IplImage* segmentImage, IplImage* segmentDisparityImage) const {
	CvScalar temp_pixel;
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			int pixelSegmentIndex = labelImage_[width_*y + x];
			double estimatedDisparity = segments_[pixelSegmentIndex].estimatedDisparity(x, y);   //在对应的superpixel找xy
			temp_pixel.val[0] = static_cast<unsigned short>(pixelSegmentIndex);
			cvSet2D(segmentImage, y, x, temp_pixel);
			if (estimatedDisparity <= 0.0 || estimatedDisparity > 255.0) {
				temp_pixel.val[0] = 0;
				cvSet2D(segmentDisparityImage, y, x, temp_pixel);
			}
			else {
				temp_pixel.val[0] = static_cast<unsigned short>(estimatedDisparity);
				cvSet2D(segmentDisparityImage, y, x, temp_pixel);
			}
		}
	}
}


void SPSStereo::makeSegmentBoundaryData(std::vector< std::vector<double> >& disparityPlaneParameters, std::vector< std::vector<int> >& boundaryLabels) const {
	int segmentTotal = static_cast<int>(segments_.size());
	disparityPlaneParameters.resize(segmentTotal);
	for (int segmentIndex = 0; segmentIndex < segmentTotal; ++segmentIndex) {
		disparityPlaneParameters[segmentIndex].resize(3);
		disparityPlaneParameters[segmentIndex][0] = segments_[segmentIndex].planeParameter(0);
		disparityPlaneParameters[segmentIndex][1] = segments_[segmentIndex].planeParameter(1);
		disparityPlaneParameters[segmentIndex][2] = segments_[segmentIndex].planeParameter(2);
	}

	int boundaryTotal = static_cast<int>(boundaries_.size());
	boundaryLabels.resize(boundaryTotal);
	for (int boundaryIndex = 0; boundaryIndex < boundaryTotal; ++boundaryIndex) {
		boundaryLabels[boundaryIndex].resize(3);
		boundaryLabels[boundaryIndex][0] = boundaries_[boundaryIndex].segmentIndex(0);
		boundaryLabels[boundaryIndex][1] = boundaries_[boundaryIndex].segmentIndex(1);
		boundaryLabels[boundaryIndex][2] = boundaries_[boundaryIndex].type();
	}
}

void SPSStereo::getSegmentAroundIndex(std::vector< std::vector<int> >& segmentAroundIndex, int segmentNums, int outputNum) {
	int boundariesTotals = boundaries_.size();
	segmentAroundIndex.resize(segmentNums);  //初始化seg周围索引变量大小为超像素的个数

	for (int boundariseIndex = 0; boundariseIndex < boundariesTotals; boundariseIndex++) {
		int temp0 = boundaries_[boundariseIndex].segmentIndex(0);
		int temp1 = boundaries_[boundariseIndex].segmentIndex(1);
		//所有边界遍历结束后。segmentAroundIndex的个数一定会满足超像素的个数，因为边界是连接了所有的seg的（也就是超像素），之后对于每个向量进行去重就可以，。这样
		//就得到了一个seg周围所有挨着的seg
		segmentAroundIndex[temp0].push_back(temp1);  //对应0的邻居1加入
		segmentAroundIndex[temp1].push_back(temp0);  //对应1的邻居0加入
	}

	//	for (int segmentIndex = 0; segmentIndex < segmentNums; segmentIndex++){
	//	sort(segmentAroundIndex[segmentIndex].begin(), segmentAroundIndex[segmentIndex].end());  //对元素进行排序
	//		segmentAroundIndex[segmentIndex].erase( unique(segmentAroundIndex[segmentIndex].begin(), segmentAroundIndex[segmentIndex].end()),
	//			segmentAroundIndex[segmentIndex].end());   //去掉重复的元素，得到唯一的相邻seg
	//}

	for (int i = 0; i < outputNum; i++) {
		std::cout << "segment " << i << ":";
		for (int j = 0; j < segmentAroundIndex[i].size(); j++)
		{
			std::cout << segmentAroundIndex[i][j] << "-";
		}
		std::cout << "" << std::endl;
	}
}
