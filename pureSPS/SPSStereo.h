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

//SPSStereo��
class SPSStereo {
public:
	SPSStereo();
	
	//��������Ӳ�����
	void setOutputDisparityFactor(const double outputDisparityFactor);
	//���õ��������������������������ڲ����������
	void setIterationTotal(const int outerIterationTotal, const int innerIterationTotal);
	//����Ȩ�ز���
	void setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight);
	//�����ڵ����ֵ
	void setInlierThreshold(const double inlierThreshold);
	//���óͷ�����
	void setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty);
	//*****���㺯��*****
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
	//segment�ࣨĿ��Ϊ�ָ���ص��ࣩ
	class Segment {
	public:
		Segment() {
			pixelTotal_ = 0;
			colorSum_[0] = 0;  colorSum_[1] = 0;  colorSum_[2] = 0;
			positionSum_[0] = 0;  positionSum_[1] = 0;
			disparityPlane_[0] = 0;  disparityPlane_[1] = 0;  disparityPlane_[2] = -1;
		}
		//��������
		void addPixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {   
			pixelTotal_ += 1;   //������Ŀ��1
			colorSum_[0] += colorL;  colorSum_[1] += colorA;  colorSum_[2] += colorB;   //LAB��ɫ�ռ�ͽ��и��£������µ�ֵ
			positionSum_[0] += x;  positionSum_[1] += y;     //λ�úͽ��и��£�����������
		}
		//�Ƴ�����
		void removePixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {
			pixelTotal_ -= 1;    //������Ŀ��1
			colorSum_[0] -= colorL;  colorSum_[1] -= colorA;  colorSum_[2] -= colorB;   //LAB��ɫ�ռ�ͽ��и��£���ȥ�µ�ֵ
			positionSum_[0] -= x;  positionSum_[1] -= y;    //λ�úͽ��и��£���ȥ������
		}

		//�����Ӳ�ƽ��
		void setDisparityPlane(const double planeGradientX, const double planeGradientY, const double planeConstant) {
			disparityPlane_[0] = planeGradientX;   //x���Ӳ��ݶ�
			disparityPlane_[1] = planeGradientY;    //y���Ӳ��ݶ�
			disparityPlane_[2] = planeConstant;    //ƽ�泣��
		}

		//����seg����������
		int pixelTotal() const { return pixelTotal_; }   
		//����һ��seg��ĳһͨ����ƽ����ɫֵ
		double color(const int colorIndex) const { return colorSum_[colorIndex]/pixelTotal_; }   
		//����һ��seg��x����y��ƽ������
		double position(const int coordinateIndex) const { return positionSum_[coordinateIndex]/pixelTotal_; }    
		//���������x��y����Ƶ��Ӳ�ֵ��Ҳ������ȣ�z =��ax+by+c)
		double estimatedDisparity(const double x, const double y) const { return disparityPlane_[0]*x + disparityPlane_[1]*y + disparityPlane_[2]; }  
		//����Ƿ��й��Ƶ�ƽ�档ֻҪa,b,c�в������ڣ�˵������ƽ��
		bool hasDisparityPlane() const { if (disparityPlane_[0] != 0.0 || disparityPlane_[1] != 0.0 || disparityPlane_[2] != -1.0) return true; else return false; }

		//�������
		void clearConfiguration() {
			neighborSegmentIndices_.clear();  //�������seg������
			boundaryIndices_.clear();    //����߽�����
			for (int i = 0; i < 9; ++i) polynomialCoefficients_[i] = 0;  //�������ʽϵ��
			for (int i = 0; i < 6; ++i) polynomialCoefficientsAll_[i] = 0;   //������ж���ʽϵ��
		}
		//����߽�����
		void appendBoundaryIndex(const int boundaryIndex) { boundaryIndices_.push_back(boundaryIndex); }
		//����seg������
		void appendSegmentPixel(const int x, const int y) {
			polynomialCoefficientsAll_[0] += x*x;    //a = a + x*x
			polynomialCoefficientsAll_[1] += y*y;    //b = b + y*y
			polynomialCoefficientsAll_[2] += x*y;    //c = c + x*y
			polynomialCoefficientsAll_[3] += x;      //d = d + x
			polynomialCoefficientsAll_[4] += y;      //e = e + y
			polynomialCoefficientsAll_[5] += 1;      //f = f + 1
		}
		//������������Ϣ��seg������
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

		//�������ڵ�seg������
		int neighborTotal() const { return static_cast<int>(neighborSegmentIndices_.size()); }
		//�������ڵ�����Ϊindex��seg��ֵ
		int neighborIndex(const int index) const { return neighborSegmentIndices_[index]; }

		//�������ڵı߽������
		int boundaryTotal() const { return static_cast<int>(boundaryIndices_.size()); }
		//�������ڵı߽�Ϊindex��seg��ֵ
		int boundaryIndex(const int index) const { return boundaryIndices_[index]; }

		//���ز����������Ϣ�ĵ�index������ʽϵ��
		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }
		//���ذ��������Ϣ�ĵ�index������ʽϵ��
		double polynomialCoefficientAll(const int index) const { return polynomialCoefficientsAll_[index]; }

		//����ƽ�������Ҳ����ȷ��ƽ���a,b,c��
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
	//Boundary�ࣨĿ����߽���أ�
	class Boundary {
	public:
		//���캯�� ��segment����������Ԫ�ض���Ϊ-1�������ϵ��
		Boundary() { segmentIndices_[0] = -1; segmentIndices_[1] = -1; clearCoefficients(); }
		//���εĹ��캯��������߽����ҵĵ�һ���ڶ�������seg������
		Boundary(const int firstSegmentIndex, const int secondSegmentIndex) {
			if (firstSegmentIndex < secondSegmentIndex) {   //�����һ��С�ڵڶ�������ô0�ǵ�һ����1�ǵڶ���
				segmentIndices_[0] = firstSegmentIndex; segmentIndices_[1] = secondSegmentIndex;
			} else {    //����0�ǵڶ�����1�ǵ�һ��
				segmentIndices_[0] = secondSegmentIndex; segmentIndices_[1] = firstSegmentIndex;
			} 
			clearCoefficients();//�������ʽϵ��
		}

		//��������������Ϣ�Ķ���ʽϵ��
		void clearCoefficients() {
			for (int i = 0; i < 6; ++i) polynomialCoefficients_[i] = 0;
		}
		
		//��������
		void setType(const int typeIndex) { type_ = typeIndex; }

		//���󲻴������Ϣ�ı߽�����
		void appendBoundaryPixel(const double x, const double y) {
			boundaryPixelXs_.push_back(x);   //ѹ��߽����ص�x
			boundaryPixelYs_.push_back(y);   //ѹ��߽����ص�y
			polynomialCoefficients_[0] += x*x;   //a = a + x*x
			polynomialCoefficients_[1] += y*y;   //b = b + y*y
			polynomialCoefficients_[2] += x*y;   //c = c + x*y
			polynomialCoefficients_[3] += x;     //d = d + x
			polynomialCoefficients_[4] += y;     //e = e + y
			polynomialCoefficients_[5] += 1;     //f = f + 1
		}

		//��������
		int type() const { return type_; }
		//���ذ���seg��������index = 0����ߵģ�index = 1���ұߵ�
		int segmentIndex(const int index) const { return segmentIndices_[index]; }

		//�������seg�Ƿ���һ���߽�����ࣨҲ���������Ƿ����ڣ����ǵĻ�����true
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

		//���������߽����е����ظ���
		int boundaryPixelTotal() const { return static_cast<int>(boundaryPixelXs_.size()); }
		//���ر߽����ƶ�index�����ص�x����
		double boundaryPixelX(const int index) const { return boundaryPixelXs_[index]; }
		//���ر߽����ƶ�index�����ص�y����
		double boundaryPixelY(const int index) const { return boundaryPixelYs_[index]; }

		//�����ƶ��ĵ�index������ʽϵ�����������9�������������6����
		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }

	private:
		int type_;
		int segmentIndices_[2];  //seg�������������ʾ���¼�˱߽���������ġ�
		std::vector<double> boundaryPixelXs_;
		std::vector<double> boundaryPixelYs_;

		double polynomialCoefficients_[6];
	};


	void allocateBuffer();   //���仺��
	void freeBuffer();    //�ͷŻ���
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
