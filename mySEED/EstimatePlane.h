#pragma once
#include <vector>
#include <stack>
#include <opencv.hpp>
#include "Vertex.h"
#include "Matrix.h"

typedef unsigned int UINT;

class EstimatePlane {
public:
	EstimatePlane();

	void setOutputDisparityFactor(const double outputDisparityFactor);
	void setIterationTotal(const int outerIterationTotal, const int innerIterationTotal);
	void setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight);
	void setInlierThreshold(const double inlierThreshold);
	void setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty);
	void setCameraParam(const double param_cu, const double param_cv, const double param_base, const double param_f) {

	}

	~EstimatePlane();
	void planeCompute(IplImage* disparityImage, int segmentTotal, UINT* labelImage, IplImage* segmentImage, IplImage* optimizedDisparityImage);
	void makeOutputImage(IplImage* segmentImage, IplImage* segmentDisparityImage) const;
	void makeSegmentBoundaryData(std::vector< std::vector<double> >& disparityPlaneParameters, std::vector< std::vector<int> >& boundaryLabels) const;
	void getDisparityPlane(double** planeFunction);
	void performSmoothingSegmentation();
	void freeBuffer();

	//得到seg的周围图
	void getSegmentAroundIndex(std::vector< std::vector<int> >& segmentAroundIndex, int segmentNums, int outputNum);

	// for plane reconstruction
	void planeGetposition(std::vector<double>& x, std::vector<double>& y) {
		x.resize(segmentTotal_);
		y.resize(segmentTotal_);
		for (int i = 0; i < segmentTotal_; ++i) {
			x[i] = segments_[i].position(0);
			y[i] = segments_[i].position(1);
		}
	}
	void planeGetParameter(std::vector<double>& p1, std::vector<double>& p2, std::vector<double>& p3) {
		p1.resize(segmentTotal_);
		p2.resize(segmentTotal_);
		p3.resize(segmentTotal_);
		for (int i = 0; i < segmentTotal_; ++i) {
			p1[i] = segments_[i].planeParameter(0);
			p2[i] = segments_[i].planeParameter(1);
			p3[i] = segments_[i].planeParameter(2);
		}
	}
	// project result
	void cameraParameterInitialization(double cu, double cv, double f, double base) {
		param_cu = cu;
		param_cv = cv;
		param_f = f;
		param_base = base;
	}
	void vertexProjectAndStore(const vector<VERTEX>& result);
	void setInputColor(const IplImage* leftImage);
	void projectPlane();
	std::vector<VERTEX3D> getProjectResult(std::vector< std::vector<double> >& planeOutput, Matrix pose);
private:
	class Segment {
	public:
		Segment() {
			pixelTotal_ = 0;
			colorSum_[0] = 0;  colorSum_[1] = 0;  colorSum_[2] = 0;
			positionSum_[0] = 0;  positionSum_[1] = 0;
			disparityPlane_[0] = 0;  disparityPlane_[1] = 0;  disparityPlane_[2] = -1;
			planeColorGray_ = 0; planeColorRGB_[0] = 0; planeColorRGB_[1] = 0; planeColorRGB_[2] = 0;
			vertexX.clear(); vertexY.clear();
			projectX.clear(); projectY.clear(); projectZ.clear();
			planeProjectCenter_[0] = 0; planeProjectCenter_[1] = 0; planeProjectCenter_[2] = 0;
		}

		void addPixel(const int x, const int y) {
			//, const float colorL, const float colorA, const float colorB
			pixelTotal_ += 1;
			//colorSum_[0] += colorL;  colorSum_[1] += colorA;  colorSum_[2] += colorB;
			positionSum_[0] += x;  positionSum_[1] += y;
		}
		void removePixel(const int x, const int y, const float colorL, const float colorA, const float colorB) {
			pixelTotal_ -= 1;
			colorSum_[0] -= colorL;  colorSum_[1] -= colorA;  colorSum_[2] -= colorB;
			positionSum_[0] -= x;  positionSum_[1] -= y;
		}
		void setDisparityPlane(const double planeGradientX, const double planeGradientY, const double planeConstant) {
			disparityPlane_[0] = planeGradientX;
			disparityPlane_[1] = planeGradientY;
			disparityPlane_[2] = planeConstant;
		}

		int pixelTotal() const { return pixelTotal_; }
		double color(const int colorIndex) const { return colorSum_[colorIndex] / pixelTotal_; }
		double position(const int coordinateIndex) const { return positionSum_[coordinateIndex] / pixelTotal_; }
		double estimatedDisparity(const double x, const double y) const { return disparityPlane_[0] * x + disparityPlane_[1] * y + disparityPlane_[2]; }
		bool hasDisparityPlane() const { if (disparityPlane_[0] != 0.0 || disparityPlane_[1] != 0.0 || disparityPlane_[2] != -1.0) return true; else return false; }

		void clearConfiguration() {
			neighborSegmentIndices_.clear();
			boundaryIndices_.clear();
			for (int i = 0; i < 9; ++i) polynomialCoefficients_[i] = 0;
			for (int i = 0; i < 6; ++i) polynomialCoefficientsAll_[i] = 0;
		}
		void appendBoundaryIndex(const int boundaryIndex) { boundaryIndices_.push_back(boundaryIndex); }
		void appendSegmentPixel(const int x, const int y) {
			polynomialCoefficientsAll_[0] += x*x;
			polynomialCoefficientsAll_[1] += y*y;
			polynomialCoefficientsAll_[2] += x*y;
			polynomialCoefficientsAll_[3] += x;
			polynomialCoefficientsAll_[4] += y;
			polynomialCoefficientsAll_[5] += 1;
		}
		void appendSegmentPixelWithDisparity(const int x, const int y, const double d) {
			polynomialCoefficients_[0] += x*x;
			polynomialCoefficients_[1] += y*y;
			polynomialCoefficients_[2] += x*y;
			polynomialCoefficients_[3] += x;
			polynomialCoefficients_[4] += y;
			polynomialCoefficients_[5] += x*d;
			polynomialCoefficients_[6] += y*d;
			polynomialCoefficients_[7] += d;
			polynomialCoefficients_[8] += 1;
		}

		int neighborTotal() const { return static_cast<int>(neighborSegmentIndices_.size()); }
		int neighborIndex(const int index) const { return neighborSegmentIndices_[index]; }

		int boundaryTotal() const { return static_cast<int>(boundaryIndices_.size()); }
		int boundaryIndex(const int index) const { return boundaryIndices_[index]; }

		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }
		double polynomialCoefficientAll(const int index) const { return polynomialCoefficientsAll_[index]; }

		double planeParameter(const int index) const { return disparityPlane_[index]; }
		double getPlaneColorRGB(const int colorIndex) const { return planeColorRGB_[colorIndex]; }
		void setPlaneColorRGB(int r, int g, int b) {
			planeColorRGB_[0] = r;
			planeColorRGB_[1] = g;
			planeColorRGB_[2] = b;
		}
		int getPlaneColorRGB(int index) { return planeColorRGB_[index];  }
		double getPlaneColorGray() const { return planeColorGray_; }
		void setPlaneColorGray(double gray) {
			planeColorGray_ = gray;
		}
		int getPlaneGray() { return planeColorGray_; }
		void setVertex(int vX, int vY, double pX, double pY, double pZ) {
			vertexX.push_back(vX);
			vertexY.push_back(vY);
			projectX.push_back(pX);
			projectY.push_back(pY);
			projectZ.push_back(pZ);
		}
		int getVertexSize() { return vertexX.size(); }
		double getProjectX(int index) { return projectX[index]; }
		double getProjectY(int index) { return projectY[index]; }
		double getProjectZ(int index) { return projectZ[index]; }

		int getVertexU(int index) { return vertexX[index]; }
		int getVertexV(int index) { return vertexY[index]; }

		void setProjectCenter(double x, double y, double z) {
			planeProjectCenter_[0] = x;
			planeProjectCenter_[1] = y;
			planeProjectCenter_[2] = z;
		}
		double getProjectCenter(int index) { return planeProjectCenter_[index]; }

		void setProjectCoefficients(double p1, double p2, double p3) {
			planeProjectParameter_[0] = p1;
			planeProjectParameter_[1] = p2;
			planeProjectParameter_[2] = p3;
		}
		void clearVertex() {
			vertexX.clear();
			vertexY.clear();
			projectX.clear();
			projectY.clear();
			projectZ.clear();
		}
		double getProjectCoefficients(int index) { return planeProjectParameter_[index];  }
	private:
		int pixelTotal_;
		double colorSum_[3];
		double positionSum_[2];
		double disparityPlane_[3];

		int planeColorRGB_[3];
		double planeColorGray_;
		double planeProjectCenter_[3];
		double planeProjectParameter_[3];

		std::vector<int> neighborSegmentIndices_;
		std::vector<int> boundaryIndices_;
		double polynomialCoefficients_[9];
		double polynomialCoefficientsAll_[6];
		std::vector<int> vertexX;
		std::vector<int> vertexY;
		std::vector<double> projectX;
		std::vector<double> projectY;
		std::vector<double> projectZ;

	};
	class Boundary {
	public:
		Boundary() { segmentIndices_[0] = -1; segmentIndices_[1] = -1; clearCoefficients(); }
		Boundary(const int firstSegmentIndex, const int secondSegmentIndex) {
			if (firstSegmentIndex < secondSegmentIndex) {
				segmentIndices_[0] = firstSegmentIndex; segmentIndices_[1] = secondSegmentIndex;
			}
			else {
				segmentIndices_[0] = secondSegmentIndex; segmentIndices_[1] = firstSegmentIndex;
			}
			clearCoefficients();
		}

		void clearCoefficients() {
			for (int i = 0; i < 6; ++i) polynomialCoefficients_[i] = 0;
		}

		void setType(const int typeIndex) { type_ = typeIndex; }
		void appendBoundaryPixel(const double x, const double y) {
			boundaryPixelXs_.push_back(x);
			boundaryPixelYs_.push_back(y);
			polynomialCoefficients_[0] += x*x;
			polynomialCoefficients_[1] += y*y;
			polynomialCoefficients_[2] += x*y;
			polynomialCoefficients_[3] += x;
			polynomialCoefficients_[4] += y;
			polynomialCoefficients_[5] += 1;
		}

		int type() const { return type_; }
		int segmentIndex(const int index) const { return segmentIndices_[index]; }
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
		int boundaryPixelTotal() const { return static_cast<int>(boundaryPixelXs_.size()); }
		double boundaryPixelX(const int index) const { return boundaryPixelXs_[index]; }
		double boundaryPixelY(const int index) const { return boundaryPixelYs_[index]; }

		double polynomialCoefficient(const int index) const { return polynomialCoefficients_[index]; }

	private:
		int type_;
		int segmentIndices_[2];
		std::vector<double> boundaryPixelXs_;
		std::vector<double> boundaryPixelYs_;

		double polynomialCoefficients_[6];
	};


	void allocateBuffer();
	void disparityImageAssignment(const IplImage* disparityImage);
	void labelImageAssignment(UINT* labelImage);
	void estimateDisparityPlaneRANSAC(const float* disparityImage);
	void solvePlaneEquations(const double x1, const double y1, const double z1, const double d1,
		const double x2, const double y2, const double z2, const double d2,
		const double x3, const double y3, const double z3, const double d3,
		std::vector<double>& planeParameter) const;
	void interpolateDisparityImage(float* interpolatedDisparityImage, float* disparityData) const;
	void initializeOutlierFlagImage();
	void planeSmoothing();
	void estimateBoundaryLabel();
	void estimateSmoothFitting();
	void buildSegmentConfiguration();
	int appendBoundary(const int firstSegmentIndex, const int secondSegmentIndex);
	bool isHorizontalBoundary(const int x, const int y) const;
	bool isVerticalBoundary(const int x, const int y) const;
	int computeRequiredSamplingTotal(const int drawTotal, const int inlierTotal, const int pointTotal, const int currentSamplingTotal, const double confidenceLevel) const;
	
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
	
	
	int segmentTotal_;
	int width_, height_;
	int stepSize_;
	int* labelImage_;
	unsigned char* outlierFlagImage_;
	std::vector<Boundary> boundaries_;
	float* disparityImage_;
	std::vector<Segment> segments_;
	std::vector< std::vector<int> > boundaryIndexMatrix_;

	

	double param_cu, param_cv, param_f, param_base;
};