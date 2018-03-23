#include "EstimatePlane.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>

EstimatePlane::EstimatePlane() {

}

EstimatePlane::~EstimatePlane() {
	delete[] labelImage_;
	delete[] disparityImage_;

}

void EstimatePlane::setOutputDisparityFactor(const double outputDisparityFactor) {
	if (outputDisparityFactor < 1) {
		throw std::invalid_argument("[SPSStereo::setOutputDisparityFactor] disparity factor is less than 1");
	}

	outputDisparityFactor_ = outputDisparityFactor;
}

void EstimatePlane::setIterationTotal(const int outerIterationTotal, const int innerIterationTotal) {
	if (outerIterationTotal < 1 || innerIterationTotal < 1) {
		throw std::invalid_argument("[SPSStereo::setIterationTotal] the number of iterations is less than 1");
	}

	outerIterationTotal_ = outerIterationTotal;
	innerIterationTotal_ = innerIterationTotal;
}

void EstimatePlane::setWeightParameter(const double positionWeight, const double disparityWeight, const double boundaryLengthWeight, const double smoothnessWeight) {
	if (positionWeight < 0 || disparityWeight < 0 || boundaryLengthWeight < 0 || smoothnessWeight < 0) {
		throw std::invalid_argument("[SPSStereo::setWeightParameter] weight value is nagative");
	}

	positionWeight_ = positionWeight;
	disparityWeight_ = disparityWeight;
	boundaryLengthWeight_ = boundaryLengthWeight;
	smoothRelativeWeight_ = smoothnessWeight / disparityWeight;
}

void EstimatePlane::setInlierThreshold(const double inlierThreshold) {
	if (inlierThreshold <= 0) {
		throw std::invalid_argument("[SPSStereo::setInlierThreshold] threshold of inlier is less than zero");
	}

	inlierThreshold_ = inlierThreshold;
}

void EstimatePlane::setPenaltyParameter(const double hingePenalty, const double occlusionPenalty, const double impossiblePenalty) {
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

void EstimatePlane::planeCompute(IplImage * disparityImage, int segmentTotal, UINT* labelImage,
	IplImage* segmentImage, IplImage* optimizedDisparityImage) {
	segmentTotal_ = segmentTotal;   //分割的seg总数
	width_ = disparityImage->width;
	height_ = disparityImage->height;
	stepSize_ = static_cast<int>(sqrt(static_cast<double>(width_*height_) / segmentTotal_) + 2.0);
	segments_.resize(segmentTotal_);

	allocateBuffer();  //分配空间
	labelImageAssignment(labelImage);    //得到每个像素（x,y）所对应的超像素分割后的序号
	/*cvNamedWindow("1");
	cvShowImage("1", disparityImage);
	cvWaitKey(0);*/
	std::cout << "Input disparity image depth " << disparityImage->depth << std::endl;
	float* disparityData;
	disparityData = reinterpret_cast<float*>(malloc(width_*height_ * sizeof(float)));
	if (disparityImage->depth == 16)  //如果位深为16位，则需要进行转换
		cvConvertScale(disparityImage, disparityImage, 255.0 / 65535.0);
	// 将IPLImage数据格式转换为float格式 transform IplImage data to float data
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			//if (cvGet2D(disparityImage, y, x).val[0] != 0)
			//std::cout << x << " " << y << " " << (int)(cvGet2D(disparityImage, y, x).val[0]) << std::endl;
			disparityData[width_*y + x] = static_cast<float>(cvGet2D(disparityImage, y, x).val[0]);  //获取对应于每个xy的颜色（视差图中的）
		}
	}
	estimateDisparityPlaneRANSAC(disparityData);   //经过以下四个函数后得到每一个平面的参数
	float* interpolatedDisparityImage = reinterpret_cast<float*>(malloc(width_*height_ * sizeof(float)));
	interpolateDisparityImage(interpolatedDisparityImage, disparityData);
	estimateDisparityPlaneRANSAC(interpolatedDisparityImage);//************
	initializeOutlierFlagImage();
	free(interpolatedDisparityImage);
	//makeOutputImage(segmentImage, optimizedDisparityImage);
	//performSmoothingSegmentation();

}


void EstimatePlane::allocateBuffer() {
	labelImage_ = reinterpret_cast<int*>(malloc(width_*height_ * sizeof(int)));
	disparityImage_ = reinterpret_cast<float*>(malloc(width_*height_ * sizeof(float)));
	outlierFlagImage_ = reinterpret_cast<unsigned char*>(malloc(width_*height_ * sizeof(unsigned char)));
}

void EstimatePlane::freeBuffer() {
	free(this->labelImage_);
	free(this->disparityImage_);
	free(this->outlierFlagImage_);
}


void EstimatePlane::vertexProjectAndStore(const vector<VERTEX>& result) {
	int size = result.size();
	if (size != segmentTotal_)
		std::cout << "快报警" << std::endl;
	for (int i = 0; i < size; ++i) {
		segments_[i].clearVertex();
		for (int t = 0; t < result[i].u.size(); ++t) {
			double disparity = segments_[i].estimatedDisparity(result[i].u[t], result[i].v[t]);
			//cout << "disparity = " << disparity << endl;
			//cout << result[i].u[t] << " " << result[i].v[t] << endl;
			double xTemp = ((double)result[i].u[t] - param_cu) * param_base / disparity;
			double yTemp = ((double)result[i].v[t] - param_cv) * param_base / disparity;
			double zTemp = param_f*param_base / disparity;
			//cout << yTemp << endl;
			segments_[i].setVertex(result[i].u[t], result[i].v[t], xTemp, yTemp, zTemp);
		}
	}
}

void EstimatePlane::setInputColor(const IplImage * leftImage) {
	for (int i = 0; i < segmentTotal_; ++i) {
		int temp_u = static_cast<int>(segments_[i].position(0));
		int temp_v = static_cast<int>(segments_[i].position(1));
		CvScalar color;
		if (temp_u >= 0 && temp_v >= 0)
			color = cvGet2D(leftImage, temp_v, temp_u);
		else
			color = cvScalar(0, 0, 0);
		if (leftImage->nChannels == 1) {
			segments_[i].setPlaneColorGray(color.val[0]);
		}
		else {
			segments_[i].setPlaneColorRGB(color.val[2], color.val[1], color.val[0]);
		}
	}
}

void EstimatePlane::projectPlane() {
	for (int i = 0; i < segmentTotal_; ++i) {
		double tempX = segments_[i].position(0);
		double tempY = segments_[i].position(1);
		double tempDisparity;
		if (_finite(tempX) != 0 && _finite(tempY) != 0)
			tempDisparity = segments_[i].estimatedDisparity(tempX, tempY);
		else
			tempDisparity = -1;
		double x1p = (tempX - param_cu) * param_base / tempDisparity;
		double y1p = (tempY - param_cv) * param_base / tempDisparity;
		double z1p = param_f * param_base / tempDisparity;
		segments_[i].setProjectCenter(x1p, y1p, z1p);

		double temp = segments_[i].planeParameter(0) * param_cu
			+ segments_[i].planeParameter(1) * param_cv
			+ segments_[i].planeParameter(2);
		double tempP1 = -param_f * segments_[i].planeParameter(0) / temp;
		double tempP2 = -param_f * segments_[i].planeParameter(1) / temp;
		double tempP3 = param_base * param_f / temp;
		//std::cout << segments_[i].planeParameter(2) << " " << tempP2 << " " << tempP3 << endl;
		segments_[i].setProjectCoefficients(tempP1, tempP2, tempP3);
	}
}

std::vector<VERTEX3D> EstimatePlane::getProjectResult(std::vector<std::vector<double>>& planeOutput, Matrix pose) {
	//planeOutput.resize(segmentTotal_);
	cout << segmentTotal_ << endl;
	std::vector<VERTEX3D> vertexProjectResult;
	vertexProjectResult.reserve(segmentTotal_);
	vector<double> temp;
	//cout << pose << endl;
	for (int i = 0; i < segmentTotal_; ++i) {
		temp.clear();
		/*if (sqrt(segments_[i].getProjectCenter(0) * segments_[i].getProjectCenter(0)
			+ segments_[i].getProjectCenter(1) * segments_[i].getProjectCenter(1)
			+ segments_[i].getProjectCenter(2) * segments_[i].getProjectCenter(2)) > 30)
			continue;
		if (segments_[i].getProjectCenter(2) > 25.0)
			continue;*/
		bool judge = false;
		double c1p = segments_[i].getProjectCenter(0),
			c2p = segments_[i].getProjectCenter(1),
			c3p = segments_[i].getProjectCenter(2);
		for (int t = 0; t < segments_[i].getVertexSize(); ++t) {
			double x1p = segments_[i].getProjectX(t),
				y1p = segments_[i].getProjectY(t),
				z1p = segments_[i].getProjectZ(t);
			//cout << z1p << endl;
			double u1t = segments_[i].getVertexU(t);
			double v1t = segments_[i].getVertexV(t);
			
			if (z1p > 16.0 || z1p < 0.0) {
				//cout << segments_[i].getVertexU(t) << " " << segments_[i].getVertexV(t) << endl;
				judge = true;
				break;
			}
			for (int j = 0; j < segments_[i].getVertexSize(); ++j) {
				double x1j = segments_[i].getProjectX(j),
					y1j = segments_[i].getProjectY(j),
					z1j = segments_[i].getProjectZ(j);
			}
		}
		if (judge){
			continue;
		}
		if (_finite(segments_[i].getProjectCoefficients(0)) == 0
			|| _finite(segments_[i].getProjectCoefficients(1)) == 0
			|| _finite(segments_[i].getProjectCoefficients(2)) == 0) {
			continue;
		}
		// center
		double c1 = segments_[i].getProjectCenter(0);
		double c2 = segments_[i].getProjectCenter(1);
		double c3 = segments_[i].getProjectCenter(2);

		double c1t = pose.val[0][0] * c1 + pose.val[0][1] * c2 + pose.val[0][2] * c3 + pose.val[0][3];
		double c2t = pose.val[1][0] * c1 + pose.val[1][1] * c2 + pose.val[1][2] * c3 + pose.val[1][3];
		double c3t = pose.val[2][0] * c1 + pose.val[2][1] * c2 + pose.val[2][2] * c3 + pose.val[2][3];

		temp.push_back(c1t);
		temp.push_back(c2t);
		temp.push_back(c3t);
		// coefficients
		double tempP1 = segments_[i].getProjectCoefficients(0);
		double tempP2 = segments_[i].getProjectCoefficients(1);
		double tempP3 = segments_[i].getProjectCoefficients(2);
		double coeff3 = pose.val[2][0] * tempP1 + pose.val[2][1] * tempP2 + pose.val[2][2] * (-1.0);
		double P1 = -(pose.val[0][0] * tempP1 + pose.val[0][1] * tempP2 + pose.val[0][2] * (-1.0)) / coeff3;
		double P2 = -(pose.val[1][0] * tempP1 + pose.val[1][1] * tempP2 + pose.val[1][2] * (-1.0)) / coeff3;
		double P3 = c3t - P1 * c1t - P2 * c2t;
		temp.push_back(P1);
		temp.push_back(P2);
		temp.push_back(P3);
		// color
		temp.push_back(segments_[i].getPlaneColorRGB(0));
		temp.push_back(segments_[i].getPlaneColorRGB(1));
		temp.push_back(segments_[i].getPlaneColorRGB(2));
		// vertex number
		int size = segments_[i].getVertexSize();
		//temp.push_back(size);
		VERTEX3D temp3D;
		temp3D.num = size;
		for (int t = 0; t < size; ++t) {
			double x1p = segments_[i].getProjectX(t),
				y1p = segments_[i].getProjectY(t),
				z1p = segments_[i].getProjectZ(t);
			double x1t = pose.val[0][0] * x1p + pose.val[0][1] * y1p + pose.val[0][2] * z1p + pose.val[0][3];
			double y1t = pose.val[1][0] * x1p + pose.val[1][1] * y1p + pose.val[1][2] * z1p + pose.val[1][3];
			double z1t = pose.val[2][0] * x1p + pose.val[2][1] * y1p + pose.val[2][2] * z1p + pose.val[2][3];
			temp3D.x.push_back(x1t);
			temp3D.y.push_back(y1t);
			temp3D.z.push_back(z1t);
		}
		planeOutput.push_back(temp);
		vertexProjectResult.push_back(temp3D);
	}
	return vertexProjectResult;
}

void EstimatePlane::disparityImageAssignment(const IplImage* disparityImage) {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			disparityImage_[y*width_ + x] = cvGet2D(disparityImage, y, x).val[0];
		}
	}
}

void EstimatePlane::labelImageAssignment(UINT* labelImage) {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			labelImage_[y*width_ + x] = labelImage[y*width_ + x];  //针对于x,y，其所在的seg
		}
	}
}

void EstimatePlane::estimateDisparityPlaneRANSAC(const float* disparityData) {
	const double confidenceLevel = 0.99;
	const double initialInlierThreshold = 1.0;

	std::vector< std::vector<int> > segmentPixelXs(segmentTotal_);
	std::vector< std::vector<int> > segmentPixelYs(segmentTotal_);
	std::vector< std::vector<float> > segmentPixelDisparities(segmentTotal_);

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (disparityData[width_*y + x] == 0) continue;
			//std::cout << disparityData[width_*y + x] << std::endl;

			int pixelSegmentIndex = labelImage_[width_*y + x];
			segmentPixelXs[pixelSegmentIndex].push_back(x);
			segmentPixelYs[pixelSegmentIndex].push_back(y);
			segmentPixelDisparities[pixelSegmentIndex].push_back(disparityData[width_*y + x]);
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
			drawIndices[0] = rand() % segmentPixelTotal;
			drawIndices[1] = rand() % segmentPixelTotal;
			while (drawIndices[1] == drawIndices[0]) drawIndices[1] = rand() % segmentPixelTotal;
			drawIndices[2] = rand() % segmentPixelTotal;
			while (drawIndices[2] == drawIndices[0] || drawIndices[2] == drawIndices[1]) drawIndices[2] = rand() % segmentPixelTotal;

			std::vector<double> planeParameter;
			solvePlaneEquations(segmentPixelXs[segmentIndex][drawIndices[0]], segmentPixelYs[segmentIndex][drawIndices[0]], 1, segmentPixelDisparities[segmentIndex][drawIndices[0]],
				segmentPixelXs[segmentIndex][drawIndices[1]], segmentPixelYs[segmentIndex][drawIndices[1]], 1, segmentPixelDisparities[segmentIndex][drawIndices[1]],
				segmentPixelXs[segmentIndex][drawIndices[2]], segmentPixelYs[segmentIndex][drawIndices[2]], 1, segmentPixelDisparities[segmentIndex][drawIndices[2]],
				planeParameter);

			// Count the number of inliers
			int inlierTotal = 0;
			std::vector<bool> inlierFlags(segmentPixelTotal);
			for (int pixelIndex = 0; pixelIndex < segmentPixelTotal; ++pixelIndex) {
				double estimateDisparity = planeParameter[0] * segmentPixelXs[segmentIndex][pixelIndex]
					+ planeParameter[1] * segmentPixelYs[segmentIndex][pixelIndex]
					+ planeParameter[2];
				if (fabs(estimateDisparity - segmentPixelDisparities[segmentIndex][pixelIndex]) <= initialInlierThreshold) {
					++inlierTotal;
					inlierFlags[pixelIndex] = true;
				}
				else {
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

void EstimatePlane::solvePlaneEquations(const double x1, const double y1, const double z1, const double d1,
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

	planeParameter[0] = ((z2*d1 - z1*d2)*(y2*z3 - y3*z2) - (z3*d2 - z2*d3)*(y1*z2 - y2*z1)) / denominatorA;

	double denominatorB = y1*z2 - y2*z1;
	if (denominatorB > epsilonValue) {
		planeParameter[1] = (z2*d1 - z1*d2 - planeParameter[0] * (x1*z2 - x2*z1)) / denominatorB;
	}
	else {
		denominatorB = y2*z3 - y3*z2;
		planeParameter[1] = (z3*d2 - z2*d3 - planeParameter[0] * (x2*z3 - x3*z2)) / denominatorB;
	}
	if (z1 > epsilonValue) {
		planeParameter[2] = (d1 - planeParameter[0] * x1 - planeParameter[1] * y1) / z1;
	}
	else if (z2 > epsilonValue) {
		planeParameter[2] = (d2 - planeParameter[0] * x2 - planeParameter[1] * y2) / z2;
	}
	else {
		planeParameter[2] = (d3 - planeParameter[0] * x3 - planeParameter[1] * y3) / z3;
	}
}

void EstimatePlane::interpolateDisparityImage(float * interpolatedDisparityImage, float* disparityData) const {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			interpolatedDisparityImage[width_*y + x] = disparityData[width_*y + x];
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
			}
			else {
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
				for (int interpolateY = y + 1; interpolateY < height_; ++interpolateY) {
					interpolatedDisparityImage[width_*interpolateY + x] = interpolatedDisparityImage[width_*y + x];
				}
				break;
			}
		}
	}
}

void EstimatePlane::initializeOutlierFlagImage() {
	memset(outlierFlagImage_, 0, width_*height_);
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			if (disparityImage_[width_*y + x] == 0) {
				outlierFlagImage_[width_*y + x] = 255;
				continue;
			}
			int pixelSegmentIndex = labelImage_[width_*y + x];
			double estimatedDisparity = segments_[pixelSegmentIndex].estimatedDisparity(x, y);
			if (fabs(disparityImage_[width_*y + x] - estimatedDisparity) > inlierThreshold_) {
				outlierFlagImage_[width_*y + x] = 255;
			}
		}
	}
}

void EstimatePlane::performSmoothingSegmentation() {
	// 这里不循环了
	buildSegmentConfiguration();
	planeSmoothing();
}

void EstimatePlane::planeSmoothing() {
	// 这里也不循环了
	estimateBoundaryLabel();
	estimateSmoothFitting();
}

void EstimatePlane::estimateBoundaryLabel() {
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
		hingeSquaredError = currentBoundary.polynomialCoefficient(0)*(ai*ai + aj*aj - 2 * ai*aj)
			+ currentBoundary.polynomialCoefficient(1)*(bi*bi + bj*bj - 2 * bi*bj)
			+ currentBoundary.polynomialCoefficient(2)*(2 * ai*bi + 2 * aj*bj - 2 * ai*bj - 2 * aj*bi)
			+ currentBoundary.polynomialCoefficient(3)*(2 * ai*ci + 2 * aj*cj - 2 * ai*cj - 2 * aj*ci)
			+ currentBoundary.polynomialCoefficient(4)*(2 * bi*ci + 2 * bj*cj - 2 * bi*cj - 2 * bj*ci)
			+ currentBoundary.polynomialCoefficient(5)*(ci*ci + cj*cj - 2 * ci*cj);
		hingeSquaredError /= currentBoundary.boundaryPixelTotal();
		boundaryEnergies[2] = hingePenalty_ + hingeSquaredError;

		// Occlusion
		if (hingeError > 0) {
			boundaryEnergies[0] = occlusionPenalty_;
			boundaryEnergies[1] = occlusionPenalty_ + impossiblePenalty_;
		}
		else {
			boundaryEnergies[0] = occlusionPenalty_ + impossiblePenalty_;
			boundaryEnergies[1] = occlusionPenalty_;
		}

		// Coplanar
		double coplanarSquaredError = 0;
		coplanarSquaredError = firstSegment.polynomialCoefficientAll(0)*(ai*ai + aj*aj - 2 * ai*aj)
			+ firstSegment.polynomialCoefficientAll(1)*(bi*bi + bj*bj - 2 * bi*bj)
			+ firstSegment.polynomialCoefficientAll(2)*(2 * ai*bi + 2 * aj*bj - 2 * ai*bj - 2 * aj*bi)
			+ firstSegment.polynomialCoefficientAll(3)*(2 * ai*ci + 2 * aj*cj - 2 * ai*cj - 2 * aj*ci)
			+ firstSegment.polynomialCoefficientAll(4)*(2 * bi*ci + 2 * bj*cj - 2 * bi*cj - 2 * bj*ci)
			+ firstSegment.polynomialCoefficientAll(5)*(ci*ci + cj*cj - 2 * ci*cj);
		coplanarSquaredError += secondSegment.polynomialCoefficientAll(0)*(ai*ai + aj*aj - 2 * ai*aj)
			+ secondSegment.polynomialCoefficientAll(1)*(bi*bi + bj*bj - 2 * bi*bj)
			+ secondSegment.polynomialCoefficientAll(2)*(2 * ai*bi + 2 * aj*bj - 2 * ai*bj - 2 * aj*bi)
			+ secondSegment.polynomialCoefficientAll(3)*(2 * ai*ci + 2 * aj*cj - 2 * ai*cj - 2 * aj*ci)
			+ secondSegment.polynomialCoefficientAll(4)*(2 * bi*ci + 2 * bj*cj - 2 * bi*cj - 2 * bj*ci)
			+ secondSegment.polynomialCoefficientAll(5)*(ci*ci + cj*cj - 2 * ci*cj);

		coplanarSquaredError /= (firstSegment.pixelTotal() + secondSegment.pixelTotal());
		boundaryEnergies[3] = coplanarSquaredError;
		int minBoundaryLabel = 0;
		if (boundaryEnergies[1] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 1;
		if (boundaryEnergies[2] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 2;
		if (boundaryEnergies[3] < boundaryEnergies[minBoundaryLabel]) minBoundaryLabel = 3;
		/*std::cout << "0 = " << boundaryEnergies[0]
			<< " 1 = " << boundaryEnergies[1]
			<< " 2 = " << boundaryEnergies[2]
			<< " 3 = " << boundaryEnergies[3] << std::endl;*/
			//std::cout << minBoundaryLabel << std::endl;
		boundaries_[boundaryIndex].setType(minBoundaryLabel);
	}
}

void EstimatePlane::estimateSmoothFitting() {
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
				double weightValue = smoothRelativeWeight_ / boundaryPixelTotal*stepSize_*stepSize_;

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

			}
			else {
				// Coplanar
				int neighborSegmentPixelTotal = neighborSegment.pixelTotal();
				double weightValue = smoothRelativeWeight_ / (segmentPixelTotal + neighborSegmentPixelTotal)*stepSize_*stepSize_;

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

void EstimatePlane::buildSegmentConfiguration() {
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
			segments_[pixelSegmentIndex].addPixel(x, y);
			if (disparityImage_[width_*y + x] > 0 && outlierFlagImage_[width_*y + x] == 0) {
				segments_[pixelSegmentIndex].appendSegmentPixelWithDisparity(x, y, disparityImage_[width_*y + x]);
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

int EstimatePlane::appendBoundary(const int firstSegmentIndex, const int secondSegmentIndex) {
	if (boundaryIndexMatrix_[firstSegmentIndex][secondSegmentIndex] >= 0) return boundaryIndexMatrix_[firstSegmentIndex][secondSegmentIndex];

	boundaries_.push_back(Boundary(firstSegmentIndex, secondSegmentIndex));
	int newBoundaryIndex = static_cast<int>(boundaries_.size()) - 1;
	boundaryIndexMatrix_[firstSegmentIndex][secondSegmentIndex] = newBoundaryIndex;
	boundaryIndexMatrix_[secondSegmentIndex][firstSegmentIndex] = newBoundaryIndex;

	segments_[firstSegmentIndex].appendBoundaryIndex(newBoundaryIndex);
	segments_[secondSegmentIndex].appendBoundaryIndex(newBoundaryIndex);

	return newBoundaryIndex;
}

bool EstimatePlane::isHorizontalBoundary(const int x, const int y) const {
	if (x >= width_ - 1) return false;

	if (labelImage_[width_*y + x] != labelImage_[width_*y + x + 1]) return true;
	return false;
}

bool EstimatePlane::isVerticalBoundary(const int x, const int y) const {
	if (y >= height_ - 1) return false;

	if (labelImage_[width_*y + x] != labelImage_[width_*(y + 1) + x]) return true;
	return false;
}

int EstimatePlane::computeRequiredSamplingTotal(const int drawTotal, const int inlierTotal, const int pointTotal, const int currentSamplingTotal, const double confidenceLevel) const {
	double ep = 1 - static_cast<double>(inlierTotal) / static_cast<double>(pointTotal);
	if (ep == 1.0) {
		ep = 0.5;
	}

	int newSamplingTotal = static_cast<int>(log(1 - confidenceLevel) / log(1 - pow(1 - ep, drawTotal)) + 0.5);
	if (newSamplingTotal < currentSamplingTotal) {
		return newSamplingTotal;
	}
	else {
		return currentSamplingTotal;
	}
}

void EstimatePlane::makeOutputImage(IplImage * segmentImage, IplImage * segmentDisparityImage) const {
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

void EstimatePlane::makeSegmentBoundaryData(std::vector<std::vector<double>>& disparityPlaneParameters, std::vector<std::vector<int>>& boundaryLabels) const {
	int segmentTotal = static_cast<int>(segments_.size());
	disparityPlaneParameters.resize(segmentTotal);
	for (int segmentIndex = 0; segmentIndex < segmentTotal; ++segmentIndex) {
		disparityPlaneParameters[segmentIndex].resize(3);
		disparityPlaneParameters[segmentIndex][0] = segments_[segmentIndex].planeParameter(0);
		disparityPlaneParameters[segmentIndex][1] = segments_[segmentIndex].planeParameter(1);
		disparityPlaneParameters[segmentIndex][2] = segments_[segmentIndex].planeParameter(2);
		/*cout << segments_[segmentIndex].planeParameter(0) << " "
			<< segments_[segmentIndex].planeParameter(1) << " "
			<< segments_[segmentIndex].planeParameter(2) << endl;*/
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

void EstimatePlane::getDisparityPlane(double ** planeFunction) {
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			int pixelSegmentIndex = labelImage_[width_*y + x];
			//std::cout << pixelSegmentIndex << std::endl;
			planeFunction[y*width_ + x][0] = segments_[pixelSegmentIndex].planeParameter(0);
			planeFunction[y*width_ + x][1] = segments_[pixelSegmentIndex].planeParameter(1);
			planeFunction[y*width_ + x][2] = segments_[pixelSegmentIndex].planeParameter(2);
			//std::cout << "p1 = " << planeFunction[y*width_ + x][0]
			//	<< " p2 = " << planeFunction[y*width_ + x][1]
			//	<< " p3 = " << planeFunction[y*width_ + x][2] << std::endl;
		}
		//std::cout << "----------------------------------------------------------------" << std::endl;
	}
}
