#pragma once
#include <vector>
#include <opencv.hpp>
#include "Matrix.h"
#include "Vertex.h"

class PlaneProject {
public:
	struct CameraParameters {
		double f;  // focal length (in pixels)
		double cu; // principal point (u-coordinate)
		double cv; // principal point (v-coordinate)
		double base;  // baseline
		CameraParameters() {
			f = 1.0;
			cu = 0.0;
			cv = 0.0;
			base = 0.0;
		}
	};
	PlaneProject(int planeTotal, CameraParameters param, Matrix pose) {
		planeTotal_ = planeTotal;
		camera_cu = param.cu;
		camera_cv = param.cv;
		camera_f = param.f;
		camera_base = param.base;
		pose_ = pose;
	}
	void initialization(const std::vector<double> centerX, const std::vector<double> centerY,
		const std::vector<double> p1, const std::vector<double> p2, const std::vector<double> p3,
		const std::vector<VERTEX> vertex);
	void setInputColor(const IplImage* leftImage);
	void projectResult(const IplImage* disparityImage);
	void getPlaneProjectionResult(std::vector< std::vector<double> >& result, int flag = 1);  // 1 rgb; 0 none; -1 gray;
private:
	class Plane {
	public:
		Plane() { planeLabel_ = -1; clearPlaneParameters(); };
		void clearPlaneParameters() {
			for (int i = 0; i < 3; ++i) {
				planeCenter_[i] = 0.0;
				planeCoefficient_[i] = 0.0;
				planeColorRGB_[i] = 0.0;
			}
			planeColorGray_ = 0.0;
			vertexX_.clear();
			vertexY_.clear();
		}
		double getPlaneCenter(const int planeIndex) const { return planeCenter_[planeIndex]; }
		void setPlaneCenter(const double x, const double y, const double z) {
			planeCenter_[0] = x, planeCenter_[1] = y, planeCenter_[2] = z;
		}

		double getPlaneCoefficient(const int CoefficientIndex) const { return planeCoefficient_[CoefficientIndex]; }
		void setPlaneCoefficient(const double parameter1, const double parameter2, const double parameter3) {
			planeCoefficient_[0] = parameter1;
			planeCoefficient_[1] = parameter2;
			planeCoefficient_[2] = parameter3;
		}

		double getVertexX(const int vertexIndex) const { return vertexX_[vertexIndex]; }
		double getVertexY(const int vertexIndex) const { return vertexY_[vertexIndex]; }
		void setVertex(const double x, const double y) {
			vertexX_.push_back(x);
			vertexY_.push_back(y);
		}

		double getPlaneColorRGB(const int colorIndex) const { return planeColorRGB_[colorIndex]; }
		void setPlaneColorRGB(double r, double g, double b) {
			planeColorRGB_[0] = r;
			planeColorRGB_[1] = g;
			planeColorRGB_[2] = b;
		}
		double getPlaneColorGray() const { return planeColorGray_; }
		void setPlaneColorGray(double gray) {
			planeColorGray_ = gray;
		}
		int getPlaneVertexNum() { return vertexX_.size(); }
	private:
		double planeCenter_[3];
		double planeCenterDisparity;
		double planeCoefficient_[3];   //disparityPlane[0] disparityPlane[1] disparityPlane[2]
		double planeColorRGB_[3];    // r g b
		double planeColorGray_;     // gray
		std::vector<double> vertexX_;     // store the x coordinate of vertex
		std::vector<double> vertexY_;     // store the y coordinate of vertex
		int planeLabel_;
	};
	std::vector<Plane> plane_;
	int planeTotal_;
	int width_, height_;
	double camera_cu, camera_cv, camera_f, camera_base;
	Matrix pose_ = Matrix::eye(4);
};