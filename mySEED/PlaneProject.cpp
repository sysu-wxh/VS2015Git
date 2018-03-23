#include "PlaneProject.h"

void PlaneProject::initialization(const std::vector<double> centerX, const std::vector<double> centerY, 
	const std::vector<double> p1, const std::vector<double> p2, const std::vector<double> p3, 
	const std::vector<VERTEX> vertex) {
	plane_.resize(planeTotal_);
	for (int i = 0; i < planeTotal_; ++i) {
		plane_[i].setPlaneCenter(centerX[i], centerY[i], -1);
		plane_[i].setPlaneCoefficient(p1[i], p2[i], p3[i]);
		for (int t = 0; t < vertex[i].num; ++t)
			plane_[i].setVertex(vertex[i].u[t], vertex[i].v[t]);
	}
}

void PlaneProject::setInputColor(const IplImage * leftImage) {
	for (int i = 0; i < planeTotal_; ++i) {
		// Gray
		//printf("1\n");
		int temp_u = static_cast<int>(plane_[i].getPlaneCenter(0));
		int temp_v = static_cast<int>(plane_[i].getPlaneCenter(1));
		//printf("%lf %lf\n", temp_u, temp_v);
		CvScalar color;
		if (temp_u >= 0 && temp_v >= 0)
			color = cvGet2D(leftImage, temp_v, temp_u);
		else
			color = cvScalar(0, 0, 0);
		//printf("2\n");
		if (leftImage->nChannels == 1) {
			plane_[i].setPlaneColorGray(color.val[0]);
		}
		else {
			plane_[i].setPlaneColorRGB(color.val[2], color.val[1], color.val[0]);
		}
	}
}

void PlaneProject::projectResult(const IplImage * disparityImage) {
	for (int i = 0; i < planeTotal_; ++i) {
		// project center point
		double tempX = plane_[i].getPlaneCenter(0);
		double tempY = plane_[i].getPlaneCenter(1);
		double tempDisparity;
		// for robustness
		if (_finite(tempX) != 0 && _finite(tempY) != 0)
			tempDisparity = cvGet2D(disparityImage, tempY, tempX).val[0];
		else
			tempDisparity = 0.0;
		// project plane center
		double x1p = (tempX - camera_cu) * camera_base / tempDisparity;
		double y1p = (tempY - camera_cv) * camera_base / tempDisparity;
		double z1p = camera_f * camera_base / tempDisparity;
		double x1t = pose_.val[0][0] * x1p + pose_.val[0][1] * y1p + pose_.val[0][2] * z1p + pose_.val[0][3];
		double y1t = pose_.val[1][0] * x1p + pose_.val[1][1] * y1p + pose_.val[1][2] * z1p + pose_.val[1][3];
		double z1t = pose_.val[2][0] * x1p + pose_.val[2][1] * y1p + pose_.val[2][2] * z1p + pose_.val[2][3];
		plane_[i].setPlaneCenter(x1t, y1t, z1t);
		// project plane coefficient
		double temp = plane_[i].getPlaneCoefficient(0) * camera_cu + plane_[i].getPlaneCoefficient(1) * camera_cv + plane_[i].getPlaneCoefficient(2);
		double tempP1 = -camera_f * plane_[i].getPlaneCoefficient(0) / temp;
		double tempP2 = -camera_f * plane_[i].getPlaneCoefficient(1) / temp;
		double tempP3 = camera_base * camera_f / temp;
		double coeff3 = pose_.val[2][0] * tempP1 + pose_.val[2][1] * tempP2 + pose_.val[2][2] * (-1.0);
		double P1 = -(pose_.val[0][0] * tempP1 + pose_.val[0][1] * tempP2 + pose_.val[0][2] * (-1.0)) / coeff3;
		double P2 = -(pose_.val[1][0] * tempP1 + pose_.val[1][1] * tempP2 + pose_.val[1][2] * (-1.0)) / coeff3;
		double P3 = z1t - P1 * x1t - P2 * y1t;
		plane_[i].setPlaneCoefficient(P1, P2, P3);
	}
}

void PlaneProject::getPlaneProjectionResult(std::vector<std::vector<double>>& result, int flag) {
	// 1 rgb; 0 none; -1 gray;
	result.resize(planeTotal_);
	for (int i = 0; i < planeTotal_; ++i) {
		//temp.clear();
		int vertexNum = plane_[i].getPlaneVertexNum();   // for both x and y
		if (flag == 0)
			result[i].resize(1 + vertexNum*2 + 3);   // 1 for the total vertex num 3 for the coefficients
		else if (flag == 1)
			result[i].resize(1 + vertexNum*2 + 3 + 3);
		else
			result[i].resize(1 + vertexNum*2 + 3 + 1);
		// plane center
		result[i][0] = plane_[i].getPlaneCenter(0);
		result[i][1] = plane_[i].getPlaneCenter(1);
		result[i][2] = plane_[i].getPlaneCenter(2);

		// plane coefficient
		result[i][3] = plane_[i].getPlaneCoefficient(0);
		result[i][4] = plane_[i].getPlaneCoefficient(1);
		result[i][5] = plane_[i].getPlaneCoefficient(2);

		// plane vertex
		for (int t = 0; t < vertexNum; t=t+2) {
			result[i][t + 6] = plane_[i].getVertexX(t);
			result[i][t + 7] = plane_[i].getVertexX(t+1);
		}

		// plane color (it is not that important)
		if (flag == 0)
			continue;
		else if (flag == 1)
			for (int t = 4 + vertexNum * 2; t < 7 + vertexNum * 2; ++t)
				result[i][t] = plane_[i].getPlaneColorRGB(t - (4 + vertexNum * 2));
		else
			result[i][4 + vertexNum * 2] = plane_[i].getPlaneColorGray();
	}
}
