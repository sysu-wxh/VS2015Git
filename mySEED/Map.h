#pragma once
#include <vector>
#include <cmath>
#include <opencv.hpp>
#include <fstream>
#include "Vertex.h"
// 一个map中包含非常多的cluster
class Map {
public:
	Map() { clusterCnt_ = 0; cluster_.clear(); };

	void setPrecision(double precision) { precision_ = precision; }
	double getPrecision() { return precision_; }

	int getClusterCnt() { return clusterCnt_; }

	int findCoplane(int c1, int c2, int c3);
	void storeCluster(double center1, double center2, double center3,    // center
		double param1, double param2, double param3,     // paramter
		int color1, int color2, int color3,  // color
		VERTEX3D vertex   // vertex
	);
	int getSegmentCnt_Map(int index) {
		return cluster_[index].getSegmentCnt();
	}
	void outputFile(string filePath, int mode = 1);
	// mode 1: 共面颜色相同(平均颜色)   mode2: 共面颜色相同(第一个segment颜色)  mode3: 共面不显示
	double RGBCompare(int aR, int aG, int aB, int bR, int bG, int bB);
	double normVectorCompare(double nv1, double nv2, double nv3, double nvp1, double nvp2, double nvp3);

private:
	class Cluster {
	public:
		Cluster(double center1, double center2, double center3,    // center
			double param1, double param2, double param3,     // paramter
			int color1, int color2, int color3,  // color
			double norm1, double norm2, double norm3,     // normal
			double normParam1, double normParam2, double normParam3,   //  normalVector
			VERTEX3D vertex3D   // vertex
			) {
			cnt_ = 1, center1_.push_back(center1), center2_.push_back(center2), center3_.push_back(center3),
				param1_.push_back(param1), param2_.push_back(param2), param3_.push_back(param3),
				color1_.push_back(color1), color2_.push_back(color2), color3_.push_back(color3),
				norm1_ = norm1, norm2_ = norm2, norm3_ = norm3,
				normParam1_.push_back(normParam1), normParam2_.push_back(normParam2), normParam3_.push_back(normParam3),
				vertex3D_.push_back(vertex3D);
		};

		int getSegmentCnt() { return cnt_; }

		void addSegment(double center1, double center2, double center3,    // center
			double param1, double param2, double param3,     // paramter
			int color1, int color2, int color3,  // color
			double normParam1, double normParam2, double normParam3,   //  normalVector
			VERTEX3D vertex3D   // vertex
		) {
			cnt_++;
			center1_.push_back(center1), center2_.push_back(center2), center3_.push_back(center3),
				param1_.push_back(param1), param2_.push_back(param2), param3_.push_back(param3),
				color1_.push_back(color1), color2_.push_back(color2), color3_.push_back(color3),
				normParam1_.push_back(normParam1), normParam2_.push_back(normParam2), normParam3_.push_back(normParam3),
				vertex3D_.push_back(vertex3D);
		}
		bool checkCoplane(int c1, int c2, int c3);

		void getSegmentCenter(int index, double& x, double& y, double& z) {
			x = center1_[index], y = center2_[index], z = center3_[index];
		}
		void getSegmentParam(int index, double& param1, double& param2, double& param3) {
			param1 = param1_[index], param2 = param2_[index], param3 = param3_[index];
		}
		void getSegmentNormParam(int index, double& normParam1, double& normParam2, double& normParam3) {
			normParam1 = normParam1_[index], normParam2 = normParam2_[index], normParam3 = normParam3_[index];
		}
		void getAverageColor(int& r, int& g, int& b) {
			int rTotal = 0, gTotal = 0, bTotal = 0;
			for (int i = 0; i < cnt_; ++i) {
				rTotal += color1_[i];
				gTotal += color2_[i];
				bTotal += color3_[i];
			}
			r = rTotal / cnt_;
			g = gTotal / cnt_;
			b = bTotal / cnt_;
		}
		void getFirstSegmentColor(double& r, double& g, double& b) {
			r = color1_[0], g = color2_[0], b = color3_[0];
		}
		void getSegmentColor(int index, int& r, int& g, int& b) {
			r = color1_[index], g = color2_[index], b = color3_[index];
		}
		int getVertex3DSize() { return vertex3D_.size(); }
		int getVertexNum(int index) { return vertex3D_[index].x.size(); }
		float getVertexX(int verIndex, int coorIndex) { return vertex3D_[verIndex].x[coorIndex]; }
		float getVertexY(int verIndex, int coorIndex) { return vertex3D_[verIndex].y[coorIndex]; }
		float getVertexZ(int verIndex, int coorIndex) { return vertex3D_[verIndex].z[coorIndex]; }
	private:
		int cnt_;
		std::vector<double> center1_, center2_, center3_;
		std::vector<double> param1_, param2_, param3_;
		std::vector<int> color1_, color2_, color3_;
		std::vector<double> normParam1_, normParam2_, normParam3_;
		std::vector<VERTEX3D> vertex3D_;
		int norm1_, norm2_, norm3_;
	};
	double precision_;
	int clusterCnt_;
	std::vector<Cluster> cluster_;
};