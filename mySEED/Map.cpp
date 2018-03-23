#include "Map.h"

void Map::storeCluster(double center1, double center2, double center3,    // center
	double param1, double param2, double param3,     // paramter
	int color1, int color2, int color3,  // color
	VERTEX3D vertex) {

	double precision = precision_;
	double normVec = sqrt(param1 * param1 + param2 * param2 + 1.0);
	double normParam1 = param1 / normVec, normParam2 = param2 / normVec, normParam3 = -1.0 / normVec;
	int nParam1 = normParam1 / precision, nParam2 = normParam2 / precision, nParam3 = normParam3 / precision;
	//std::cout << precision << std::endl;
	//std::cout << nParam1 << " " << nParam2 << " " << nParam3 << std::endl;

	int index = findCoplane(nParam1, nParam2, nParam3);
	//index = -1;
	//cout << index << endl;&& center2 > -1.5
	if (index != -1 && center2 > -1.5) {
		cluster_[index].addSegment(center1, center2, center3, param1, param2, param3, color1, color2, color3, normParam1, normParam2, normParam3, vertex);
	}
	else {
		clusterCnt_++;
		cluster_.push_back(Cluster::Cluster(center1, center2, center3, param1, param2, param3, color1, color2, color3,
			nParam1, nParam2, nParam3, normParam1, normParam2, normParam3, vertex));
	}
}

void Map::outputFile(string filePath, int mode) {
	ofstream output(filePath.c_str(), ios::out);
	std::cout << clusterCnt_ << std::endl;
	double colorErrRate = 0.0;
	double thetaErrRate = 0.0;
	int rPre, gPre, bPre;
	int rTemp, gTemp, bTemp;
	double normParam1Pre = 0.0, normParam2Pre = 0.0, normParam3Pre = 0.0;
	cluster_[0].getAverageColor(rPre, gPre, bPre);
	cluster_[0].getAverageColor(rTemp, gTemp, bTemp);
	cluster_[0].getSegmentNormParam(0, normParam1Pre, normParam2Pre, normParam3Pre);
	for (int i = 0; i < clusterCnt_; ++i) {
		int r, g, b;
		double normParam1, normParam2, normParam3;
		int vertex3DSize = cluster_[i].getVertex3DSize();
		
		for (int t = 0; t < vertex3DSize; ++t) {
			int vertexNum = cluster_[i].getVertexNum(t);
			if (vertexNum == 0)
				continue;
			else {
				cluster_[i].getAverageColor(r, g, b);
				cluster_[i].getSegmentNormParam(t,normParam1, normParam2, normParam3);
				output << vertexNum << " ";
				for (int j = 0; j < vertexNum; ++j)
					output << cluster_[i].getVertexX(t, j) << " "
					<< cluster_[i].getVertexY(t, j) << " "    
					<< cluster_[i].getVertexZ(t, j) << " ";
				colorErrRate = RGBCompare(r, g, b, rPre, gPre, bPre);
				thetaErrRate = normVectorCompare(normParam1, normParam2, normParam3, normParam1Pre, normParam2Pre, normParam3Pre);
				if (colorErrRate < 0.1 && thetaErrRate < 7.0)   //说明颜色的相似度为96%以内且法向量的差值在10度以内
				{
					output << rTemp << " " << gTemp << " " << bTemp << endl;
					rPre = r;
					gPre = g;
					bPre = b;                                                                                                                                    
				}
				else
				{
					output << r << " " << g << " " << b << endl;
					rPre = r;
					gPre = g;
					bPre = b;
					rTemp = r;
					rTemp = g;
					rTemp = b;
				}
				normParam1Pre = normParam1;
				normParam2Pre = normParam2;
				normParam3Pre = normParam3;
				
			}
		}
	}
	output.close();
}

int Map::findCoplane(int c1, int c2, int c3) {
	for (int i = 0; i < clusterCnt_; ++i) {
		int currentC1, currentC2, currentC3;
		if (cluster_[i].checkCoplane(c1, c2, c3))
			return i;
	}
	return -1;
}

bool Map::Cluster::checkCoplane(int c1, int c2, int c3) {
	return (norm1_ == c1 && norm2_ == c2 && norm3_ == c3);
}

double Map::RGBCompare(int aR, int aG, int aB, int bR, int bG, int bB) {  // 比价颜色的相似程度
	int absR = aR - bR, absG = aG - bG, absB = aB - bB;

	return sqrt(absR*absR + absG*absG + absB*absB) / sqrt(255 * 255 + 255 * 255 + 255 * 255);   //返回的是差错度，越小越相近
}

double Map::normVectorCompare(double nv1, double nv2, double nv3, double nvp1, double nvp2, double nvp3) {
	double norm1by = nv1*nvp1, norm2by = nv2*nvp2, norm3by = nv3*nvp3;
	double sqrtNv = sqrt(nv1*nv1 + nv2*nv2 + nv3*nv3), sqrtNvp = sqrt(nvp1*nvp1 + nvp2*nvp2 + nvp3*nvp3);
	double normVecCos = (norm1by + norm2by + norm3by) / (sqrtNv*sqrtNvp);
	

	return fabs(acos(normVecCos) * 180 / 3.1415);
}