#pragma once
#include <vector>
#include <cstdio>
using namespace std;

typedef unsigned int UINT;

struct VERTEX {
	int num;
	vector<float> u, v;
	VERTEX() {
		num = 0;
		u.clear();
		v.clear();
	}
	VERTEX(vector<float> x, vector<float> y) {
		num = x.size();
		u.clear();
		v.clear();
		for (int i = 0; i < num; ++i) {
			u.push_back(x[i]);
			v.push_back(y[i]);
		}
	}
};

struct VERTEX3D {
	int num;
	vector<float> x, y, z;
	VERTEX3D() {
		num = 0;
		x.clear();
		y.clear();
		z.clear();
	}
	VERTEX3D(vector<float> tempX, vector<float> tempY, vector<float> tempZ) {
		num = tempX.size();
		x.clear();
		y.clear();
		z.clear();
		for (int i = 0; i < num; ++i) {
			x.push_back(tempX[i]);
			y.push_back(tempY[i]);
			z.push_back(tempZ[i]);
		}
	}
};