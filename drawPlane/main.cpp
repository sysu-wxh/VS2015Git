#include <Windows.h>
#include <opencv.hpp>
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <GL/freeglut.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

const int numOfEdge = 100;
const float Pi = 3.1415926536f;
static float c = Pi / 180.0f; //弧度和角度转换参数  
static int du = 90, oldmy = -1, oldmx = -1; //du是视点绕y轴的角度,opengl里默认y轴是上方向  
static float r = 1.5f, h = 0.0f; //r是视点绕y轴的半径,h是视点高度即在y轴上的坐标  
float rotatex = 0, rotatey = 0, rotatez = 0;
int eyex = 0, eyez = 0, atx = 0, atz = 0;
float scalar = 1;
int s = 1;
int tx = 0, ty = 0, tz = 0;
int mx = 0, my = 0;
bool mouseisdown = false;
double moveStep = 0.2;
double moveStep2 = 1.0;
struct Vertex {
	int cnt;
	std::vector<double> x, y, z;
	Vertex() {
		cnt = 0;
		x.clear();
		y.clear();
		z.clear();
	}
};

struct Plane {
	//double centerX, centerY, centerZ;
	//double param1, param2, param3;   // 注意此处的param3并不影响，因为有了中心点坐标和另外两个参数，param3是可以反推出来的
	double r, g, b;
	Vertex vertex;

	Plane() {
		//centerX = 0.0, centerY = 0.0, centerZ = 0.0;
		//param1 = 0.0, param2 = 0.0, param3 = 0.0;
		r = 0.0, g = 0.0, b = 0.0;
		vertex.cnt = 0;
		vertex.x.clear();
		vertex.y.clear();
		vertex.z.clear();
	}
	Plane(double c1, double c2, double c3,
		int cnt, Vertex v) {
		r = c1, g = c2, b = c3;
		vertex.cnt = cnt;
		for (int i = 0; i < cnt; ++i) {
			vertex.x.push_back(v.x[i]);
			vertex.y.push_back(v.y[i]);
			vertex.z.push_back(v.z[i]);
		}
	}
};
std::vector<Plane> plane;

void readInput(std::vector<Plane>& plane, const std::string planeSegmentationFileStr);
void wininit();
void draw();
void Mouse(int button, int state, int x, int y);


void reshape(int w, int h) {
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-50, 50, -50, 50, -500, 500);
	//gluPerspective (60, (GLfloat)w / (GLfloat)h, 1.0, 500.0);    // 显示 1 - 500 距离单位（这里是 cm）内的点云
	glMatrixMode(GL_MODELVIEW);
}

void special(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_LEFT:
		rotatey -= moveStep;
		glutPostRedisplay();
		break;
	case GLUT_KEY_RIGHT:
		rotatey += moveStep;
		glutPostRedisplay();
		break;
	case GLUT_KEY_UP:
		rotatex += moveStep;
		glutPostRedisplay();
		break;
	case GLUT_KEY_DOWN:
		rotatex -= moveStep;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F7:
		rotatez -= moveStep;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F8:
		rotatez += moveStep;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F1:
		tx -= moveStep2;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F2:
		tx += moveStep2;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F3:
		ty -= moveStep2;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F4:
		ty += moveStep2;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F5:
		tz -= moveStep2;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F6:
		tz += moveStep2;
		glutPostRedisplay();
		break;
	case GLUT_KEY_PAGE_UP:
		scalar *= 1.1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_PAGE_DOWN:
		scalar /= 1.1;
		glutPostRedisplay();
		break;
	}
}

void motion(int x, int y)
{
	if (mouseisdown == true)
	{
		rotatey += ((x - mx) / 400.0);
		rotatex += ((y - my) / 400.0);
		mx = x;
		my = -y;
		glutPostRedisplay();
	}
	//glutReshapeFunc(reshape);            // 窗口变化时重构图像  
	//glutDisplayFunc(&renderScene);        // 显示三维图像  
}

void mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			mouseisdown = true;
		}
		else
		{
			mouseisdown = false;
		}
	}
}

void reconstrction3D() {
	while (1) {
		glutReshapeFunc(reshape);            // 窗口变化时重构图像  
		glutDisplayFunc(&draw);        // 显示三维图像  
		glutSpecialFunc(special);                // 响应方向键按键消息  
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutMainLoopEvent();
		glutPostRedisplay();                        // 刷新画面（不用此语句则不能动态更新图像）  
		cvWaitKey(10);
	}
}

int main(int argc, char *argv[]) {
	std::string planeSegmentationFileStr = "I://Master//dataset//08-2018-03-15-LW//output00_2b.txt";
	//std::string planeSegmentationFileStr = "F://dataset//data_stereo_flow//training//planeReconstructionResult//000098_10.txt";
	//std::string planeSegmentationFileStr = "F://dataset//middlebury//2001//png//barn2//output.txt";
	//std::string planeSegmentationFileStr = "F://dataset//data_stereo_flow//testing//planeReconstructionResult//000000_10.txt";
	/*std::string rectanSegmentationFileStr;
	std::cin >> rectanSegmentationFileStr;
	for (int i = 0; i < rectanSegmentationFileStr.length(); ++i) {
	if (rectanSegmentationFileStr[i] == '\\')
	rectanSegmentationFileStr.replace(i, 1, "//");
	}*/
	//std::string circleSegmentationFileStr = "G://dataset//data_stereo_flow//testing//result.txt";
	readInput(plane, planeSegmentationFileStr);
	std::cout << "read file done!" << std::endl;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(800, 700);
	glutCreateWindow("CircleSegmentationDraw");
	wininit();
	glutReshapeFunc(reshape);
	//while (1){
	glutDisplayFunc(draw);
	glutIdleFunc(draw);  //设置不断调用显示函数
	reconstrction3D();
	//glutMouseFunc(Mouse);
	//glutMotionFunc(onMouseMove);
	//glutMainLoop();
	//glutPostRedisplay();
	//}
	//system("pause");
	return 0;
}

void Mouse(int button, int state, int x, int y) //处理鼠标点击  
{
	if (state == GLUT_DOWN) //第一次鼠标按下时,记录鼠标在窗口中的初始坐标  
		oldmx = x, oldmy = y;
}

void readInput(std::vector<Plane>& plane, const std::string planeSegmentationFileStr) {
	std::ifstream inputFileStream(planeSegmentationFileStr.c_str(), std::ios_base::in);
	if (inputFileStream.fail()) {
		std::cerr << "error: can't open file (" << planeSegmentationFileStr << ")" << std::endl;
		exit(0);
	}
	int cnt;
	double r, g, b;
	double tempX, tempY, tempZ;
	Vertex temp;
	int index = 0;
	while (!inputFileStream.eof()) {
		inputFileStream >> cnt;
		//std::cout << a1 << " " << a2 << " " << a3 << " " << a4 << " " << a5
		//	<< " " << a6 << " " << a7 << " " << a8 << " " << a9 << " " << a10 << std::endl;
		temp.cnt = cnt;
		temp.x.clear(), temp.y.clear(), temp.z.clear();
		for (int i = 0; i < cnt; ++i) {
			inputFileStream >> tempX >> tempY >> tempZ;
			//std::cout << tempX << " " << tempY << " " << tempZ << std::endl;
			temp.x.push_back(tempX);
			temp.y.push_back(tempY);
			temp.z.push_back(tempZ);
		}
		inputFileStream >> r >> g >> b;
		//std::cout << r << " " << g << " " << b << std::endl;
		//std::cout << index << std::endl;
		index++;
		Plane tempPlane(r, g, b, cnt, temp);
		plane.push_back(tempPlane);
	}
}

void wininit() {
	gluLookAt(0, 0, 1, 0, 0, 0.0, 0.0, 1.0, 0.0);
	glClearColor(0.0, 0.0, 0.0, 0.0);   //clear the color of the window with black
										//glScalef(scalar, scalar, scalar);
	glMatrixMode(GL_MODELVIEW);    // projection mode
	glLoadIdentity();
	glOrtho(-160, 160, -10, 50, -200, 200);   //set the Clipping window
}

void draw() {
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	//glScalef(scalar, scalar, scalar);
	glRotatef(rotatey, 0.0, 1.0, 0.0); //rotate about the y axis            // 根据键盘方向键按键消息变换摄像机视角
	glRotatef(rotatex, 1.0, 0.0, 0.0); //rotate about the x axis
	glRotatef(rotatez, 0.0, 0.0, 1.0); //rotate about the z axis
	glTranslatef(tx, ty, tz);
	glScalef(scalar, scalar, scalar);
	glMatrixMode(GL_MODELVIEW);
	int size = plane.size();
	for (int planeIndice = 0; planeIndice < size; ++planeIndice) {
		/*double x = plane[planeIndice].centerX;
		double y = plane[planeIndice].centerY;
		double z = plane[planeIndice].centerZ;
		double p1 = plane[planeIndice].param1;
		double p2 = plane[planeIndice].param2;
		double p3 = plane[planeIndice].param3;
		if (p1 == 0 && p2 == 0 && p3 == 0)
		continue;*/
		//glLoadIdentity();
		//std::cout << x << " " << y << " " << z << std::endl;
		/*glTranslatef(x, y, z);
		double tempDenominator = sqrt(p1 * p1 + p2 * p2 + 1.0);
		double midLineX = p1 / (tempDenominator * 2.0);
		double midLineY = p2 / (tempDenominator * 2.0);
		double midLineZ = (-1.0 / tempDenominator + 1.0) / 2.0;
		glRotated(180.0, midLineX, midLineY, midLineZ);*/
		glBegin(GL_POLYGON);
		glColor3f(plane[planeIndice].r / 255.0, plane[planeIndice].g / 255.0, plane[planeIndice].b / 255.0); //set the color of circle
		for (int i = 0; i < plane[planeIndice].vertex.cnt; ++i)
			glVertex3f(plane[planeIndice].vertex.x[i], plane[planeIndice].vertex.y[i], plane[planeIndice].vertex.z[i] - 10.0);
		/*glVertex3f(plane[planeIndice].vertex.x[0], plane[planeIndice].vertex.y[0], 0.0);*/
		/*glVertex3f(-0.3, -0.3, 0.0);
		glVertex3f(-0.3, 0.3, 0.0);
		glVertex3f(0.3, 0.3, 0.0);
		glVertex3f(0.3, -0.3, 0.0);*/
		/*for (int i = 0; i < numOfEdge; ++i){
		glVertex3f(maxR*cos(2 * Pi / numOfEdge * i), maxR*sin(2 * Pi / numOfEdge * i), 0.0);
		}*/
		glEnd();
		//glRotated(180.0, -midLineX, -midLineY, -midLineZ);
		//glTranslatef(-x, -y, -z);
	}
	glFlush();
	glutSwapBuffers();
}

