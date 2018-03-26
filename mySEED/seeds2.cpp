// ****************************************************************************** 
// SEEDS Superpixels
// ******************************************************************************
// Author: Beat Kueng based on Michael Van den Bergh's code
// Contact: vamichae@vision.ee.ethz.ch
//
// This code implements the superpixel method described in:
// M. Van den Bergh, X. Boix, G. Roig, B. de Capitani and L. Van Gool, 
// "SEEDS: Superpixels Extracted via Energy-Driven Sampling",
// ECCV 2012
// 
// Copyright (c) 2012 Michael Van den Bergh (ETH Zurich). All rights reserved.
// ******************************************************************************

#include "seeds2.h"
#include "math.h"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

#include <cstring>

// Choose color space, LAB is better, HSV is faster (initialization)
#define LAB_COLORSPACE
//#define HSV_COLORSPACE

// Enables means post-processing
#define MEANS

// Enables 3x3 smoothing prior
#define PRIOR

// If enabled, each block-updating level will be iterated twice.
// Disable this to speed up the algorithm.
#define DOUBLE_STEPS
#define REQ_CONF 0.3

#define MINIMUM_NR_SUBLABELS 1

const double inlier_threshold = 3.0;

void SEEDS::iterate()
{
	// block updates
	while (seeds_current_level >= 0)
	{
#ifdef DOUBLE_STEPS
		update_blocks(seeds_current_level, REQ_CONF);
#endif
		update_blocks(seeds_current_level, 0.15);
		seeds_current_level = go_down_one_level();
	}

#ifdef MEANS
	/*compute_means();
	update_pixels_means();
	update_pixels_means();
	update_pixels_means();
	update_pixels_means();*/
#else
	update_pixels();
	update_pixels();
	update_pixels();
	update_pixels();
#endif
}

SEEDS::SEEDS(int width, int height, int nr_channels, int nr_bins, float disparity_factor)
{
	this->width = width;
	this->height = height;
	this->nr_channels = nr_channels;
	this->nr_bins = nr_bins;
	lambda = disparity_factor;

	image_bins = new UINT[width*height];
	image_l = new float[width*height];
	image_a = new float[width*height];
	image_b = new float[width*height];
	forwardbackward = true;
	histogram_size = nr_bins*nr_bins*nr_bins;
	initialized = false;
}

SEEDS::~SEEDS()
{
	deinitialize();
	// from constructor
	vertex.clear();
	delete[] image_bins;
	delete[] image_l;
	delete[] image_a;
	delete[] image_b;
}

void SEEDS::update_blocks_d(double** planeFunction) {
	int labelA;
	int labelB;
	int sublabel;
	bool done;
	int level = 0; //只更新最小的block
	int req_confidence = 0;
	int step = nr_w[level];
	double inlierThresholdSquare = inlier_threshold * inlier_threshold;
	// horizontal bidirectional block updating
	for (int y = 1; y < nr_h[level] - 1; y++)
		for (int x = 1; x < nr_w[level] - 2; x++)
		{
			// choose a label at the current level
			sublabel = y*step + x;
			// get the label at the top level (= superpixel label)
			labelA = parent[level][y*step + x];
			// get the neighboring label at the top level (= superpixel label)
			labelB = parent[level][y*step + x + 1];

			int positionA = y_mean[seeds_top_level][labelA] * width + x_mean[seeds_top_level][labelA];
			int positionB = y_mean[seeds_top_level][labelB] * width + x_mean[seeds_top_level][labelB];
			double planeA0 = planeFunction[positionA][0],
				planeA1 = planeFunction[positionA][1],
				planeA2 = planeFunction[positionA][2];
			double planeB0 = planeFunction[positionB][0],
				planeB1 = planeFunction[positionB][1],
				planeB2 = planeFunction[positionB][2];
			double positionCurrentX = (x + 0.5) * seeds_w, positionCurrentY = (y + 0.5) * seeds_h;
			double disparityCurrent = disparity_data[(int)(positionCurrentY*width + positionCurrentX)];
			double estimatedDisparityA = 0.0, estimatedDisparityB = 0.0;
			double distanceA = inlierThresholdSquare;
			double distanceB = inlierThresholdSquare;
			if (planeA0 != 0 && planeA1 != 0 && planeA2 != -1) {
				estimatedDisparityA = planeA0 * positionCurrentX + planeA1 * positionCurrentY + planeA2;
				if (estimatedDisparityA > 0) {
					distanceA = (disparityCurrent - estimatedDisparityA)*(disparityCurrent - estimatedDisparityA);
					if (distanceA > inlierThresholdSquare) distanceA = inlierThresholdSquare;
				}
			}
			if (planeB0 != 0 && planeB1 != 0 && planeB2 != -1) {
				estimatedDisparityB = planeB0 * positionCurrentX + planeB1 * positionCurrentY + planeB2;
				if (estimatedDisparityB > 0) {
					distanceB = (disparityCurrent - estimatedDisparityB)*(disparityCurrent - estimatedDisparityB);
					if (distanceB > inlierThresholdSquare) distanceB = inlierThresholdSquare;
				}
			}
			//cout << distanceA << " " << distanceB << endl;

			if (labelA != labelB){
				// get the surrounding labels at the top level, to check for splitting 
				int a11 = parent[level][(y - 1)*step + (x - 1)];
				int a12 = parent[level][(y - 1)*step + (x)];
				int a13 = parent[level][(y - 1)*step + (x + 1)];
				int a14 = parent[level][(y - 1)*step + (x + 2)];
				int a21 = parent[level][(y)*step + (x - 1)];
				int a22 = parent[level][(y)*step + (x)];
				int a23 = parent[level][(y)*step + (x + 1)];
				int a24 = parent[level][(y)*step + (x + 2)];
				int a31 = parent[level][(y + 1)*step + (x - 1)];
				int a32 = parent[level][(y + 1)*step + (x)];
				int a33 = parent[level][(y + 1)*step + (x + 1)];
				int a34 = parent[level][(y + 1)*step + (x + 2)];

				done = false;
				//if (a13 != labelA)


				if (nr_partitions[seeds_top_level][labelA] > MINIMUM_NR_SUBLABELS)
				{
					// remove sublabel from A, get intersections with A and B
					if (nr_partitions[seeds_top_level][labelA] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelA, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						float confidence = fabs(intA - intB);
						//cout << "intA = " << intA << " distanceA = " << distanceA << endl;
						//cout << "intB = " << intB << " distanceB = " << distanceB << endl;
						float energyA = intA + lambda*distanceA;
						float energyB = intB + lambda*distanceB;
						// add to label with highest intersection
						if ((energyB > energyA) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelB, level, sublabel);
							done = true;
						}
						else
						{
							add_block(seeds_top_level, labelA, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelA] > 2) // 3 or more partitions
					{
						if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, true, true))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelA, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							float energyA = intA + lambda*distanceA;
							float energyB = intB + lambda*distanceB;
							float confidence = fabs(intA - intB);
							// add to label with highest intersection
							if ((energyB > energyA) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelB, level, sublabel);
								done = true;
							}
							else
							{
								add_block(seeds_top_level, labelA, level, sublabel);
							}
						}
					}
				}

				if ((!done) && (nr_partitions[seeds_top_level][labelB] > MINIMUM_NR_SUBLABELS))
				{
					// try opposite direction
					sublabel = y*step + x + 1;
					positionCurrentX = (x + 1.5) * seeds_w, positionCurrentY = (y + 0.5) * seeds_h;
					disparityCurrent = disparity_data[(int)(positionCurrentY*width + positionCurrentX)];
					distanceA = inlierThresholdSquare;
					distanceB = inlierThresholdSquare;
					if (planeA0 != 0 && planeA1 != 0 && planeA2 != -1) {
						estimatedDisparityA = planeA0 * positionCurrentX + planeA1 * positionCurrentY + planeA2;
						if (estimatedDisparityA > 0) {
							distanceA = (disparityCurrent - estimatedDisparityA)*(disparityCurrent - estimatedDisparityA);
							if (distanceA > inlierThresholdSquare) distanceA = inlierThresholdSquare;
						}
					}
					if (planeB0 != 0 && planeB1 != 0 && planeB2 != -1) {
						estimatedDisparityB = planeB0 * positionCurrentX + planeB1 * positionCurrentY + planeB2;
						if (estimatedDisparityB > 0) {
							distanceB = (disparityCurrent - estimatedDisparityB)*(disparityCurrent - estimatedDisparityB);
							if (distanceB > inlierThresholdSquare) distanceB = inlierThresholdSquare;
						}
					}
					if (nr_partitions[seeds_top_level][labelB] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelB, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
						//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
						float energyA = intA + lambda*distanceA;
						float energyB = intB + lambda*distanceB;
						float confidence = fabs(intA - intB);
						if ((energyA > energyB) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelA, level, sublabel);
							x++;
						}
						else
						{
							add_block(seeds_top_level, labelB, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelB] > 2)
					{
						if (!check_split(a12, a13, a14, a22, a23, a24, a32, a33, a34, true, false))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelB, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							float energyA = intA + lambda*distanceA;
							float energyB = intB + lambda*distanceB;
							float confidence = fabs(intA - intB);
							if ((energyA > energyB) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelA, level, sublabel);
								x++;
							}
							else
							{
								add_block(seeds_top_level, labelB, level, sublabel);
							}
						}
					}
				}
			}
		}
	update_labels(level);

	// vertical bidirectional
	for (int x = 1; x < nr_w[level] - 1; x++)
		for (int y = 1; y < nr_h[level] - 2; y++)
		{
			// choose a label at the current level
			sublabel = y*step + x;
			// get the label at the top level (= superpixel label)
			labelA = parent[level][y*step + x];
			// get the neighboring label at the top level (= superpixel label)
			labelB = parent[level][(y + 1)*step + x];

			int positionA = y_mean[seeds_top_level][labelA] * width + x_mean[seeds_top_level][labelA];
			int positionB = y_mean[seeds_top_level][labelB] * width + x_mean[seeds_top_level][labelB];
			double planeA0 = planeFunction[positionA][0],
				planeA1 = planeFunction[positionA][1],
				planeA2 = planeFunction[positionA][2];
			double planeB0 = planeFunction[positionB][0],
				planeB1 = planeFunction[positionB][1],
				planeB2 = planeFunction[positionB][2];
			double positionCurrentX = (x+0.5) * seeds_w, positionCurrentY = (y+0.5) * seeds_h;
			double disparityCurrent = disparity_data[(int)(positionCurrentY*width + positionCurrentX)];
			
			double estimatedDisparityA = 0.0, estimatedDisparityB = 0.0;
			double distanceA = inlierThresholdSquare;
			double distanceB = inlierThresholdSquare;
			if (planeA0 != 0 && planeA1 != 0 && planeA2 != -1) {
				estimatedDisparityA = planeA0 * positionCurrentX + planeA1 * positionCurrentY + planeA2;
				if (estimatedDisparityA > 0) {
					distanceA = (disparityCurrent - estimatedDisparityA)*(disparityCurrent - estimatedDisparityA);
					if (distanceA > inlierThresholdSquare) distanceA = inlierThresholdSquare;
				}
			}
			if (planeB0 != 0 && planeB1 != 0 && planeB2 != -1) {
				estimatedDisparityB = planeB0 * positionCurrentX + planeB1 * positionCurrentY + planeB2;
				if (estimatedDisparityB > 0) {
					distanceB = (disparityCurrent - estimatedDisparityB)*(disparityCurrent - estimatedDisparityB);
					if (distanceB > inlierThresholdSquare) distanceB = inlierThresholdSquare;
				}
			}

			if (labelA != labelB)
			{
				int a11 = parent[level][(y - 1)*step + (x - 1)];
				int a12 = parent[level][(y - 1)*step + (x)];
				int a13 = parent[level][(y - 1)*step + (x + 1)];
				int a21 = parent[level][(y)*step + (x - 1)];
				int a22 = parent[level][(y)*step + (x)];
				int a23 = parent[level][(y)*step + (x + 1)];
				int a31 = parent[level][(y + 1)*step + (x - 1)];
				int a32 = parent[level][(y + 1)*step + (x)];
				int a33 = parent[level][(y + 1)*step + (x + 1)];
				int a41 = parent[level][(y + 2)*step + (x - 1)];
				int a42 = parent[level][(y + 2)*step + (x)];
				int a43 = parent[level][(y + 2)*step + (x + 1)];

				done = false;
				if (nr_partitions[seeds_top_level][labelA] > MINIMUM_NR_SUBLABELS)
				{
					// remove sublabel from A, get intersections with A and B
					if (nr_partitions[seeds_top_level][labelA] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelA, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						float confidence = fabs(intA - intB);
						//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
						//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
						float energyA = intA + lambda*distanceA;
						float energyB = intB + lambda*distanceB;
						// add to label with highest intersection
						if ((energyB > energyA) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelB, level, sublabel);
							//y++;
							done = true;
						}
						else
						{
							add_block(seeds_top_level, labelA, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelA] > 2) // 3 or more partitions
					{
						if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, false, true))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelA, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							float confidence = fabs(intA - intB);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							float energyA = intA + lambda*distanceA;
							float energyB = intB + lambda*distanceB;
							// add to label with highest intersection
							if ((energyB > energyA) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelB, level, sublabel);
								//y++;
								done = true;
							}
							else
							{
								add_block(seeds_top_level, labelA, level, sublabel);
							}
						}
					}
				}

				if ((!done) && (nr_partitions[seeds_top_level][labelB] > MINIMUM_NR_SUBLABELS))
				{
					// try opposite direction
					sublabel = (y + 1)*step + x;
					positionCurrentX = (x+0.5) * seeds_w, positionCurrentY = (y+1.5) * seeds_h;
					disparityCurrent = disparity_data[(int)(positionCurrentY*width + positionCurrentX)];
					estimatedDisparityA = 0.0, estimatedDisparityB = 0.0;
					distanceA = inlierThresholdSquare;
					distanceB = inlierThresholdSquare;
					if (planeA0 != 0 && planeA1 != 0 && planeA2 != -1) {
						estimatedDisparityA = planeA0 * positionCurrentX + planeA1 * positionCurrentY + planeA2;
						if (estimatedDisparityA > 0) {
							distanceA = (disparityCurrent - estimatedDisparityA)*(disparityCurrent - estimatedDisparityA);
							if (distanceA > inlierThresholdSquare) distanceA = inlierThresholdSquare;
						}
					}
					if (planeB0 != 0 && planeB1 != 0 && planeB2 != -1) {
						estimatedDisparityB = planeB0 * positionCurrentX + planeB1 * positionCurrentY + planeB2;
						if (estimatedDisparityB > 0) {
							distanceB = (disparityCurrent - estimatedDisparityB)*(disparityCurrent - estimatedDisparityB);
							if (distanceB > inlierThresholdSquare) distanceB = inlierThresholdSquare;
						}
					}
					if (nr_partitions[seeds_top_level][labelB] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelB, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						float confidence = fabs(intA - intB);
						//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
						//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
						float energyA = intA + lambda*distanceA;
						float energyB = intB + lambda*distanceB;
						if ((energyA > energyB) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelA, level, sublabel);
							y++;
						}
						else
						{
							add_block(seeds_top_level, labelB, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelB] > 2)
					{
						if (!check_split(a21, a22, a23, a31, a32, a33, a41, a42, a43, false, false))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelB, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							float confidence = fabs(intA - intB);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							float energyA = intA + lambda*distanceA;
							float energyB = intB + lambda*distanceB;
							if ((energyA > energyB) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelA, level, sublabel);
								y++;
							}
							else
							{
								add_block(seeds_top_level, labelB, level, sublabel);
							}
						}
					}
				}
			}
		}
	update_labels(level);
}

void SEEDS::getParrentLabel(int level, UINT * parentLabel){
	int size = sizeof(parent[level]) / sizeof(parent[level][0]);
	parentLabel = reinterpret_cast<UINT*>(malloc(size * sizeof(UINT)));
	for (int i = 0; i < size; ++i) {
		parentLabel[i] = parent[level][i];
	}
}

void SEEDS::deinitialize()
{
	if (initialized)
	{
		initialized = false;
		for (int level = 0; level < seeds_nr_levels; level++)
		{
			for (UINT label = 0; label < nr_labels[level]; label++)
			{
				delete[] histogram[level][label];
			}
			delete[]  histogram[level];
			delete[] T[level];
			delete[] P[level];
			delete[] labels[level];
			delete[] parent[level];
			delete[] nr_partitions[level];

		}
		delete[] histogram;
		delete[] T;
		delete[] P;
		delete[] labels;
		delete[] parent;
		delete[] nr_partitions;
		delete[] nr_labels;
		delete[] nr_w;
		delete[] nr_h;

#ifdef LAB_COLORSPACE
		delete[] bin_cutoff1;
		delete[] bin_cutoff2;
		delete[] bin_cutoff3;
#endif

#ifdef MEANS
		delete[] L_channel;
		delete[] A_channel;
		delete[] B_channel;
#endif
	}
}



void SEEDS::initialize(int seeds_w, int seeds_h, int nr_levels)
{
	deinitialize();

	this->seeds_w = seeds_w;
	this->seeds_h = seeds_h;
	this->seeds_nr_levels = nr_levels;
	this->seeds_top_level = nr_levels - 1;

	// init labels
	labels = new UINT*[seeds_nr_levels];
	parent = new UINT*[seeds_nr_levels];
	nr_partitions = new UINT*[seeds_nr_levels];
	nr_labels = new UINT[seeds_nr_levels];
	nr_w = new int[seeds_nr_levels];
	nr_h = new int[seeds_nr_levels];
	int level = 0;
	int nr_seeds_w = floor(width / seeds_w);
	int nr_seeds_h = floor(height / seeds_h);
	int nr_seeds = nr_seeds_w*nr_seeds_h;
	nr_labels[level] = nr_seeds;
	nr_w[level] = nr_seeds_w;
	nr_h[level] = nr_seeds_h;
	labels[level] = new UINT[width*height];
	parent[level] = new UINT[nr_seeds];
	nr_partitions[level] = new UINT[nr_seeds];
	// allocate disparity array
	disparity_data = reinterpret_cast<float*>(malloc(width * height * sizeof(float)));
	for (level = 1; level < seeds_nr_levels; level++)
	{
		nr_seeds_w /= 2; // always partitioned in 2x2 sub-blocks
		nr_seeds_h /= 2; // always partitioned in 2x2 sub-blocks
		nr_seeds = nr_seeds_w*nr_seeds_h;
		labels[level] = new UINT[width*height];
		parent[level] = new UINT[nr_seeds];
		nr_partitions[level] = new UINT[nr_seeds];
		nr_labels[level] = nr_seeds;
		nr_w[level] = nr_seeds_w;
		nr_h[level] = nr_seeds_h;
	}


#ifdef MEANS
	L_channel = new float[nr_labels[seeds_top_level]];
	A_channel = new float[nr_labels[seeds_top_level]];
	B_channel = new float[nr_labels[seeds_top_level]];
#endif


	// create histogram buffers
	histogram = new int**[seeds_nr_levels]; // histograms are kept for each level and each label [level][label][bin]
	T = new int*[seeds_nr_levels]; // block sizes are kept at each level [level][label]
	P = new int*[seeds_nr_levels];
	for (int level = 0; level < seeds_nr_levels; level++) {
		histogram[level] = new int*[nr_labels[level]]; // histograms are kept for each level and each label [level][label][bin]
		T[level] = new int[nr_labels[level]]; // block sizes are kept at each level [level][label]
		P[level] = new int[nr_labels[level]];
		for (UINT label = 0; label < nr_labels[level]; label++){
			histogram[level][label] = new int[histogram_size]; // histogram bins
		}
	}

	// create disparity mean buffers
	disparity_mean = new float*[seeds_nr_levels];
	x_mean = new float*[seeds_nr_levels];
	y_mean = new float*[seeds_nr_levels];
	for (int level = 0; level < seeds_nr_levels; level++) {
		disparity_mean[level] = new float[nr_labels[level]];
		x_mean[level] = new float[nr_labels[level]];
		y_mean[level] = new float[nr_labels[level]];
	}

#ifdef LAB_COLORSPACE
	bin_cutoff1 = new float[nr_bins];
	bin_cutoff2 = new float[nr_bins];
	bin_cutoff3 = new float[nr_bins];
#endif

	initialized = true;
}

void SEEDS::update_image_ycbcr(UINT* image) {

	seeds_current_level = seeds_nr_levels - 2;

	assign_labels();

	// adaptive histograms
#ifdef LAB_COLORSPACE
	lab_get_histogram_cutoff_values(image);
#endif

	// convert input image from YCbCr to LAB or HSV


	unsigned char r, g, b;
	float L, A, B;

	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
		{
			int i = y * width + x;
			r = image[i] >> 16;
			g = (image[i] >> 8) & 0xff;
			b = (image[i]) & 0xff;
#ifdef LAB_COLORSPACE
			image_bins[i] = RGB2LAB_special((int)r, (int)g, (int)b, &L, &A, &B);
			image_l[i] = L / 100.0;
			image_a[i] = (A + 128.0) / 255.0;
			image_b[i] = (B + 128.0) / 255.0;
#endif
#ifdef HSV_COLORSPACE
			image_bins[i] = RGB2HSV((int)r, (int)g, (int)b, &L, &A, &B);
			image_l[i] = L;
			image_a[i] = A;
			image_b[i] = B;
#endif
		}


	compute_histograms();
}

void SEEDS::update_disparityImage(IplImage * disparityImage){
	int index = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			disparity_data[index] = static_cast<float>(cvGet2D(disparityImage, y, x).val[0]);   //得到每一个像素的深度
			index++;
		}
	}
}




// Assign Labels
// adds labeling to all the blocks at all levels and sets the correct parents
void SEEDS::assign_labels()
{
	// level 0 (base level): seeds_w x seeds_h
	int level = 0;
	int nr_seeds_w = floor(width / seeds_w);
	int nr_seeds_h = floor(height / seeds_h);
	int step_w = seeds_w;
	int step_h = seeds_h;
	int nr_seeds = nr_seeds_w*nr_seeds_h;
	for (int i = 0; i < nr_seeds; i++) nr_partitions[level][i] = 1; // base level: no further partitioning
	for (int y = 0; y < height; y++) {
		int label_y = y / step_h;
		if (label_y >= nr_seeds_h) label_y = nr_seeds_h - 1;
		const int y_width = y*width;
		for (int x = 0; x < width; x++)
		{
			int label_x = x / step_w;
			if (label_x >= nr_seeds_w) label_x = nr_seeds_w - 1;
			labels[level][y_width + x] = label_y*nr_seeds_w + label_x;
			// parents will be set in the next loop
		}
	}

	// level 1, 2, 3 ... until top level
	for (int level = 1; level < seeds_nr_levels; level++)
	{
		nr_seeds_w /= 2; // always partitioned in 2x2 sub-blocks
		nr_seeds_h /= 2; // always partitioned in 2x2 sub-blocks
		step_w *= 2;
		step_h *= 2;
		nr_seeds = nr_seeds_w*nr_seeds_h;
		for (int i = 0; i < nr_seeds; i++) nr_partitions[level][i] = 0; // will be added in add_blocks
		for (int y = 0; y < height; y++) {
			int label_y = y / step_h;
			if (label_y >= nr_seeds_h) label_y = nr_seeds_h - 1;
			const int y_width = y*width;
			for (int x = 0; x < width; x++)
			{
				int label_x = x / step_w;
				if (label_x >= nr_seeds_w) label_x = nr_seeds_w - 1;
				labels[level][y_width + x] = label_y*nr_seeds_w + label_x;
				parent[level - 1][labels[level - 1][y_width + x]] = labels[level][y_width + x]; // set parent
			}
		}
	}

}


#ifdef LAB_COLORSPACE
void SEEDS::lab_get_histogram_cutoff_values(const UINT* image)
{
	// get image lists and histogram cutoff values
	vector<float> list_channel1;
	vector<float> list_channel2;
	vector<float> list_channel3;
	vector<float>::iterator it;
	int samp = 5;
	int ctr = 0;
	for (int x = 0; x < width; x += samp)
		for (int y = 0; y < height; y += samp)
		{
			int i = y*width + x;
			//const Image::Pixel& pix = image.image[y][x];
			unsigned char r, g, b;
			//ColorModelConversions::fromYCbCrToRGB(pix.y, pix.cb, pix.cr, r, g, b);
			r = image[ctr] >> 16;
			g = image[ctr] << 8 >> 16;
			b = image[ctr] << 16 >> 16;
			float L = 0;
			float A = 0;
			float B = 0;
			RGB2LAB((int)r, (int)g, (int)b, &L, &A, &B);
			list_channel1.push_back(L);
			list_channel2.push_back(A);
			list_channel3.push_back(B);
			ctr++;
		}
	for (int i = 1; i < nr_bins; i++)
	{
		int N = (int)floor((float)(i*ctr) / (float)nr_bins);
		nth_element(list_channel1.begin(), list_channel1.begin() + N, list_channel1.end());
		it = list_channel1.begin() + N;
		bin_cutoff1[i - 1] = *it;
		nth_element(list_channel2.begin(), list_channel2.begin() + N, list_channel2.end());
		it = list_channel2.begin() + N;
		bin_cutoff2[i - 1] = *it;
		nth_element(list_channel3.begin(), list_channel3.begin() + N, list_channel3.end());
		it = list_channel3.begin() + N;
		bin_cutoff3[i - 1] = *it;
	}
	bin_cutoff1[nr_bins - 1] = 300;
	bin_cutoff2[nr_bins - 1] = 300;
	bin_cutoff3[nr_bins - 1] = 300;
}

#endif

void SEEDS::compute_means()
{
	// clear counted LAB values
	for (UINT label = 0; label < nr_labels[seeds_top_level]; label++)
	{
		L_channel[label] = 0;
		A_channel[label] = 0;
		B_channel[label] = 0;
	}

	// sweep image
	for (int i = 0; i < width*height; i++)
	{
		int label = labels[seeds_top_level][i];
		// add LAB values to total
		L_channel[label] += image_l[i];
		A_channel[label] += image_a[i];
		B_channel[label] += image_b[i];
	}
}

void SEEDS::compute_histograms(int until_level)
{
	if (until_level == -1) until_level = seeds_nr_levels - 1;
	until_level++;


	// clear histograms
	for (int level = 0; level < seeds_nr_levels; level++)
	{
		for (UINT label = 0; label < nr_labels[level]; label++)
		{
			memset(histogram[level][label], 0, sizeof(int)*histogram_size);
		}
		memset(T[level], 0, sizeof(int)*nr_labels[level]);
		memset(P[level], 0, sizeof(int)*nr_labels[level]);
		memset(disparity_mean[level], 0, sizeof(float)*nr_labels[level]);
		memset(x_mean[level], 0, sizeof(float)*nr_labels[level]);
		memset(y_mean[level], 0, sizeof(float)*nr_labels[level]);
	}

	// build histograms on the first level by adding the pixels to the blocks
	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
		{
			int i = y*width + x;
			add_pixel(0, labels[0][i], x, y);
		}

	// build histograms on the upper levels by adding the histogram from the level below
	for (int level = 1; level < until_level; level++)
	{
		for (UINT label = 0; label < nr_labels[level - 1]; label++)
		{
			add_block(level, parent[level - 1][label], level - 1, label);
		}
	}
}

// Merging Algorithm
void SEEDS::update_blocks(int level, float req_confidence)
{
	int labelA;
	int labelB;
	int sublabel;
	bool done;

	int step = nr_w[level];

	// horizontal bidirectional block updating
	for (int y = 1; y < nr_h[level] - 1; y++)
		for (int x = 1; x < nr_w[level] - 2; x++)
		{
			// choose a label at the current level
			sublabel = y*step + x;
			// get the label at the top level (= superpixel label)
			labelA = parent[level][y*step + x];
			// get the neighboring label at the top level (= superpixel label)
			labelB = parent[level][y*step + x + 1];

			if (labelA != labelB)
			{
				// get the surrounding labels at the top level, to check for splitting 
				int a11 = parent[level][(y - 1)*step + (x - 1)];
				int a12 = parent[level][(y - 1)*step + (x)];
				int a13 = parent[level][(y - 1)*step + (x + 1)];
				int a14 = parent[level][(y - 1)*step + (x + 2)];
				int a21 = parent[level][(y)*step + (x - 1)];
				int a22 = parent[level][(y)*step + (x)];
				int a23 = parent[level][(y)*step + (x + 1)];
				int a24 = parent[level][(y)*step + (x + 2)];
				int a31 = parent[level][(y + 1)*step + (x - 1)];
				int a32 = parent[level][(y + 1)*step + (x)];
				int a33 = parent[level][(y + 1)*step + (x + 1)];
				int a34 = parent[level][(y + 1)*step + (x + 2)];

				done = false;

				if (nr_partitions[seeds_top_level][labelA] > MINIMUM_NR_SUBLABELS)
				{
					// remove sublabel from A, get intersections with A and B
					if (nr_partitions[seeds_top_level][labelA] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelA, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
						//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
						float confidence = fabs(intA - intB);
						//cout << "intA = " << intA << " disparityA = " << disparity_diffA << endl;
						//cout << "intB = " << intB << " disparityB = " << disparity_diffB << endl;
						//float energyA = intA + lambda*disparity_diffA;
						//float energyB = intB + lambda*disparity_diffB;
						// add to label with highest intersection
						//cout << "confidence = " << confidence << endl;
						if ((intB > intA) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelB, level, sublabel);
							done = true;
						}
						else
						{
							add_block(seeds_top_level, labelA, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelA] > 2) // 3 or more partitions
					{
						if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, true, true))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelA, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							//float energyA = intA + lambda*disparity_diffA;
							//float energyB = intB + lambda*disparity_diffB;
							float confidence = fabs(intA - intB);
							// add to label with highest intersection
							if ((intB > intA) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelB, level, sublabel);
								done = true;
							}
							else
							{
								add_block(seeds_top_level, labelA, level, sublabel);
							}
						}
					}
				}

				if ((!done) && (nr_partitions[seeds_top_level][labelB] > MINIMUM_NR_SUBLABELS))
				{
					// try opposite direction
					sublabel = y*step + x + 1;
					if (nr_partitions[seeds_top_level][labelB] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelB, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
						//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
						//float energyA = intA + lambda*disparity_diffA;
						//float energyB = intB + lambda*disparity_diffB;
						float confidence = fabs(intA - intB);
						if ((intA > intB) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelA, level, sublabel);
							x++;
						}
						else
						{
							add_block(seeds_top_level, labelB, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelB] > 2)
					{
						if (!check_split(a12, a13, a14, a22, a23, a24, a32, a33, a34, true, false))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelB, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							//float energyA = intA + lambda*disparity_diffA;
							//float energyB = intB + lambda*disparity_diffB;
							float confidence = fabs(intA - intB);
							if ((intA > intB) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelA, level, sublabel);
								x++;
							}
							else
							{
								add_block(seeds_top_level, labelB, level, sublabel);
							}
						}
					}
				}
			}
		}
	update_labels(level);

	// vertical bidirectional
	for (int x = 1; x < nr_w[level] - 1; x++)
		for (int y = 1; y < nr_h[level] - 2; y++)
		{
			// choose a label at the current level
			sublabel = y*step + x;
			// get the label at the top level (= superpixel label)
			labelA = parent[level][y*step + x];
			// get the neighboring label at the top level (= superpixel label)
			labelB = parent[level][(y + 1)*step + x];

			if (labelA != labelB)
			{
				int a11 = parent[level][(y - 1)*step + (x - 1)];
				int a12 = parent[level][(y - 1)*step + (x)];
				int a13 = parent[level][(y - 1)*step + (x + 1)];
				int a21 = parent[level][(y)*step + (x - 1)];
				int a22 = parent[level][(y)*step + (x)];
				int a23 = parent[level][(y)*step + (x + 1)];
				int a31 = parent[level][(y + 1)*step + (x - 1)];
				int a32 = parent[level][(y + 1)*step + (x)];
				int a33 = parent[level][(y + 1)*step + (x + 1)];
				int a41 = parent[level][(y + 2)*step + (x - 1)];
				int a42 = parent[level][(y + 2)*step + (x)];
				int a43 = parent[level][(y + 2)*step + (x + 1)];

				done = false;
				if (nr_partitions[seeds_top_level][labelA] > MINIMUM_NR_SUBLABELS)
				{
					// remove sublabel from A, get intersections with A and B
					if (nr_partitions[seeds_top_level][labelA] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelA, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						float confidence = fabs(intA - intB);
						//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
						//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
						//float energyA = intA + lambda*disparity_diffA;
						//float energyB = intB + lambda*disparity_diffB;
						// add to label with highest intersection
						if ((intB > intA) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelB, level, sublabel);
							//y++;
							done = true;
						}
						else
						{
							add_block(seeds_top_level, labelA, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelA] > 2) // 3 or more partitions
					{
						if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, false, true))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelA, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							float confidence = fabs(intA - intB);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							//float energyA = intA + lambda*disparity_diffA;
							//float energyB = intB + lambda*disparity_diffB;
							// add to label with highest intersection
							if ((intB > intA) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelB, level, sublabel);
								//y++;
								done = true;
							}
							else
							{
								add_block(seeds_top_level, labelA, level, sublabel);
							}
						}
					}
				}

				if ((!done) && (nr_partitions[seeds_top_level][labelB] > MINIMUM_NR_SUBLABELS))
				{
					// try opposite direction
					sublabel = (y + 1)*step + x;
					if (nr_partitions[seeds_top_level][labelB] <= 2) // == 2
					{
						// run algorithm as usual
						delete_block(seeds_top_level, labelB, level, sublabel);
						float intA = intersection(seeds_top_level, labelA, level, sublabel);
						float intB = intersection(seeds_top_level, labelB, level, sublabel);
						float confidence = fabs(intA - intB);
						//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
						//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
						//float energyA = intA + lambda*disparity_diffA;
						//float energyB = intB + lambda*disparity_diffB;
						if ((intA > intB) && (confidence > req_confidence))
						{
							add_block(seeds_top_level, labelA, level, sublabel);
							y++;
						}
						else
						{
							add_block(seeds_top_level, labelB, level, sublabel);
						}
					}
					else if (nr_partitions[seeds_top_level][labelB] > 2)
					{
						if (!check_split(a21, a22, a23, a31, a32, a33, a41, a42, a43, false, false))
						{
							// run algorithm as usual
							delete_block(seeds_top_level, labelB, level, sublabel);
							float intA = intersection(seeds_top_level, labelA, level, sublabel);
							float intB = intersection(seeds_top_level, labelB, level, sublabel);
							float confidence = fabs(intA - intB);
							//float disparity_diffA = calculateDisparityDifference(seeds_top_level, labelA, level, sublabel);
							//float disparity_diffB = calculateDisparityDifference(seeds_top_level, labelB, level, sublabel);
							//float energyA = intA + lambda*disparity_diffA;
							//float energyB = intB + lambda*disparity_diffB;
							if ((intA > intB) && (confidence > req_confidence))
							{
								add_block(seeds_top_level, labelA, level, sublabel);
								y++;
							}
							else
							{
								add_block(seeds_top_level, labelB, level, sublabel);
							}
						}
					}
				}
			}
		}
	update_labels(level);
}



int SEEDS::go_down_one_level()
{
	int old_level = seeds_current_level;
	int new_level = seeds_current_level - 1;

	if (new_level < 0) return -1;

	// go through labels of top level
	for (int x = 0; x < nr_w[seeds_top_level]; x++)
		for (int y = 0; y < nr_h[seeds_top_level]; y++)
		{
			// reset nr_partitions
			nr_partitions[seeds_top_level][y*nr_w[seeds_top_level] + x] = 0;
		}

	// go through labels of new_level
	for (int x = 0; x < nr_w[new_level]; x++)
		for (int y = 0; y < nr_h[new_level]; y++)
		{
			// assign parent = parent of old_label
			int p = parent[old_level][parent[new_level][y*nr_w[new_level] + x]];
			parent[new_level][y*nr_w[new_level] + x] = p;
			// add nr_partitions[label] to parent
			nr_partitions[seeds_top_level][p] += nr_partitions[new_level][y*nr_w[new_level] + x];
		}

	return new_level;
}



// Border Updating Algorithm
void SEEDS::update_pixels()
{
	int labelA;
	int labelB;
	int priorA = 0;
	int priorB = 0;

	if (forwardbackward)
	{
		forwardbackward = false;
		for (int y = 1; y < height - 1; y++)
			for (int x = 1; x < width - 2; x++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a14 = labels[seeds_top_level][(y - 1)*width + (x + 2)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a24 = labels[seeds_top_level][(y)*width + (x + 2)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a34 = labels[seeds_top_level][(y + 1)*width + (x + 2)];

				// horizontal bidirectional
				labelA = a22;
				labelB = a23;
				if (labelA != labelB)
				{
					if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, true, true))
					{
#ifdef PRIOR
						priorA = threebyfour(x, y, labelA);
						priorB = threebyfour(x, y, labelB);
#endif

						if (probability(image_bins[y*width + x], labelA, labelB, priorA, priorB))
						{
							update(seeds_top_level, labelB, x, y);
						}
						else if (!check_split(a12, a13, a14, a22, a23, a24, a32, a33, a34, true, false))
						{
							if (probability(image_bins[y*width + x + 1], labelB, labelA, priorB, priorA))
							{
								update(seeds_top_level, labelA, x + 1, y);
								x++;
							}
						}
					}
				}
			}


		for (int x = 1; x < width - 1; x++)
			for (int y = 1; y < height - 2; y++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a41 = labels[seeds_top_level][(y + 2)*width + (x - 1)];
				int a42 = labels[seeds_top_level][(y + 2)*width + (x)];
				int a43 = labels[seeds_top_level][(y + 2)*width + (x + 1)];

				// vertical bidirectional
				labelA = a22;
				labelB = a32;
				if (labelA != labelB)
				{
					if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, false, true))
					{
#ifdef PRIOR
						priorA = fourbythree(x, y, labelA);
						priorB = fourbythree(x, y, labelB);
#endif

						if (probability(image_bins[y*width + x], labelA, labelB, priorA, priorB))
						{
							update(seeds_top_level, labelB, x, y);
						}
						else if (!check_split(a21, a22, a23, a31, a32, a33, a41, a42, a43, false, false))
						{
							if (probability(image_bins[(y + 1)*width + x], labelB, labelA, priorB, priorA))
							{
								update(seeds_top_level, labelA, x, y + 1);
								y++;
							}
						}
					}
				}
			}
	}
	else
	{
		forwardbackward = true;
		for (int y = 1; y < height - 1; y++)
			for (int x = 1; x < width - 2; x++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a14 = labels[seeds_top_level][(y - 1)*width + (x + 2)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a24 = labels[seeds_top_level][(y)*width + (x + 2)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a34 = labels[seeds_top_level][(y + 1)*width + (x + 2)];

				// horizontal bidirectional
				labelA = a22;
				labelB = a23;
				if (labelA != labelB)
				{
					if (!check_split(a12, a13, a14, a22, a23, a24, a32, a33, a34, true, false))
					{
#ifdef PRIOR
						priorA = threebyfour(x, y, labelA);
						priorB = threebyfour(x, y, labelB);
#endif

						if (probability(image_bins[y*width + x + 1], labelB, labelA, priorB, priorA))
						{
							update(seeds_top_level, labelA, x + 1, y);
							x++;
						}
						else if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, true, true))
						{
							if (probability(image_bins[y*width + x], labelA, labelB, priorA, priorB))
							{
								update(seeds_top_level, labelB, x, y);
							}
						}
					}
				}
			}


		for (int x = 1; x < width - 1; x++)
			for (int y = 1; y < height - 2; y++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a41 = labels[seeds_top_level][(y + 2)*width + (x - 1)];
				int a42 = labels[seeds_top_level][(y + 2)*width + (x)];
				int a43 = labels[seeds_top_level][(y + 2)*width + (x + 1)];

				// vertical bidirectional
				labelA = a22;
				labelB = a32;
				if (labelA != labelB)
				{
					if (!check_split(a21, a22, a23, a31, a32, a33, a41, a42, a43, false, false))
					{
#ifdef PRIOR						
						priorA = fourbythree(x, y, labelA);
						priorB = fourbythree(x, y, labelB);
#endif

						if (probability(image_bins[(y + 1)*width + x], labelB, labelA, priorB, priorA))
						{
							update(seeds_top_level, labelA, x, y + 1);
							y++;
						}
						else if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, false, true))
						{
							if (probability(image_bins[y*width + x], labelA, labelB, priorA, priorB))
							{
								update(seeds_top_level, labelB, x, y);
							}
						}
					}
				}
			}

	}

	// update border pixels
	for (int x = 0; x < width; x++)
	{
		labelA = labels[seeds_top_level][x];
		labelB = labels[seeds_top_level][width + x];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, x, 0);
		}
		labelA = labels[seeds_top_level][(height - 1)*width + x];
		labelB = labels[seeds_top_level][(height - 2)*width + x];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, x, height - 1);
		}
	}
	for (int y = 0; y < height; y++)
	{
		labelA = labels[seeds_top_level][y*width];
		labelB = labels[seeds_top_level][y*width + 1];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, 0, y);
		}
		labelA = labels[seeds_top_level][y*width + width - 1];
		labelB = labels[seeds_top_level][y*width + width - 2];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, width - 1, y);
		}
	}
}


// Border Updating Algorithm
void SEEDS::update_pixels_means()
{
	int labelA;
	int labelB;
	int priorA = 0;
	int priorB = 0;

	if (forwardbackward)
	{
		forwardbackward = false;
		for (int y = 1; y < height - 1; y++)
			for (int x = 1; x < width - 2; x++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a14 = labels[seeds_top_level][(y - 1)*width + (x + 2)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a24 = labels[seeds_top_level][(y)*width + (x + 2)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a34 = labels[seeds_top_level][(y + 1)*width + (x + 2)];

				// horizontal bidirectional
				labelA = a22;
				labelB = a23;
				if (labelA != labelB)
				{
					if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, true, true))
					{
#ifdef PRIOR
						priorA = threebyfour(x, y, labelA);
						priorB = threebyfour(x, y, labelB);
#endif

						if (probability_means(image_l[y*width + x], image_a[y*width + x], image_b[y*width + x], labelA, labelB, priorA, priorB))
						{
							update(seeds_top_level, labelB, x, y);
						}
						else if (!check_split(a12, a13, a14, a22, a23, a24, a32, a33, a34, true, false))
						{
							if (probability_means(image_l[y*width + x + 1], image_a[y*width + x + 1], image_b[y*width + x + 1], labelB, labelA, priorB, priorA))
							{
								update(seeds_top_level, labelA, x + 1, y);
								x++;
							}
						}
					}
				}
			}


		for (int x = 1; x < width - 1; x++)
			for (int y = 1; y < height - 2; y++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a41 = labels[seeds_top_level][(y + 2)*width + (x - 1)];
				int a42 = labels[seeds_top_level][(y + 2)*width + (x)];
				int a43 = labels[seeds_top_level][(y + 2)*width + (x + 1)];

				// vertical bidirectional
				labelA = a22;
				labelB = a32;
				if (labelA != labelB)
				{
					if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, false, true))
					{
#ifdef PRIOR
						priorA = fourbythree(x, y, labelA);
						priorB = fourbythree(x, y, labelB);
#endif

						if (probability_means(image_l[y*width + x], image_a[y*width + x], image_b[y*width + x], labelA, labelB, priorA, priorB))
						{
							update(seeds_top_level, labelB, x, y);
						}
						else if (!check_split(a21, a22, a23, a31, a32, a33, a41, a42, a43, false, false))
						{
							if (probability_means(image_l[(y + 1)*width + x], image_a[(y + 1)*width + x], image_b[(y + 1)*width + x], labelB, labelA, priorB, priorA))
							{
								update(seeds_top_level, labelA, x, y + 1);
								y++;
							}
						}
					}
				}
			}
	}
	else
	{
		forwardbackward = true;
		for (int y = 1; y < height - 1; y++)
			for (int x = 1; x < width - 2; x++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a14 = labels[seeds_top_level][(y - 1)*width + (x + 2)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a24 = labels[seeds_top_level][(y)*width + (x + 2)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a34 = labels[seeds_top_level][(y + 1)*width + (x + 2)];

				// horizontal bidirectional
				labelA = a22;
				labelB = a23;
				if (labelA != labelB)
				{
					if (!check_split(a12, a13, a14, a22, a23, a24, a32, a33, a34, true, false))
					{
#ifdef PRIOR
						priorA = threebyfour(x, y, labelA);
						priorB = threebyfour(x, y, labelB);
#endif

						if (probability_means(image_l[y*width + x + 1], image_a[y*width + x + 1], image_b[y*width + x + 1], labelB, labelA, priorB, priorA))
						{
							update(seeds_top_level, labelA, x + 1, y);
							x++;
						}
						else if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, true, true))
						{
							if (probability_means(image_l[y*width + x], image_a[y*width + x], image_b[y*width + x], labelA, labelB, priorA, priorB))
							{
								update(seeds_top_level, labelB, x, y);
							}
						}
					}
				}
			}


		for (int x = 1; x < width - 1; x++)
			for (int y = 1; y < height - 2; y++)
			{
				int a11 = labels[seeds_top_level][(y - 1)*width + (x - 1)];
				int a12 = labels[seeds_top_level][(y - 1)*width + (x)];
				int a13 = labels[seeds_top_level][(y - 1)*width + (x + 1)];
				int a21 = labels[seeds_top_level][(y)*width + (x - 1)];
				int a22 = labels[seeds_top_level][(y)*width + (x)];
				int a23 = labels[seeds_top_level][(y)*width + (x + 1)];
				int a31 = labels[seeds_top_level][(y + 1)*width + (x - 1)];
				int a32 = labels[seeds_top_level][(y + 1)*width + (x)];
				int a33 = labels[seeds_top_level][(y + 1)*width + (x + 1)];
				int a41 = labels[seeds_top_level][(y + 2)*width + (x - 1)];
				int a42 = labels[seeds_top_level][(y + 2)*width + (x)];
				int a43 = labels[seeds_top_level][(y + 2)*width + (x + 1)];

				// vertical bidirectional
				labelA = a22;
				labelB = a32;
				if (labelA != labelB)
				{
					if (!check_split(a21, a22, a23, a31, a32, a33, a41, a42, a43, false, false))
					{
#ifdef PRIOR
						priorA = fourbythree(x, y, labelA);
						priorB = fourbythree(x, y, labelB);
#endif

						if (probability_means(image_l[(y + 1)*width + x], image_a[(y + 1)*width + x], image_b[(y + 1)*width + x], labelB, labelA, priorB, priorA))
						{
							update(seeds_top_level, labelA, x, y + 1);
							y++;
						}
						else if (!check_split(a11, a12, a13, a21, a22, a23, a31, a32, a33, false, true))
						{
							if (probability_means(image_l[y*width + x], image_a[y*width + x], image_b[y*width + x], labelA, labelB, priorA, priorB))
							{
								update(seeds_top_level, labelB, x, y);
							}
						}
					}
				}
			}

	}

	// update border pixels
	for (int x = 0; x < width; x++)
	{
		labelA = labels[seeds_top_level][x];
		labelB = labels[seeds_top_level][width + x];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, x, 0);
		}
		labelA = labels[seeds_top_level][(height - 1)*width + x];
		labelB = labels[seeds_top_level][(height - 2)*width + x];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, x, height - 1);
		}
	}
	for (int y = 0; y < height; y++)
	{
		labelA = labels[seeds_top_level][y*width];
		labelB = labels[seeds_top_level][y*width + 1];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, 0, y);
		}
		labelA = labels[seeds_top_level][y*width + width - 1];
		labelB = labels[seeds_top_level][y*width + width - 2];
		if (labelA != labelB)
		{
			update(seeds_top_level, labelB, width - 1, y);
		}
	}
}



void SEEDS::update(int level, int label_new, int x, int y)
{
	int label_old = labels[level][y*width + x];
	delete_pixel_m(level, label_old, x, y);
	add_pixel_m(level, label_new, x, y);
	labels[level][y*width + x] = label_new;
}

void SEEDS::add_pixel(int level, int label, int x, int y)
{
	histogram[level][label][image_bins[y*width + x]]++;
	T[level][label]++;
	if (disparity_data[y*width + x] > 0) {
		disparity_mean[level][label] = static_cast<float>((disparity_mean[level][label] * P[level][label] + disparity_data[y*width + x]) / (P[level][label] + 1));
		x_mean[level][label] = static_cast<float>((x_mean[level][label] * P[level][label] + x) / (P[level][label] + 1));
		y_mean[level][label] = static_cast<float>((y_mean[level][label] * P[level][label] + y) / (P[level][label] + 1));

		P[level][label]++;
	}
}

void SEEDS::add_pixel_m(int level, int label, int x, int y)
{
	histogram[level][label][image_bins[y*width + x]]++;
	T[level][label]++;

#ifdef MEANS
	L_channel[label] += image_l[y*width + x];
	A_channel[label] += image_a[y*width + x];
	B_channel[label] += image_b[y*width + x];
#endif
}

void SEEDS::delete_pixel(int level, int label, int x, int y)
{
	histogram[level][label][image_bins[y*width + x]]--;
	if (disparity_data[y*width + x] > 0) {
		if (P[level][label] - 1 > 0) {
			disparity_mean[level][label] = static_cast<float>((disparity_mean[level][label] * P[level][label] - disparity_data[y*width + x]) / (P[level][label] - 1));
			x_mean[level][label] = static_cast<float>((x_mean[level][label] * P[level][label] - x) / (P[level][label] - 1));
			y_mean[level][label] = static_cast<float>((y_mean[level][label] * P[level][label] - y) / (P[level][label] - 1));
		}
		else {
			disparity_mean[level][label] = 0.0;
			x_mean[level][label] = 0.0;
			y_mean[level][label] = 0.0;
		}
		P[level][label]--;
	}
	T[level][label]--;
}

void SEEDS::delete_pixel_m(int level, int label, int x, int y)
{
	histogram[level][label][image_bins[y*width + x]]--;
	T[level][label]--;

#ifdef MEANS
	L_channel[label] -= image_l[y*width + x];
	A_channel[label] -= image_a[y*width + x];
	B_channel[label] -= image_b[y*width + x];
#endif
}


void SEEDS::add_block(int level, int label, int sublevel, int sublabel)
{
	parent[sublevel][sublabel] = label;

	for (int n = 0; n < histogram_size; n++)
	{
		histogram[level][label][n] += histogram[sublevel][sublabel][n];
	}
	if (P[level][label] + P[sublevel][sublabel] != 0) {
		disparity_mean[level][label] = static_cast<float>((disparity_mean[level][label] * P[level][label]
			+ disparity_mean[sublevel][sublabel] * P[sublevel][sublabel])
			/ (P[level][label] + P[sublevel][sublabel]));
		x_mean[level][label] = static_cast<float>((x_mean[level][label] * P[level][label]
			+ x_mean[sublevel][sublabel] * P[sublevel][sublabel])
			/ (P[level][label] + P[sublevel][sublabel]));
		y_mean[level][label] = static_cast<float>((y_mean[level][label] * P[level][label]
			+ y_mean[sublevel][sublabel] * P[sublevel][sublabel])
			/ (P[level][label] + P[sublevel][sublabel]));
	}
	else {
		disparity_mean[level][label] = 0.0;
		x_mean[level][label] = 0.0;
		y_mean[level][label] = 0.0;
	}
	T[level][label] += T[sublevel][sublabel];
	P[level][label] += P[sublevel][sublabel];
	nr_partitions[level][label]++;
}



void SEEDS::delete_block(int level, int label, int sublevel, int sublabel)
{
	parent[sublevel][sublabel] = -1;

	for (int n = 0; n < histogram_size; n++)
	{
		histogram[level][label][n] -= histogram[sublevel][sublabel][n];
	}
	if (P[level][label] - P[sublevel][sublabel] != 0) {
		disparity_mean[level][label] = static_cast<float>((disparity_mean[level][label] * P[level][label]
			- disparity_mean[sublevel][sublabel] * P[sublevel][sublabel])
			/ (P[level][label] - P[sublevel][sublabel]));
		x_mean[level][label] = static_cast<float>((x_mean[level][label] * P[level][label]
			- x_mean[sublevel][sublabel] * P[sublevel][sublabel])
			/ (P[level][label] - P[sublevel][sublabel]));
		y_mean[level][label] = static_cast<float>((y_mean[level][label] * P[level][label]
			- y_mean[sublevel][sublabel] * P[sublevel][sublabel])
			/ (P[level][label] - P[sublevel][sublabel]));
	}
	else {
		disparity_mean[level][label] = 0.0;
		x_mean[level][label] = 0.0;
		y_mean[level][label] = 0.0;
	}
	T[level][label] -= T[sublevel][sublabel];
	P[level][label] -= P[sublevel][sublabel];

	nr_partitions[level][label]--;
}



void SEEDS::update_labels(int level)
{
	for (int i = 0; i < width*height; i++)
	{
		labels[seeds_top_level][i] = parent[level][labels[level][i]];
	}
}




bool SEEDS::probability(int color, int label1, int label2, int prior1, int prior2)
{
	float P_label1 = (float)histogram[seeds_top_level][label1][color] / (float)T[seeds_top_level][label1];
	float P_label2 = (float)histogram[seeds_top_level][label2][color] / (float)T[seeds_top_level][label2];

#ifdef PRIOR
	P_label1 *= (float)prior1;
	P_label2 *= (float)prior2;
#endif

	return (P_label2 > P_label1);
}

bool SEEDS::probability_means(float L, float a, float b, int label1, int label2, int prior1, int prior2)
{
	float L1 = L_channel[label1] / T[seeds_top_level][label1];
	float a1 = A_channel[label1] / T[seeds_top_level][label1];
	float b1 = B_channel[label1] / T[seeds_top_level][label1];
	float L2 = L_channel[label2] / T[seeds_top_level][label2];
	float a2 = A_channel[label2] / T[seeds_top_level][label2];
	float b2 = B_channel[label2] / T[seeds_top_level][label2];

	float P_label1 = (L - L1)*(L - L1) + (a - a1)*(a - a1) + (b - b1)*(b - b1);
	float P_label2 = (L - L2)*(L - L2) + (a - a2)*(a - a2) + (b - b2)*(b - b2);

#ifdef PRIOR
	P_label1 /= prior1;
	P_label2 /= prior2;
#endif

	return (P_label1 > P_label2);
}



void SEEDS::LAB2RGB(float L, float a, float b, int* R, int* G, int* B)
{
	float T1 = 0.008856;
	float T2 = 0.206893;

	float fY = pow((L + 16.0) / 116.0, 3);
	bool YT = fY > T1;
	fY = (!YT) * (L / 903.3) + YT * fY;
	float Y = fY;

	// Alter fY slightly for further calculations
	fY = YT * pow(fY, 1.0 / 3.0) + (!YT) * (7.787 * fY + 16.0 / 116.0);

	float fX = a / 500.0 + fY;
	bool XT = fX > T2;
	float X = (XT * pow(fX, 3) + (!XT) * ((fX - 16.0 / 116.0) / 7.787));

	float fZ = fY - b / 200.0;
	bool ZT = fZ > T2;
	float Z = (ZT * pow(fZ, 3) + (!ZT) * ((fZ - 16.0 / 116.0) / 7.787));

	X = X * 0.950456 * 255.0;
	Y = Y * 255.0;
	Z = Z * 1.088754 * 255.0;

	*R = (int)(3.240479*X - 1.537150*Y - 0.498535*Z);
	*G = (int)(-0.969256*X + 1.875992*Y + 0.041556*Z);
	*B = (int)(0.055648*X - 0.204043*Y + 1.057311*Z);
}


int SEEDS::RGB2LAB(const int& r, const int& g, const int& b, float* lval, float* aval, float* bval)
{
	float xVal = 0.412453 * r + 0.357580 * g + 0.180423 * b;
	float yVal = 0.212671 * r + 0.715160 * g + 0.072169 * b;
	float zVal = 0.019334 * r + 0.119193 * g + 0.950227 * b;

	xVal /= (255.0 * 0.950456);
	yVal /= 255.0;
	zVal /= (255.0 * 1.088754);

	float fY, fZ, fX;
	float lVal, aVal, bVal;
	float T = 0.008856;

	bool XT = (xVal > T);
	bool YT = (yVal > T);
	bool ZT = (zVal > T);

	fX = XT * pow(xVal, 1.0 / 3.0) + (!XT) * (7.787 * xVal + 16.0 / 116.0);

	// Compute L
	float Y3 = pow(yVal, 1.0 / 3.0);
	fY = YT*Y3 + (!YT)*(7.787*yVal + 16.0 / 116.0);
	lVal = YT * (116 * Y3 - 16.0) + (!YT)*(903.3*yVal);

	fZ = ZT*pow(zVal, 1.0 / 3.0) + (!ZT)*(7.787*zVal + 16.0 / 116.0);

	// Compute a and b
	aVal = 500 * (fX - fY);
	bVal = 200 * (fY - fZ);

	*lval = lVal;
	*aval = aVal;
	*bval = bVal;

	return 1; //LAB2bin(lVal, aVal, bVal);
}



int SEEDS::RGB2HSV(const int& r, const int& g, const int& b, float* hval, float* sval, float* vval)
{
	float r_ = r / 256.0;
	float g_ = g / 256.0;
	float b_ = b / 256.0;

	float min_rgb = min(r_, min(g_, b_));
	float max_rgb = max(r_, max(g_, b_));
	float V = max_rgb;
	float H = 0.0;
	float S = 0.0;

	float delta = max_rgb - min_rgb;

	if ((delta > 0.0) && (max_rgb > 0.0))
	{
		S = delta / max_rgb;
		if (max_rgb == r_)
			H = (g_ - b_) / delta;
		else if (max_rgb == g_)
			H = 2 + (b_ - r_) / delta;
		else
			H = 4 + (r_ - g_) / delta;
	}

	H /= 6;

	/*float V = max(r_,max(g_,b_));
	float S = 0.0;
	if (V != 0.0) S = (V - min(r_, min(g_,b_)))/V;
	float H = 0.0;
	if (S != 0.0)
	{
		if (V == r_)
			H = ((g_ - b_)/6.0)/S;
		else if (V == g_)
			H = 1.0/2.0 + ((b_-r_)/6.0)/S;
		else
			H = 2.0/3.0 + ((r_-g_)/6.0)/S;
	}*/

	if (H < 0.0) H += 1.0;

	*hval = H;
	*sval = S;
	*vval = V;

	int hbin = int(H * nr_bins);
	int sbin = int(S * nr_bins);
	int vbin = int(V * nr_bins);
	if (sbin == nr_bins) --sbin; //S can be equal to 1.0

	//printf("%d %d %d -- %f %f %f -- bins %d %d %d\n", r, g, b, H, S, V, hbin, sbin, vbin);

	return hbin + nr_bins*(sbin + nr_bins*vbin);
}

int SEEDS::LAB2bin(float l, float a, float b)
{
	// binning
	int bin_l = floor(l / 100.1*nr_bins);

	int bin_a = floor(a / 100 * (nr_bins - 2) + (nr_bins - 2) / 2 + 1);
	if (bin_a < 0) bin_a = 0;
	if (bin_a >= nr_bins) bin_a = nr_bins - 1;

	int bin_b = floor(b / 100 * (nr_bins - 2) + (nr_bins - 2) / 2 + 1);
	if (bin_b < 0) bin_b = 0;
	if (bin_b >= nr_bins) bin_b = nr_bins - 1;

	// encoding
	return bin_l + nr_bins*bin_a + nr_bins*nr_bins*bin_b;
}


int SEEDS::RGB2LAB_special(int r, int g, int b, float* lval, float* aval, float* bval)
{
	float xVal = 0.412453 * r + 0.357580 * g + 0.180423 * b;
	float yVal = 0.212671 * r + 0.715160 * g + 0.072169 * b;
	float zVal = 0.019334 * r + 0.119193 * g + 0.950227 * b;

	xVal /= (255.0 * 0.950456);
	yVal /= 255.0;
	zVal /= (255.0 * 1.088754);

	float fY, fZ, fX;
	float lVal, aVal, bVal;
	float T = 0.008856;

	bool XT = (xVal > T);
	bool YT = (yVal > T);
	bool ZT = (zVal > T);

	fX = XT * pow(xVal, 1.0 / 3.0) + (!XT) * (7.787 * xVal + 16 / 116);

	// Compute L
	float Y3 = pow(yVal, 1.0 / 3.0);
	fY = YT*Y3 + (!YT)*(7.787*yVal + 16 / 116);
	lVal = YT * (116 * Y3 - 16.0) + (!YT)*(903.3*yVal);

	fZ = ZT*pow(zVal, 1.0 / 3.0) + (!ZT)*(7.787*zVal + 16 / 116);

	// Compute a and b
	aVal = 500 * (fX - fY);
	bVal = 200 * (fY - fZ);

	*lval = lVal;
	*aval = aVal;
	*bval = bVal;

	int bin1 = 0;
	int bin2 = 0;
	int bin3 = 0;

	while (lVal > bin_cutoff1[bin1]) {
		bin1++;
	}
	while (aVal > bin_cutoff2[bin2]) {
		bin2++;
	}
	while (bVal > bin_cutoff3[bin3]) {
		bin3++;
	}

	return bin1 + nr_bins*bin2 + nr_bins*nr_bins*bin3;
}


int SEEDS::RGB2LAB_special(int r, int g, int b, int* bin_l, int* bin_a, int* bin_b)
{
	float xVal = 0.412453 * r + 0.357580 * g + 0.180423 * b;
	float yVal = 0.212671 * r + 0.715160 * g + 0.072169 * b;
	float zVal = 0.019334 * r + 0.119193 * g + 0.950227 * b;

	xVal /= (255.0 * 0.950456);
	yVal /= 255.0;
	zVal /= (255.0 * 1.088754);

	float fY, fZ, fX;
	float lVal, aVal, bVal;
	float T = 0.008856;

	bool XT = (xVal > T);
	bool YT = (yVal > T);
	bool ZT = (zVal > T);

	fX = XT * pow(xVal, 1.0 / 3.0) + (!XT) * (7.787 * xVal + 16 / 116);

	// Compute L
	float Y3 = pow(yVal, 1.0 / 3.0);
	fY = YT*Y3 + (!YT)*(7.787*yVal + 16 / 116);
	lVal = YT * (116 * Y3 - 16.0) + (!YT)*(903.3*yVal);

	fZ = ZT*pow(zVal, 1.0 / 3.0) + (!ZT)*(7.787*zVal + 16 / 116);

	// Compute a and b
	aVal = 500 * (fX - fY);

	bVal = 200 * (fY - fZ);

	//*lval = lVal;
	//*aval = aVal;
	//*bval = bVal;

	int bin1 = 0;
	int bin2 = 0;
	int bin3 = 0;

	while (lVal > bin_cutoff1[bin1]) {
		bin1++;
	}
	while (aVal > bin_cutoff2[bin2]) {
		bin2++;
	}
	while (bVal > bin_cutoff3[bin3]) {
		bin3++;
	}

	*bin_l = bin1;
	*bin_a = bin2;
	*bin_b = bin3;

	return bin1 + nr_bins*bin2 + nr_bins*nr_bins*bin3;
}

int SEEDS::threebythree(int x, int y, UINT label)
{
	int count = 0;
	const UINT* ptr_label = labels[seeds_top_level];
	const int y_width = y*width;
	if (ptr_label[y_width - width + x - 1] == label) count++;
	if (ptr_label[y_width - width + x] == label) count++;
	if (ptr_label[y_width - width + x + 1] == label) count++;

	if (ptr_label[y_width + x - 1] == label) count++;
	if (ptr_label[y_width + x + 1] == label) count++;

	if (ptr_label[y_width + width + x - 1] == label) count++;
	if (ptr_label[y_width + width + x] == label) count++;
	if (ptr_label[y_width + width + x + 1] == label) count++;

	return count;
}

int SEEDS::threebyfour(int x, int y, UINT label)
{
	int count = 0;
	const UINT* ptr_label = labels[seeds_top_level];
	const int y_width = y*width;
	if (ptr_label[y_width - width + x - 1] == label) count++;
	if (ptr_label[y_width - width + x] == label) count++;
	if (ptr_label[y_width - width + x + 1] == label) count++;
	if (ptr_label[y_width - width + x + 2] == label) count++;

	if (ptr_label[y_width + x - 1] == label) count++;
	if (ptr_label[y_width + x + 2] == label) count++;

	if (ptr_label[y_width + width + x - 1] == label) count++;
	if (ptr_label[y_width + width + x] == label) count++;
	if (ptr_label[y_width + width + x + 1] == label) count++;
	if (ptr_label[y_width + width + x + 2] == label) count++;

	return count;
}

int SEEDS::fourbythree(int x, int y, UINT label)
{
	int count = 0;
	const UINT* ptr_label = labels[seeds_top_level];
	const int y_width = y*width;
	if (ptr_label[y_width - width + x - 1] == label) count++;
	if (ptr_label[y_width - width + x] == label) count++;
	if (ptr_label[y_width - width + x + 1] == label) count++;

	if (ptr_label[y_width + x - 1] == label) count++;
	if (ptr_label[y_width + x + 2] == label) count++;

	if (ptr_label[y_width + width + x - 1] == label) count++;
	if (ptr_label[y_width + width + x + 2] == label) count++;

	if (ptr_label[y_width + width * 2 + x - 1] == label) count++;
	if (ptr_label[y_width + width * 2 + x] == label) count++;
	if (ptr_label[y_width + width * 2 + x + 1] == label) count++;

	return count;
}


float SEEDS::intersection(int level1, int label1, int level2, int label2)
{
	//intersection of 2 histograms: take the smaller value in each bin
	//and return the sum
	int sum1 = 0, sum2 = 0;
	int* h1 = histogram[level1][label1];
	int* h2 = histogram[level2][label2];
	const int count1 = T[level1][label1];
	const int count2 = T[level2][label2];

	for (int n = 0; n < histogram_size; n++)
	{
		if (*h1 * count2 < *h2 * count1) sum1 += *h1;
		else sum2 += *h2;
		++h1;
		++h2;
	}

	return ((float)sum1) / (float)count1 + ((float)sum2) / (float)count2;
}

float SEEDS::calculateDisparityDifference(int level1, int label1, int level2, int label2){
	
	float mean1 = disparity_mean[level1][label1];
	float mean2 = disparity_mean[level2][label2];
	return fabs(mean1 - mean2);
}

bool SEEDS::check_split(int a11, int a12, int a13, int a21, int a22, int a23, int a31, int a32, int a33, bool horizontal, bool forward)
{
	if (horizontal)
	{
		if (forward)
		{
			if ((a22 != a21) && (a22 == a12) && (a22 == a32)) return true;
			if ((a22 != a11) && (a22 == a12) && (a22 == a21)) return true;
			if ((a22 != a31) && (a22 == a32) && (a22 == a21)) return true;
		}
		else /*backward*/
		{
			if ((a22 != a23) && (a22 == a12) && (a22 == a32)) return true;
			if ((a22 != a13) && (a22 == a12) && (a22 == a23)) return true;
			if ((a22 != a33) && (a22 == a32) && (a22 == a23)) return true;
		}
	}
	else /*vertical*/
	{
		if (forward)
		{
			if ((a22 != a12) && (a22 == a21) && (a22 == a23)) return true;
			if ((a22 != a11) && (a22 == a21) && (a22 == a12)) return true;
			if ((a22 != a13) && (a22 == a23) && (a22 == a12)) return true;
		}
		else /*backward*/
		{
			if ((a22 != a32) && (a22 == a21) && (a22 == a23)) return true;
			if ((a22 != a31) && (a22 == a21) && (a22 == a32)) return true;
			if ((a22 != a33) && (a22 == a23) && (a22 == a32)) return true;
		}
	}

	return false;
}

int SEEDS::count_superpixels()
{
	int* count_labels = new int[nr_labels[seeds_top_level]];
	for (UINT i = 0; i < nr_labels[seeds_top_level]; i++)
		count_labels[i] = 0;
	for (int i = 0; i < width*height; i++)
		count_labels[labels[seeds_top_level][i]] = 1;
	int count = 0;
	for (UINT i = 0; i < nr_labels[seeds_top_level]; i++)
		count += count_labels[i];

	delete count_labels;
	return count;
}


void SEEDS::SaveLabels_Text(string filename)
{

	ofstream outfile;
	outfile.open(filename.c_str());
	int i = 0;
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width - 1; w++)
		{
			outfile << labels[seeds_top_level][i] << " ";
			i++;
		}
		outfile << labels[seeds_top_level][i] << endl;
		i++;
	}
	outfile.close();
}

void SEEDS::SaveParentLabels_Text(string filename, int level){
	ofstream outfile;
	outfile.open(filename.c_str());
	for (int i = 0; i < sizeof(parent[level]) / sizeof(parent[level][0]); ++i) {
		outfile << parent[level][i] << " ";
	}
	outfile << endl;
	outfile.close();
}


void SEEDS::compute_mean_map()
{
	means = new UINT[width*height];

	for (int i = 0; i < width*height; i++)
	{
		int label = labels[seeds_top_level][i];
		float L = 100.0 * ((float)L_channel[label]) / T[seeds_top_level][label];
		float a = 255.0 * ((float)A_channel[label]) / T[seeds_top_level][label] - 128.0;
		float b = 255.0 * ((float)B_channel[label]) / T[seeds_top_level][label] - 128.0;
		int R, G, B;
		LAB2RGB(L, a, b, &R, &G, &B);
		means[i] = B | (G << 8) | (R << 16);
	}
}


void SEEDS::findcontour()
{
	int sz = width*height;
	UINT label_count = labels[seeds_top_level][sz - 1];
	vertex.resize(label_count + 1);

	vector<int> istaken(sz, 0);
	vector<bool> isfirst(label_count + 2, false);
	vector<int > labelfirst(label_count + 2, 0);
	int mainindex = 0;
	int* tempLabel;
	tempLabel = new int[(width + 2)*(height + 2)];
	memset(tempLabel, -1, sizeof(int)*(width + 2)*(height + 2));
	for (int i = 1; i < width+1; ++i)
		for (int t = 1; t < height+1; ++t)
			tempLabel[t*(width+2) + i] = labels[seeds_top_level][(t-1)*width + i-1];
	for (int j = 0; j < height; j++) {
		for (int k = 0; k < width; k++) {
			int num = labels[seeds_top_level][mainindex];
			if (isfirst[num] == false) {
				isfirst[num] = true;
				labelfirst[num] = mainindex;
			}
			if (j == 0 || j == height - 1 || k == 0 || k == width - 1) {
				if ((j == 0 && k == 0) || (j == 0 && k == width - 1) || (j == height - 1 && k == 0) || (j == height - 1 && k == width - 1)) {
					istaken[mainindex] = 1;
				}
				else if (j == 0 || j == height - 1) {
					if (labels[seeds_top_level][mainindex] != labels[seeds_top_level][mainindex - 1] || labels[seeds_top_level][mainindex] != labels[seeds_top_level][mainindex + 1])
						istaken[mainindex] = 1;
					else istaken[mainindex] = 2;
				}
				else if (k == 0 || k == width - 1) {
					if (labels[seeds_top_level][mainindex] != labels[seeds_top_level][mainindex - width] || labels[seeds_top_level][mainindex] != labels[seeds_top_level][mainindex + width])
						istaken[mainindex] = 1;
					else istaken[mainindex] = 2;
				}
				else {
					istaken[mainindex] = 2;
				}
			}
			else {
				int count_temp = 0;
				UINT p_center = labels[seeds_top_level][mainindex];
				UINT p_up = labels[seeds_top_level][mainindex - width];
				UINT p_down = labels[seeds_top_level][mainindex + width];
				UINT p_right = labels[seeds_top_level][mainindex + 1];
				UINT p_left = labels[seeds_top_level][mainindex - 1];
				if (labels[seeds_top_level][mainindex - width - 1] != p_center) count_temp++;
				if (p_up != p_center)count_temp++;
				if (labels[seeds_top_level][mainindex - width + 1] != p_center) count_temp++;
				if (p_left != p_center)count_temp++;
				if (p_right != p_center)count_temp++;
				if (labels[seeds_top_level][mainindex + width - 1] != p_center)count_temp++;
				if (p_down != p_center)count_temp++;
				if (labels[seeds_top_level][mainindex + width + 1] != p_center)count_temp++;
				int num = labels[seeds_top_level][mainindex];
				if (count_temp >= 1) {
					istaken[mainindex] = count_temp;
					if (p_up != p_down&&p_left != p_right) {
						istaken[mainindex] = 4;
					}
				}
			}
			mainindex++;

		}
	}
	for (int i = 0; i <= label_count; i++) {
		int now = labelfirst[i];
		int last;
		int temp;
		if (labels[seeds_top_level][now + 1] == i && istaken[now + 1] >0) {//right
			if (istaken[now] == 1 || istaken[now] > 3) {
				vertex[i].u.push_back(now % width);
				vertex[i].v.push_back(now / width);
				vertex[i].num++;
			}
			temp = now;
			now = now + 1;
			last = temp;
		}
		//else if (labels[seeds_top_level][now + width] == i && istaken[now + width] >0) {//down
		//	if (istaken[now] == 1 || istaken[now]>3) {
		//		vertex[i].u.push_back(now % width - 0.5);
		//		vertex[i].v.push_back(now / width - 0.5);
		//		vertex[i].num++;
		//	}
		//	temp = now;
		//	now = now + width;
		//	last = temp;
		//	cout << "1111111111111111" << endl;
		//}

		while (now != labelfirst[i]) {
			
			if (now < width*height && labels[seeds_top_level][now + 1] == i && istaken[now + 1] >0 && last - 1 != now) {//right
				if (istaken[now] == 1 || istaken[now] > 3) {
					vertex[i].u.push_back(now % width);
					vertex[i].v.push_back(now / width);
					vertex[i].num++;
				}

				temp = now;
				now = now + 1;
				last = temp;
			}
			else if (now <= (width * (height - 1)) && labels[seeds_top_level][now + width] == i && istaken[now + width] >0 && last - width != now) {//down
				if (istaken[now] == 1 || istaken[now]>3) {
					vertex[i].u.push_back(now % width);
					vertex[i].v.push_back(now / width);
					vertex[i].num++;
				}
				temp = now;
				now = now + width;
				last = temp;
			}
			else if (now != 0 && labels[seeds_top_level][now - 1] == i && istaken[now - 1]>0 && last + 1 != now) {//left
				if (istaken[now] == 1 || istaken[now]>3) {
					vertex[i].u.push_back(now % width);
					vertex[i].v.push_back(now / width);
					vertex[i].num++;
				}
				temp = now;
				now = now - 1;
				last = temp;
			}
			else if (now > width - 1 && labels[seeds_top_level][now - width] == i && istaken[now - width] >0 && last + width != now) {//up
				if (istaken[now] == 1 || istaken[now] >3) {
					vertex[i].u.push_back(now % width);
					vertex[i].v.push_back(now / width);
					vertex[i].num++;
				}
				temp = now;
				now = now - width;
				last = temp;
			}
			/*else {
				printf("%d %d\n", i, now);
			}*/

		}
		int vertexSize = vertex[i].num;
		double in = 1.0;
		for (int vz = 0;vz < vertexSize;vz++) {
			int verX = vertex[i].u[vz], verY = vertex[i].v[vz];
			// 左上
			if (tempLabel[verY * (width + 2) + verX] !=  labels[seeds_top_level][verY*width + verX]) {
				vertex[i].u[vz] -= in;
				vertex[i].v[vz] -= in;
			}
			// 右上
			if (tempLabel[verY * (width + 2) + verX + 2] != labels[seeds_top_level][verY*width + verX]) {
				vertex[i].u[vz] += in;
				vertex[i].v[vz] -= in;
			}
			// 左下
			if (tempLabel[(verY+2) * (width + 2) + verX] != labels[seeds_top_level][verY*width + verX]) {
				vertex[i].u[vz] -= in;
				vertex[i].v[vz] += in;
			}
			// 右下
			if (tempLabel[(verY+2) * (width + 2) + verX+2] != labels[seeds_top_level][verY*width + verX]) {
				vertex[i].u[vz] += in;
				vertex[i].v[vz] += in;
			}
		}
	}
	/*string outfileName = "F://dataset//data_stereo_flow//testing//vertex//000001_10.txt";
	ofstream outfile(outfileName.c_str(), ios::out);
	for (int j = 0; j <= label_count; j++) {
		int vertexnum = vertex[j].num;
		outfile << j << " " << vertexnum << endl;
		for (int k = 0; k < vertexnum - 1; k++) {
			outfile << vertex[j].u[k] << " " << vertex[j].v[k] << "     ";
		}
		outfile << vertex[j].u[vertexnum - 1] << " " << vertex[j].v[vertexnum - 1];
		outfile << endl << endl;
	}
	outfile.close();*/
}

void SEEDS::planeCalculateVertex(std::vector<VERTEX>& result) {
	int size = vertex.size();
	result.resize(size);
	for (int i = 0; i < size; ++i) {
		for (int t = 0; t < vertex[i].u.size(); ++t) {
			/*double disparity = (double)(cvGet2D(disparityImage, vertex[i].v[t], vertex[i].u[t]).val[0]);
			double xTemp = ((double)vertex[i].u[t] - param_cu) * param_base / disparity;
			double yTemp = ((double)vertex[i].v[t] - param_cv) * param_base / disparity;
			double zTemp = param_f*param_base / disparity;*/
			//if (vertex[i].u[t] < 0 || vertex[i].v[t] < 0)
			//	cout << vertex[i].u[t] << " " << vertex[i].v[t] << endl;
			result[i].u.push_back(vertex[i].u[t]);   //将每一个seg边界的uv放入result中。也就是第i个seg所具有的的轮廓都加进去
			result[i].v.push_back(vertex[i].v[t]);
		}
	}
}
