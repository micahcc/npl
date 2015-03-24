/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file indexMapToOffsetMap.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "kernel_slicer.h"
#include "kdtree.h"
#include "slicer.h"
#include "kernel_slicer.h"

using std::string;
using namespace npl;
using std::shared_ptr;

template <typename T>
ostream& operator<<(ostream& out, const std::vector<T>& v)
{
	out << "[ ";
	for(size_t ii=0; ii<v.size()-1; ii++)
		out << v[ii] << ", ";
	out << v[v.size()-1] << " ]";
	return out;
}

void invertDeform(shared_ptr<MRImage> inimg, size_t vdim,
		shared_ptr<MRImage> outimg)
{
	// create KDTree
	KDTree<3, 3, float, float> tree;
	const double MINERR = .1;
	const double LAMBDA = .1;
	const size_t MAXITERS = 300;
	double prevdist = 0;
	vector<float> err(3,0);
	vector<float> tmp(3,0);
	vector<float> src(3,0);
	vector<float> trg(3,0);
	vector<int64_t> index(inimg->ndim(), 0);
	vector<double> cindex(inimg->ndim(), 0);

	// map every current point back to its source, in the output image
	for(index[0]=0; index[0]<inimg->dim(0); index[0]++) {
		for(index[1]=0; index[1]<inimg->dim(1); index[1]++) {
			for(index[2]=0; index[2]<inimg->dim(2); index[2]++) {

				// get full vector,
				for(int64_t ii=0; ii<3; ii++) {
					index[vdim] = ii;
					src[ii] = inimg->get_dbl(index);
					trg[ii] = index[ii];
				}

				// add point to kdtree
				tree.insert(src, trg);
			}
		}
	}

	// Tree Stores [target] = source
	std::cerr << "Building Tree" << endl;
	tree.build();
	cerr << "Done" << endl;

	cerr << "Inverting Map" << endl;
	// go through output image and find nearby points that map
	for(index[0]=0; index[0] < outimg->dim(0); index[0]++) {
		std::cerr << index[0] << "/" << outimg->dim(0) << "\r";
		for(index[1]=0; index[1] < outimg->dim(1); index[1]++) {
			for(index[2]=0; index[2] < outimg->dim(2); index[2]++) {
				for(int64_t ii=0; ii<3; ii++)
					trg[ii] = index[ii];

				// find point that maps near the current in
				double dist = INFINITY;
				auto result = tree.nearest(trg, dist);

				if(!result)
					throw std::logic_error("Deformation too large!");

#ifdef DEBUG
				std::cerr  << "Found" << std::endl;
				for(int ii=0 ; ii < 3 ;ii++) {
					std::cerr << result->m_point[ii] << ",";
				}
				std::cerr << " <- ";
				for(int ii=0 ; ii < 3 ;ii++) {
					std::cerr << result->m_data[ii] << ",";
				}
				std::cerr << std::endl;
#endif //DEBUG

				// intiailize theoretical source
				for(int64_t jj=0; jj<3; jj++) {
					src[jj] = result->m_data[jj];
				}

				// use the error in app_target vs target to find a
				// source that fits
				prevdist = dist+1;
				for(size_t ii = 0 ; fabs(prevdist-dist) > 0 &&
						dist > MINERR && ii < MAXITERS; ii++) {
#ifdef DEBUG
					std::cerr << trg  << " <- ?? " << src << " ?? " << endl;
#endif //DEBUG

					// forward map computed src (fwdtrg)
					// load new source into cindex
					for(size_t jj=0; jj<3; jj++) {
						cindex[jj] = src[jj];
					}

					bool outside = false;
					for(size_t jj=0; jj<3; jj++) {
						cindex[vdim] = jj;
						tmp[jj] = inimg->linSampleInd(cindex, CONSTZERO, outside);
					}
#ifdef DEBUG
					std::cerr << trg  << " vs " << tmp << " <- " << src << endl;
#endif //DEBUG
					prevdist = dist;
					dist = 0;
					for(size_t jj=0; jj<3; jj++) {
						err[jj] = trg[jj]-tmp[jj];
						dist += err[jj]*err[jj];
						src[jj] = src[jj]+LAMBDA*err[jj];
					}
					dist = sqrt(dist);
#ifdef DEBUG
				std::cerr << "Err (" << dist << ") " << err << endl;
#endif //DEBUG

				}

#ifdef DEBUG
				std::cerr << "Found: " << trg  << " <- " << src << endl;
#endif //DEBUG

				// set the inverted source in the output
				for(size_t ii=0; ii<3; ii++) {
					index[vdim] = ii;
					outimg->set_dbl(index, src[ii]);
				}

			}
		}
	}

	outimg->write("omap.nii.gz");
	cerr << "Done" << endl;
}

/**
 * @brief This function sets distortion levels outside of the masked region
 * to the nearest within-mask value
 *
 * @param deform
 * @param mask
 */
void growFromMask(shared_ptr<MRImage> deform, shared_ptr<MRImage> mask, size_t vdim)
{
	cerr << "Changing Outside Points to match nearest inside-mask point" << endl;

	// create kernel iterator, radius 1
	std::vector<size_t> krad(mask->ndim(), 1);
	KSlicer kern(mask->ndim(), mask->dim(), krad);
	std::vector<int64_t> index;
	std::vector<int64_t> dindex1(deform->ndim(), 0);
	std::vector<int64_t> dindex2(deform->ndim(), 0);
	size_t minval;
	int nchange = 1;
	int visitations = 1;

	// need an image to write to so that we don't detect our own changes
	auto mask_trg = mask->cloneImage();

	while(nchange > 0) {
		nchange = 0;
		visitations = 0;
		for(kern.goBegin(); !kern.isEnd(); ++kern) {

			// skip values in mask, or that have been reached
			if(mask->get_int(kern.center()) != 0)
				continue;

			visitations++;

			// find minimum non-zero point in kernel
			minval = SIZE_MAX;
			std::vector<int64_t> minind(mask->ndim());
			for(size_t ii=0; ii<kern.ksize(); ii++) {
				int v = mask->get_int(kern[ii]);
				if(v > 0 && v < minval) {
					minval = v;
					minind = kern.offset_index(ii);
				}
			}

			// copy value at index to deform, and set middle to minv+1
			if(minval < SIZE_MAX) {
				mask_trg->set_int(kern.center(), minval+1);

				// convert to point
				index = kern.center_index();

				// move from 1 (min) to 2 (center)
				for(size_t ii=0; ii<3; ii++) {
					dindex1[ii] = minind[ii];
					dindex2[ii] = index[ii];
				}

				for(size_t ii=0; ii<3; ii++) {
					dindex1[vdim] = ii;
					dindex2[vdim] = ii;
					deform->set_dbl(dindex2, deform->get_dbl(dindex1));
				}

				nchange++;
			}
		}

		// copy changes back to source
		for(size_t ii=0; ii<mask->elements(); ii++)
			mask->set_int(ii, mask_trg->get_int(ii));

		cerr << "Number Changed: " << nchange << endl;
		cerr << "Visited: " << visitations << endl;
	}
}

shared_ptr<MRImage> erode(shared_ptr<MRImage> in, size_t reps)
{
	auto out = in->cloneImage();
	std::vector<size_t> krad(out->ndim(), 1);
	KSlicer kernel(in->ndim(), in->dim(), krad);

	for(size_t ii=0; ii<reps; ii++) {
		// for each pixels neighborhood, smooth neightbors
		for(kernel.goBegin(); !kernel.isEnd(); ++kernel) {

			// if any of the neighbors are 0, then set to 0
			bool erodeme = false;
			for(size_t ii=0; ii<kernel.ksize(); ii++) {
				if(in->get_int(kernel.offset(ii)) == 0)
					erodeme = true;
			}

			if(erodeme)
				out->set_int(*kernel, 0);
			else
				out->set_int(*kernel, in->get_int(*kernel));

		}
	}

	return out;
}

shared_ptr<MRImage> voteDilate(shared_ptr<MRImage> in, size_t reps)
{
	std::vector<size_t> krad(in->ndim(), 1);
	KSlicer kernel(in->ndim(), in->dim(), krad);

	auto out = in->cloneImage();
	for(size_t ii=0; ii<reps; ii++) {
		cerr << "Dilate " << ii << endl;
		auto prev = out->cloneImage();
		// for each pixels neighborhood, smooth neightbors
		for(kernel.goBegin(); !kernel.isEnd(); ++kernel) {

			// if any of the neighbors are 0, then set to 0
			int vote = 0;
			int dilval = 0;
			for(size_t ii=0; ii<kernel.ksize(); ii++) {
				if(prev->get_int(kernel.offset(ii)) != 0) {
					dilval = prev->get_int(kernel.offset(ii));
					vote++;
				}
			}

			if(vote >= (kernel.ksize()+1)/2) {
				out->set_int(*kernel, dilval);
			}
		}

		prev = out;
	}

	return out;
}

shared_ptr<MRImage> dilate(shared_ptr<MRImage> in, size_t reps)
{
	std::vector<size_t> krad(in->ndim(), 1);
	KSlicer kernel(in->ndim(), in->dim(), krad);

	auto out = in->cloneImage();
	for(size_t ii=0; ii<reps; ii++) {
		cerr << "Dilate " << ii << endl;
		auto prev = out->cloneImage();
		// for each pixels neighborhood, smooth neightbors
		for(kernel.goBegin(); !kernel.isEnd(); ++kernel) {

			// if any of the neighbors are 0, then set to 0
			bool dilateme = false;
			int dilval = 0;
			for(size_t ii=0; ii<kernel.ksize(); ii++) {
				if(prev->get_int(kernel.offset(ii)) != 0) {
					dilval = prev->get_int(kernel.offset(ii));
					dilateme = true;
				}
			}

			if(dilateme) {
				out->set_int(*kernel, dilval);
			}
		}

		prev = out;
	}

	return out;
}

shared_ptr<MRImage> smoothOutsideMask(shared_ptr<MRImage> deform,
		shared_ptr<MRImage> mask, int vdim)
{
	auto odeform = deform->cloneImage();
	std::vector<size_t> krad(odeform->ndim(), 1);
	krad[vdim] = 0;

	KSlicer kernel(odeform->ndim(), odeform->dim(), krad);

	// for each pixels neighborhood, smooth neightbors
	for(kernel.goBegin(); !kernel.isEnd(); ++kernel) {

		// if the center is non-zero masked then aveage from neighbors
		if(mask->get_int(kernel.center_index()) == 0) {
			double sum = 0;
			for(size_t ii=0; ii<kernel.ksize(); ii++) {
				sum += deform->get_dbl(kernel.offset(ii));
			}
			odeform->set_dbl(*kernel, sum/kernel.ksize());
		}
	}


	return odeform;
}

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Often nonlinear registration produces deformation fields"
			" which store within each point a source point. Further these are"
			" often in index coordinates rather than RAS coordinates. This tool "
			"will convert a field of source indices into a field of offsets in "
			"world coordinates. For example [x,y,z,] contains a vector with value "
			"[x+a,y+b,z+c]. This is the sum of 3 vectors (x+a)i+(y+b)j+(z+c)k, we "
			"change that to A[a,b,c]^T. Its also possible to invert (-I) .",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			false, "", "*.nii.gz", cmd);
//	TCLAP::ValueArg<string> a_offset("O", "offset-map", "Output offset map.",
//			false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_mask("m", "mask", "Input mask image. Values "
			"outside the masked regions will be recreated by expanding from "
			"masked regions. The resutling corrected distortion field can be "
			"written with -c.", false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_omask("", "updated-mask", "Updated mask image. "
			"Result of dilation.", false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_corr("c", "corrected", "Corrected deformation "
			"field", false, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_invert("I", "invert", "Whether to invert.", cmd);
	TCLAP::ValueArg<string> a_atlas("a", "atlas", "Atlas which will be used "
			"for field of view during inversion.", true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<size_t> a_dilate("d", "dilate", "Number of times to dilate "
			"the input mask.", false, 1, "int", cmd);

	cmd.parse(argc, argv);

	// read input
	std::shared_ptr<MRImage> deform(readMRImage(a_in.getValue()));

	// ensure the image dimensions match our expectations
	size_t vdim = 4;
	if(deform->ndim() == 4 && deform->dim(3) == 3)
		vdim = 3;
	else if(deform->ndim() == 5 && deform->dim(4) == 3)
		vdim = 4;
	else {
		cerr << "Not sure how to handle the dimensionality of this image"
			<< endl;
		cerr << deform->ndim() << " [";
		for(size_t ii=0; ii< deform->ndim(); ii++) {
			cerr << deform->dim(ii) << ",";
		}
		cerr << endl;
		return -1;
	}

	// check that it is 4D or 5D (with time=1)
	vector<int64_t> defInd;
	list<size_t> order({vdim});
	Slicer it(deform->ndim(), deform->dim(), order);

	// propogate values out from masked region
	if(a_mask.isSet()) {

		// convert deform to offsets, otherwise propagating distortion doesn't
		// make sense
		for(it.goBegin(); !it.isEnd(); ) {
			defInd = it.index();
			for(int ii=0; ii<3; ii++, ++it) {
				deform->set_dbl(*it, deform->get_dbl(*it)-defInd[ii]);
			}
		}

		// everything outside the mask takes on the nearest value
		// from inside the mask
		std::shared_ptr<MRImage> mask(readMRImage(a_mask.getValue()));
		if(mask->ndim() != 3) {
			cerr << "Mask should be 3D!" << endl;
			return -1;
		}

		// force the mask to be binary
		for(size_t ii=0; ii<mask->elements(); ii++) {
			if(mask->get_int(ii) != 0)
				mask->set_int(ii, 1);
		}

		cerr << "Dilating: " << a_dilate.getValue() << endl;
		mask = dilate(mask, a_dilate.getValue());
		cerr << "Filling Holes: " << a_dilate.getValue() << endl;
		mask = voteDilate(mask, a_dilate.getValue());
		if(a_omask.isSet()) {
			mask->write(a_omask.getValue());
		}

		// TODO just resample
		// check that orientation/size match
		for(size_t ii = 0; ii<3; ii++) {
			for(size_t jj = 0; jj<3; jj++) {
				if(mask->direction()(ii,jj) != deform->direction()(ii,jj)) {
					std::cerr << "Mask and Deform do not have matching "
						"orientation" << endl;
					return -1;
				}
			}
			if(mask->origin()[ii] != deform->origin()[ii]) {
				std::cerr << "Mask and Deform do not have matching "
					"origin" << endl;
				return -1;
			}
			if(mask->spacing()[ii] != deform->spacing()[ii]) {
				std::cerr << "Mask and Deform do not have matching "
					"spacing" << endl;
				return -1;
			}

			if(mask->dim(ii) != deform->dim(ii)) {
				std::cerr << "Mask and Deform do not have matching "
					"size " << endl;
				return -1;
			}
		}

		auto tmpmask = mask->cloneImage();
		growFromMask(deform, tmpmask, vdim);
		cerr << "Done" << endl;

//		// smooth masked deform
//		size_t SMOOTH_ITERS = 4;
//		for(size_t ii=0; ii<SMOOTH_ITERS; ii++)
//			deform = smoothOutsideMask(deform, mask, vdim);
//		deform = smooth(deform, vdim);

		// convert offset back to deform,
		for(it.goBegin(); !it.isEnd(); ) {
			defInd = it.index();
			for(int ii=0; ii<3; ii++, ++it) {
				deform->set_dbl(*it, deform->get_dbl(*it)+defInd[ii]);
			}
		}

		// write out the corrected deform
		if(a_corr.isSet()) {
			cerr << "Writing smoothed/expanded deform" << endl;
			deform->write(a_corr.getValue());
		}
	}

	shared_ptr<MRImage> newdeform;
	if(a_invert.isSet()) {
		shared_ptr<MRImage> atlas;
		atlas = readMRImage(a_atlas.getValue());

		if(!atlas || atlas->ndim() != 3)
			cerr << "Atlas should be 3D!" << endl;

		// copy size and orientation from atlas
		std::vector<size_t> sz(deform->ndim());
		for(size_t ii=0; ii<3; ii++)
			sz[ii] = atlas->dim(ii);
		for(size_t ii=3; ii<deform->ndim(); ii++)
			sz[ii] = deform->dim(ii);

		cerr << sz << endl;
		newdeform = createMRImage(sz, FLOAT32);
		newdeform->setOrient(atlas->origin(), deform->spacing(),
				atlas->direction(), true);

		invertDeform(deform, vdim, newdeform);
		newdeform->write("inverted.nii.gz");
	} else {
		newdeform = deform;
	}

	// convert each coordinate to an offset
	it.updateDim(newdeform->ndim(), newdeform->dim());
	it.goBegin();
	vector<double> pt1, pt2;
	while(!it.isEnd()) {

		// convert location to point
		auto index = it.index();
		newdeform->indexToPoint(index, pt1);

		// get point at source (value in newdeform, in index space of original)
		for(size_t ii=0; ii<3; ii++, ++it)
			index[ii] = newdeform->get_dbl(*it);

		deform->indexToPoint(index, pt2);

		// set deform to the difference in point locations
		it--; it--; it--;
		for(size_t ii=0; ii<3; ii++, ++it)
			newdeform->set_dbl(*it, pt2[ii]-pt1[ii]);
	}

	newdeform->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
