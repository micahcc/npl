/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file convertDeform.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "nplio.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using std::string;
using namespace npl;
using std::shared_ptr;

template <typename T>
ostream& operator<<(ostream& out, const std::vector<T>& v)
{
	out << "[ ";
	for(size_t ii=0; ii<v.size()-1; ++ii)
		out << v[ii] << ", ";
	out << v[v.size()-1] << " ]";
	return out;
}

/**
 * @brief Inverts a deformation, given a mask in the target space.
 *
 * Mathematical basis for this function is that in image S (subject)
 * we have a mapping from S to A (atlas), which we will call u,
 * and in image A we have a map from A to S (v) that we want to optimize. The
 * goal is to :
 *
 * minimize
 * ||u+v||^2
 *
 * @param deform Maps points, vector is assumed to be in physical space
 *
 * @return
 */
ptr<MRImage> invertForwardBack(ptr<MRImage> deform,
		size_t MAXITERS, double MINERR)
{
	if(deform->ndim() != 3 || deform->tlen() != 3) {
		cerr << "Error invalid deform image, needs 3 points in the 4th or "
			"5th dim" << endl;
		return NULL;
	}

	// create output the size of atlas, with 3 volumes in the 4th dimension
	auto idef = dPtrCast<MRImage>(deform->createAnother(FLOAT64));

	LinInterp3DView<double> definterp(deform);
	const double LAMBDA = 0.1;
	double cind[3];
	double err[3];
	double origpt[3]; // point in origin 
	double defpt[3]; // deformed point
	double fwd[3]; // forward vector
	double rev[3]; // reverse vector
	KDTree<3,3,double, double> tree;

	/*
	 * Construct KDTree indexed by atlas-space indices, and storing indices
	 * in subject (mask) space. Only do it if mask value is non-zero however
	 */
	for(Vector3DIter<double> dit(idef); !dit.eof(); ++dit) {
		dit.index(3, cind);
		deform->indexToPoint(3, cind, origpt);

		for(int ii=0; ii<3; ++ii) {
			fwd[ii] = dit[ii];
			rev[ii] = -dit[ii];
			defpt[ii] = origpt[ii] + fwd[ii];
		}

		// add point to kdtree, use atl2sub since this will be our best guess
		// of atl2sub when we pull the point out of the tree
		tree.insert(3, defpt, 3, rev);
	}
	tree.build();

	/*
	 * In atlas image try to find the correct source in the subject by first
	 * finding a point from the KDTree then improving on that result
	 */
	// set atlas deform to NANs
	for(FlatIter<double> iit(idef); !iit.eof(); ++iit)
		iit.set(NAN);

	// at each point in atlas try to find the best mapping in the subject by
	// going back and forth. Since the mapping from sub to atlas is ground truth
	// we need to keep checking until we find a point in the subject that maps
	// to our current atlas location
	assert(iit.tlen() == 3);
	for(Vector3DIter<double> iit(idef); !iit.eof(); ++iit) {

		iit.index(3, cind);
		idef->indexToPoint(3, cind, defpt);

		// that map from outside, then compute the median of the remaining
		double dist = INFINITY;
		auto results = tree.nearest(3, defpt, dist);

		// ....no...sort
		if(!results)
			continue;

		for(size_t dd=0; dd<3; dd++)
			rev[dd] = results->m_data[dd];

		// SUB <- ATLAS (given)
		//    atl2sub
		double prevdist = dist+1;
		size_t iters = 0;
		for(iters = 0 ; fabs(prevdist-dist) > 0 && dist > MINERR &&
						iters < MAXITERS; iters++) {

			for(size_t ii=0; ii<3; ii++)
				origpt[ii] = defpt[ii] + rev[ii];

			// (estimate) SUB <- ATLAS (given)
			//              offset
			// interpolate new offset at subpoint
			deform->pointToIndex(3, origpt, cind);

			/* Update */

			// update sub2atl
			for(size_t ii=0; ii<3; ii++)
				fwd[ii] = definterp(cind[0], cind[1], cind[2], ii);

			// update image with the error, using the derivative to estimate
			// where error crosses 0
			prevdist = dist;
			dist = 0;
			for(size_t ii=0; ii<3; ii++)
				err[ii] = rev[ii]+fwd[ii];

			for(size_t ii=0; ii<3; ii++) {
				rev[ii] -= LAMBDA*err[ii];
				dist += err[ii]*err[ii];
			}
			dist = sqrt(dist);
		}


		// save out final deform
		for(size_t ii=0; ii<3; ++ii)
			iit.set(ii, rev[ii]);

	}

	return idef;
}

/**
 * @brief Computes the overlap of the two images' in 3-space.
 *
 * @param a Image
 * @param b Image
 *
 * @return Ratio of b that overlaps with a's grid
 */
double overlapRatio(shared_ptr<MRImage> a, shared_ptr<MRImage> b)
{
	int64_t index[3];
	double point[3];
	size_t incount = 0;
	size_t maskcount = 0;
	for(OrderIter<int64_t> it(a); !it.eof(); ++it) {
		it.index(3, index);
		a->indexToPoint(3, index, point);
		++maskcount;
		incount += (b->pointInsideFOV(3, point));
	}
	return (double)(incount)/(double)(maskcount);
}

shared_ptr<MRImage> indexMapToOffsetMap(shared_ptr<const MRImage> defimg,
		shared_ptr<const MRImage> atlas, bool one_based_indexing = false)
{
	shared_ptr<MRImage> out = dynamic_pointer_cast<MRImage>(defimg->copy());

	Vector3DIter<double> it(out);
	if(it.tlen() != 3)
		cerr << "Error expected 3 volumes!" << endl;
	int64_t index[3];
	double cindex[3];
	double pointS[3]; //subject
	double pointA[3]; //atlas
	for(it.goBegin(); !it.eof(); ++it) {
		// fill index with coordinate
		it.index(3, index);
		out->indexToPoint(3, index, pointS);

		// convert source pixel to pointA
		for(size_t ii=0; ii<3; ++ii) {
			cindex[ii] = it[ii]-one_based_indexing;
		}
		//since values in deformation vector are indices in atlas space
		atlas->indexToPoint(3, cindex, pointA);
		// set value to offset
		// store sub2atl (vector going from subject to atlas space)
		for(int64_t ii=0; ii < 3; ++ii)
			it.set(ii, pointA[ii] - pointS[ii]);
	}

	return out;
}

void binarize(shared_ptr<MRImage> in)
{
	OrderIter<int> it(in);
	for(it.goBegin(); !it.eof(); ++it) {
		if(*it != 0)
			it.set(1);
	}

}

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Converts 3D deformation to a map of offsets. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_indef("i", "input", "Input deformation.",
			true, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_atlas("a", "atlas", "Atlas (mask) image. Needed "
			"if you provide an index-based deformation or if you intend to "
			"extrapolate '-E'. Note that if you already have a valid "
			"deformation for the entire FOV, and the input is a offset type "
			"deform, then neither this nor mask '-m' are needed", false, "",
			"*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);

	TCLAP::SwitchArg a_invert("I", "invert", "index "
			"lookup type deform to an offset (in mm) type deform", cmd);
	
	TCLAP::SwitchArg a_in_offset("", "in-offset", "Input an offset (in mm) "
			"type deform", cmd);
	TCLAP::SwitchArg a_in_index("", "in-index", "Input an index type deform. "
			"Must provide an atlas if this is the case '-a'", cmd);
	
	TCLAP::SwitchArg a_out_offset("", "out-offset", "Output an offset "
			"type deform. (in mm/physical units)", cmd);
	TCLAP::SwitchArg a_out_index("", "out-index", "Output an input "
			"type deform", cmd);

	TCLAP::SwitchArg a_one_index("1", "one-based-index", "When referring to "
			"indexes make them one-based (MATLAB standard).", cmd);

	TCLAP::ValueArg<double> a_improve("", "delta", "Amount of improvement in "
			"estimate before stopping during inverse.", false, 0.1, "DNORM", cmd);

	cmd.parse(argc, argv);
	std::shared_ptr<MRImage> deform;
	std::shared_ptr<MRImage> atlas;

	/**********
	 * Input
	 *********/
	deform = readMRImage(a_indef.getValue());
	if(deform->ndim() < 4 || deform->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}
	
	if(a_atlas.isSet()) {
		atlas = readMRImage(a_atlas.getValue());
		binarize(atlas);
	}

	// convert to offset
	if(a_in_index.isSet()) {
		if(!atlas) {
			cerr << "Must provide an atlas for index-based deformations, "
				"otherwise there is no way to know what the indexes mean!"
				<< endl;
			return -1;
		}

		cerr << "Converting from index map." << endl;
		if(a_one_index.isSet()) {
			cerr << "Indexes start at 1" << endl;
		} else {
			cerr << "Indexes start at 0" << endl;
		}
		// convert deform to RAS space offsets
		deform = indexMapToOffsetMap(deform, atlas, a_one_index.isSet());
		deform->write("offset.nii.gz");
	}

	// invert
	if(a_invert.isSet()) {
		cerr << "Inverting" << endl;
		deform = invertForwardBack(deform, 15, 0.1);
		cerr << "Done" << endl;

		deform->write("invert.nii.gz");
	}

	// convert to correct output type
	if(a_out_index.isSet()) {
		cerr << "Changing back to index not yet implemented " << endl;
		return -1;
	}

	// write
	deform->write(a_out.getValue());
	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

