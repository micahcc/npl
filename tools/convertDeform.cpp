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

void parseDir(string idir, bool& flip, int& dim)
{
	/*
	 * Convert x/y/z to dimension with flip
	 */
	flip = false;
	dim = 0;
	int pos = 0;
	if(idir.empty()) {
		throw INVALID_ARGUMENT("Error Invalid Dimension: "+idir);
	} else if(idir[0] == '-') {
		flip = true;
		pos++;
	} else if(idir[0] == '+') {
		flip = false;
		pos++;
	}

	if(pos >= idir.size())
		throw INVALID_ARGUMENT("Error Invalid Dimension: "+idir);
	else if(idir[pos] == 'x')
		dim = 0;
	else if(idir[pos] == 'y')
		dim = 1;
	else if(idir[pos] == 'z')
		dim = 2;
	else
		throw INVALID_ARGUMENT("Error Invalid Dimension: "+idir);
}

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
	if(deform->ndim() != 4 || deform->tlen() != 3) {
		cerr << "Error invalid deform image, needs 3 points in the 4th or "
			"5th dim" << endl;
		return NULL;
	}

	// create output the size of atlas, with 3 volumes in the 4th dimension
	auto idef = dPtrCast<MRImage>(deform->createAnother(FLOAT64));

	LinInterp3DView<double> definterp(deform);
	const double LAMBDA = 0.5;
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
	for(Vector3DConstIter<double> dit(deform); !dit.eof(); ++dit) {
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
//		cerr << "Deformed Point: ";
//		for(size_t dd=0; dd<3; dd++) {
//			if(dd != 0) cerr << ", ";
//			cerr << defpt[dd];
//		}
//		cerr << endl << "Deformed Value: ";
//		for(size_t dd=0; dd<3; dd++) {
//			if(dd != 0) cerr << ", ";
//			cerr << rev[dd];
//		}
//		cerr << endl;
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
	for(Vector3DIter<double> iit(idef); !iit.eof(); ++iit) {

		iit.index(3, cind);
		idef->indexToPoint(3, cind, defpt);

		// that map from outside, then compute the median of the remaining
		double dist = INFINITY;
		auto results = tree.nearest(3, defpt, dist);

		// ....no...sort
		if(!results) {
			cerr << "No results within distance!" << endl;
			continue;
		}

//		cerr << "Initial Reverse: " << endl;
		for(size_t dd=0; dd<3; dd++) {
//			if(dd != 0) cerr << ", ";
			rev[dd] = results->m_data[dd];
//			cerr << rev[dd];
		}
//		cerr << endl;

//		cerr << "Target Point: ";
//		for(size_t dd=0; dd<3; dd++) {
//			if(dd != 0) cerr << ", ";
//			cerr << defpt[dd];
//		}
//		cerr << endl;

		// SUB <- ATLAS (given)
		//    atl2sub
		double prevdist = dist+1;
		size_t iters = 0;
		for(iters = 0 ; fabs(prevdist-dist) > 0 && dist > MINERR &&
						iters < MAXITERS; iters++) {

//			cerr << "Estimated Source: ";
			for(size_t ii=0; ii<3; ii++) {
//				if(ii != 0) cerr << ", ";
				origpt[ii] = defpt[ii] + rev[ii];
//				cerr << origpt[ii];
			}
//			cerr << endl;

			// (estimate) SUB <- ATLAS (given)
			//              offset
			// interpolate new offset at subpoint
			deform->pointToIndex(3, origpt, cind);

			/* Update */

			// update sub2atl
//			cerr << "Fwd at Source: ";
			for(size_t ii=0; ii<3; ii++) {
//				if(ii != 0) cerr << ", ";
				fwd[ii] = definterp(cind[0], cind[1], cind[2], ii);
//				cerr << fwd[ii];
			}
//			cerr << endl;

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

//			cerr << "New Rev: ";
//			for(size_t ii=0; ii<3; ++ii) {
//				if(ii != 0) cerr << ", ";
//				cerr << rev[ii];
//			}
//			cerr << endl;
			dist = sqrt(dist);
//			cerr << "Err: " << dist << endl;
		}
		if(iters == MAXITERS) {
			cerr << "Warning, failed to converge at " << cind[0] << ", " <<
				cind[1] << ", " << cind[2] << endl;
		}


		// save out final deform
//		cerr << "Estimate: ";
		for(size_t ii=0; ii<3; ++ii) {
//			if(ii != 0) cerr << ", ";
//			cerr << rev[ii];
			iit.set(ii, rev[ii]);
		}
//		cerr << endl;
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
	auto out = dPtrCast<MRImage>(defimg->copy());

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

shared_ptr<MRImage> reorientVectors(shared_ptr<const MRImage> defimg)
{
	auto out = dPtrCast<MRImage>(defimg->createAnother());

	Vector3DConstIter<double> iit(defimg);
	Vector3DIter<double> oit(out);
	if(defimg->tlen() != 3)
		cerr << "Error expected 3 volumes!" << endl;
	double vec[3]; //subject
	for(; !iit.eof() && !oit.eof(); ++oit, ++iit) {
		for(size_t dd=0; dd<3; dd++)
			vec[dd] = iit[dd];

		defimg->orientVector(3, vec, vec);

		for(size_t dd=0; dd<3; dd++)
			oit.set(dd, vec[dd]);
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
	cerr << "Version: " << __version__ << endl;
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

	std::vector<string> allowed({"+x","x", "-x", "+y", "y", "-y", "+z", "z", "-z"});
	TCLAP::ValuesConstraint<string> allowedDirs( allowed );
	TCLAP::ValueArg<string> a_in_dir_space("", "in-uni-space", "Input is a uni-"
			"directional distortion field in spacing (mm) coords. "
			"Direction of distortion may be "
			"-x +x x -y +y y -z z +z, where no +/- implies +. Thus a positive "
			"distortion value for a '+x' direction would mean the image is "
			"shifted in the positive direction of +x (in index space).",
			false, "dir", &allowedDirs, cmd);

	TCLAP::ValueArg<string> a_in_dir_index("", "in-uni-index", "Input is a uni-"
			"directional distortion field in index coords. "
			"Direction of distortion may be "
			"-x +x x -y +y y -z z +z, where no +/- implies +. Thus a positive "
			"distortion value for a '+x' direction would mean the image is "
			"shifted in the positive direction of +x (in index space).",
			false, "dir", &allowedDirs, cmd);

	TCLAP::ValueArg<string> a_out_dir_space("", "out-uni-space", "Output is a uni-"
			"directional distortion field in spacing (mm) coords. "
			"Direction of distortion may be "
			"-x +x x -y +y y -z z +z, where no +/- implies +. Thus a positive "
			"distortion value for a '+x' direction would mean the image is "
			"shifted in the positive direction of +x (in index space).",
			false, "dir", &allowedDirs, cmd);

	TCLAP::ValueArg<string> a_out_dir_index("", "out-uni-index", "Output is a uni-"
			"directional distortion field in index coords. "
			"Direction of distortion may be "
			"-x +x x -y +y y -z z +z, where no +/- implies +. Thus a positive "
			"distortion value for a '+x' direction would mean the image is "
			"shifted in the positive direction of +x (in index space).",
			false, "dir", &allowedDirs, cmd);

	TCLAP::SwitchArg a_invert("I", "invert", "index "
			"lookup type deform to an offset (in mm) type deform", cmd);

	TCLAP::SwitchArg a_in_offset("", "in-offset", "Input an offset (in mm) "
			"type deform", cmd);
	TCLAP::SwitchArg a_in_index("", "in-index", "Input an index type deform. "
			"Must provide an atlas if this is the case '-a'", cmd);
	TCLAP::SwitchArg a_apply_orient("r", "reorient", "Input deformation "
			"is in index coordinates (rather than physical coordinates). "
			"This will cause the vectors to be re-oriented to physical space",
			cmd);

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
	cerr << "Deform: " << *deform << endl;
	if(a_atlas.isSet()) {
		atlas = readMRImage(a_atlas.getValue());
		binarize(atlas);
	}

	// convert to offset
	if(a_in_index.isSet()) {
		if(deform->ndim() < 4 || deform->tlen() != 3) {
			cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
			return -1;
		}

		cerr << "Converting Index Lookup to Offset Map" << endl;
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
	} else if(a_apply_orient.isSet()) {
		if(deform->ndim() < 4 || deform->tlen() != 3) {
			cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
			return -1;
		}
		cerr << "Reorienting Vectors to RAS Space" << endl;
		deform = reorientVectors(deform);
	} else if(a_in_dir_space.isSet() || a_in_dir_index.isSet()) {
		if(deform->tlen() == 3) {
			cerr << "Input is already 3D!" << endl;
			return -1;
		}

		bool flip = false;
		int dim = 0;
		if(a_in_dir_space.isSet()) parseDir(a_in_dir_space.getValue(), flip, dim);
		if(a_in_dir_index.isSet()) parseDir(a_in_dir_index.getValue(), flip, dim);

		/*
		 * Now Convert 1D Vector to 3D oriented Vector
		 */
		auto olddef = deform;
		vector<size_t> newdim(4);
		for(size_t ii=0; ii<4; ii++) {
			if(ii<deform->ndim() && ii<3)
				newdim[ii] = deform->dim(ii);
			else if(ii == 3)
				newdim[ii] = 3;
			else
				newdim[ii] = 1;
		}
		deform = dPtrCast<MRImage>(deform->createAnother(4, newdim.data()));

		// FILL, setting vector to be in direction of dimension, then orienting
		// it (multiplying by orientation matrix)
		double vec[3] = {0,0,0};
		for(Vector3DIter<double> oit(deform), iit(olddef);
					!iit.eof() && !oit.eof(); ++iit, ++oit) {
			// zero vector
			for(size_t dd=0; dd<3; dd++) vec[dd] = 0;

			// Set Value in Direction of distortion
			if(flip)
				vec[dim] = -iit[0];
			else
				vec[dim] = iit[0];

			if(a_in_dir_space.isSet())
				vec[dim] /= deform->spacing(dim);

			// Convert
			deform->orientVector(3, vec, vec);

			// set in output
			for(size_t dd=0; dd<3; dd++)
				oit.set(dd, vec[dd]);
		}
	}
	if(deform->ndim() < 4 || deform->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes! If you "
			"have a 1D transform provide a direction with -u " << endl;
		return -1;
	}

	// invert
	if(a_invert.isSet()) {
		cerr << "Inverting" << endl;
		deform = invertForwardBack(deform, 100, 0.05);
		cerr << "Done" << endl;

		deform->write("invert.nii.gz");
	}

	// convert to correct output type
	if(a_out_index.isSet()) {
		cerr << "Changing back to index not yet implemented " << endl;
		return -1;
	} else if(a_out_dir_space.isSet()) {
		bool flip = false;
		int dim = 0;
		if(a_out_dir_space.isSet()) parseDir(a_out_dir_space.getValue(), flip, dim);

		cerr << "Creating 1D deformation in index space (output will be 3 "
			"Dimeions with scalar pixels indicating offset (in mm) "
			"projected in " << (flip ? "-" : "+") << (char)(dim+'x') << " Direction)" << endl;

		double vec[3];
		auto out = dPtrCast<MRImage>(deform->copyCast(3, deform->dim()));
		for(Vector3DIter<double> dit(deform), oit(out); !dit.eof() ; ++dit, ++oit) {
			for(size_t dd=0; dd<3; dd++)
				vec[dd] = dit[dd];
			deform->disOrientVector(3, vec, vec);
			if(flip) vec[dim] = -vec[dim];
			vec[dim] *= out->spacing(dim);
			oit.set(vec[dim]);
		}
		deform = out;
	} else if(a_out_dir_index.isSet()) {
		bool flip = false;
		int dim = 0;
		if(a_out_dir_index.isSet()) parseDir(a_out_dir_index.getValue(), flip, dim);
		cerr << "Creating 1D deformation in index space (output will be 3 "
			"Dimeions with scalar pixels indicating offset (in #pixels) "
			"projected in " << (flip ? "-" : "+") << (char)(dim+'x') << " Direction)" << endl;

		double vec[3];
		auto out = dPtrCast<MRImage>(deform->copyCast(3, deform->dim()));
		for(Vector3DIter<double> dit(deform), oit(out); !dit.eof() ; ++dit, ++oit) {
			for(size_t dd=0; dd<3; dd++)
				vec[dd] = dit[dd];
			deform->disOrientVector(3, vec, vec);
			if(flip) vec[dim] = -vec[dim];
			oit.set(vec[dim]);
		}
		deform = out;
	}

	// write
	deform->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

