/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "kernel_slicer.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using std::string;
using namespace npl;
using std::shared_ptr;

double gaussKern(double x)
{
	const double PI = acos(-1);
	const double den = 1./sqrt(2*PI);
	return den*exp(-x*x/(2));
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
 * @param mask Mask in target space (where the current deform maps from)
 * @param deform Maps from mask space.
 * @param atldef Maps to mask space, should be same size as mask
 *
 * @return 
 */
shared_ptr<MRImage> invertForwardBack(shared_ptr<MRImage> deform, 
		shared_ptr<MRImage> atlas, size_t MAXITERS, double MINERR)
{
	// create output the size of atlas, with 3 volumes in the 4th dimension
	auto atldef = createMRImage({atlas->dim(0), atlas->dim(1), 
				atlas->dim(2), 3}, FLOAT64);
	atldef->setDirection(atlas->direction(), true);
	atldef->setSpacing(atlas->spacing(), true);
	atldef->setOrigin(atlas->origin(), true);

	LinInterp3DView<double> definterp(deform);
	const double LAMBDA = 0.1;
	int64_t index[3];
	double err[3];
	double subpoint[3];
	double cindex[3];
	double sub2atl[3];
	vector<double> atlpoint(3);
	vector<double> atl2sub(3);
	KDTree<3,3,double, double> tree;

	if(deform->tlen() != 3) {
		cerr << "Error invalid deform image, needs 3 points in the 4th or "
			"5th dim" << endl;
		return NULL;
	}

	/* 
	 * Construct KDTree indexed by atlas-space indices, and storing indices
	 * in subject (mask) space. Only do it if mask value is non-zero however
	 */
	Vector3DIter<double> dit(deform);
	for(dit.goBegin(); !dit.eof(); ++dit) {
		dit.index(3, index);
		deform->indexToPoint(3, index, subpoint);

		for(int ii=0; ii<3; ++ii) {
			sub2atl[ii] = dit[ii];
			atl2sub[ii] = -dit[ii];
			atlpoint[ii] = subpoint[ii] + sub2atl[ii];
		}

		// add point to kdtree, use atl2sub since this will be our best guess
		// of atl2sub when we pull the point out of the tree
		tree.insert(atlpoint, atl2sub);
	}
	tree.build();

	/* 
	 * In atlas image try to find the correct source in the subject by first
	 * finding a point from the KDTree then improving on that result 
	 */
	// set atlas deform to NANs
	for(FlatIter<double> ait(atldef); !ait.eof(); ++ait) 
		ait.set(NAN);

	// at each point in atlas try to find the best mapping in the subject by
	// going back and forth. Since the mapping from sub to atlas is ground truth
	// we need to keep checking until we find a point in the subject that maps
	// to our current atlas location
	Vector3DIter<double> ait(atldef);
	assert(ait.tlen() == 3);

	for(ait.goBegin(); !ait.eof(); ++ait) {

		ait.index(3, index);
		atldef->indexToPoint(3, index, atlpoint.data());


		// that map from outside, then compute the median of the remaining
		double dist = INFINITY;
		auto results = tree.nearest(atlpoint, dist);

		// sort 
		if(!results)
			continue;

		for(size_t dd=0; dd<3; dd++)
			atl2sub[dd] = results->m_data[dd];

		// SUB <- ATLAS (given)
		//    atl2sub
		double prevdist = dist+1;
		size_t iters = 0;
		for(iters = 0 ; fabs(prevdist-dist) > 0 && dist > MINERR && 
						iters < MAXITERS; iters++) {

			for(size_t ii=0; ii<3; ii++) 
				subpoint[ii] = atlpoint[ii] + atl2sub[ii];

			// (estimate) SUB <- ATLAS (given)
			//              offset
			// interpolate new offset at subpoint
			deform->pointToIndex(3, subpoint, cindex);

			/* Update */

			// update sub2atl
			for(size_t ii=0; ii<3; ii++) {
				sub2atl[ii] = definterp(cindex[0], cindex[1], cindex[2], ii);
			}

			// update image with the error, using the derivative to estimate
			// where error crosses 0
			prevdist = dist;
			dist = 0;
			for(size_t ii=0; ii<3; ii++) {
				err[ii] = atl2sub[ii]+sub2atl[ii];
			}

			for(size_t ii=0; ii<3; ii++) {
				atl2sub[ii] -= LAMBDA*err[ii];
				dist += err[ii]*err[ii];
			}
			dist = sqrt(dist);
		}


		// save out final deform
		for(size_t ii=0; ii<3; ++ii) 
			ait.set(ii, atl2sub[ii]);
		cout << setw(10) << index[0] << setw(10) << index[1] << setw(10) 
					<< index[2] << setw(10) << "\r";

	}

	return atldef;
}

/**
 * @brief Smooths an image in 1 dimension, masked version. Only updates pixels
 * within masked region.
 *
 * @param in Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 * @param mask Only smooth (alter) point within the mask, inverted by 'invert'
 * @param invert only smooth points outside the mask
 */
void gaussianSmooth1D(shared_ptr<MRImage> inout, size_t dim, 
		double stddev, shared_ptr<MRImage> mask, bool invert)
{
	//TODO figure out how to scale this properly, including with stddev and 
	//spacing
	if(dim >= inout->ndim()) {
		throw std::out_of_range("Invalid dimension specified for 1D gaussian "
				"smoothing");
	}

	NNInterp3DView<int> maskinterp(mask);

	std::vector<int64_t> index(inout->ndim(), 0);
	std::vector<double> cindex(inout->ndim(), 0);
	std::vector<double> point(inout->ndim(), 0);
	stddev /= inout->spacing()[dim];
	std::vector<double> buff(inout->dim(dim));

	// for reading have the kernel iterator
	KernelIter<double> kit(inout);
	std::vector<size_t> radius(inout->ndim(), 0);
	for(size_t dd=0; dd<inout->ndim(); dd++) {
		if(dd == dim)
			radius[dd] = round(2*stddev);
	}
	kit.setRadius(radius);
	kit.goBegin();

	// calculate normalization factor
	double normalize = 0;
	int64_t rad = radius[dim];
	for(int64_t ii=-rad; ii<=rad; ii++)
		normalize += gaussKern(ii/stddev);

	// for writing, have the regular iterator
	OrderIter<double> oit(inout);
	oit.setOrder(kit.getOrder());
	oit.goBegin();
	
	while(!oit.eof()) {
		// perform kernel math, writing to buffer
		for(size_t ii=0; ii<inout->dim(dim); ii++, ++kit) {
			kit.center_index(index.size(), index.data());
			inout->indexToPoint(index.size(), index.data(), point.data());
			mask->pointToIndex(point.size(), point.data(), cindex.data());

			int m = maskinterp(cindex[0], cindex[1], cindex[2]);
			if((m != 0 && !invert) || (m == 0 && invert)) {
				double tmp = 0;
				for(size_t kk=0; kk<kit.ksize(); kk++) {
					double dist = kit.from_center(kk, dim);
					double nval = kit[kk];
					double stddist = dist/stddev;
					double weighted = gaussKern(stddist)*nval/normalize;
					tmp += weighted;
				}
				buff[ii] = tmp;
			} else {
				buff[ii] = kit.center();
			}
		}

		// write back out
		for(size_t ii=0; ii<inout->dim(dim); ii++, ++oit) 
			oit.set(buff[ii]);
	}
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
 * @param mask Mask in target space (where the current deform maps from)
 * @param deform Maps from mask space.
 * @param atldef Maps to mask space, should be same size as mask
 *
 * @return 
 */
shared_ptr<MRImage> invertMedianSmooth(shared_ptr<MRImage> deform, 
		shared_ptr<MRImage> atlas, size_t MAXITERS, double CHNORM, double MINDIST)
{
	// create output the size of atlas, with 3 volumes in the 4th dimension
	auto atldef = createMRImage({atlas->dim(0), atlas->dim(1), 
				atlas->dim(2), 3}, FLOAT64);
	atldef->setDirection(atlas->direction(), true);
	atldef->setSpacing(atlas->spacing(), true);
	atldef->setOrigin(atlas->origin(), true);

	const double MINNORM = 0.000001;
	int64_t index[3];
	double subpoint[3];
//	double cindex[3];
	double sub2atl[3];
	vector<double> atlpoint(3);
	vector<double> atl2sub(3);
	KDTree<3,3,double, double> tree;

	if(deform->tlen() != 3) {
		cerr << "Error invalid deform image, needs 3 points in the 4th or "
			"5th dim" << endl;
		return NULL;
	}

	/* 
	 * Construct KDTree indexed by atlas-space indices, and storing indices
	 * in subject (mask) space. Only do it if mask value is non-zero however
	 */
	Vector3DIter<double> dit(deform);
	for(dit.goBegin(); !dit.eof(); ++dit) {
		dit.index(3, index);
		deform->indexToPoint(3, index, subpoint);

		for(int ii=0; ii<3; ++ii) {
			sub2atl[ii] = dit[ii];
			atl2sub[ii] = -dit[ii];
			atlpoint[ii] = subpoint[ii] + sub2atl[ii];
		}

		// add point to kdtree, use atl2sub since this will be our best guess
		// of atl2sub when we pull the point out of the tree
		tree.insert(atlpoint, atl2sub);
	}
	tree.build();

	/* 
	 * In atlas image try to find the correct source in the subject by first
	 * finding a point from the KDTree then improving on that result 
	 */
	// set atlas deform to NANs
	for(FlatIter<double> ait(atldef); !ait.eof(); ++ait) 
		ait.set(NAN);

	// at each point in atlas try to find the best mapping in the subject by
	// going back and forth. Since the mapping from sub to atlas is ground truth
	// we need to keep checking until we find a point in the subject that maps
	// to our current atlas location
	Vector3DIter<double> ait(atldef);
	assert(ait.tlen() == 3);

	for(ait.goBegin(); !ait.eof(); ++ait) {

		ait.index(3, index);
		atldef->indexToPoint(3, index, atlpoint.data());


		// that map from outside, then compute the median of the remaining
		double dist = MINDIST;
		auto results = tree.withindist(atlpoint, dist);

		// sort 
		if(results.empty())
			continue;

		/*****************************
		 * find the geometric median 
		 *****************************/

		// intiialize with the mean
		for(size_t ii=0; ii<3; ++ii)
			atl2sub[ii] = 0;

		for(auto lit = results.begin(); lit != results.end(); ++lit) {
			for(size_t ii=0; ii<3; ++ii)
				atl2sub[ii] += (*lit)->m_data[ii];
		}
		for(size_t ii=0; ii<3; ++ii)
			atl2sub[ii] /= results.size();

		// iteratively reweight least squares solution
		double norm = CHNORM+1;
		size_t iter=0;
		for(iter = 0; iter < MAXITERS && norm > CHNORM; ++iter) {
			double sumnorm = 0;
			double prev[3];

			// copy current best into previous
			for(size_t ii=0; ii<3; ++ii) {
				prev[ii] = atl2sub[ii];
				atl2sub[ii] = 0;
			}

			for(auto lit = results.begin(); lit != results.end(); ++lit) {
				// compute distance between point and current best
				norm = 0;
				for(size_t ii=0; ii<3; ++ii) {
					norm += (prev[ii]-(*lit)->m_data[ii])*(prev[ii]-
							(*lit)->m_data[ii]);
				}
				norm = sqrt(norm);
				if(norm == 0) 
					norm = MINNORM;

				// add up total weights
				sumnorm += (1./norm);

				for(size_t ii=0; ii<3; ++ii)
					atl2sub[ii] += (*lit)->m_data[ii]/norm;
			}

			// divide by total weights
			for(size_t ii=0; ii<3; ++ii)
				atl2sub[ii] /= sumnorm;

			// compute difference from previous
			norm = 0;
			for(size_t ii=0; ii<3; ++ii)
				norm += (atl2sub[ii] - prev[ii])*(atl2sub[ii] - prev[ii]);
			norm = sqrt(norm);
		}

		// save out final deform
		for(size_t ii=0; ii<3; ++ii) 
			ait.set(ii, atl2sub[ii]);
		cout << setw(10) << index[0] << setw(10) << index[1] << setw(10) 
					<< index[2] << setw(10) << iter << "\r";


	}

	return atldef;
}

void binarize(shared_ptr<MRImage> in)
{
	OrderIter<int> it(in);
	for(it.goBegin(); !it.eof(); ++it) {
		if(*it != 0) 
			it.set(1);
	}

}

/**
 * @brief Takes an input and output mask and deformation field, then
 * extrapolates outside-the-brain deformations, which 1) do not map into output
 * masked region and 2) are continuous with within-the brain region
 *
 * @param def input deformation
 * @param omask mask in output (space that the points in the deform refer to)
 *
 * @return 
 */
shared_ptr<MRImage> extrapolateFromMasked(shared_ptr<MRImage> def, 
		shared_ptr<MRImage> omask)
{
	cerr << "Extrapolating Outside Masked Region" << endl;

	// we are going to spread out to outside-the mask regions until there are
	// no untouched regions left
	NNInterp3DView<int> omask_interp(omask);
	auto maskprev = dynamic_pointer_cast<MRImage>(omask->copyCast(INT8));
	auto maskcur = dynamic_pointer_cast<MRImage>(omask->copyCast(INT8));
	auto outdef = dynamic_pointer_cast<MRImage>(def->copyCast(FLOAT32));
	Vector3DView<double> odview(outdef);
	double offset[3];
	double point[3];
	int64_t index[3];

	// construct tree from points currently outside mask omask
	KDTree<3, 0> tree;
	for(OrderIter<int> it(omask); !it.eof(); ++it) {
		if(*it == 0) {
			it.index(3, index);
			omask->indexToPoint(3, index, point);
			tree.insert(3, point, 0, point);
		}
	}
	// rebuild tree
	tree.build();
		
	// construct iterators
	KernelIter<int> pmit(maskprev);
	pmit.setRadius(1);
	OrderIter<int> cmit(maskcur);
	cmit.setOrder(pmit.getOrder());

	size_t changed = 1;
	for(size_t iters = 0; changed; ++iters) {
		cerr << iters << setw(10) << changed << "\r";
		changed = 0;

		// copy cur into previous
		for(FlatIter<int> pit(maskprev), cit(maskcur); !cit.eof(); ++cit, ++pit) 
			pit.set(*cit);

		for(pmit.goBegin(), cmit.goBegin(); !cmit.eof(); ++pmit, ++cmit) {
			// skip regions inside the mask
			if(pmit.center() != 0)
				continue;
			++changed;

			// get the current index in deform/mask
			pmit.center_index(3, index);

			// zero continuous index
			std::fill(offset, offset+3, 0);

			// look at all the members of the kernel, compute the average deform
			// among the non-masked kernel members
			size_t count = 0;
			for(size_t ii=0; ii<pmit.ksize(); ++ii) {
				if(pmit[ii] != 0) {
					++count;
				}
			}

			// skip if there were no neighbors inside the mask (we'll get 
			// back to it later)
			if(count == 0)
				continue;
			
			for(size_t ii=0; ii<pmit.ksize(); ++ii) {
				int64_t tmp[3];
				if(pmit[ii] != 0) {
					pmit.offset_index(ii, 3, tmp);
					for(size_t jj=0; jj<3; ++jj) 
						offset[jj] += odview(tmp[0], tmp[1], tmp[2], jj);
				}
			}

			for(size_t jj=0; jj<3; ++jj) 
				offset[jj] /= count;
			
			// mark this pixel as valid, set value in deform
			cmit.set(1);
			for(size_t ii=0; ii<3; ++ii) {
				odview.set(offset[ii], index[0], index[1], index[2], ii);
			}
		}

		assert(pmit.isEnd() && cmit.isEnd());
	}

	cerr << "Smoothing Results outside mask" << endl;
	// smooth extrapolated points, we repeat rather than doing a larger kernel
	// to improve mask boundaries, since the support is smaller, the oddness
	// due to internal pointers not being smoothed versus the neighboring 
	// smoothed points is decreased
//	for(size_t ii=0; ii<4; ii++) {
//		gaussianSmooth1D(outdef, 0, 1.0, omask, true);
//		gaussianSmooth1D(outdef, 1, 1.0, omask, true);
//		gaussianSmooth1D(outdef, 2, 1.0, omask, true);
//	}

	cerr << "Done with extrapolation" << endl;
	return dynamic_pointer_cast<MRImage>(outdef);
}

shared_ptr<MRImage> medianSmooth(shared_ptr<MRImage> deform, double radius)
{
	const double CHNORM = .1;
	const size_t MAXITERS = 100;
	const double MINNORM = 0.000001;

	// create output
	auto out = dynamic_pointer_cast<MRImage>(deform->copy());

	// create KDTree of the grid
	KDTree<3,3,double, double> tree;
	double point[3]; 
	double sub2atl[3]; 
	int64_t index[3]; 

	/* 
	 * Construct KDTree indexed by atlas-space indices, and storing indices
	 * in subject (mask) space. Only do it if mask value is non-zero however
	 */
	Vector3DIter<double> dit(deform);
	for(dit.goBegin(); !dit.eof(); ++dit) {
		dit.index(3, index);
		deform->indexToPoint(3, index, point);

		for(int ii=0; ii<3; ++ii) {
			sub2atl[ii] = dit[ii];
		}

		// add point to kdtree, use atl2sub since this will be our best guess
		// of atl2sub when we pull the point out of the tree
		tree.insert(3, point, 3, sub2atl);
	}
	tree.build();

	Vector3DIter<double> oit(out);
	for(dit.goBegin(), oit.goBegin(); !oit.eof(); ++dit, ++oit) {

		oit.index(3, index);
		deform->indexToPoint(3, index, point);

		double dist = radius;
		auto results = tree.withindist(3, point, dist);

		assert(!results.empty());

		/*****************************
		 * find the geometric median 
		 *****************************/

		// intiialize with the mean
		for(size_t ii=0; ii<3; ++ii)
			sub2atl[ii] = 0;

		for(auto lit = results.begin(); lit != results.end(); ++lit) {
			for(size_t ii=0; ii<3; ++ii)
				sub2atl[ii] += (*lit)->m_data[ii];
		}
		for(size_t ii=0; ii<3; ++ii)
			sub2atl[ii] /= results.size();

		// iteratively reweight least squares solution
		double norm = CHNORM+1;
		size_t iter=0;
		for(iter = 0; iter < MAXITERS && norm > CHNORM; ++iter) {
			double sumnorm = 0;
			double prev[3];

			// copy current best into previous
			for(size_t ii=0; ii<3; ++ii) {
				prev[ii] = sub2atl[ii];
				sub2atl[ii] = 0;
			}

			for(auto lit = results.begin(); lit != results.end(); ++lit) {
				// compute distance between point and current best
				norm = 0;
				for(size_t ii=0; ii<3; ++ii) {
					norm += (prev[ii]-(*lit)->m_data[ii])*(prev[ii]-
							(*lit)->m_data[ii]);
				}
				norm = sqrt(norm);
				if(norm < MINNORM) 
					norm = MINNORM;

				// add up total weights
				sumnorm += (1./norm);

				for(size_t ii=0; ii<3; ++ii)
					sub2atl[ii] += (*lit)->m_data[ii]/norm;
			}

			// divide by total weights
			for(size_t ii=0; ii<3; ++ii)
				sub2atl[ii] /= sumnorm;

			// compute difference from previous
			norm = 0;
			for(size_t ii=0; ii<3; ++ii)
				norm += (sub2atl[ii] - prev[ii])*(sub2atl[ii] - prev[ii]);
			norm = sqrt(norm);
		}

		// save out final deform
		for(size_t ii=0; ii<3; ++ii) 
			oit.set(ii, sub2atl[ii]);
		
		cout << setw(10) << index[0] << setw(10) << index[1] << setw(10) 
					<< index[2] << setw(10) << iter << "\r";

	}
	
	return out;
}


int main(int argc, char** argv)
{
	try {
	/* 
	 * Command Line 
	 */

	TCLAP::CmdLine cmd("Applies 3D deformation to volume or time-series of "
			"volumes. Deformation should be a map of offsets. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_indef("i", "input", "Input deformation.", 
			true, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_mask("m", "mask", "Mask image in "
			"deform space.", false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_atlas("a", "atlas", "Atlas (mask) image. Needed "
			"if you provide an index-based deformation or if you intend to "
			"extrapolate '-E'. Note that if you already have a valid "
			"deformation for the entire FOV, and the input is a offset type "
			"deform, then neither this nor mask '-m' are needed", false, "", 
			"*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);

	TCLAP::SwitchArg a_extrapolate("E", "extrapolate", "Extrapolate "
			"deformations outside of masked regions ", cmd);
	TCLAP::SwitchArg a_invert("I", "invert", "index "
			"lookup type deform to an offset (in mm) type deform", cmd);
	
	TCLAP::SwitchArg a_in_offset("", "in-offset", "Input an offset (in mm) "
			"type deform", cmd);
	TCLAP::SwitchArg a_in_index("", "in-index", "Input an index type deform. "
			"Must provide an atlas if this is the case '-a'", cmd);
	
	TCLAP::SwitchArg a_out_offset("", "out-offset", "Output an index "
			"lookup type deform.", cmd);
	TCLAP::SwitchArg a_out_index("", "out-index", "Output an offset "
			"type deform (in mm/physical units)", cmd);

	TCLAP::SwitchArg a_one_index("1", "one-based-index", "When referring to "
			"indexes make them one-based.", cmd);


	TCLAP::ValueArg<size_t> a_iters("", "iters", "Number of iterations during "
			"median-smoothing of input deform during inverse", false, 100, "iters", cmd);
	TCLAP::ValueArg<double> a_improve("", "delta", "Amount of improvement in "
			"estimate before stopping during inverse.", false, 0.1, "DNORM", cmd);
//	TCLAP::ValueArg<double> a_radius("R", "radius", "Radius to search "
//			"for points that may map to a coordinate in the output image. "
//			"We compute the geometric median of the points to give a smoothed "
//			"deformation.", false, 10, "mm", cmd);

	TCLAP::ValueArg<double> a_gaussian_sigma("s", "sigma", "Sigma of gaussian "
			"smoothing kernel in mm.", false, 5, "mm", cmd);
	TCLAP::ValueArg<double> a_median_radius("M", "median-radius", "Radius of "
			"median smoothing.", false, 2, "mm", cmd);


	cmd.parse(argc, argv);
	std::shared_ptr<MRImage> deform;
	std::shared_ptr<MRImage> atlas;
	std::shared_ptr<MRImage> mask;
	bool extrapolate = a_extrapolate.isSet();

	/**********
	 * Input
	 *********/
	deform = readMRImage(a_indef.getValue());
	if(deform->ndim() < 4 || deform->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}
	if(a_mask.isSet()) {
		mask = readMRImage(a_mask.getValue());
		binarize(mask);

		// check if mask has same dimensions as first DIM dimensions of deform
		bool samedim = true;
		for(size_t dd=0; dd<mask->ndim() && dd<deform->ndim(); dd++) {
			if(mask->dim(dd) != deform->dim(dd))
				samedim = false;
		}

		double overlap = overlapRatio(mask, deform);
		if(overlap < 0.5 && samedim) {
			cerr << "WARNING it looks like you provided a mask with different "
				"orientation than the deformation. We are going to alter the "
				"orientation of the mask to match the deform. THIS IS BAD. You "
				"should fix your input mask." << endl;
			mask->setOrigin(deform->origin());
			mask->setSpacing(deform->spacing());
			mask->setDirection(deform->direction());
		} else if(overlap < 0.5) {
			cerr << "Error there is not a large enough overlap between the "
				"input  deform and the mask. You should provide a mask that "
				"overlaps!" << endl;
			return -1;
		}
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


	if(mask && a_invert.isSet()) {
		cerr << "Forcing extrapolation, because the input deform is masked. "
			"Note you need to provide an atlas." << endl;
		extrapolate = true;
	}

	// extrapolate 
	if(extrapolate) {
		if(!atlas || !mask) {
			cerr << "Must provide an atlas mask and deform mask together, for "
				"extrapolation!" 
				<< endl;
			return -1;
		}
		deform = extrapolateFromMasked(deform, mask);
		deform->write("extrapolated.nii.gz");
		// don't need mask anymore
		mask.reset();
	}

	// invert
	if(a_invert.isSet()) {
		if(!atlas) {
			cerr << "Must provide atlas for inversion!" << endl;
		}

		cerr << "Median Smoothing..." << endl;
		deform = medianSmooth(deform, a_median_radius.getValue());
		cerr << "Gaussian Smoothing..." << endl;
		deform->write("median_smoothed.nii.gz");
		gaussianSmooth1D(deform, 0, a_gaussian_sigma.getValue());
		gaussianSmooth1D(deform, 1, a_gaussian_sigma.getValue());
		gaussianSmooth1D(deform, 2, a_gaussian_sigma.getValue());
		cerr << "Done Smoothing." << endl;
		deform->write("smoothed.nii.gz");

		cerr << "Inverting" << endl;
		deform = invertForwardBack(deform, atlas, 15, 0.1); 
		cerr << "Done" << endl;

		deform->write("invert.nii.gz");
		gaussianSmooth1D(deform, 0, a_gaussian_sigma.getValue());
		gaussianSmooth1D(deform, 1, a_gaussian_sigma.getValue());
		gaussianSmooth1D(deform, 2, a_gaussian_sigma.getValue());
		// smooth masked regions
		deform->write("invert_smooth.nii.gz");
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

