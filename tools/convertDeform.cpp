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

template <typename T>
ostream& operator<<(ostream& out, const std::vector<T>& v)
{
	out << "[ ";
	for(size_t ii=0; ii<v.size()-1; ++ii)
		out << v[ii] << ", ";
	out << v[v.size()-1] << " ]";
	return out;
}
//
//void applyDeform(shared_ptr<MRImage> in, shared_ptr<MRImage> deform,
//		shared_ptr<MRImage> out, size_t vdim)
//{
//	vector<double> in_index(in->ndim());
//	vector<double> in_point(3);
//	vector<double> deform_index(deform->ndim());
//	vector<double> deform_point(deform->ndim());
//	vector<int64_t> out_index(out->ndim());
//	vector<double> out_point(out->ndim());
//	vector<double> offset(3);
//	bool outside = false;
//
//	// map all the points
//	list<size_t> slcorder;
//	if(in->ndim() == 4)
//		slcorder.push_back(3); // iterate through time the fastest
//	
//	Slicer fit(out->ndim(), out->dim(), slcorder);
//	for(fit.goBegin(); !fit.isEnd(); ) {
//		// for point in deform image
//		out_index = fit.index();
//		out->indexToPoint(out_index, out_point);
//#ifdef DEBUG
//		cerr << "Out: " << out_index << " | " << out_point << endl;
//#endif //DEBUG
//
//		for(size_t ii=0; ii<3; ++ii)
//			deform_point[ii] = out_point[ii];
//		for(size_t ii=3; ii<deform_point.size(); ++ii)
//			deform_point[ii] = 0;
//
//		// sampe offset at point
//		deform->pointToIndex(deform_point, deform_index);
//#ifdef DEBUG
//		cerr << "Deform:" << deform_point << " | " << deform_index << endl;
//#endif //DEBUG
//		for(size_t ii=0; ii<3; ++ii) {
//			deform_index[vdim] = ii;
//			offset[ii] = deform->linSampleInd(deform_index, ZEROFLUX, outside);
//			in_point[ii] = deform_point[ii] + offset[ii];
//		}
//
//#ifdef DEBUG
//		cerr << "Offset: " << offset << endl;
//#endif //DEBUG
//		in->pointToIndex(in_point, in_index);
//#ifdef DEBUG
//		cerr << "In: " << in_point << " | " << in_index << endl; 
//#endif //DEBUG
//
//		if(in->ndim() == 4){
//
//			// run through time
//			assert(in->dim(3) == out->dim(3));
//			for(size_t tt=0; tt < in->dim(3); ++tt, ++fit) {
//				in_index[3] = tt;
//				out->set_dbl(*fit, in->linSampleInd(in_index,ZEROFLUX, outside));
//			}
//		} else {
//			
//			// just do one
//			assert(in->dim(3) == out->dim(3));
//			out->set_dbl(*fit, in->linSampleInd(in_index, ZEROFLUX, outside));
//			++fit; 
//		}
//	}
//}

shared_ptr<MRImage> erode(shared_ptr<MRImage> in, size_t reps)
{
	std::vector<int64_t> index1(in->ndim(), 0);
	std::vector<int64_t> index2(in->ndim(), 0);
	auto prev = in->cloneImage();
	auto out = in->cloneImage();
	shared_ptr<MRImage> tmp; 
	for(size_t rr=0; rr<reps; ++rr) {
		cerr << "Erode " << rr << endl;

		tmp = prev;
		prev = out;
		out = tmp;
		
		KernelIter<int> it(prev);
		it.setRadius(1);
		OrderIter<int> oit(out);
		oit.setOrder(it.getOrder());
		// for each pixels neighborhood, smooth neightbors
		for(oit.goBegin(), it.goBegin(); !it.eof(); ++it, ++oit) {
			oit.index(index1.size(), index1.data());
			it.center_index(index2.size(), index2.data());

			// if any of the neighbors are 0, then set to 0
			bool erodeme = false;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.offset(ii) == 0) {
					erodeme = true;
				}
			}

			if(erodeme) 
				oit.set(0);
		}
	}

	return out;
}

shared_ptr<MRImage> dilate(shared_ptr<MRImage> in, size_t reps)
{
	std::vector<int64_t> index1(in->ndim(), 0);
	std::vector<int64_t> index2(in->ndim(), 0);
	auto prev = in->cloneImage();
	auto out = in->cloneImage();
	shared_ptr<MRImage> tmp; 
	for(size_t rr=0; rr<reps; ++rr) {
		cerr << "Dilate " << rr << endl;

		tmp = prev;
		prev = out;
		out = tmp;
		
		KernelIter<int> it(prev);
		it.setRadius(1);
		OrderIter<int> oit(out);
		oit.setOrder(it.getOrder());
		// for each pixels neighborhood, smooth neightbors
		for(oit.goBegin(), it.goBegin(); !it.eof(); ++it, ++oit) {
			oit.index(index1.size(), index1.data());
			it.center_index(index2.size(), index2.data());
			for(size_t ii=0; ii<in->ndim(); ++ii) {
				if(index1[ii] != index2[ii]) {
					throw std::logic_error("Error differece in iteration!");
				}
			}

			// if any of the neighbors are 0, then set to 0
			bool dilateme = false;
			int dilval = 0;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.offset(ii) != 0) {
					dilval = it.offset(ii);
					dilateme = true;
				}
			}

			if(dilateme) 
				oit.set(dilval);
		}
	}

	return out;
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
shared_ptr<MRImage> invertUnMasked(shared_ptr<MRImage> deform, 
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

	cout << setw(30) << "Inverting: "  << "Median Iterations" << endl; 
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
shared_ptr<MRImage> invertMasked(shared_ptr<MRImage> mask, 
		shared_ptr<MRImage> deform, shared_ptr<MRImage> atlas, size_t MAXITERS,
		double CHNORM, double MINDIST)
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
	double cindex[3];
	double sub2atl[3];
	vector<double> atlpoint(3);
	vector<double> atl2sub(3);
	NNInterp3DView<double> maskinterp(mask);
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
		mask->pointToIndex(3, subpoint, cindex);

		// skip 0s mask
		if(maskinterp(cindex[0], cindex[1], cindex[2]) == 0)
			continue;

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

	cout << setw(30) << "Inverting: "  << "Median Iterations" << endl; 
	for(ait.goBegin(); !ait.eof(); ++ait) {

		ait.index(3, index);
		atldef->indexToPoint(3, index, atlpoint.data());


		// that map from outside, then compute the median of the remaining
		double dist = MINDIST;
		auto results = tree.withindist(atlpoint, dist);

		// remove elements that map outside the masked region
		for(auto rit=results.begin(); rit != results.end(); ++rit) {
			for(size_t ii=0; ii<3; ++ii) 
				subpoint[ii] = atlpoint[ii] + (*rit)->m_data[ii];

			mask->pointToIndex(3, subpoint, cindex);
			if(maskinterp(cindex[0], cindex[1], cindex[2]) == 0)
				rit = results.erase(rit);
		} 

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
 * @param dmask mask in deform space 
 * @param omask mask in output (space that the points in the deform refer to)
 *
 * @return 
 */
shared_ptr<MRImage> extrapolateFromMasked(shared_ptr<MRImage> def, 
		shared_ptr<MRImage> dmask, shared_ptr<MRImage> omask)
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
	double cindex[3];
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

			/*
			 * force the offset to map into unmasked space
			 */

//			// get the mapped point
//			dmask->indexToPoint(3, index, point);
//			for(size_t jj=0; jj<3; ++jj)
//				point[jj] += offset[jj];
//
//			omask->pointToIndex(3, point, cindex);
//			if(omask_interp(cindex[0], cindex[1], cindex[2]) != 0) {
//				// find the nearest point outside
//				double dist = INFINITY;
//				auto nearby = tree.nearest(3, point, dist);
//				
//				// add (point to found) offset
//				// offset += found - point 
//				for(size_t ii=0; ii<3; ++ii)
//					offset[ii] += nearby->m_point[ii]-point[ii];
//			}
//
			// mark this pixel as valid, set value in deform
			cmit.set(1);
			for(size_t ii=0; ii<3; ++ii) {
				odview.set(offset[ii], index[0], index[1], index[2], ii);
			}
		}

		assert(pmit.isEnd() && cmit.isEnd());
	}

	// smooth extrapolated points
	gaussianSmooth1D(outdef, 0, 3.0);
	gaussianSmooth1D(outdef, 1, 3.0);
	gaussianSmooth1D(outdef, 2, 3.0);

	outdef->write("extrap.nii.gz");
	cerr << "Done with extrapolation" << endl;
	return dynamic_pointer_cast<MRImage>(outdef);
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

	// TODO IMPELEMENT
//	TCLAP::SwitchArg a_svreg("O", "offset", "Indicates that the input is "
//			"already an offset map (rather than a exact position.", cmd);
//	TCLAP::SwitchArg a_svreg("R", "realspace", "Indicates that the input is "
//			"already in real space (rather than index space).", cmd);


	TCLAP::ValueArg<size_t> a_iters("", "iters", "Number of iterations during "
			"median-smoothing of input deform during inverse", false, 100, "iters", cmd);
	TCLAP::ValueArg<double> a_improve("", "delta", "Amount of improvement in "
			"estimate before stopping during inverse.", false, 0.1, "DNORM", cmd);
	TCLAP::ValueArg<double> a_radius("R", "radius", "Radius to search "
			"for points that may map to a coordinate in the output image. "
			"We compute the geometric median of the points to give a smoothed "
			"deformation.", false, 10, "mm", cmd);

//	TCLAP::ValueArg<size_t> a_dilate("D", "dilate", "Number of times to "
//			"dilate mask.", false, 0, "iters", cmd);
//	TCLAP::ValueArg<size_t> a_erode("E", "erode", "Number of times to "
//			"erode mask.", false, 1, "iters", cmd);


	cmd.parse(argc, argv);
	std::shared_ptr<MRImage> indef;
	std::shared_ptr<MRImage> atlas;
	std::shared_ptr<MRImage> mask;
	bool extrapolate = a_extrapolate.isSet();

	/**********
	 * Input
	 *********/
	indef = readMRImage(a_indef.getValue());
	if(indef->ndim() < 4 || indef->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}
	if(a_mask.isSet()) {
		mask = readMRImage(a_mask.getValue());
		binarize(mask);
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
		indef = indexMapToOffsetMap(indef, atlas, a_one_index.isSet());
		indef->write("offset.nii.gz");
	}


//	if(mask && a_invert.isSet()) {
//		cerr << "Forcing extrapolation, because the input deform is masked. "
//			"Note you need to provide an atlas." << endl;
//		extrapolate = true;
//	}
//
	// extrapolate 
	if(extrapolate) {
		if(!atlas || !mask) {
			cerr << "Must provide an atlas mask and deform mask together, for "
				"extrapolation!" 
				<< endl;
			return -1;
		}
		indef = extrapolateFromMasked(indef, atlas, mask);
		indef->write("extrapolated.nii.gz");
		// don't need mask anymore
		mask.reset();
	}

	// invert
	if(a_invert.isSet()) {
		if(!atlas) {
			cerr << "Must provide atlas for inversion!" << endl;
		}

		cerr << "Inverting" << endl;
		if(mask) {
			indef = invertMasked(mask, indef, atlas, a_iters.getValue(), 
					a_improve.getValue(), a_radius.getValue());
		} else {
			indef = invertUnMasked(indef, atlas, a_iters.getValue(), 
					a_improve.getValue(), a_radius.getValue());
		}
		cerr << "Done" << endl;

		indef->write("invert.nii.gz");
	}

	// convert to correct output type
	if(a_out_index.isSet()) {
		cerr << "Changing back to index not yet implemented " << endl;
		return -1;
	}

	// write
	indef->write(a_out.getValue());
	
//	std::shared_ptr<MRImage> mask(readMRImage(a_mask.getValue()));
//	if(mask->ndim() != 3) {
//		cerr << "Expected mask to be 3D Image!" << endl;
//		return -1;
//	}
//	binarize(mask);
//
//	// dilate then erode mask
//	if(a_dilate.isSet()) 
//		mask = dilate(mask, a_dilate.getValue());
//	if(a_erode.isSet()) 
//		mask = erode(mask, a_erode.getValue());
//
//	// ensure the the images overlap sufficiently, lack of overlap may indicate
//	// incorrect orientation
//	double f = overlapRatio(mask, inimg);
//	if(f < .5)  {
//		cerr << "Warning the input and mask images do not overlap very much."
//			" This could indicate bad orientation, overlap: " << f << endl;
//		return -1;
//	}
//	
//	std::shared_ptr<MRImage> atlas(readMRImage(a_atlas.getValue()));
//	if(atlas->ndim() != 3) {
//		cerr << "Expected mask to be 3D Image!" << endl;
//		return -1;
//	}
//	binarize(atlas);
//
//	std::shared_ptr<MRImage> defimg(readMRImage(a_deform.getValue()));
//
//	// perform interpolation to estimate outside-the brain deformations that
//	// are continuous with the within-brain deformations
//	defimg = extrapolate(defimg, mask, atlas);
//	defimg->write("extrapolated.nii.gz");
//
//	if(a_invert.isSet()) {
//		// create output the size of atlas, with 3 volumes in the 4th dimension
//		auto idef = createMRImage({atlas->dim(0), atlas->dim(1), 
//					atlas->dim(2), 3}, FLOAT64);
//		idef->setDirection(atlas->direction(), true);
//		idef->setSpacing(atlas->spacing(), true);
//		idef->setOrigin(atlas->origin(), true);
//		invert(mask, defimg, idef, a_iters.getValue(), 
//				a_improve.getValue(), a_radius.getValue());
//		idef->write("inversedef.nii.gz");
//	}
//
//	/*********
//	 * Output
//	 ********/
//	// figure out field of view
//	vector<double> lowerbound(3, INFINITY);
//	vector<double> upperbound(3, -INFINITY);
//	getBounds(deform, lowerbound, upperbound);
//
//	cerr << "Computed Bound 1: " << lowerbound << endl;
//	cerr << "Computed Bound 2: " << upperbound << endl;
//
//	// convert bounds to indices, so we can figure out image size
//	std::vector<int64_t> index1;
//	std::vector<int64_t> index2;
//	in->pointToIndex(lowerbound, index1);
//	in->pointToIndex(upperbound, index2);
//	for(size_t ii=0; ii<3; ++ii) {
//		int64_t tmp1 = index1[ii];
//		int64_t tmp2 = index2[ii];
//		index1[ii] = std::min(tmp1, tmp2);
//		index2[ii] = std::max(tmp1, tmp2);
//	}
//
//	cerr << "Bounding Index 1: " << index1 << endl;
//	cerr << "Bounding Index 2: " << index2 << endl;
//	// create output image that covers FOV
//	
//	std::vector<size_t> sz;
//	if(in->ndim() == 3) {
//		sz.resize(3);
//	} else {
//		sz.resize(4);
//		sz[3] = in->dim(3);
//	}
//	for(size_t ii=0; ii<3; ++ii) 
//		sz[ii] = index2[ii]-index1[ii];
//
//	// force size from input
//	if(a_xsize.isSet())
//		sz[0] = a_xsize.getValue();
//	if(a_ysize.isSet())
//		sz[1] = a_ysize.getValue();
//	if(a_zsize.isSet())
//		sz[2] = a_zsize.getValue();
//
//	cerr << "Size: " << sz << endl; 
//	auto out = createMRImage(sz, FLOAT32);
//
//	std::cerr << "Min Index: " << index1 << endl;
//	
//	// origin is minimum index in point space:
//	std::vector<double> point;
//	in->indexToPoint(index1, point);
//	for(size_t ii=0; ii<3; ++ii) 
//		out->origin()[ii] = point[ii];
//
//	// force size from input
//	if(a_xorigin.isSet()) 
//		out->origin()[0] = a_xorigin.getValue();
//	if(a_yorigin.isSet()) 
//		out->origin()[1] = a_yorigin.getValue();
//	if(a_zorigin.isSet()) 
//		out->origin()[2] = a_zorigin.getValue();
//
//	
//	std::cerr << "New Origin:\n" << out->origin() << endl;
//
//	out->setSpacing(in->spacing());
//	out->setDirection(in->direction());
//
//	applyDeform(in, deform, out, vdim);
//	
//	out->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

//		// SUB <- ATLAS (given)
//		//    atl2sub
//		double prevdist = dist+1;
//		size_t iters = 0;
//		for(iters = 0 ; fabs(prevdist-dist) > 0 && dist > MINERR && 
//						iters < MAXITERS; ++iters) {
//
//			for(size_t ii=0; ii<3; ++ii) 
//				subpoint[ii] = atlpoint[ii] + atl2sub[ii];
//
//			// ignore points that map outside the mask, just accept this as the
//			// best approximate deform
//			mask->pointToIndex(3, subpoint, cindex);
//			if(maskinterp(cindex[0], cindex[1], cindex[2]) == 0)
//				break;
//
//			// (estimate) SUB <- ATLAS (given)
//			//              offset
//			// interpolate new offset at subpoint
//			deform->pointToIndex(3, subpoint, cindex);
//
//			/* 
//			 * Compute teh Derivative of the Vector Field
//			 */
//
//			/* Update */
//
//			// update sub2atl
//			for(size_t ii=0; ii<3; ++ii) {
//				sub2atl[ii] = definterp(cindex[0], cindex[1], cindex[2], ii);
//			}
//
//			for(size_t ii=0; ii<3; ++ii) {
//				for(size_t jj=0; jj<3; ++jj) {
//					cindex[jj]+=0.00001;
//					dVdX(ii,jj) = (definterp(cindex[0], cindex[1], cindex[2], ii)-sub2atl[ii])/0.00001;
//					cindex[jj]-=0.00001;
//				}
//			}
//			for(size_t ii=0; ii<3; ++ii)
//				dVdX(ii,ii) += 1;
//
//			// update image with the error, using the derivative to estimate
//			// where error crosses 0
//			prevdist = dist;
//			dist = 0;
//			for(size_t ii=0; ii<3; ++ii) {
//				err[ii] = atl2sub[ii]+sub2atl[ii];
//			}
//
//			err = inverse(dVdX)*err;
//			for(size_t ii=0; ii<3; ++ii) {
//				atl2sub[ii] -= LAMBDA*err[ii];
//				dist += err[ii]*err[ii];
//			}
//			dist = sqrt(dist);
//		}

