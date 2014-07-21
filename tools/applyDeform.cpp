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
	for(size_t ii=0; ii<v.size()-1; ii++)
		out << v[ii] << ", ";
	out << v[v.size()-1] << " ]";
	return out;
}

//void getBounds(shared_ptr<MRImage> deform, std::vector<double>& lowerbound, 
//		std::vector<double>& upperbound)
//{
//	lowerbound.resize(3);
//	upperbound.resize(3);
//
//	for(size_t ii=0; ii<3; ii++) {
//		lowerbound[ii] = INFINITY;
//		upperbound[ii] = -INFINITY;
//	}
//
//	vector<double> offset(3);
//	vector<int64_t> index;
//	vector<double> point;
//	Slicer it(deform->ndim(), deform->dim());
//	for(it.goBegin(); !it.isEnd(); ) {
//		for(size_t ii=0; ii<3; ii++, ++it) {
//			offset[ii] = deform->get_dbl(*it);
//		}
//
//		index = it.index();
//		deform->indexToPoint(index, point);
//		for(size_t ii=0; ii<3; ii++) {
//			point[ii] -= offset[ii];
//			if(point[ii] < lowerbound[ii]) 
//				lowerbound[ii] = point[ii];
//			if(point[ii] > upperbound[ii]) 
//				upperbound[ii] = point[ii];
//		}
//	}
//}
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
//		for(size_t ii=0; ii<3; ii++)
//			deform_point[ii] = out_point[ii];
//		for(size_t ii=3; ii<deform_point.size(); ii++)
//			deform_point[ii] = 0;
//
//		// sampe offset at point
//		deform->pointToIndex(deform_point, deform_index);
//#ifdef DEBUG
//		cerr << "Deform:" << deform_point << " | " << deform_index << endl;
//#endif //DEBUG
//		for(size_t ii=0; ii<3; ii++) {
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
//			for(size_t tt=0; tt < in->dim(3); tt++, ++fit) {
//				in_index[3] = tt;
//				out->set_dbl(*fit, in->linSampleInd(in_index,ZEROFLUX, outside));
//			}
//		} else {
//			
//			// just do one
//			assert(in->dim(3) == out->dim(3));
//			out->set_dbl(*fit, in->linSampleInd(in_index, ZEROFLUX, outside));
//			fit++; 
//		}
//	}
//}

shared_ptr<MRImage> dilate(shared_ptr<MRImage> in, size_t reps)
{
	std::vector<int64_t> index(in->ndim(), 0);
	auto prev = in->cloneImage();
	auto out = in->cloneImage();
	shared_ptr<MRImage> tmp; 
	for(size_t rr=0; rr<reps; rr++) {
		cerr << "Dilate " << rr << endl;
		
		KernelIter<int> it(prev);
		it.setRadius(1);
		OrderIter<int> oit(out);
		// for each pixels neighborhood, smooth neightbors
		for(oit.goBegin(), it.goBegin(); !it.eof(); ++it, ++oit) {

			// if any of the neighbors are 0, then set to 0
			bool dilateme = false;
			int dilval = 0;
			for(size_t ii=0; ii<it.ksize(); ii++) {
				if(it.offset(ii) != 0) {
					dilval = it.offset(ii);
					dilateme = true;
				}
			}

			if(dilateme) 
				oit.set(dilval);
		}

		tmp = prev;
		prev = out;
		out = tmp;
	}

	return out;
}

int invert(shared_ptr<MRImage> mask, shared_ptr<MRImage> deform,
		shared_ptr<MRImage> atldef)
{
	const double MINERR = .1;
	const double LAMBDA = .1;
	const size_t MAXITERS = 300;
	int64_t index[3];
	double subpoint[3];
	double cindex[3];
	double sub2atl[3];
	double err[3];
	vector<double> atlpoint(3);
	vector<double> atl2sub(3);
	OrderIter<int> mit(mask);
	LinInterp3DView<double> definterp(deform);
	NNInterp3DView<double> maskinterp(mask);

	if(deform->tlen() != 3) {
		cerr << "Error invalid deform image, needs 3 points in the 4th or "
			"5th dim" << endl;
		return -1;
	}

	/* 
	 * Construct KDTree indexed by atlas-space indices, and storing indices
	 * in subject (mask) space. Only do it if mask value is non-zero however
	 */
	KDTree<3,3,double, double> tree;
	for(mit.goBegin(); !mit.eof(); ++mit) {
		// ignore zeros in mask
		if(*mit == 0)
			continue;

		// target is current coordinate
		mit.index(3, index);
		mask->indexToPoint(3, index, subpoint);
		deform->pointToIndex(3, subpoint, cindex);
		for(int ii=0; ii<3; ii++) {
			sub2atl[ii] = definterp(cindex[0], cindex[1], cindex[2], ii);
			atl2sub[ii] = sub2atl[ii];
			atlpoint[ii] = subpoint[ii] + atl2sub[ii];
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
	OrderIter<double> ait(atldef);
	// set atlas deform to NANs
	for(ait.goBegin(); !ait.eof(); ++ait) 
		ait.set(NAN);

	// at each point in atlas try to find the best mapping in the subject by
	// going back and forth. Since the mapping from sub to atlas is ground truth
	// we need to keep checking until we find a point in the subject that maps
	// to our current atlas location
	for(ait.goBegin(); !ait.eof(); ) {
		ait.index(3, index);
		cout << index[0] << "|" << index[1] << "|" << index[2] << "\r";
		atldef->indexToPoint(3, index, atlpoint.data());

		double dist = INFINITY;
		auto result = tree.nearest(atlpoint, dist);
			
		for(size_t ii=0; ii<3; ii++) {
			atl2sub[ii] = result->m_data[ii];
		}

		// SUB <- ATLAS (given)
		//    atl2sub
		double prevdist = dist+1;
		for(size_t jj = 0 ; fabs(prevdist-dist) > 0 && dist > MINERR && 
						jj < MAXITERS; jj++) {

			for(size_t ii=0; ii<3; ii++) 
				subpoint[ii] = atlpoint[ii] + atl2sub[ii];

//			// atlpoint-sub2atl = subpoint
//			for(size_t ii=0; ii<3; ii++) 
//				subpoint[ii] = atlpoint[ii] + atl2sub[ii];
			mask->pointToIndex(3, subpoint, cindex);
			if(maskinterp(cindex[0], cindex[1], cindex[2]) == 0) 
				break;

			// (estimate) SUB <- ATLAS (given)
			//              offset
			// interpolate new offset at subpoint
			deform->pointToIndex(3, subpoint, cindex);
			for(size_t ii=0; ii<3; ii++) {
				sub2atl[ii] = definterp(cindex[0], cindex[1], cindex[2], ii);
				err[ii] = atl2sub[ii]+sub2atl[ii];
			}

			// compute error
			prevdist = dist;
			dist = 0;
			for(size_t ii=0; ii<3; ii++) {
				atl2sub[ii] += -LAMBDA*err[ii];
				dist += err[jj]*err[jj];
			}
			dist = sqrt(dist);
		}

		// save out final deform
		for(size_t ii=0; ii<3; ii++, ait++) 
			ait.set(atl2sub[ii]);

	}

	return 0;
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

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.", 
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_mask("m", "mask", "Mask image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_deform("d", "deform", "Deformation field.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_atlas("a", "atlas", "Atlas image (image where "
			"indices in the deform refer to).",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<size_t> a_xsize("x", "xsize", "output image size in x dim.",
			false, 0, "size", cmd);
	TCLAP::ValueArg<size_t> a_ysize("y", "ysize", "Output image size y dim.",
			false, 0, "size", cmd);
	TCLAP::ValueArg<size_t> a_zsize("z", "zsize", "Output image size z dim.",
			false, 0, "size", cmd);
	TCLAP::ValueArg<double> a_xorigin("X", "xorigin", "output image origin [x dim].",
			false, 0, "xcoord", cmd);
	TCLAP::ValueArg<double> a_yorigin("Y", "yorigin", "Output image origin [y dim].",
			false, 0, "ycoord", cmd);
	TCLAP::ValueArg<double> a_zorigin("Z", "zorigin", "Output image origin [z dim].",
			false, 0, "zcoord", cmd);
	TCLAP::SwitchArg a_invert("I", "invert", "Invert deformatoin.", cmd);
	TCLAP::ValueArg<int> a_dilate("D", "dilate", "Number of times to "
			"dilate mask.", false, 2, "iters", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	std::shared_ptr<MRImage> inimg(readMRImage(a_in.getValue()));
	if(inimg->ndim() > 4 || inimg->ndim() < 3) {
		cerr << "Expected input to be 3D/4D Image!" << endl;
		return -1;
	}
	
	std::shared_ptr<MRImage> maskimg(readMRImage(a_mask.getValue()));
	if(maskimg->ndim() != 3) {
		cerr << "Expected mask to be 3D Image!" << endl;
		return -1;
	}

	// make mask image binary
	{
		OrderIter<int> it(maskimg);
		for(it.goBegin(); !it.eof(); ++it) {
			if(*it != 0) 
				it.set(1);
		}
	}

	maskimg = dilate(maskimg, a_dilate.getValue());
	maskimg->write("maskdilate.nii.gz");
	
	std::shared_ptr<MRImage> defimg(readMRImage(a_deform.getValue()));
	if(defimg->ndim() > 5 || defimg->ndim() < 4 || defimg->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}

	std::shared_ptr<MRImage> atlas(readMRImage(a_atlas.getValue()));
	if(atlas->ndim() != 3) {
		cerr << "Expected atlas to be 3D!" << endl;
		return -1;
	}
	// convert deform to RAS space offsets
	{
		OrderIter<double> it(defimg);
		int64_t index[3];
		double cindex[3];
		double pointS[3]; //subject
		double pointA[3]; //atlas
		for(it.goBegin(); !it.eof(); ) {
			// fill index with coordinate
			it.index(3, index);
			defimg->indexToPoint(3, index, pointS);

			// convert source pixel to pointA
			for(size_t ii=0; ii<3; ii++, ++it) 
				cindex[ii] = *it;
			atlas->pointToIndex(3, cindex, pointA);
			
			// set value to offset
			// store sub2atl (vector going from subject to atlas space)
			for(int64_t ii=2; ii >= 0; --ii, --it) 
				it.set(pointA[ii] - pointS[ii]);
			++it; ++it; ++it;
		}
	}
	defimg->write("deform.nii.gz");

	if(a_invert.isSet()) {
		// create output the size of atlas, with 3 volumes in the 4th dimension
		auto idef = createMRImage({atlas->dim(0), atlas->dim(1), 
				atlas->dim(2), 3}, FLOAT64);
		//
		invert(maskimg, defimg, idef);
		idef->write("inversedef.nii.gz");
	}

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
//	for(size_t ii=0; ii<3; ii++) {
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
//	for(size_t ii=0; ii<3; ii++) 
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
//	for(size_t ii=0; ii<3; ii++) 
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

