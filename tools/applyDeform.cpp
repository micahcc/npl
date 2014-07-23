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

double gaussian(double x)
{
	const double PI = acos(-1);
	const double den = sqrt(2*PI);
	return exp(x*x)*den;
}

double gaussianFilter(shared_ptr<MRImage> in, double sd)
{
	std::vector<size_t> rvector(in->ndim(), 0);
	auto out = in->cloneImage();

	// split up and perform on each dimension separately
	for(size_t dd=0; dd<in->ndim(); dd++) {

		// create iterators, make them have the same order
		KernelIter it(in);
		OrderIter oit(out);
		oit.setOrder(it.getOrder());

		// construct weights array
		double normsd = sd/in->spacing()[dd];
		int64_t rad = 2*normsd;
		double weights[rad*2+1];
		for(int64_t oo = -rad; oo <= rad; oo++) 
			weights[oo+rad] = gaussian(oo/normsd);

		// set radius
		rvector[dd] = rad;
		it.setRadius(rvector);

		// apply kernel
		for(it.goBegin(); !it.eof(); ++it) {
			double vv = 0;
			assert(it.ksize() == rad*2+1);
			for(int64_t oo=-rad; oo <= rad; oo++) {
				vv += weights[oo+rad]*it[oo+rad];
			}
				
			oit.set(vv);
		}

		// reset radius vector to 0
		rvector[dd] = 0;
	}

	return out;
}

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
	std::vector<int64_t> index1(in->ndim(), 0);
	std::vector<int64_t> index2(in->ndim(), 0);
	auto prev = in->cloneImage();
	auto out = in->cloneImage();
	shared_ptr<MRImage> tmp; 
	for(size_t rr=0; rr<reps; rr++) {
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
			for(size_t ii=0; ii<in->ndim(); ii++) {
				if(index1[ii] != index2[ii]) {
					throw std::logic_error("Error differece in iteration!");
				}
			}

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
	}

	return out;
}

void deriv(shared_ptr<MRImage> in, shared_ptr<MRImage>& dx, 
		shared_ptr<MRImage>& dy, shared_ptr<MRImage>& dz)
{
	Vector3DView<double> inview(in);
	int64_t index[3];
	dy = in->cloneImage();
	dz = in->cloneImage();

	// dx
	dx = in->cloneImage();
	for(Vector3DIter<double> it(dx); !it.eof(); ++it) {
		it.index(3, index);
		if(index[0] == 0) {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, inview(index[0]+1, index[1], index[2], tt) -
						inview(index[0], index[1], index[2], tt));
			}
		} else if(index[0] >= in->dim(0)) {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, inview(index[0], index[1], index[2], tt) -
						inview(index[0]-1, index[1], index[2], tt));
			}
		} else {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, .5*(inview(index[0]+1, index[1], index[2], tt) -
						inview(index[0]-1, index[1], index[2], tt)));
			}
		}
	}
	
	// dy
	dy = in->cloneImage();
	for(Vector3DIter<double> it(dy); !it.eof(); ++it) {
		it.index(3, index);
		if(index[1] == 0) {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, inview(index[0], index[1]+1, index[2], tt) -
						inview(index[0], index[1], index[2], tt));
			}
		} else if(index[1] >= in->dim(1)) {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, inview(index[0], index[1], index[2], tt) -
						inview(index[0], index[1]-1, index[2], tt));
			}
		} else {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, .5*(inview(index[0], index[1]+1, index[2], tt) -
						inview(index[0], index[1]-1, index[2], tt)));
			}
		}
	}
	
	// dz
	dz = in->cloneImage();
	for(Vector3DIter<double> it(dz); !it.eof(); ++it) {
		it.index(3, index);
		if(index[2] == 0) {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, inview(index[0], index[1], index[2]+1, tt) -
						inview(index[0], index[1], index[2], tt));
			}
		} else if(index[2] >= in->dim(2)) {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, inview(index[0], index[1], index[2], tt) -
						inview(index[0], index[1], index[2]-1, tt));
			}
		} else {
			for(size_t tt=0; tt<it.tlen(); tt++) {
				it.set(tt, .5*(inview(index[0], index[1], index[2]-1, tt) -
						inview(index[0], index[1], index[2]+1, tt)));
			}
		}
	}
}

/**
 * @brief Inverts a deformation, given a mask in the target space.
 *
 * @param mask Mask in target space (where the current deform maps from)
 * @param deform Maps from mask space.
 * @param atldef Maps to mask space, should be same size as mask
 *
 * @return 
 */
int invert(shared_ptr<MRImage> mask, shared_ptr<MRImage> deform, 
		shared_ptr<MRImage> atldef)
{

	shared_ptr<MRImage> imgdx, imgdy, imgdz;
	deriv(deform, imgdx, imgdy, imgdz);

	const double MINERR = .1;
	const double LAMBDA = .2;
	const size_t MAXITERS = 300;
	int64_t index[3];
	double subpoint[3];
	double cindex[3];
	double sub2atl[3];
	Matrix<3,1> err;
	vector<double> atlpoint(3);
	vector<double> atl2sub(3);
	Vector3DView<double> defview(deform);
	LinInterp3DView<double> definterp(deform);
	NNInterp3DView<double> maskinterp(mask);
	KDTree<3,3,double, double> tree;
	int64_t neighbors[3][2];
	Matrix<3,3> dVdX;

	if(deform->tlen() != 3) {
		cerr << "Error invalid deform image, needs 3 points in the 4th or "
			"5th dim" << endl;
		return -1;
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

		for(int ii=0; ii<3; ii++) {
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
		
		cout << setw(10) << index[0] << setw(10) << index[1] << setw(10) << index[2] << "\r";

		ait.index(3, index);
		atldef->indexToPoint(3, index, atlpoint.data());

		double dist = 50;
		auto result = tree.nearest(atlpoint, dist);
		if(!result)
			continue;

		for(size_t ii=0; ii<3; ii++) 
			atl2sub[ii] = result->m_data[ii];

//		// SUB <- ATLAS (given)
//		//    atl2sub
//		double prevdist = dist+1;
//		size_t iters = 0;
//		for(iters = 0 ; fabs(prevdist-dist) > 0 && dist > MINERR && 
//						iters < MAXITERS; iters++) {
//
//			for(size_t ii=0; ii<3; ii++) 
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
//			// compute the neighbors of the nearest point, and update sub2atl
//			// with interpolation
//			for(size_t ii=0; ii<3; ii++) {
//				neighbors[ii][0] = floor(cindex[ii]);
//				neighbors[ii][1] = neighbors[ii][0]+1;
//			}
//
//			/* Update */
//
//			// update sub2atl
//			for(size_t ii=0; ii<3; ii++) {
//				sub2atl[ii] = definterp(cindex[0], cindex[1], cindex[2], ii);
//			}
//
//			for(size_t ii=0; ii<3; ii++) {
//				for(size_t jj=0; jj<3; jj++) {
//					cindex[jj]+=0.00001;
//					dVdX(ii,jj) = (definterp(cindex[0], cindex[1], cindex[2], ii)-sub2atl[ii])/0.00001;
//					cindex[jj]-=0.00001;
//				}
//			}
//			for(size_t ii=0; ii<3; ii++)
//				dVdX(ii,ii) += 1;
//
//			// update image with the error, using the derivative to estimate
//			// where error crosses 0
//			prevdist = dist;
//			dist = 0;
//			for(size_t ii=0; ii<3; ii++) {
//				err[ii] = atl2sub[ii]+sub2atl[ii];
//			}
//
//			err = inverse(dVdX)*err;
//			for(size_t ii=0; ii<3; ii++) {
//				atl2sub[ii] -= LAMBDA*err[ii];
//				dist += err[ii]*err[ii];
//			}
//			dist = sqrt(dist);
//		}

		// save out final deform
		for(size_t ii=0; ii<3; ii++) 
			ait.set(ii, atl2sub[ii]);

	}

	return 0;
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

	TCLAP::CmdLine cmd("Applies 3D deformation to volume or time-series of "
			"volumes. Deformation should be a map of offsets. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.", 
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_mask("m", "mask", "Mask image in "
			"deform space.", true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_deform("d", "deform", "Deformation field.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_atlas("a", "atlas", "Atlas image.",
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
	
	std::shared_ptr<MRImage> mask(readMRImage(a_mask.getValue()));
	if(mask->ndim() != 3) {
		cerr << "Expected mask to be 3D Image!" << endl;
		return -1;
	}
	binarize(mask);

	////////////////////////////////////////////////////////////////////////////
	//   HACK
	////////////////////////////////////////////////////////////////////////////
	// calculate overlap of input and mask, sometimes brainsuite produces images
	// without correct origin, so this should throw an error if thats the case.
	{
	int64_t index[3];
	double point[3];
	size_t incount = 0;
	size_t maskcount = 0;
	for(OrderIter<int64_t> it(mask); !it.eof(); ++it) {
		it.index(3, index);
		mask->indexToPoint(3, index, point);
		maskcount++;
		incount += (inimg->pointInsideFOV(3, point));
	}
	double f = (double)(incount)/(double)(maskcount);
	if(f < .5)  {
		cerr << "Warning the input and mask images do not overlap very much."
			" This could indicate bad orientation, overlap: " << f << endl;
		return -1;
	}
	}
	////////////////////////////////////////////////////////////////////////////
	
	std::shared_ptr<MRImage> atlas(readMRImage(a_atlas.getValue()));
	if(atlas->ndim() != 3) {
		cerr << "Expected mask to be 3D Image!" << endl;
		return -1;
	}

	std::shared_ptr<MRImage> defimg(readMRImage(a_deform.getValue()));
	if(defimg->ndim() > 5 || defimg->ndim() < 4 || defimg->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}

	// convert deform to RAS space offsets
	{
		Vector3DIter<double> it(defimg);
		if(it.tlen() != 3) 
			cerr << "Error expected 3 volumes!" << endl;
		int64_t index[3];
		double cindex[3];
		double pointS[3]; //subject
		double pointA[3]; //atlas
		for(it.goBegin(); !it.eof(); ++it) {
			// fill index with coordinate
			it.index(3, index);
			defimg->indexToPoint(3, index, pointS);

			// convert source pixel to pointA
			for(size_t ii=0; ii<3; ii++) {
				cindex[ii] = it[ii];
			}
			//since values in deformation vector are indices in atlas space
			atlas->indexToPoint(3, cindex, pointA);
			// set value to offset
			// store sub2atl (vector going from subject to atlas space)
			for(int64_t ii=0; ii < 3; ii++) 
				it.set(ii, pointA[ii] - pointS[ii]);
		}
	}
	defimg->write("deform.nii.gz");

	if(a_invert.isSet()) {
		// create output the size of atlas, with 3 volumes in the 4th dimension
		auto idef = createMRImage({atlas->dim(0), atlas->dim(1), 
					atlas->dim(2), 3}, FLOAT64);
		idef->setDirection(atlas->direction(), true);
		idef->setSpacing(atlas->spacing(), true);
		idef->setOrigin(atlas->origin(), true);
		invert(mask, defimg, idef);
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

