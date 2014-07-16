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

void getBounds(shared_ptr<MRImage> deform, std::vector<double>& lowerbound, 
		std::vector<double>& upperbound)
{
	lowerbound.resize(3);
	upperbound.resize(3);

	for(size_t ii=0; ii<3; ii++) {
		lowerbound[ii] = INFINITY;
		upperbound[ii] = -INFINITY;
	}

	vector<double> offset(3);
	vector<int64_t> index;
	vector<double> point;
	Slicer it(deform->ndim(), deform->dim());
	for(it.goBegin(); !it.isEnd(); ) {
		for(size_t ii=0; ii<3; ii++, ++it) {
			offset[ii] = deform->get_dbl(*it);
		}

		index = it.index();
		deform->indexToPoint(index, point);
		for(size_t ii=0; ii<3; ii++) {
			point[ii] -= offset[ii];
			if(point[ii] < lowerbound[ii]) 
				lowerbound[ii] = point[ii];
			if(point[ii] > upperbound[ii]) 
				upperbound[ii] = point[ii];
		}
	}
}

void applyDeform(shared_ptr<MRImage> in, shared_ptr<MRImage> deform,
		shared_ptr<MRImage> out, size_t vdim)
{
	vector<double> in_index(in->ndim());
	vector<double> in_point(3);
	vector<double> deform_index(deform->ndim());
	vector<double> deform_point(deform->ndim());
	vector<int64_t> out_index(out->ndim());
	vector<double> out_point(out->ndim());
	vector<double> offset(3);
	bool outside = false;

	// map all the points
	list<size_t> slcorder;
	if(in->ndim() == 4)
		slcorder.push_back(3); // iterate through time the fastest
	
	Slicer fit(out->ndim(), out->dim(), slcorder);
	for(fit.goBegin(); !fit.isEnd(); ) {
		// for point in deform image
		out_index = fit.index();
		out->indexToPoint(out_index, out_point);
#ifdef DEBUG
		cerr << "Out: " << out_index << " | " << out_point << endl;
#endif //DEBUG

		for(size_t ii=0; ii<3; ii++)
			deform_point[ii] = out_point[ii];
		for(size_t ii=3; ii<deform_point.size(); ii++)
			deform_point[ii] = 0;

		// sampe offset at point
		deform->pointToIndex(deform_point, deform_index);
#ifdef DEBUG
		cerr << "Deform:" << deform_point << " | " << deform_index << endl;
#endif //DEBUG
		for(size_t ii=0; ii<3; ii++) {
			deform_index[vdim] = ii;
			offset[ii] = deform->linSampleInd(deform_index, ZEROFLUX, outside);
			in_point[ii] = deform_point[ii] + offset[ii];
		}

#ifdef DEBUG
		cerr << "Offset: " << offset << endl;
#endif //DEBUG
		in->pointToIndex(in_point, in_index);
#ifdef DEBUG
		cerr << "In: " << in_point << " | " << in_index << endl; 
#endif //DEBUG

		if(in->ndim() == 4){

			// run through time
			assert(in->dim(3) == out->dim(3));
			for(size_t tt=0; tt < in->dim(3); tt++, ++fit) {
				in_index[3] = tt;
				out->set_dbl(*fit, in->linSampleInd(in_index,ZEROFLUX, outside));
			}
		} else {
			
			// just do one
			assert(in->dim(3) == out->dim(3));
			out->set_dbl(*fit, in->linSampleInd(in_index, ZEROFLUX, outside));
			fit++; 
		}
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
	TCLAP::ValueArg<string> a_deform("d", "deform", "Deformation field.",
			true, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	std::shared_ptr<MRImage> in(readMRImage(a_in.getValue()));
	if(in->ndim() > 4 || in->ndim() < 3) {
		cerr << "Expected input to be 3D/4D Image!" << endl;
		return -1;
	}
	
	/**********
	 * Deform
	 *********/
	// ensure the image dimensions match our expectations
	std::shared_ptr<MRImage> deform(readMRImage(a_deform.getValue()));
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

	/*********
	 * Output
	 ********/
	// figure out field of view
	vector<double> lowerbound(3, INFINITY);
	vector<double> upperbound(3, -INFINITY);
	getBounds(deform, lowerbound, upperbound);

	cerr << "Computed Bound 1: " << lowerbound << endl;
	cerr << "Computed Bound 2: " << upperbound << endl;

	// convert bounds to indices, so we can figure out image size
	std::vector<int64_t> index1;
	std::vector<int64_t> index2;
	in->pointToIndex(lowerbound, index1);
	in->pointToIndex(upperbound, index2);
	for(size_t ii=0; ii<3; ii++) {
		int64_t tmp1 = index1[ii];
		int64_t tmp2 = index2[ii];
		index1[ii] = std::min(tmp1, tmp2);
		index2[ii] = std::max(tmp1, tmp2);
	}

	cerr << "Bounding Index 1: " << index1 << endl;
	cerr << "Bounding Index 2: " << index2 << endl;
	// create output image that covers FOV
	
	std::vector<size_t> sz;
	if(in->ndim() == 3) {
		sz.resize(3);
	} else {
		sz.resize(4);
		sz[3] = in->dim(3);
	}
	for(size_t ii=0; ii<3; ii++) 
		sz[ii] = index2[ii]-index1[ii];

	// force size from input
	if(a_xsize.isSet())
		sz[0] = a_xsize.getValue();
	if(a_ysize.isSet())
		sz[1] = a_ysize.getValue();
	if(a_zsize.isSet())
		sz[2] = a_zsize.getValue();

	cerr << "Size: " << sz << endl; 
	auto out = createMRImage(sz, FLOAT32);

	std::cerr << "Min Index: " << index1 << endl;
	
	// origin is minimum index in point space:
	std::vector<double> point;
	in->indexToPoint(index1, point);
	for(size_t ii=0; ii<3; ii++) 
		out->origin()[ii] = point[ii];

	// force size from input
	if(a_xorigin.isSet()) 
		out->origin()[0] = a_xorigin.getValue();
	if(a_yorigin.isSet()) 
		out->origin()[1] = a_yorigin.getValue();
	if(a_zorigin.isSet()) 
		out->origin()[2] = a_zorigin.getValue();

	
	std::cerr << "New Origin:\n" << out->origin() << endl;

	out->setSpacing(in->spacing());
	out->setDirection(in->direction());

	applyDeform(in, deform, out, vdim);
	
	out->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
