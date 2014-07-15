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

#include "mrimage.h"
#include "kdtree.h"

using std::string;
using namespace npl;
using std::shared_ptr;

#define DEBUG

int64_t clamp(int64_t low, int64_t high, int64_t v)
{
	return std::max(low, std::min(high, v));
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

/* Linear Kernel Sampling */
double linKern(double x, double a)
{
	return fabs(1-fmin(1,fabs(x/a)))/a;
}

/* Linear Kernel Sampling */
double dLinKern(double x, double a)
{
	if(x < -a || x > a)
		return 0;
	if(x < 0)
		return 1/a;
	else
		return -1/a; 
}

double interp(MRImage* img, std::vector<float> cindex, 
			double rad, double(*kfunc)(double, double))
{
	if(cindex.size() != img->ndim()) {
		throw std::out_of_range("cindex size does not match image dimensions");
	}

	int DIM = std::max(cindex.size(), img->ndim());
	std::vector<size_t> index(cindex.size(), 0);
	
	//kernels are essentially 1D, so we can save time by combining 1D kernls
	//rather than recalculating
	int kpoints = rad*2+1;
	vector<double> karray(DIM*kpoints);
	for(int dd = 0; dd < DIM; dd++) {
		for(double ii = -rad; ii <= rad; ii++) {
			int nearpoint = round(cindex[dd]+ii);
			karray[dd*kpoints+(int)(ii+rad)] = kfunc(nearpoint-cindex[dd], rad);
		}
	}

	double pixval = 0;
	double weight = 0;
	div_t result;
	//iterator over points in the neighborhood
	for(int ii = 0 ; ii < pow(kpoints, DIM); ii++) {
		weight = 1;
		
		//convert to local index, compute weight
		result.quot = ii;
		for(int dd = 0; dd < DIM; dd++) {
			result = std::div(result.quot, kpoints);
			weight *= karray[dd*kpoints+result.rem];
			index[dd] = clamp(0, img->dim(dd), round(cindex[dd]+result.rem-rad));
		}
	
//		std::cerr << "Point: " << ii << " weight: " << weight << " Cont. Index: " 
//			<< cindex << ", Index: " << index << endl;
		pixval += weight*img->get_dbl(index.size(), index.data());
	}

	return pixval;
}

shared_ptr<MRImage> invertDeform(shared_ptr<MRImage> in, size_t vdim)
{
	// create KDTree
	KDTree<3, 3, float, float> tree;
	const double MINERR = .5;
	const size_t MAXITERS = 100;
	vector<float> err(3,0);
	vector<float> tmp(3,0);
	vector<float> src(3,0);
	vector<float> trg(3,0);
	vector<size_t> index(in->ndim(), 0);
	vector<float> cindex(in->ndim(), 0);

	// map every current point back to its source
	for(index[0]=0; index[0]<in->dim(0); index[0]++) {
		for(index[1]=0; index[1]<in->dim(1); index[1]++) {
			for(index[2]=0; index[2]<in->dim(2); index[2]++) {

				// get full vector, 
				for(size_t ii=0; ii<3; ii++) {
					index[vdim] = ii;
					src[ii] = in->get_dbl(index.size(), index.data());
					trg[ii] = index[ii];
				}

				// add point to kdtree
				tree.insert(src, trg);
			}
		}
	}

	// Tree Stores [target] = source
	tree.build();

	shared_ptr<MRImage> out = in->cloneImg();

	// go through output image and find nearby points that map 
	for(index[0]=0; index[0]<in->dim(0); index[0]++) {
		for(index[1]=0; index[1]<in->dim(1); index[1]++) {
			for(index[2]=0; index[2]<in->dim(2); index[2]++) {
				for(size_t ii=0; ii<3; ii++)
					trg[ii] = index[ii];

				// find point that maps near the current in
				double dist = INFINITY;
				auto result = tree.nearest(trg, dist);

				if(!result) 
					throw std::logic_error("Deformation too large!");

				// we want the forward source = reverse target 
				// and reverse source = forward target
//				fwdtrg.assign(result->m_point, result->m_point+3);
//				fwdsrc.assign(result->m_data, result->m_data+3);

				// intiailize theoretical source
				for(size_t jj=0; jj<3; jj++) {
					src[jj] = result->m_data[jj];
				}

				// use the error in app_target vs target to find a 
				// source that fits
				for(size_t ii = 0 ; dist > MINERR && ii < MAXITERS; ii++) {
#ifdef DEBUG
					std::cerr << trg  << " <- ?? " << src << " ?? " << endl;
#endif //DEBUG

					// forward map computed src (fwdtrg)
					// load new source into cindex
					for(size_t jj=0; jj<3; jj++) {
						cindex[jj] = src[jj];
					}

					for(size_t jj=0; jj<3; jj++) {
						cindex[vdim] = jj;
						tmp[jj] = interp(in.get(), cindex, 1, linKern);
					}
#ifdef DEBUG
					std::cerr << trg  << " vs " << tmp << " <- " << src << endl;
#endif //DEBUG
					dist = 0;
					for(size_t jj=0; jj<3; jj++) {
						err[jj] = trg[jj]-tmp[jj];
						dist += err[jj]*err[jj];
						src[jj] = src[jj]+.5*err[jj];
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
					out->set_dbl(index.size(), index.data(), src[ii]);
				}

			}
		}
	}
	return out;
}

int main(int argc, char** argv)
{
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
	TCLAP::ValueArg<string> a_mask("m", "mask", "Input image.", 
			true, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_invert("I", "invert", "Whether to invert.", cmd);
	TCLAP::SwitchArg a_nohack("H", "dont-hack-ones", "Brainsute make 1 the "
			"default for self-maps. However for the purpose of inversion this "
			"will screw things up. Most the times making values that map to "
			"1,1,1 self map instead is harmless. If you are using a map that "
			"doesn't do this, then you can safely disable this (-H)." , cmd);

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
	std::vector<size_t> index;
	std::list<size_t> order({vdim});
	Slicer it(deform->ndim(), deform->dim(), order);

	// convert deform to offsets

	if(a_mask.isSet()) {

		// everything outside the mask takes on the nearest value
		// from inside the mask
		
		std::shared_ptr<MRImage> mask(readMRImage(a_mask.getValue()));

		//TODO
		// for each point in mask add point and offset to KDTree
		//
		// for each point outside mask
		// find nearest point in kdtree
		// copy offset to point

		cerr << "Making [1,1,1] values self-mapping" << endl;

		it.goBegin();
		while(!it.isEnd()) {
			bool allones = true;
			for(size_t ii=0; ii<3; ii++, ++it) {
				if(fabs(deform->get_dbl(*it) - 1) > 1) 
					allones = false;
			}

			// if allones, convert deform value to current index
			if(allones) {
				--it; --it; --it;
				it.get_index(index);
				for(size_t ii=0; ii<3; ii++, ++it) 
					deform->set_dbl(*it, index[ii]);
			}
		}
		cerr << "Done" << endl;
	}

	deform->write("fixed.nii.gz");

	if(a_invert.isSet()) {
		deform = invertDeform(deform, vdim);
	}
	deform->write("inverted.nii.gz");
	
	// convert each coordinate to an offset
	
	const auto& spacing = deform->space();
	const auto& dir = deform->direction();

	std::vector<double> offset(deform->ndim(), 0);
	std::vector<double> pointoffset(deform->ndim(), 0);
	it.goBegin();
	while(!it.isEnd()) {
		it.get_index(index);

		// each value in deform is an index, so subtract current index from the
		// source index to get the offset
		for(size_t ii=0; ii<3; ii++, ++it) 
			offset[ii] = (deform->get_dbl(*it)-index[ii])*spacing[ii];

		dir.mvproduct(offset, pointoffset);

		--it; --it; --it;
		for(size_t ii=0; ii<3; ii++, ++it) {
			deform->set_dbl(*it, pointoffset[ii]);
		}
	}

	deform->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
