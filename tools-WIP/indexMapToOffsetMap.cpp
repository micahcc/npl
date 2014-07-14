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

shared_ptr<MRImage> invertDeform(shared_ptr<MRImage> in)
{
	// create KDTree
	KDTree<3, 3, float, float> tree;

	// create output image
//	shared_ptr<MRImage> out((NDArray*)in->clone().get());

	std::list<size_t> order({0,1,2});
//	itt.GoToBegin();
//	for(int rr = 0 ; !itt.IsAtEnd(); rr++) {
//		tfm->TransformIndexToPhysicalPoint(itt.GetIndex(), point);
//		def = itt.Value();
//
//		for(int dd = 0 ; dd < DIM; dd++) {
//			dpoints(rr, dd) = point[dd] + def[dd];
//			dpoints(rr, DIM+dd) = -def[dd];
//		}
//
//		++itt;
//	}
//
//	// need to interpolate when a deformation misses
//	auto interp = itk::LinearInterpolateImageFunction<V>::New();
//	interp->SetInputImage(tfm);
//
//	// build kd-tree mapping destination points to the deformation
//	// at the target point
//	std::cerr << "Building KDTree" << std::endl;
//	alglib::kdtreebuild(dpoints, nterms, DIM, DIM, 2, kdtree);
//	std::cerr << "Done" << std::endl;
//	alglib::real_1d_array px;
//	px.setlength(DIM);
////	alglib::minlmstate minstate;
////	alglib::minlmreport minrep;
//
//	int ii = 0;
//	double dist = ERROR;
//	ito.GoToBegin();
//	while(!ito.IsAtEnd()) {
//		out->TransformIndexToPhysicalPoint(ito.GetIndex(), point);
//
//		if(verbose) 
//			std::cerr << "Querying " << point << endl;
//
//		for(int dd = 0 ; dd < DIM; dd++)
//			searchpoint[dd] = point[dd];
//
//		// find point that maps to the output point
//		alglib::kdtreequeryknn(kdtree, searchpoint, 1);
//		alglib::kdtreequeryresultsxy(kdtree, dpoints);
//			
//		if(verbose) {
//			for(int dd = 0 ; dd < DIM; dd++) 
//				pt1[dd] = dpoints(0, dd);
//			std::cerr << "\tFound" << pt1 << endl;
//			for(int dd = 0 ; dd < DIM; dd++) 
//				pt1[dd] = dpoints(0, dd) + dpoints(0, DIM+dd);
//			std::cerr << "\tMapped From" << pt1 << endl;
//		}
//
//		for(int dd = 0 ; dd < DIM; dd++) 
//			def[dd] = dpoints(0, DIM+dd);
//
//		for(ii = 0 ; dist < ERROR && ii < MAXITERS; ii++) {
//			if(verbose)
//				std::cerr << "\tApprox Def:" << def << endl;
//
//			// check to see where the found point comes from
//			// check the deformation to see if this fits the minimum error
//			for(int dd = 0 ; dd < DIM; dd++) 
//				cpoint[dd] = point[dd] + def[dd];
//			
//			if(verbose)
//				std::cerr << "\tRevMap:" << point << " -> " << cpoint << endl;
//
//			if(interp->IsInsideBuffer(cpoint)) {
//				auto testdef = -interp->Evaluate(cpoint);
//				if(verbose)
//					cerr << "\tRevMapDef: " << testdef << endl;
//				
//				// should map back to point
//				for(int dd = 0 ; dd < DIM; dd++) 
//					cpoint[dd] -= testdef[dd];
//
//				if(verbose)
//					std::cerr << "\tSelfMap:" << point << " vs. " << cpoint << endl;
//				
//				// calculate selfmap error, and add it to searchpoint to improve 
//				// the result
//				dist = 0;
//				for(int dd = 0 ; dd < DIM; dd++) {
//					dist += pow(cpoint[dd]-point[dd],2);
//					def[dd] += .5*(point[dd]-cpoint[dd]);
//				}
//				dist = sqrt(dist);
//			} else {
//				// just ignore
//				dist = 0;
//			}
//		} 
//
//		if(ii == MAXITERS) 
//			cerr << "Failed, using last: " << def << endl;
//
//		ito.Set(def);
//		++ito;
//	}
//
////	error = 0;
////	ito.GoToBegin();
////	while(!ito.IsAtEnd()) {
////		out->TransformIndexToPhysicalPoint(ito.GetIndex(), point);
////
////		for(int ii = 0 ; ii < DIM; ii++)
////			pt1[ii] = point[ii] + ito.Value()[ii];
////
////		auto def = bsplineSample<V>(tfm, pt1);
////
////		for(int ii = 0 ; ii < DIM; ii++)
////			error += pow(def[ii]+ito.Value()[ii], 2);
////
////		++ito;
////	}
////	cerr << "Forward-Back Deformation Error: " << sqrt(error) << endl;
////	error = 0;
////	ito.GoToBegin();
////	while(!ito.IsAtEnd()) {
////		out->TransformIndexToPhysicalPoint(ito.GetIndex(), point);
////
////		auto def1 = bsplineSample<V>(out, point);
////		for(int ii = 0 ; ii < DIM; ii++)
////			pt1[ii] = point[ii] + def1[ii];
////
////		auto def2 = bsplineSample<V>(tfm, pt1);
////
////		for(int ii = 0 ; ii < DIM; ii++)
////			error += pow(def1[ii]+def2[ii], 2);
////
////		++ito;
////	}
////	cerr << "Pre-Parameter Error: " << sqrt(error) << endl;
////
////	
////	alglib::real_2d_array unmix;
////	alglib::real_1d_array targs;
////	alglib::real_1d_array terms;
////	unmix.setlength(nterms, nterms);
////	targs.setlength(nterms);
////	terms.setlength(nterms);
////	
////	itk::ImageRegionIteratorWithIndex<V> itii(out, out->GetRequestedRegion());
////	itk::ImageRegionIteratorWithIndex<V> itjj(out, out->GetRequestedRegion());
////	
////	// create matrix that weights points based on their distance from each other
////	itii.GoToBegin();
////	for(int ii = 0; ii < nterms; ii++) {
////		auto indexii = itii.GetIndex();
////
////		itjj.GoToBegin();
////		for(int jj = 0 ; jj < nterms; jj++) {
////			auto indexjj = itjj.GetIndex();
////			double w = 1;
////			for(int dd = 0 ; dd < 3; dd++) 
////				w *= B3kern((double)indexjj[dd] - (double)indexii[dd]);
////
////			unmix(ii, jj) = w;
////			++itjj;
////		}
////		++itii;
////	}
////
////
////	// Invert Smoothing Matrix
////	alglib::ae_int_t info;
////	alglib::matinvreport irep;
////
////	alglib::integer_1d_array pivots;
////	alglib::rmatrixlu(unmix, nterms, nterms, pivots);
////	alglib::rmatrixluinverse(unmix, pivots, info, irep);
////
////	for(int dim = 0; dim < 3; dim++) {
////		// create matrix that weights points based on their distance from each other
////		itii.GoToBegin();
////		for(int ii = 0; ii < nterms; ii++) {
////			targs[ii] = itii.Value()[dim];
////
////			++itii;
////		}
////
////		// Invert Smoothing Matrix
////		alglib::rmatrixmv(nterms, nterms, unmix,0,0,0, targs,0, terms,0);
////		
////		// Un-smooth the input
////		itii.GoToBegin();
////		for(int ii = 0; ii < nterms; ii++) {
////			itii.Value()[dim] = terms[ii];
////			++itii;
////		}
////	}
////
////	error = 0;
////	ito.GoToBegin();
////	while(!ito.IsAtEnd()) {
////		out->TransformIndexToPhysicalPoint(ito.GetIndex(), point);
////
////		auto def1 = bsplineSample<V>(out, point);
////		for(int ii = 0 ; ii < DIM; ii++)
////			pt1[ii] = point[ii] + def1[ii];
////
////		auto def2 = bsplineSample<V>(tfm, pt1);
////
////		for(int ii = 0 ; ii < DIM; ii++)
////			error += pow(def1[ii]+def2[ii], 2);
////
////		++ito;
////	}
////	cerr << "Estimated Parameter Error: " << sqrt(error) << endl;
	return NULL;
}
int main()
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

	TCLAP::ValueArg<string> a_fmri("i", "input", "Input image.", 
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_invert("I", "invert", "Whether to invert.", cmd);

	// read input
	std::shared_ptr<MRImage> deform(readMRImage(a_fmri.getValue()));

	// check that it is 4D or 5D (with time=1)
	

	if(a_invert.isSet()) {
		deform= invertDeform(deform);
	}
	

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
