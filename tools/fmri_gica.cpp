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
 * @file fmri_ica.cpp Tool for performing ICA on a fMRI image.
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"
#include "macros.h"

using std::string;
using std::shared_ptr;
using std::to_string;

using namespace Eigen;
using namespace npl;

MatrixXd reduce(shared_ptr<const MRImage> in)
{
    if(in->ndim() != 4)
        throw INVALID_ARGUMENT(": Input mmust be 4D!");
	
    return X_ic;
}

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Perform ICA analysis on an image, or group of images. "
			"By default this will be a temporal ICA. You can perform "
			"group-analysis by concatinating in space or time. ORDER OF INPUTS "
			"MATTERS IF YOU CONCATINATE IN BOTH. Concatination in time occurs "
			"for adjacent arguments. So for -t 2 -s 2 -i a -i b -i c -i d"
			"a and b will be concatinated in time and c and d will be "
			"concatinated in time then ab and cd will be concatined in space.",
			' ', __version__ );

	TCLAP::SwitchArg a_spatial_ica("S", "spatial-ica", "Perform a spatial ICA"
			", reducing unmixing timepoints to produce spatially independent "
			"maps.", cmd);

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI image.",
			true, "*.nii.gz", cmd);
	TCLAP::ValueArg<int> a_time_append("t", "time-appends", "Number of images "
			"to append in the matrix of images, in the time direction.", false,
			0, "int", cmd);
	TCLAP::ValueArg<int> a_space_append("s", "space-appends", "Number of images "
			"to append in the matrix of images, in the space direction.", false,
			0, "int", cmd);

    TCLAP::ValueArg<string> a_components("o", "out-components", "Output "
            "Independent Components as a 1x1xCxT image.",
			true, "", "*.nii.gz", cmd);
    TCLAP::ValueArg<string> a_mapdir("d", "mapdir", "Output "
            "directory for ICA significance maps. The number of maps will "
            "depend on the number of components, and will be in the same "
            "space as the input fMRI image. names will be the "
            "$mapdir/$input_$num.nii.gz where $mapdir is the mapdir, $input "
            "is the basename from -i and $num is the component number",
            true, "./", "/", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	if(!a_in.isSet()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

	if(a_spatial_ica.isSet()) {
		cout << "Performing Spatial ICA" << endl;
		// Covariance Method (since presumably there will be more voxels
		// than timepoints
		MatrixXd cov;
		VectorXd row;
		VectorXd mu;
		MatrixXd timecat;
		
		// fill Matrix with values from inputs, skip first (just use already-read)
		size_t subT = 0;
		size_t subV = 0;
		size_t totalT = 0;
		size_t totalV = 0;
		size_t ii = 0;
		for(size_t ss = 0; ss < a_space_append.getValue(); ss++) {
			for(size_t tt = 0; tt < a_time_append.getValue(); tt++, ii++) {
				auto img = readMRImage(a_in.getValue()[ii]);

				// Initialize subT/subV on first round
				if(subT == 0 && subV == 0) {
					subT = img->tlen();
					subV = img->elements()/img->tlen();
					cov.resize(subT, subT);
					row.resize(subT);
					mu.resize(subT);
					timecat.resize(subV*a_space_append.getValue(), subT);
				}

				// Check Input Size
				if(img->tlen() != subT) {
					cerr << "Image has different number of timepoints from the "
						"rest " << a_in.getValue()[ii] << endl;
					return -1;
				}
				if(img->elements()/img->tlen() != subV) {
					cerr << "Image has different number of voxels from the "
						"rest " << a_in.getValue()[ii] << endl;
					return -1;
				}

				// Add each timeseries to the matrix
				size_t r = 0;
				for(Vector3DIter<double> it(img); !it.eof(); ++r, ++it) {
					for(size_t c=0; c<img->tlen(); c++) 
						row[c] = it[c];
					mu += row;
					cov += row.transpose()*row;
				}
			}
		}

		// fill, zero mean the timeseries
		MatrixXd data(T, N);
		ChunkConstIter<double> it(in); 
		it.setLineChunk(3);
		for(size_t xx=0; !it.eof(); it.nextChunk(), ++xx) {
			double tmp = 0;
			for(size_t tt=0; !it.eoc(); ++it) {
				data(tt,xx) = *it;
				tmp += *it;
			}
			tmp = 1./tmp;
			for(size_t tt=0; !it.eoc(); ++it) 
				data(tt,xx) = data(tt,xx)*tmp;
		}

		// perform PCA
		std::cerr << "PCA...";
		MatrixXd X_pc = pca(data, 0.01);
		std::cerr << "Done " << endl;
    
		// perform ICA
		std::cerr << "ICA...";
		MatrixXd X_ic = ica(X_pc, 0.01);
		std::cerr << "Done" << endl;

	} else {
		// SVD Method (since presumably there will be more timepoints than
		// voxels)
		cout << "Performing Temporal ICA" << endl;

		MatrixXd data;
		size_t times = 0;
		size_t voxels = 0;
		size_t row = 0;
		size_t col = 0;
		size_t rr = 0;
		size_t cc = 0;
		for(size_t ii = 0; ii<a_in.getValue().size(); ++ii) {
			auto tmp = readMRImage(a_in.getValue()[ii]);
			data.resize(data.rows()+times, data.cols()+voxels);

			rr = 0;
			cc = 0;
			for(Vector3DIter<double> it(tmp); !it.eof(); ++it) {
				cc = 0;
				for(size_t tt=0; tt<tmp->tlen(); cc++, tt++) {
					data(times+tt, 
				}
			}
			
			// fill starting at T, V
			times += tmp->tlen();
			voxels += tmp->elements()/tmp->tlen();
			
			col++;
			if(col >= a_time_append.getValue())
				col = 0;
		}
		
		// fill Matrix with values from input
		size_t T = in->tlen();
		size_t N = in->elements()/T;

		// fill, zero mean the timeseries
		MatrixXd data(T, N);
		ChunkConstIter<double> it(in); 
		it.setLineChunk(3);
		for(size_t xx=0; !it.eof(); it.nextChunk(), ++xx) {
			double tmp = 0;
			for(size_t tt=0; !it.eoc(); ++it) {
				data(tt,xx) = *it;
				tmp += *it;
			}
			tmp = 1./tmp;
			for(size_t tt=0; !it.eoc(); ++it) 
				data(tt,xx) = data(tt,xx)*tmp;
		}

		// perform PCA
		std::cerr << "PCA...";
		MatrixXd X_pc = pca(data, 0.01);
		std::cerr << "Done " << endl;
    
		// perform ICA
		std::cerr << "ICA...";
		MatrixXd X_ic = ica(X_pc, 0.01);
		std::cerr << "Done" << endl;
	}

//    // 
//	MatrixXd regressors = reduce(inimg);
//    for(size_t cc = 0; cc < regressors.cols(); cc++) {
//        // perform regression
//        //RegrResult tmp = regress(inimg, regressors.row(cc));
//
//        // write out each of the images
//        //tmp.rsqr->write("rsqr_"+to_string(cc)+".nii.gz");
//        //tmp.T->write("T_"+to_string(cc)+".nii.gz");
//        //tmp.p->write("p_"+to_string(cc)+".nii.gz");
//        //tmp.beta->write("beta_"+to_string(cc)+".nii.gz");
//    }

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return 0;
}


