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
#include "iterators.h"
#include "statistics.h"

using std::string;
using std::shared_ptr;

using namespace npl;

Matrix reduce(shared_ptr<const MRImage> in)
{
    if(in->ndim() != 4)
        throw std::invalid_argument("Input mmust be 4D!");
    
    // fill Matrix with values from input
    size_t T = in->tlen();
    size_t N = in->elements()/T;

    // fill, zero mean the timeseries
    MatrixXd data(T, N);
    ChunkConstIter<double> it(in); 
    it->setLineChunk(3);
    for(size_t xx=0; !it.eof(); it.nextChunk(), ++xx) {
        norm = 0;
        for(size_t tt=0; !it.eoc(); ++it) {
            data(tt,xx) = *it;
            norm += *it;
        }
        norm = 1./norm;
        for(size_t tt=0; !it.eoc(); ++it) 
            data(tt,xx) = data(tt,xx)*norm;
    }

    // perform PCA
	std::cerr << "PCA...";
	Matrix X_pc = pca(X);
	std::cerr << "Done " << endl;
	
    // perform ICA
	std::cerr << "ICA...";
	Matrix X_ic = ica(X_pc);
	std::cerr << "Done" << endl;

    return X_ic;
}

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Perform ICA analysis on an image. ",
            ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input fMRI image.",
			true, "", "*.nii.gz", cmd);
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
	auto inimg = readMRImage(a_in.getValue());
	if(inimg->ndim() != 4) {
		cerr << "Expected input to be 4D Image!" << endl;
		return -1;
	}

    // 
	


	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


