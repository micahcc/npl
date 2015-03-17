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
#include "basic_functions.h"
#include "macros.h"

using std::string;
using std::shared_ptr;
using std::to_string;

using namespace Eigen;
using namespace npl;

MatrixXd reduce(shared_ptr<const MRImage> in, double expvar)
{
    if(in->ndim() != 4)
        throw INVALID_ARGUMENT("Input mmust be 4D!");

    // fill Matrix with values from input
    size_t T = in->tlen();
    size_t N = in->elements()/T;

    // fill, zero mean the timeseries
	cerr << "Filling "<<T<<"x"<<N<<" matrix"<<endl;
	MatrixXd data(T, N);
	size_t cc = 0;
	for(Vector3DConstIter<double> it(in); !it.eof(); ++it, ++cc) {
		for(size_t rr=0; rr<T; rr++)
			data(rr, cc) = it[rr];
	}
	cerr << "Done"<<endl<<"Standardizing"<<endl;
	for(size_t cc=0; cc<data.cols(); cc++) {
		double ssq = data.col(cc).squaredNorm();
		double sum = data.col(cc).sum();
		ssq = sqrt(sample_var(data.rows(), sum, ssq)); // stddev
		sum /= data.rows(); // mean

		for(size_t rr=0; rr<data.rows(); rr++) {
			if(!std::isnormal(ssq) || std::isinf(sum) || std::isnan(sum))
				data(rr,cc) = 0;
			else
				data(rr,cc) = (data(rr,cc)-sum)/ssq;
		}
	}
	cerr << "Done"<<endl;

    // perform PCA
	std::cerr << "PCA...";
	MatrixXd X_pc = pca(data, expvar);
	cerr << X_pc.col(0).transpose() << endl;
	std::cerr << "Done " << endl;

    // perform ICA
	std::cerr << "ICA...";
	MatrixXd X_ic = symICA(X_pc);
	cerr << X_pc.col(0).transpose() << endl;
	std::cerr << "Done" << endl;

    return X_ic;
}

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Perform ICA analysis on an image. ",
            ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input fMRI image.",
			true, "", "*.nii.gz", cmd);
//    TCLAP::ValueArg<string> a_components("o", "out-components", "Output "
//            "Independent Components as a 1x1xCxT image.",
//			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<double> a_varthresh("v", "var", "Variance (percentage) to "
			"keep during PCA", false, 0.99, "pct", cmd);
	TCLAP::ValueArg<string> a_pmap("p", "pmap", "Output probability map, "
			"result of regressing each IC", false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_tmap("t", "tmap", "Output t-score map, "
			"result of regressing each IC", false, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	auto inimg = readMRImage(a_in.getValue());
	if(inimg->ndim() != 4) {
		cerr << "Expected input to be 4D Image!" << endl;
		return -1;
	}

	MatrixXd X = reduce(inimg, a_varthresh.getValue());
	VectorXd y(X.rows());
	size_t nics = X.cols();

	// Add Intercept
	X.conservativeResize(NoChange, nics+1);
	X.col(nics).setOnes();

	// Cache Constants Across All Regressions
	auto Xinv = pseudoInverse(X);
	auto covInv = pseudoInverse(X.transpose()*X);
	cerr << Xinv.col(0).transpose() << endl;
	cerr << covInv.col(0).transpose() << endl;
	const double MAX_T = 1000;
	const double STEP_T = 0.1;
	StudentsT distrib(X.rows()-1, STEP_T, MAX_T);

	// Create Output Images
	vector<size_t> osize(inimg->dim(), inimg->dim()+3);
	osize.push_back(nics);

	auto timg = createMRImage(4, osize.data(), FLOAT32);
	auto pimg = createMRImage(4, osize.data(), FLOAT32);

	// Load Image as Matrix
	for(Vector3DIter<double> it(inimg), tit(timg), pit(pimg);
				!it.eof(); ++it, ++tit, ++pit) {
		// Fill Y
		for(size_t rr=0; rr<y.rows(); rr++)
			y[rr] = it[rr];

		// Perform Regression
		RegrResult reg;
		regress(&reg, y, X, covInv, Xinv, distrib);

		// Save to T,P,R images
		for(size_t cc=0; cc<nics; cc++) {
			tit.set(cc, reg.t[cc]);
			pit.set(cc, reg.p[cc]);
		}
	}

	timg->write(a_tmap.getValue());
	pimg->write(a_pmap.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return 0;
}


