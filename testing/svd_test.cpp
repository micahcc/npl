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
 * @file svd_test.cpp Test of truncated SVD algorithm from Eigen/unsupported
 *
 *****************************************************************************/

#include "version.h"
#include "statistics.h"
#include "npltypes.h"
#include "fmri_inference.h"

#include <iostream>
#include <iomanip>
#include <map>
#include <fstream>
#include <tclap/CmdLine.h>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

using namespace std;
using namespace npl;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Performs an SVD. If an input matrix is given then it "
			"performs the SVD on that matrix, otherwise it creates a random "
			"SVD-able matrix and compares the true SVD with the estimated. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input Matrix. Format: "
			"size_t size_t double double ...., where the first two size_t's "
			"are the number rows and columns and then the matrix data "
			"immediately follow.", false, "", "mat.bin", cmd);
	TCLAP::SwitchArg a_verbose("v", "verbose", "Print matrices", cmd);

	TCLAP::ValueArg<size_t> a_rows("r", "rows", "Rows in random matrix (only "
		"applied if no input matrix given.", false, 10, "rows", cmd);
	TCLAP::ValueArg<size_t> a_cols("c", "cols", "Cols in random matrix (only "
		"applied if no input matrix given.", false, 10, "cols", cmd);
	TCLAP::ValueArg<size_t> a_rank("k", "rank", "Rank of random matrix (only "
		"applied if no input matrix given.", false, 10, "rank", cmd);

	TCLAP::ValueArg<double> a_svthresh("", "sv-thresh", "During dimension "
			"reduction, A singular value will be considered nonzero if its "
			"value is strictly greater than "
			"|singular value| < threshold x |max singular value|. "
			"By default this is 0.01.",
			false, 0.01, "ratio", cmd);
	TCLAP::ValueArg<double> a_deftol("", "dtol", "Deflation tolerance in "
			"eigenvalue computation. Larger values will result in fewer "
			"singular values. ", false, 1e-8, "tol", cmd);
	TCLAP::ValueArg<int> a_iters("", "iters", "Maximum number of iterations "
			"during eigenvalue computation of U or V matrix. Default: -1 "
			"(max of rows/cols)", false, -1, "iters", cmd);
	TCLAP::ValueArg<int> a_startvecs("", "nsimul", "Number of simultaneus "
			"vectors to expand in Kyrlov Subspace. ", false, -1, "#vec", cmd);

	TCLAP::ValueArg<string> a_out("R", "randmat", "Output generated (random) "
			"matrix.", false, "", "mat.bin", cmd);

	cmd.parse(argc, argv);

	MatrixXd A;
	MatrixXd U;
	MatrixXd V;
	VectorXd E;
	if(a_in.isSet()) {
		MatMap mat(a_in.getValue());
		A = mat.mat;
	} else {
		U.resize(a_rows.getValue(), a_rank.getValue());
		V.resize(a_cols.getValue(), a_rank.getValue());
		E.resize(a_rank.getValue());

		// Create Sorted (Decreasing) Singular Values
		cerr<<"Creating Sorted E"<<endl;
		E.setRandom();
		E = E.cwiseAbs();
		std::sort(E.data(), E.data()+E.rows(),
				[](double l, double r){return r < l;});

		// Create Orthogonal U
		cerr<<"Creating U"<<endl;
		U.setRandom();
		for(size_t cc=0; cc<U.cols(); cc++){

			// Modified Gram-Schmidt
			for(size_t c1 = 0; c1 < cc; c1++)
				U.col(cc) -= U.col(cc).dot(U.col(c1))*U.col(c1);
			U.col(cc).normalize();
		}

		// Create Orthogonal V
		cerr<<"Creating V"<<endl;
		V.setRandom();
		for(size_t cc=0; cc<V.cols(); cc++){

			// Modified Gram-Schmidt
			for(size_t c1 = 0; c1 < cc; c1++)
				V.col(cc) -= V.col(cc).dot(V.col(c1))*V.col(c1);
			V.col(cc).normalize();
		}

		cerr<<"Creating A"<<endl;
		if(a_verbose.isSet()) {
			cerr<<"U:\n"<<U<<endl;
			cerr<<"E:\n"<<E.transpose()<<endl;
			cerr<<"V:\n"<<V<<endl;
			cerr<<"A:\n"<<A<<endl;
		}
		A = U*E.asDiagonal()*V.transpose();

	}

	cerr<<"Performing SVD ("<<A.rows()<<"x"<<A.cols()<<")"<<endl;
	Eigen::BDCSVD<MatrixXd> svd(A, Eigen::ComputeThinV|Eigen::ComputeThinU);
//	Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinV|Eigen::ComputeThinU);
//	Eigen::TruncatedLanczosSVD<MatrixXd> svd;
//	svd.setThreshold(a_svthresh.getValue());
//	svd.setDeflationTol(a_deftol.getValue());
//	svd.setLanczosBasis(a_startvecs.getValue());
//	svd.setMaxIters(a_iters.getValue());
//	svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	cerr<<"Done"<<endl;

	cerr<<"Rank: "<<svd.rank()<<endl;
	if(a_verbose.isSet()) {
		cerr<<"Estimate"<<endl;
		cerr<<"U:\n"<<svd.matrixU()<<endl;
		cerr<<"E:\n"<<svd.singularValues().transpose()<<endl;
		cerr<<"V:\n"<<svd.matrixV()<<endl;
	}

	if(a_in.isSet()) {
		// Talk about results
	} else {
		// Comare
		double err;
		err = 0;
		for(size_t ii=0; ii<min<int>(U.cols(), svd.rank()); ii++)
			err += std::abs(U.col(ii).dot(svd.matrixU().col(ii)))-1;
		cerr<<"U Error: "<<err<<endl;

		err = 0;
		for(size_t ii=0; ii<min<int>(V.cols(), svd.rank()); ii++)
			err += std::abs(V.col(ii).dot(svd.matrixV().col(ii)))-1;
		cerr<<"V Error: "<<err<<endl;

		err = 0;
		for(size_t ii=0; ii<min<int>(V.cols(), svd.rank()); ii++)
			err += std::abs(svd.singularValues()[ii]-E[ii]);
		cerr<<"S Error: "<<err<<endl;
	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr<<"error: "<<e.error()<<" for arg "<<e.argId()<<std::endl;}

	return 0;
}

