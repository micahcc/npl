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
 * @file prob_svd_test.cpp test of probabilistic SVD (from Halko et al)
 *
 *****************************************************************************/

#include "version.h"
#include "statistics.h"
#include "npltypes.h"
#include "ica_helpers.h"

#include <iostream>
#include <iomanip>
#include <map>
#include <fstream>
#include <tclap/CmdLine.h>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

using namespace std;
using namespace npl;

TCLAP::SwitchArg a_verbose("v", "verbose", "Write more output");

template <typename T>
int testConstErr(size_t rows, size_t cols, size_t rank, size_t startrank, double esterr)
{
	double thresh = 0.01;

	MatrixXd A(rows, cols);
	MatrixXd U(rows, rank);
	MatrixXd V(cols, rank);
	VectorXd E(rank);

	// Create Sorted (Decreasing) Singular Values
	cerr<<"Creating Sorted E"<<endl;
	E.setRandom();
	E = E.cwiseAbs();
	std::sort(E.data(), E.data()+E.rows(),
			[](double l, double r){return r < l;});

	// Create Orthonormal U
	cerr<<"Creating U"<<endl;
	U.setRandom();
	for(size_t cc=0; cc<U.cols(); cc++){
		// Modified Gram-Schmidt
		for(size_t c1 = 0; c1 < cc; c1++)
			U.col(cc) -= U.col(cc).dot(U.col(c1))*U.col(c1);
		U.col(cc).normalize();
	}

	// Create Orthonormal V
	cerr<<"Creating V"<<endl;
	V.setRandom();
	for(size_t cc=0; cc<V.cols(); cc++){
		// Modified Gram-Schmidt
		for(size_t c1 = 0; c1 < cc; c1++)
			V.col(cc) -= V.col(cc).dot(V.col(c1))*V.col(c1);
		V.col(cc).normalize();
	}

	cerr<<"Creating A"<<endl;
	A = U*E.asDiagonal()*V.transpose();
	if(a_verbose.isSet()) {
		cerr<<"U:\n"<<U<<endl;
		cerr<<"E:\n"<<E.transpose()<<endl;
		cerr<<"V:\n"<<V<<endl;
		cerr<<"A:\n"<<A<<endl;
	}

	MatrixXd estU, estV;
	VectorXd estE;
	randomizePowerIterationSVD(A, esterr, startrank,
			std::max(A.rows(), A.cols()), 2, estU, estE, estV);
	cerr << E.transpose() << endl;
	cerr << estE.transpose() << endl;

	// Comare
	double err;
	err = 0;
	for(size_t ii=0; ii<min<int>(U.cols(), estU.cols()); ii++)
		err += std::abs(U.col(ii).dot(estU.col(ii)))-1;
	cerr<<"U Error: "<<err<<endl;
	if(err > thresh)
		return -1;

	err = 0;
	for(size_t ii=0; ii<min<int>(V.cols(), estV.cols()); ii++)
		err += std::abs(V.col(ii).dot(estV.col(ii)))-1;
	cerr<<"V Error: "<<err<<endl;
	if(err > thresh)
		return -1;

	err = 0;
	for(size_t ii=0; ii<min<int>(V.cols(), estV.cols()); ii++)
		err += std::abs(estE[ii]-E[ii]);
	cerr<<"S Error: "<<err<<endl;
	if(err > thresh)
		return -1;

	return 0;
}

template <typename T>
int testConstRank(size_t rows, size_t cols, size_t rank, size_t estrank)
{
	double thresh = 0.01;

	MatrixXd A(rows, cols);
	MatrixXd U(rows, rank);
	MatrixXd V(cols, rank);
	VectorXd E(rank);

	// Create Sorted (Decreasing) Singular Values
	cerr<<"Creating Sorted E"<<endl;
	E.setRandom();
	E = E.cwiseAbs();
	std::sort(E.data(), E.data()+E.rows(),
			[](double l, double r){return r < l;});

	// Create Orthonormal U
	cerr<<"Creating U"<<endl;
	U.setRandom();
	for(size_t cc=0; cc<U.cols(); cc++){
		// Modified Gram-Schmidt
		for(size_t c1 = 0; c1 < cc; c1++)
			U.col(cc) -= U.col(cc).dot(U.col(c1))*U.col(c1);
		U.col(cc).normalize();
	}

	// Create Orthonormal V
	cerr<<"Creating V"<<endl;
	V.setRandom();
	for(size_t cc=0; cc<V.cols(); cc++){
		// Modified Gram-Schmidt
		for(size_t c1 = 0; c1 < cc; c1++)
			V.col(cc) -= V.col(cc).dot(V.col(c1))*V.col(c1);
		V.col(cc).normalize();
	}

	cerr<<"Creating A"<<endl;
	A = U*E.asDiagonal()*V.transpose();
	if(a_verbose.isSet()) {
		cerr<<"U:\n"<<U<<endl;
		cerr<<"E:\n"<<E.transpose()<<endl;
		cerr<<"V:\n"<<V<<endl;
		cerr<<"A:\n"<<A<<endl;
	}

	MatrixXd estU, estV;
	VectorXd estE;
	randomizePowerIterationSVD(A, estrank, 2, estU, estE, estV);
	cerr << E.transpose() << endl;
	cerr << estE.transpose() << endl;

	// Comare
	double err;
	err = 0;
	for(size_t ii=0; ii<min<int>(U.cols(), estU.cols()); ii++)
		err += std::abs(U.col(ii).dot(estU.col(ii)))-1;
	cerr<<"U Error: "<<err<<endl;
	if(err > thresh)
		return -1;

	err = 0;
	for(size_t ii=0; ii<min<int>(V.cols(), estV.cols()); ii++)
		err += std::abs(V.col(ii).dot(estV.col(ii)))-1;
	cerr<<"V Error: "<<err<<endl;
	if(err > thresh)
		return -1;

	err = 0;
	for(size_t ii=0; ii<min<int>(V.cols(), estV.cols()); ii++)
		err += std::abs(estE[ii]-E[ii]);
	cerr<<"S Error: "<<err<<endl;
	if(err > thresh)
		return -1;

	return 0;
}

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

	TCLAP::ValueArg<size_t> a_rows("r", "rows", "Rows in random matrix (only "
		"applied if no input matrix given.", false, 15, "rows", cmd);
	TCLAP::ValueArg<size_t> a_cols("c", "cols", "Cols in random matrix (only "
		"applied if no input matrix given.", false, 10, "cols", cmd);
	TCLAP::ValueArg<size_t> a_rank("k", "rank", "Rank of random matrix (only "
		"applied if no input matrix given.", false, 5, "rank", cmd);
	TCLAP::ValueArg<size_t> a_estrank("K", "estrank", "Estimate rank for "
			"matrix decomposition.", false, 12, "rank", cmd);
	TCLAP::ValueArg<double> a_esterr("E", "esterr", "Error threshold for "
			"decomposition.", false, 0.001, "mval", cmd);


	TCLAP::ValueArg<string> a_out("R", "randmat", "Output generated (random) "
			"matrix.", false, "", "mat.bin", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	if(testConstRank<MatrixXd>(a_rows.getValue(), a_cols.getValue(),
			a_rank.getValue(), a_estrank.getValue()) != 0)
		return -1;

	if(testConstErr<MatrixXd>(a_rows.getValue(), a_cols.getValue(),
			a_rank.getValue(), 2, a_esterr.getValue()) != 0)
		return -1;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr<<"error: "<<e.error()<<" for arg "<<e.argId()<<std::endl;}

	return 0;
}


