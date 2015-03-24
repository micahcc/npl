/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file big_pca_test1.cpp Test merging of smaller PCA solutions into a larger
 * one using concatination of cols
 *
 *****************************************************************************/

#include <string>

#include "fmri_inference.h"
#include "mrimage_utils.h"
#include "mrimage.h"
#include "iterators.h"
#include "utility.h"
#include "nplio.h"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

using namespace npl;
using namespace std;

using Eigen::ComputeThinV;
using Eigen::ComputeThinU;
using Eigen::JacobiSVD;

//int testWidePCAJoin(const MatrixReorg& reorg, std::string prefix, double svt)
//{
//	double thresh = 0.00001;
//
//	size_t totrows = reorg.rows();
//	size_t totcols = reorg.cols();
//
//	MatrixXd full(totrows, totcols);
//
//	size_t currow = 0;
//	for(size_t ii=0; ii<reorg.nwide(); ++ii) {
//		MatMap mat(prefix+to_string(ii));
//
//		full.middleRows(currow, reorg.wideMatRows()[ii]) = mat.mat;
//		currow += reorg.wideMatRows()[ii];
//	}
//
//	// Perform Full SVD
//	cerr<<"Full SVD:"<<full.rows()<<"x"<<full.cols()<<endl;
//	JacobiSVD<MatrixXd> fullsvd(full, ComputeThinU | ComputeThinV);
//
//	// Maximum number of columns/rows in sigma
//	size_t outrows = 0;
//	vector<MatrixXd> Umats(reorg.nwide());
//	vector<MatrixXd> Vmats(reorg.nwide());
//	vector<VectorXd> Smats(reorg.nwide());
//	for(size_t rr=0; rr<reorg.nwide(); rr++) {
//		MatMap diskmat(prefix+to_string(rr));
//		cerr<<"Chunk SVD:"<<diskmat.mat.rows()<<"x"<<diskmat.mat.cols()<<endl;
//		JacobiSVD<MatrixXd> svd(diskmat.mat, ComputeThinU | ComputeThinV);
//		svd.setThreshold(svt);
//
//		cerr << "SVD Rank: " << svd.rank() << endl;
//		Umats[rr] = svd.matrixU().leftCols(svd.rank());
//		Vmats[rr] = svd.matrixV().leftCols(svd.rank());
//		Smats[rr] = svd.singularValues().head(svd.rank());
//
//		outrows += Smats[rr].rows();
//	}
//
//	// Merge / Construct Column(EV^T, EV^T, ... )
//	MatrixXd mergedEVt(outrows, totcols);
//	currow = 0;
//	for(size_t rr=0; rr<reorg.nwide(); rr++) {
//		mergedEVt.middleRows(currow, Smats[rr].rows()) =
//			Smats[rr].asDiagonal()*Vmats[rr].transpose();
//		currow += Smats[rr].rows();
//	}
//
//	cerr<<"Merge SVD:"<<mergedEVt.rows()<<"x"<<mergedEVt.cols()<<endl;
//	JacobiSVD<MatrixXd> mergesvd(mergedEVt, ComputeThinU | ComputeThinV);
//
//	cerr<<"Comparing Full S with Merge S"<<endl;
//	const auto& fullS = fullsvd.singularValues();
//	const auto& mergeS = mergesvd.singularValues();
//	for(size_t ii=0; ii<min(fullS.rows(), mergeS.rows()); ++ii) {
//		if(fabs(mergeS[ii] - fullS[ii]) > thresh) {
//			cerr<<"Difference in Singular Value "<<ii<<": "<<mergeS[ii]<<" vs "
//				<<fullS[ii]<<endl;
//			return -1;
//		}
//	}
//
//	cerr<<"Comparing Full V with Merge V"<<endl;
//	const auto& fullV = fullsvd.matrixV();
//	const auto& mergeV = mergesvd.matrixV();
//	for(size_t ii=0; ii<min(fullV.cols(), mergeV.cols());  ++ii) {
//		if(fabs(mergeV.col(ii).dot(fullV.col(ii)))-1 > thresh) {
//			cerr<<"Difference in V col "<<ii<<": "<<mergeV.col(ii)<<" vs "
//				<<fullV.col(ii)<<endl;
//			return -1;
//		}
//	}
//
//	return 0;
//}

int testTallPCAJoin(const MatrixReorg& reorg, std::string prefix, double svt)
{
	double thresh = 0.00001;

	size_t totrows = reorg.rows();
	size_t totcols = reorg.cols();
	size_t curcol = 0;

	// Create Full Matrix to Compare Against
	MatrixXd full(totrows, totcols);
	for(size_t ii=0; ii<reorg.ntall(); ++ii) {
		MatMap mat(prefix+to_string(ii));

		full.middleCols(curcol, reorg.tallMatCols()[ii]) = mat.mat;
		curcol += reorg.tallMatCols()[ii];
	}

	// Perform Full SVD
	cerr<<"Full SVD:"<<full.rows()<<"x"<<full.cols()<<endl;
	JacobiSVD<MatrixXd> fullsvd(full, ComputeThinU | ComputeThinV);

	// Maximum number of columns/rows in sigma
	size_t outcols = 0;
	vector<MatrixXd> Umats(reorg.ntall());
	vector<MatrixXd> Vmats(reorg.ntall());
	vector<VectorXd> Smats(reorg.ntall());
	for(size_t ii=0; ii<reorg.ntall(); ii++) {
		MatMap diskmat(prefix+to_string(ii));
		cerr<<"Chunk SVD:"<<diskmat.mat.rows()<<"x"<<diskmat.mat.cols()<<endl;
		JacobiSVD<MatrixXd> svd(diskmat.mat, ComputeThinU | ComputeThinV);
		svd.setThreshold(svt);

		cerr << "SVD Rank: " << svd.rank() << endl;
		Umats[ii] = svd.matrixU().leftCols(svd.rank());
		Vmats[ii] = svd.matrixV().leftCols(svd.rank());
		Smats[ii] = svd.singularValues().head(svd.rank());

		outcols += Smats[ii].rows();
	}

	// Merge / Construct Column(EV^T, EV^T, ... )
	MatrixXd mergedUE(totrows, outcols);
	curcol = 0;
	for(size_t ii=0; ii<reorg.ntall(); ii++) {
		mergedUE.middleCols(curcol, Smats[ii].rows()) =
			Umats[ii]*Smats[ii].asDiagonal();
		curcol += Smats[ii].rows();
	}

	cerr<<"Merge SVD:"<<mergedUE.rows()<<"x"<<mergedUE.cols()<<endl;
	JacobiSVD<MatrixXd> mergesvd(mergedUE, ComputeThinU | ComputeThinV);

	cerr<<"Comparing Full S with Merge S"<<endl;
	const auto& fullS = fullsvd.singularValues();
	const auto& mergeS = mergesvd.singularValues();
	for(size_t ii=0; ii<min(fullS.rows(), mergeS.rows()); ++ii) {
		if(fabs(mergeS[ii] - fullS[ii]) > thresh) {
			cerr<<"Difference in Singular Value "<<ii<<": "<<mergeS[ii]<<" vs "
				<<fullS[ii]<<endl;
			return -1;
		}
	}

	cerr<<"Comparing Full V with Merge V"<<endl;
	const auto& fullU = fullsvd.matrixU();
	const auto& mergeU = mergesvd.matrixU();
	for(size_t ii=0; ii<min(fullU.cols(), mergeU.cols());  ++ii) {
		if(fabs(mergeU.col(ii).dot(fullU.col(ii)))-1 > thresh) {
			cerr<<"Difference in V col "<<ii<<": "<<mergeU.col(ii)<<" vs "
				<<fullU.col(ii)<<endl;
			return -1;
		}
	}

	return 0;
}

int main()
{
	std::string pref = "reorg2";
	size_t timepoints = 5;
	size_t ncols = 3;
	size_t nrows = 4;

	// create random images
	vector<ptr<MRImage>> inputs(ncols*nrows);
	vector<ptr<MRImage>> masks(ncols);
	vector<std::string> fn_inputs(ncols*nrows);
	vector<std::string> fn_masks(ncols);
	for(size_t cc = 0; cc<ncols; cc++) {
		masks[cc] = randImage(INT8, 5, 1, 1, 1, 4, 0);
		fn_masks[cc] = pref+"mask_"+to_string(cc)+".nii.gz";
		masks[cc]->write(fn_masks[cc]);

		for(size_t rr = 0; rr<nrows; rr++) {
			inputs[rr+cc*nrows] = randImage(FLOAT64, 0, 1, 1, 1, 4, timepoints);
			int count = 0;
			for(Vector3DIter<double> it(inputs[rr+cc*nrows]); !it.eof(); ++it) {
				for(size_t tt=0; tt<timepoints; tt++) {
					it.set(tt, cc*1e6+rr*1e4+count*1e2+tt);
				}
				count++;
			}

			fn_inputs[rr+cc*nrows] = pref+to_string(cc)+"_"+
						to_string(rr)+".nii.gz";
			inputs[rr+cc*nrows]->write(fn_inputs[rr+cc*nrows]);
		}
	}

	MatrixReorg reorg(pref, 45000, true);
	if(reorg.createMats(nrows, ncols, fn_masks, fn_inputs) != 0)
		return -1;

//	// use Matrix
//	if(testWidePCAJoin(reorg, pref+"_wide_", 1e-20) != 0)
//		return -1;

	if(testTallPCAJoin(reorg, pref+"_tall_", 1e-20) != 0)
		return -1;


}

