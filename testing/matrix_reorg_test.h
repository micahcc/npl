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
 * @file matrix_reorg_test.h Tests for matrix reorganization
 *
 *****************************************************************************/
#ifndef MATRIX_REORG_TEST_H
#define MATRIX_REORG_TEST_H

#include "ica_helpers.h"
#include "nplio.h"
#include "mrimage.h"
#include "iterators.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace npl;

double approxNorm(const Ref<const MatrixXd> in)
{
	MatrixXd randv(in.cols(), 5);
	randv.setRandom();
	double norm = 0;
	for(size_t cc=0; cc<randv.cols(); cc++) {
		randv.col(cc) /= randv.norm();
		norm = std::max((in*randv.col(cc)).norm(), norm);
	}
	return norm;
}

/**
 * @brief Note that this presupposes the reorganized data matches (use the other
 * functions to test that).
 *
 * @param nrows
 * @param ncols
 * @param reorg
 * @param masks
 * @param inputs
 *
 * @return
 */
int testProducts(size_t nrows, size_t ncols,
	const MatrixReorg& reorg, const vector<ptr<MRImage>>& masks,
	const vector<ptr<MRImage>>& inputs)
{
	MatrixXd full(reorg.rows(), reorg.cols());

	double err;
	size_t globcol = 0;
	size_t globrow = 0;
	for(size_t cc=0; cc<ncols; cc++) {
		auto mask = masks[cc];
		NDIter<int> mit(mask);
		globrow = 0;
		size_t localcols = 0;
		for(size_t rr=0; rr<nrows; rr++) {
			localcols = 0;
			auto img = inputs[cc*nrows + rr];
			size_t tlen = img->tlen();
			Vector3DIter<double> it(img);
			mit.goBegin();

			for(size_t t=0; t<tlen; t++) {
				for(size_t s=0; !it.eof(); ++it, ++mit) {
					if(*mit != 0) {
						full(globrow+t, globcol+s) = it[t];
						s++;
						localcols++;
					}
				}
			}
			globrow += tlen;
		}
		globcol += localcols;
	}

	MatrixXd m(reorg.cols(), reorg.cols()/2+1);
	m.setRandom();
	MatrixXd a, b;

	b = full*m;
	a.resize(b.rows(), b.cols());
	reorg.postMult(a, m);
	err = (b - a).cwiseAbs().sum()/(a.rows()*a.cols());
	if(err > 0.1)  {
		cerr << "Mismatch of product "<<endl;
		return -1;
	}

	m.resize(reorg.rows(), reorg.cols()/2+1);
	m.setRandom();
	b = full.transpose()*m;
	a.resize(b.rows(), b.cols());
	reorg.postMult(a, m, true);
	err = (b - a).cwiseAbs().sum()/(a.rows()*a.cols());
	if(err >  0.1)  {
		cerr << "Mismatch of product (transpose)"<<endl;
		return -1;
	}

	m.resize(reorg.cols()/2+1, reorg.rows());
	m.setRandom();
	b = m*full;
	a.resize(b.rows(), b.cols());
	reorg.preMult(a, m);
	err = (b - a).cwiseAbs().sum()/(a.rows()*a.cols());
	if(err > 0.1)  {
		cerr << "Mismatch of pre-preoduct "<<endl;
		return -1;
	}

	m.resize(reorg.cols()/2+1, reorg.cols());
	m.setRandom();
	b = m*full.transpose();
	a.resize(b.rows(), b.cols());
	reorg.preMult(a, m, true);
	err = (b - a).cwiseAbs().sum()/(a.rows()*a.cols());
	if(err > 0.1)  {
		cerr << "Mismatch of pre-product (transpose)"<<endl;
		return -1;
	}

	return 0;
}

//int testWideMats(size_t nrows, size_t ncols,
//	const MatrixReorg& reorg, const vector<ptr<MRImage>>& masks,
//	const vector<ptr<MRImage>>& inputs, std::string prefix)
//{
//	MatrixXd full(reorg.rows(), reorg.cols());
//
//	size_t currow = 0;
//	for(size_t ii=0; ii<reorg.nwide(); ++ii) {
//		MatMap mat(prefix+to_string(ii));
//
//		full.middleRows(currow, reorg.wideMatRows()[ii]) = mat.mat;
//		currow += reorg.wideMatRows()[ii];
//	}
//
//	size_t globcol = 0;
//	size_t globrow = 0;
//	for(size_t cc=0; cc<ncols; cc++) {
//		auto mask = masks[cc];
//		NDIter<int> mit(mask);
//		globrow = 0;
//		size_t localcols = 0;
//		for(size_t rr=0; rr<nrows; rr++) {
//			localcols = 0;
//			auto img = inputs[cc*nrows + rr];
//			size_t tlen = img->tlen();
//			Vector3DIter<double> it(img);
//			mit.goBegin();
//
//			for(size_t t=0; t<tlen; t++) {
//				for(size_t s=0; !it.eof(); ++it, ++mit) {
//					if(*mit != 0) {
//						if(full(globrow+t, globcol+s) != it[t]) {
//							cerr << "Mismatch in wide mats!" << endl;
//							cerr << "Outer Column: " << cc << endl;
//							cerr << "Outer Row: " << rr << endl;
//							cerr << "Inner Column: " << s << endl;
//							cerr << "Inner Row: " << t << endl;
//							cerr << full(globrow+t, globcol+s) << " vs " << it[t] << endl;
//							cerr << "Full Matrix:\n\n" << full << endl;
//							return -1;
//						}
//						s++;
//						localcols++;
//					}
//				}
//			}
//			globrow += tlen;
//		}
//		globcol += localcols;
//	}
//	return 0;
//}

int testTallMats(size_t nrows, size_t ncols,
		const MatrixReorg& reorg, const vector<ptr<MRImage>>& masks,
		const vector<ptr<MRImage>>& inputs, std::string prefix)
{
	MatrixXd full(reorg.rows(), reorg.cols());

	size_t curcol = 0;
	for(size_t ii=0; ii<reorg.ntall(); ++ii) {
		MatMap mat(prefix+to_string(ii));

		full.middleCols(curcol, reorg.tallMatCols()[ii]) = mat.mat;
		curcol += reorg.tallMatCols()[ii];
	}

	size_t globcol = 0;
	size_t globrow = 0;
	for(size_t cc=0; cc<ncols; cc++) {
		auto mask = masks[cc];
		NDIter<int> mit(mask);
		globrow = 0;
		size_t localcols = 0;
		for(size_t rr=0; rr<nrows; rr++) {
			localcols = 0;
			auto img = inputs[cc*nrows + rr];
			size_t tlen = img->tlen();
			Vector3DIter<double> it(img);
			mit.goBegin();

			for(size_t t=0; t<tlen; t++) {
				for(size_t s=0; !it.eof(); ++it, ++mit) {
					if(*mit != 0) {
						if(full(globrow+t, globcol+s) != it[t]) {
							cerr << "Mismatch in tall mats!" << endl;
							cerr << "Outer Column: " << cc << endl;
							cerr << "Outer Row: " << rr << endl;
							cerr << "Inner Column: " << s << endl;
							cerr << "Inner Row: " << t << endl;
							cerr << full(globrow+t, globcol+s) << " vs " << it[t] << endl;
							cerr << "Full Matrix:\n\n" << full << endl;
							return -1;
						}
						s++;
						localcols++;
					}
				}
			}
			globrow += tlen;
		}
		globcol += localcols;
	}
	return 0;
}

#endif // MATRIX_REORG_TEST_H

