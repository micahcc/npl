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
 * @file matrix_reorg_test1.cpp Test reorganization of MRImages into matrices
 * Simple version with small matrices
 *
 *****************************************************************************/

#include <string>

#include "ica_helpers.h"
#include "mrimage_utils.h"
#include "mrimage.h"
#include "iterators.h"
#include "utility.h"
#include "nplio.h"

using namespace npl;
using namespace std;

int testWideMats(size_t nrows, size_t ncols, const MatrixReorg& reorg, const
		vector<ptr<MRImage>>& masks, const vector<ptr<MRImage>>& inputs,
		std::string prefix)
{
	MatrixXd full(reorg.rows(), reorg.cols());

	size_t currow = 0;
	for(size_t ii=0; ii<reorg.nwide(); ++ii) {
		MatMap mat(prefix+to_string(ii));
		
		full.middleRows(currow, reorg.wideMatRows()[ii]) = mat.mat;
		currow += reorg.wideMatRows()[ii];
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
							cerr << "Mismatch in wide mats!" << endl;
							cerr << "Outer Column: " << cc << endl;
							cerr << "Outer Row: " << rr << endl;
							cerr << "Inner Column: " << s << endl;
							cerr << "Inner Row: " << t << endl;
							cerr << full(globrow+t, globcol+s) << " vs " << it[t] << endl;
							cerr << "Full Matrix:\n\n" << full << endl;
							return -1;
						}
						localcols++;
						s++;
					}
				}
			}
			globrow += tlen;
		}
		globcol += localcols;
	}
	return 0;
}

int testTallMats(size_t nrows, size_t ncols, const MatrixReorg& reorg, const
		vector<ptr<MRImage>>& masks, const vector<ptr<MRImage>>& inputs,
		std::string prefix)
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
	
	// use Matrix
	if(testTallMats(nrows, ncols, reorg, masks, inputs, pref+"_tall_") != 0)
		return -1;
	
	if(testWideMats(nrows, ncols, reorg, masks, inputs, pref+"_wide_") != 0)
		return -1;

}
