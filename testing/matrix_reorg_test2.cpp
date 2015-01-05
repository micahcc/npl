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
 * @file matrix_reorg_test2.cpp Test reorganization of MRImages into matrices,
 * this is a more complicated, larger version of test 1
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

int testWideMats(const MatrixReorg& reorg, const vector<ptr<MRImage>>& masks,
		const vector<ptr<MRImage>>& inputs, size_t timecat, size_t spacecat,
		std::string prefix)
{
	MatrixXd full(reorg.m_totalrows, reorg.m_totalcols);

	size_t currow = 0;
	for(size_t ii=0; ii<reorg.m_outcols.size(); ++ii) {
		MatMap mat(prefix+to_string(ii));
		
		full.middleRows(currow, reorg.m_outrows[ii]) = mat.mat;
		currow += reorg.m_outrows[ii];
	}

	size_t globcol = 0;
	size_t globrow = 0;
	for(size_t cc=0; cc<spacecat; cc++) {
		auto mask = masks[cc];
		NDIter<int> mit(mask);
		globrow = 0;
		for(size_t rr=0; rr<timecat; rr++) {
			auto img = inputs[cc*timecat + rr];
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
					}
				}
			}
			globrow += reorg.m_inrows[rr];
		}
		globcol += reorg.m_incols[cc];
	}
	return 0;
}

int testTallMats(const MatrixReorg& reorg, const vector<ptr<MRImage>>& masks,
		const vector<ptr<MRImage>>& inputs, size_t timecat, size_t spacecat,
		std::string prefix)
{
	MatrixXd full(reorg.m_totalrows, reorg.m_totalcols);

	size_t curcol = 0;
	for(size_t ii=0; ii<reorg.m_outcols.size(); ++ii) {
		MatMap mat(prefix+to_string(ii));
		
		full.middleCols(curcol, reorg.m_outcols[ii]) = mat.mat;
		curcol += reorg.m_outcols[ii];
	}

	size_t globcol = 0;
	size_t globrow = 0;
	for(size_t cc=0; cc<spacecat; cc++) {
		auto mask = masks[cc];
		NDIter<int> mit(mask);
		globrow = 0;
		for(size_t rr=0; rr<timecat; rr++) {
			auto img = inputs[cc*timecat + rr];
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
					}
				}
			}

			globrow += reorg.m_inrows[rr];
		}
		globcol += reorg.m_incols[cc];
	}
	return 0;
}

int main()
{
	std::string pref = "reorg_test1";
	size_t timepoints = 5;
	size_t ncols = 3;
	size_t nrows = 4;

	// create random images
	vector<ptr<MRImage>> inputs(ncols*nrows);
	vector<ptr<MRImage>> masks(ncols);
	vector<std::string> fn_inputs(ncols*nrows);
	vector<std::string> fn_masks(ncols);
	for(size_t cc = 0; cc<ncols; cc++) {
		masks[cc] = randImage(INT8, 0, 1, 10, 11, 13, 0);
		fn_masks[cc] = pref+"mask_"+to_string(cc)+".nii.gz";
		masks[cc]->write(fn_masks[cc]);

		for(size_t rr = 0; rr<nrows; rr++) {
			inputs[rr+cc*nrows] = randImage(FLOAT64, 0, 1, 10, 11, 13, timepoints);
			
			fn_inputs[rr+cc*nrows] = pref+to_string(cc)+"_"+
						to_string(rr)+".nii.gz";
			inputs[rr+cc*nrows]->write(fn_inputs[rr+cc*nrows]);
		}
	}
	
	MatrixReorg reorg(pref, 45000, true);
	if(reorg.createMats(nrows, ncols, fn_masks, fn_inputs) != 0) 
		return -1;
	
	// use Matrix
	if(testTallMats(reorg, masks, inputs, nrows, ncols, pref+"_tall_") != 0)
		return -1;
	
	if(testWideMats(reorg, masks, inputs, nrows, ncols, pref+"_wide_") != 0)
		return -1;

}
