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

#include "matrix_reorg_test.h"
#include "fmri_inference.h"
#include "mrimage_utils.h"
#include "mrimage.h"
#include "iterators.h"
#include "utility.h"
#include "nplio.h"

using namespace npl;
using namespace std;

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
		masks[cc] = randImage(INT8, 5, 1, 1, 1, 4, 0);
		fn_masks[cc] = pref+"mask_"+to_string(cc)+".nii.gz";
		masks[cc]->write(fn_masks[cc]);

		for(size_t rr = 0; rr<nrows; rr++) {
			inputs[rr+cc*nrows] = randImage(FLOAT64, 0, 1, 1, 1, 4, timepoints);
			int count = 0;
			for(Vector3DIter<double> it(inputs[rr+cc*nrows]); !it.eof(); ++it) {
				for(size_t tt=0; tt<timepoints; tt++) {
					it.set(tt, (double)rand()/RAND_MAX);
				}
				count++;
			}

			fn_inputs[rr+cc*nrows] = pref+to_string(cc)+"_"+
						to_string(rr)+".nii.gz";
			inputs[rr+cc*nrows]->write(fn_inputs[rr+cc*nrows]);
		}
	}

	MatrixReorg reorg(pref, 45000, true);
	if(reorg.createMats(nrows, ncols, fn_masks, fn_inputs, false) != 0)
		return -1;

	// use Matrix
	if(testTallMats(nrows, ncols, reorg, masks, inputs, pref+"_tall_") != 0)
		return -1;

	if(testProducts(nrows, ncols, reorg, masks, inputs) != 0)
		return -1;
}
