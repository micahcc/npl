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
 * @file melodic_subjects.cpp Tool to recreate individual subject estimates from
 * melodic output.
 *
 *****************************************************************************/

#include "version.h"
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "nplio.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"

using std::string;
using namespace npl;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Takes activation maps and a matrix and multiplies them "
			"out to produce estimated fMRI images. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image, spatial maps.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_mat("m", "matrix", "Matrix of values, should have "
			"1 row for each input timepoint, 1 column for each voxel ",
			true, "", "file", cmd);
	TCLAP::ValueArg<int> a_split("s", "split", "Split every so many time "
			"points ", true, 1, "t", cmd);

	TCLAP::ValueArg<string> a_out("p", "prefix", "Output prefix",
			true, "", "pref\%n.nii.gz", cmd);


	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	ptr<MRImage> inimg(readMRImage(a_in.getValue()));
	if(inimg->ndim() != 4) {
		cerr << "Expected input to be 4D Image!" << endl;
		return -1;
	}
	size_t comps = inimg->tlen();
	size_t voxels = inimg->elements()/inimg->tlen();
	cerr<<"Image components: " << comps << endl;
	cerr<<"Image voxels: " << voxels << endl;

	vector<vector<double>> mat = readNumericCSV(a_mat.getValue());
	size_t tlen = mat.size();
	cerr<<"Time Length: " << tlen <<endl;
	cerr<<"Found Columns in Matrix:" << mat[0].size() << endl;

	vector<size_t> sz(4);
	for(size_t ii=0; ii<3; ii++) sz[ii] = inimg->dim(ii);
	sz[3] = a_split.getValue();

	size_t subtlen = a_split.getValue();
	size_t subjects = tlen/a_split.getValue();
	if(tlen%a_split.getValue() != 0) {
		cerr << "Cannot split images if the total time is not divisible by "
			"the input split len" << endl;
		return -1;
	}

	// Y = W * S, rows of S are timepoints in inimg
	MatrixXd ic(comps, voxels);
	MatrixXd ts(subtlen, voxels);
	MatrixXd W(subtlen, mat[0].size());
	for(size_t ss=0; ss<subjects; ss++) {
		cerr << "Filling " << ss << endl;
		// load components
		size_t vv = 0;
		for(Vector3DIter<double> iit(inimg); !iit.eof(); ++iit, ++vv) {
			for(size_t cc=0; cc<comps; cc++)
				ic(cc, vv) = iit[cc];
		}

		cerr << "Filling Subject Mixing Matrix" << endl;
		for(size_t rr=0; rr<W.rows(); rr++) {
			for(size_t cc=0; cc<W.cols(); cc++)
				W(rr,cc) = mat[rr+ss*subtlen][cc];
		}

		cerr << "Multiplying " << ss << endl;
		// multiply out to produce time-series
		ts = W*ic;

		cerr << "Writing " << ss << endl;
		vv = 0;
		auto out = inimg->copyCast(4, sz.data(), FLOAT32);
		for(Vector3DIter<double> oit(out); !oit.eof(); ++oit, ++vv) {
			// write out timeseries
			for(size_t cc=0; cc<subtlen; cc++)
				oit.set(cc, ts(cc, vv));
		}
		cerr<<"Writing " << a_out.getValue()<<ss<<".nii.gz" << endl;
		out->write(a_out.getValue()+to_string(ss)+".nii.gz");
	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


