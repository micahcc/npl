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
 * @file copyOrientation.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>
#include <iostream>

#include "mrimage.h"
#include "nplio.h"

using std::string;
using namespace npl;
using std::shared_ptr;
using std::endl;
using std::cerr;
using std::cin;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Copies orientation from one image to another. WARNING "
			"THIS IS AN EXTREME MEASURE.", ' ', __version__ );

	TCLAP::ValueArg<string> a_to("t", "to", "Image to copy orientation into.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image, identical to '-t' "
			" but with orientation matching '-f' image.", true, "",
			"*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_from("f", "from", "Image to copy orientation from",
			false, "", "*.nii.gz");
	TCLAP::SwitchArg a_stdin("s", "stdin", "Read orientation from stdin");
	cmd.xorAdd(a_from, a_stdin);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	ptr<MRImage> to(readMRImage(a_to.getValue()));
	auto out = to->cloneImage();
	if(a_from.isSet()) {
		ptr<MRImage> from(readMRImage(a_from.getValue()));

		cerr << "From: " << endl << *from << endl;
		cerr << "To: " << endl << *to<< endl;

		out->setOrient(from->getOrigin(), from->getSpacing(),
				from->getDirection(), false, from->m_coordinate);
	} else {
		MatrixXd direction(to->ndim(), to->ndim());
		VectorXd spacing(to->ndim());
		VectorXd origin(to->ndim());
		cerr << "Origin? " << endl;
		for(size_t ii=0; ii<origin.rows(); ii++)
			cin >> origin[ii];
		cerr << "Spacing? " << endl;
		for(size_t ii=0; ii<spacing.rows(); ii++)
			cin >> spacing[ii];
		cerr << "Direction (Row-Major)? " << endl;
		for(size_t ii=0; ii<spacing.rows(); ii++) {
			for(size_t jj=0; jj<spacing.rows(); jj++) {
				cin >> direction(ii,jj);
			}
		}

		out->setOrient(origin, spacing, direction);
	}

	cerr << "Out: " << endl << *out << endl;
	out->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


