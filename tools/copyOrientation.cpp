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

#include "mrimage.h"
#include "mrimage_utils.h"

using std::string;
using namespace npl;
using std::shared_ptr;

int main(int argc, char** argv)
{
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
			true, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	std::shared_ptr<MRImage> from(readMRImage(a_from.getValue()));
	std::shared_ptr<MRImage> to(readMRImage(a_to.getValue()));
	
	auto out = to->cloneImage();
    out->setOrient(from->getOrigin(), from->getSpacing(),
            from->getDirection());
	out->write(a_out.getValue());
	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


