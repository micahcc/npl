/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"

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
	out->setOrigin(from->origin());
	out->setSpacing(from->spacing());
	out->setDirection(from->direction());
	out->write(a_out.getValue());
	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


