/*******************************************************************************
This file is part of Neural Program Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Program Library is free software: you can redistribute it and/or
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

using std::string;

int main()
{
	try {
	/* 
	 * Command Line 
	 */

	TCLAP::CmdLine cmd("Often nonlinear registration produces deformation fields"
			" which store within each point a source point. Further these are"
			" often in index coordinates rather than RAS coordinates. This tool "
			"will convert a field of source indices into a field of offsets in "
			"world coordinates. For example [x,y,z,] contains a vector with value "
			"[x+a,y+b,z+c]. This is the sum of 3 vectors (x+a)i+(y+b)j+(z+c)k, we "
			"change that to A[a,b,c]^T. Its also possible to invert (-I) .",
			' ', __version__ );

	TCLAP::ValueArg<string> a_fmri("i", "input", "Input image.", 
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_invert("I", "invert", "Whether to invert.", cmd);


	

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
