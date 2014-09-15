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
 * @file applyDeform.cpp Tool to apply a deformation field to another image. 
 * Not yet functional
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "kernel_slicer.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using std::string;
using namespace npl;
using std::shared_ptr;

/**
 * @brief Computes the overlap of the two images' in 3-space.
 *
 * @param a Image
 * @param b Image
 *
 * @return Ratio of b that overlaps with a's grid
 */
double overlapRatio(shared_ptr<MRImage> a, shared_ptr<MRImage> b)
{
	int64_t index[3];
	double point[3];
	size_t incount = 0;
	size_t maskcount = 0;
	for(OrderIter<int64_t> it(a); !it.eof(); ++it) {
		it.index(3, index);
		a->indexToPoint(3, index, point);
		maskcount++;
		incount += (b->pointInsideFOV(3, point));
	}
	return (double)(incount)/(double)(maskcount);
}


void binarize(shared_ptr<MRImage> in)
{
	OrderIter<int> it(in);
	for(it.goBegin(); !it.eof(); ++it) {
		if(*it != 0)
			it.set(1);
	}

}

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Applies 3D deformation to volume or time-series of "
			"volumes. Deformation should be a map of offsets. If you have a "
			"*.svreg.map.* file use '$ convertDeform --in-index -1 --out-offset"
			" -E -i *.svreg.map.nii.gz -a atlas.bfc.nii.gz -m "
			"*.cerebrum.mask.nii.gz -o offset.nii.gz' to generate an "
			"appropriate input (offset.nii.gz).", ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_deform("d", "deform", "Deformation field.",
			true, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);


	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	std::shared_ptr<MRImage> inimg(readMRImage(a_in.getValue()));
	if(inimg->ndim() > 4 || inimg->ndim() < 3) {
		cerr << "Expected input to be 3D/4D Image!" << endl;
		return -1;
	}
	
	std::shared_ptr<MRImage> mask(readMRImage(a_mask.getValue()));
	if(mask->ndim() != 3) {
		cerr << "Expected mask to be 3D Image!" << endl;
		return -1;
	}
	binarize(mask);

	// dilate then erode mask
	if(a_dilate.isSet())
		mask = dilate(mask, a_dilate.getValue());
	if(a_erode.isSet())
		mask = erode(mask, a_erode.getValue());

	// ensure the the images overlap sufficiently, lack of overlap may indicate
	// incorrect orientation
	double f = overlapRatio(mask, inimg);
	if(f < .5)  {
		cerr << "Warning the input and mask images do not overlap very much."
			" This could indicate bad orientation, overlap: " << f << endl;
		return -1;
	}
	
	std::shared_ptr<MRImage> atlas(readMRImage(a_atlas.getValue()));
	if(atlas->ndim() != 3) {
		cerr << "Expected mask to be 3D Image!" << endl;
		return -1;
	}
	binarize(atlas);

	std::shared_ptr<MRImage> defimg(readMRImage(a_deform.getValue()));
	if(defimg->ndim() > 5 || defimg->ndim() < 4 || defimg->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}

	// convert deform to RAS space offsets
	defimg = indexMapToOffsetMap(defimg, atlas, true);
	defimg->write("deform.nii.gz");

	// perform interpolation to estimate outside-the brain deformations that
	// are continuous with the within-brain deformations
	defimg = extrapolate(defimg, mask, atlas);
	defimg->write("extrapolated.nii.gz");

	if(a_invert.isSet()) {
		// create output the size of atlas, with 3 volumes in the 4th dimension
		auto idef = createMRImage({atlas->dim(0), atlas->dim(1),
					atlas->dim(2), 3}, FLOAT64);
        idef->setOrient(atlas->getOrigin(), atlas->getSpacing(),
                atlas->getDirection(), true);
		invert(mask, defimg, idef, a_iters.getValue(),
				a_improve.getValue(), a_radius.getValue());
		idef->write("inversedef.nii.gz");
	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

