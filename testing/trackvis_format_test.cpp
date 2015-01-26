/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * trackvis_format_test.cpp Implementation of track class for loading tracks
 * from trackvis, or DFC formats.
 *****************************************************************************/

#include <vector>
#include <string>
#include <fstream>

#include "tracks.h"
#include "byteswap.h"
#include "trackfile_headers.h"
#include "nplio.h"
#include "mrimage.h"
#include "macros.h"

using namespace std;
using namespace npl;

int main(int argc, char** argv)
{
	string filename = "../../testing/trackvis_format_test.trk";
	if(argc > 1) {
		filename = argv[1];
	}
	cerr << "Reading "<<filename<<endl;

	auto tracks = readTracks(filename);

	cerr << "Check coordinates with ../../testing/trackvis_format_test.nii.gz "
		"tracks should be predominantly right side"<<endl;
	for(size_t tt=0; tt<tracks.size(); tt++) {
		cerr<<"Track "<<tt<<endl;
		for(size_t ii=0; ii<tracks[tt].size(); ii++) {
			cerr<<"\t("<<tracks[tt][ii][0]<<", "<<tracks[tt][ii][1]<<", "
				<<tracks[tt][ii][2]<<")"<<endl;
		}
	}

	return 0;
}
