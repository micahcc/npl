/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * trackvis_format_test.cpp test trk reader
 *
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
