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
 * @file chunkslicer_test.cpp Tests region-of-interest specification in Slicer 
 * class
 *
 *****************************************************************************/

#include <iostream>

#include "iterators.h"
#include "accessors.h"
#include "mrimage.h"
#include "mrimage_utils.h"

using namespace std;
using namespace npl;

int test1(shared_ptr<MRImage> img)
{
	cerr << "test1" << endl;
	int64_t roi1[3] = {3, 5, 4};
	int64_t roi2[3] = {7, 8, 6};
	int64_t breaks[] = {1, 1, 0};

	ChunkIter<int> slicer(img);
	slicer.setROI(3, roi1, roi2);
	slicer.setChunkSize(3, breaks);
	slicer.goBegin();
	for(int64_t xx=3; xx<=7; xx++) {
		for(int64_t yy=5; yy<=8; yy++) {
			for(int64_t zz=4; zz<=6; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks" << endl;
					return -1;
				}
			}
			if(!slicer.isChunkEnd()) {
				cerr << "Chunk end not reached at the expected point" << endl;
				return -1;
			}
			slicer.nextChunk();
		}
	}

	return 0;
}

int test2(shared_ptr<MRImage> img)
{
	cerr << "test2" << endl;
	ChunkIter<int> slicer(img);

	int64_t roi1[3] = {3, 5, 4};
	int64_t roi2[3] = {7, 8, 6};
	slicer.setROI(3, roi1, roi2);
	int64_t breaks[] = {1, 0, 0};
	slicer.setChunkSize(3, breaks);
	slicer.goBegin();
	for(int64_t xx=3; xx<=7; xx++) {
		for(int64_t yy=5; yy<=8; yy++) {
			for(int64_t zz=4; zz<=6; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks" << endl;
					return -1;
				}
			}
		}
		if(!slicer.isChunkEnd()) {
			cerr << "Chunk end not reached at the expected point" << endl;
			return -1;
		}
		slicer.nextChunk();
	}

	return 0;
}

int test3(shared_ptr<MRImage> img)
{
	cerr << "test3" << endl;
	ChunkIter<int> slicer(img);

	int64_t roi1[3] = {3, 5, 4};
	int64_t roi2[3] = {7, 8, 6};
	int64_t breaks[] = {1, 0, 0};
	slicer.setChunkSize(3, breaks);
	slicer.setROI(3, roi1, roi2);
	slicer.goBegin();
	for(int64_t xx=3; xx<=7; xx++) {
		for(int64_t yy=5; yy<=8; yy++) {
			for(int64_t zz=4; zz<=6; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks" << endl;
					return -1;
				}
			}
		}
		if(!slicer.isChunkEnd()) {
			cerr << "Chunk end not reached at the expected point" << endl;
			return -1;
		}
		slicer.nextChunk();
	}

	return 0;
}

int test4(shared_ptr<MRImage> img)
{
	cerr << "test4" << endl;
	ChunkIter<int> slicer(img);

	int64_t breaks[] = {1, 0, 0};
	slicer.setChunkSize(3, breaks);
	slicer.goBegin();
	for(int64_t xx=0; xx<=9; xx++) {
		for(int64_t yy=0; yy<=9; yy++) {
			for(int64_t zz=0; zz<=9; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks" << endl;
					return -1;
				}
			}
		}
		if(!slicer.isChunkEnd()) {
			cerr << "Chunk end not reached at the expected point" << endl;
			return -1;
		}
		slicer.nextChunk();
	}

	return 0;
}

int test5(shared_ptr<MRImage> img)
{
	cerr << "test5" << endl;
	ChunkIter<int> slicer(img);

	int64_t roi1[3] = {1, 1, 2};
	int64_t roi2[3] = {6, 8, 7};
	slicer.setROI(3, roi1, roi2);

	int64_t breaks[] = {2, 3, 7};
	slicer.setChunkSize(3, breaks);
	slicer.goBegin();
	
	// y steps
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 1" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 2" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 3" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	// x step 
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 4" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 5" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 6" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	// x steps
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 7" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 8" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 9" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	return 0;
}

int test6(shared_ptr<MRImage> img)
{
	cerr << "test6" << endl;
	ChunkIter<int> slicer(img);

	int64_t roi1[3] = {1, 1, 2};
	int64_t roi2[3] = {6, 8, 7};
	slicer.setROI(3, roi1, roi2);

	int64_t breaks[] = {2, 3, 7};
	slicer.setChunkSize(3, breaks);
	slicer.goBegin();
	
	// y steps
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 1" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	// repeat the last using previousChunk
	slicer.prevChunk();
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 1b" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 2" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 3" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	// x step 
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 4" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 5" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 6" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	// repeat using prevChunk()
	slicer.prevChunk();
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 6b" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	// x steps
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 7" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 8" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 9" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	// repeat using prevChunk()
	slicer.prevChunk();
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 9b" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	return 0;
}

int test7(shared_ptr<MRImage> img)
{
	cerr << "test7" << endl;
	ChunkIter<int> slicer(img);

	int64_t roi1[3] = {1, 1, 2};
	int64_t roi2[3] = {6, 8, 7};
	slicer.setROI(3, roi1, roi2);

	int64_t breaks[] = {2, 3, 7};
	slicer.setChunkSize(3, breaks);
	slicer.goBegin();
	
	// y steps
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 1" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	// repeat the last using goChunkBegin()
	slicer.goChunkBegin();

	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 1b" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 2" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=1; xx<=2; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 3" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	// x step 
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 4" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 5" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 6" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	// repeat using goChunkBegin()
	slicer.goChunkBegin();
	slicer.goChunkBegin();
	slicer.nextChunk();
	slicer.prevChunk();
	slicer.goChunkBegin();
	for(int64_t xx=3; xx<=4; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 6b" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	// x steps
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=1; yy<=3; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 7" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=4; yy<=6; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 8" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();
	
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 9" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	// repeat using prevChunk()
	slicer.goChunkBegin();
	slicer.goChunkEnd();
	slicer.prevChunk();
	slicer.goChunkBegin();
	slicer.nextChunk();
	for(int64_t xx=5; xx<=6; xx++) {
		for(int64_t yy=7; yy<=8; yy++) {
			for(int64_t zz=2; zz<=7; zz++, ++slicer) {
				if(100*xx+10*yy+zz != *slicer) {
					cerr << "Error slicer mismatch with known values during "
						"3D roi, with 1D chunks. Chunk 9b" << endl;
					return -1;
				}
			}
		}
	}
	if(!slicer.isChunkEnd()) {
		cerr << "Chunk end not reached at the expected point" << endl;
		return -1;
	}
	slicer.nextChunk();

	return 0;
}
int main()
{
	// create image with values set by the index
	size_t dim[] = {10, 10, 10};
	auto img = createMRImage(3, dim, INT64);
	Pixel3DView<int> acc(img);

	for(int64_t xx=0; xx<10; xx++) {
		for(int64_t yy=0; yy<10; yy++) {
			for(int64_t zz=0; zz<10; zz++) {
				acc.set(xx*100+yy*10+zz, xx,yy,zz);
			}
		}
	}

	cerr << "Testing unity/full sized chunks" << endl;
	if(test1(img) != 0) 
		return -1;
	if(test2(img) != 0) 
		return -1;
	if(test3(img) != 0) 
		return -1;
	if(test4(img) != 0) 
		return -1;

	cerr << "Testing odd sized chunks" << endl;
	if(test5(img) != 0) 
		return -1;
	if(test6(img) != 0) 
		return -1;
	if(test7(img) != 0) 
		return -1;

	return 0;
}

