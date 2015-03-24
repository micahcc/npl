/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file chunkslicer_test.cpp Tests region-of-interest specification in Slicer
 * class
 *
 *****************************************************************************/

#include <iostream>

#include "slicer.h"

using namespace std;
using namespace npl;

int test1()
{
	cerr << "test1" << endl;
	size_t dim[] = {10, 10, 10};
	ChunkSlicer slicer(3, dim);

	int64_t roi1[3] = {3, 5, 4};
	size_t roiz[3] = {5, 4, 3};
	slicer.setROI(3, roiz, roi1);
	int64_t breaks[] = {1, 1, 0};
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

int test2()
{
	cerr << "test2" << endl;
	size_t dim[] = {10, 10, 10};
	ChunkSlicer slicer(3, dim);

	int64_t roi1[3] = {3, 5, 4};
	size_t roiz[3] = {5, 4, 3};
	slicer.setROI(3, roiz, roi1);
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

int test3()
{
	cerr << "test3" << endl;
	size_t dim[] = {10, 10, 10};
	ChunkSlicer slicer(3, dim);

	int64_t roi1[3] = {3, 5, 4};
	size_t roiz[3] = {5, 4, 3};
	int64_t breaks[] = {1, 0, 0};
	slicer.setChunkSize(3, breaks);
	slicer.setROI(3, roiz, roi1);
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

int test4()
{
	cerr << "test4" << endl;
	size_t dim[] = {10, 10, 10};
	ChunkSlicer slicer(3, dim);

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

int test5()
{
	cerr << "test5" << endl;
	size_t dim[] = {10, 10, 10};
	ChunkSlicer slicer(3, dim);

	int64_t roi1[3] = {1, 1, 2};
	size_t roiz[3] = {6, 8, 6};
	slicer.setROI(3, roiz, roi1);

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

int test6()
{
	cerr << "test6" << endl;
	size_t dim[] = {10, 10, 10};
	ChunkSlicer slicer(3, dim);

	int64_t roi1[3] = {1, 1, 2};
	size_t roiz[3] = {6, 8, 6};
	slicer.setROI(3, roiz, roi1);

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

int test7()
{
	cerr << "test7" << endl;
	size_t dim[] = {10, 10, 10};
	ChunkSlicer slicer(3, dim);

	int64_t roi1[3] = {1, 1, 2};
	size_t roiz[3] = {6, 8, 6};
	slicer.setROI(3, roiz, roi1);

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
	cerr << "Testing unity/full sized chunks" << endl;
	if(test1() != 0)
		return -1;
	if(test2() != 0)
		return -1;
	if(test3() != 0)
		return -1;
	if(test4() != 0)
		return -1;

	cerr << "Testing odd sized chunks" << endl;
	if(test5() != 0)
		return -1;
	if(test6() != 0)
		return -1;
	if(test7() != 0)
		return -1;

	return 0;
}



