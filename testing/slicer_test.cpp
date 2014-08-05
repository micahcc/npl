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
 * @file slicer_test.cpp
 *
 *****************************************************************************/


#include "slicer.h"

#include <vector>
#include <list>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstdint>

using namespace std;
using namespace npl;

void linToIndex(int64_t lpos, int64_t& x, int64_t& y, int64_t& z, int64_t& w,
		int64_t sx, int64_t sy, int64_t sz, int64_t sw)
{
	w = lpos%sw;
	lpos = lpos/sw;

	z = lpos%sz;
	lpos = lpos/sz;

	y = lpos%sy;
	lpos = lpos/sy;
	
	x = lpos%sx;
}

void indexToLin(int64_t& lpos, int64_t x, int64_t y, int64_t z, int64_t w,
		int64_t sx, int64_t sy, int64_t sz, int64_t sw)
{
	(void)sx;

	lpos = 0;
	lpos = x*sy*sz*sw + y*sz*sw + z*sw + w;
}

int main()
{
	int64_t X=2, Y=3, Z=4, W=5;
	std::vector<std::pair<int64_t, int64_t>> roi({{0,1},{0,1},{1,2},{1,3}});
	double array[X*Y*Z*W];
	int64_t tx, ty, tz, tw;

	// canonical ordering
	int64_t ii=0;
	for(int64_t xx=0; xx<X; ++xx) {
		for(int64_t yy=0; yy<Y; ++yy) {
			for(int64_t zz=0; zz<Z; ++zz) {
				for(int64_t ww=0; ww<W; ++ww) {
					int64_t p;
					indexToLin(p, xx,yy,zz,ww, X,Y,Z,W);
					array[p] = ii++;

					linToIndex(p, tx,ty,tz,tw, X,Y,Z,W);
					if(tx != xx || ty != yy || tz != zz || tw != ww) {
						cerr << "Error in nd pos -> index -> nd pos conversion"
							<< endl;
						return -1;
					}
				}
			}
		}
	}

	ii = 0;
	vector<size_t> order({3,2,1,0});
	vector<int64_t> pos(4);
	int64_t p, xx, yy, zz, ww;

	std::vector<size_t> tdim({(size_t)X,(size_t)Y,(size_t)Z,(size_t)W});
	Slicer slicer(tdim.size(), tdim.data());

	cerr << "Classic Ordering" << endl;
	slicer.setOrder(order);
	for(slicer.goBegin(); !slicer.isEnd(); ++slicer, ++ii) {
		
		int64_t sp = *slicer;
		slicer.index(pos.size(), pos.data());
		xx = pos[0]; yy = pos[1]; zz = pos[2]; ww = pos[3];

		indexToLin(p, xx,yy,zz,ww, X,Y,Z,W);
		if(p != sp) {
			cerr << "Disagreement on linear position" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}

		linToIndex(sp, tx,ty,tz,tw, X,Y,Z,W);

		if(tx != xx || ty != yy || tz != zz || tw != ww) {
			cerr << "Disagreement on ND position!" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << tx << "," << ty << "," << tz << "," << tw << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}
		
		if(*slicer != array[sp]) {
			cerr << "Error, incorrect ordering!" << endl;
			return -1;
		}
	}
	cerr << "Done" << endl;
	
	cerr << "Classic Ordering (Default)" << endl;
	slicer.setOrder({});
	for(slicer.goBegin(); !slicer.isEnd(); ++slicer, ++ii) {
		
		int64_t sp = *slicer;
		slicer.index(pos.size(), pos.data());
		xx = pos[0]; yy = pos[1]; zz = pos[2]; ww = pos[3];

		indexToLin(p, xx,yy,zz,ww, X,Y,Z,W);
		if(p != sp) {
			cerr << "Disagreement on linear position" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}

		linToIndex(sp, tx,ty,tz,tw, X,Y,Z,W);

		if(tx != xx || ty != yy || tz != zz || tw != ww) {
			cerr << "Disagreement on ND position!" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << tx << "," << ty << "," << tz << "," << tw << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}
		
		if(*slicer != array[sp]) {
			cerr << "Error, incorrect ordering!" << endl;
			return -1;
		}
	}
	cerr << "Done" << endl;
	
	cerr << endl << "Rotated 1" << endl;
	std::rotate(order.begin(), order.begin()+1, order.end());

	slicer.setOrder(order);
	for(slicer.goBegin(); !slicer.isEnd(); ++slicer, ++ii) {
		
		int64_t sp = *slicer;
		slicer.index(pos.size(), pos.data());
		xx = pos[0]; yy = pos[1]; zz = pos[2]; ww = pos[3];
		cerr << xx << "," << yy << "," << zz << "," << ww << endl;

		indexToLin(p, xx,yy,zz,ww, X,Y,Z,W);
		if(p != sp) {
			cerr << "Disagreement on linear position" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}

		linToIndex(sp, tx,ty,tz,tw, X,Y,Z,W);

		if(tx != xx || ty != yy || tz != zz || tw != ww) {
			cerr << "Disagreement on ND position!" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << tx << "," << ty << "," << tz << "," << tw << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}
		
		if(*slicer != array[sp]) {
			cerr << "Error, incorrect ordering!" << endl;
			return -1;
		}
	}
	cerr << endl;
	
	cerr << endl << "Rotated 2" << endl;
	std::rotate(order.begin(), order.begin()+1, order.end());

	slicer.setOrder(order);
	for(slicer.goBegin(); !slicer.isEnd(); ++slicer, ++ii) {
		
		int64_t sp = *slicer;
		slicer.index(pos.size(), pos.data());

		xx = pos[0]; yy = pos[1]; zz = pos[2]; ww = pos[3];
		cerr << xx << "," << yy << "," << zz << "," << ww << endl;

		indexToLin(p, xx,yy,zz,ww, X,Y,Z,W);
		if(p != sp) {
			cerr << "Disagreement on linear position" << endl;
			return -1;
		}

		linToIndex(sp, tx,ty,tz,tw, X,Y,Z,W);

		if(tx != xx || ty != yy || tz != zz || tw != ww) {
			cerr << "Disagreement on ND position!" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << tx << "," << ty << "," << tz << "," << tw << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}
		
		if(*slicer != array[sp]) {
			cerr << "Error, incorrect ordering!" << endl;
			return -1;
		}
	}
	cerr << endl;
	
	cerr << endl << "Rotated 3" << endl;
	std::rotate(order.begin(), order.begin()+1, order.end());
	slicer.setOrder(order);
	for(slicer.goBegin(); !slicer.isEnd(); ++slicer, ++ii) {
		
		int64_t sp = *slicer;
		slicer.index(pos.size(), pos.data());

		xx = pos[0]; yy = pos[1]; zz = pos[2]; ww = pos[3];
		cerr << xx << "," << yy << "," << zz << "," << ww << endl;

		indexToLin(p, xx,yy,zz,ww, X,Y,Z,W);
		if(p != sp) {
			cerr << "Disagreement on linear position" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}

		linToIndex(sp, tx,ty,tz,tw, X,Y,Z,W);

		if(tx != xx || ty != yy || tz != zz || tw != ww) {
			cerr << "Disagreement on ND position!" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << tx << "," << ty << "," << tz << "," << tw << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}
		
		if(*slicer != array[sp]) {
			cerr << "Error, incorrect ordering!" << endl;
			return -1;
		}
	}
	cerr << endl;

	// roi
	cerr << endl << "Previous, With ROI" << endl;
	slicer.setROI(roi);
	for(slicer.goBegin(); !slicer.isEnd(); ++slicer, ++ii) {
		
		int64_t sp = *slicer;
		slicer.index(pos.size(), pos.data());
		xx = pos[0]; yy = pos[1]; zz = pos[2]; ww = pos[3];
		cerr << xx << "," << yy << "," << zz << "," << ww << endl;

		indexToLin(p, xx,yy,zz,ww, X,Y,Z,W);
		if(p != sp) {
			cerr << "Disagreement on linear position" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}

		linToIndex(sp, tx,ty,tz,tw, X,Y,Z,W);

		if(tx != xx || ty != yy || tz != zz || tw != ww) {
			cerr << "Disagreement on ND position!" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << tx << "," << ty << "," << tz << "," << tw << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}
		
		if(*slicer != array[sp]) {
			cerr << "Error, incorrect ordering!" << endl;
			cerr << xx << "," << yy << "," << zz << "," << ww << endl;
			cerr << tx << "," << ty << "," << tz << "," << tw << endl;
			cerr << p << " vs " << sp << endl;
			return -1;
		}
	}
	cerr << endl;

	size_t ITERS = 100000;
	size_t sum = 0;
	cerr << "Speed Test!" << endl;
	clock_t t = clock();
	for(size_t ii=0 ; ii < ITERS; ++ii) {
		for(slicer.goBegin(); !slicer.isEnd(); ++slicer, ++ii)
			sum += *slicer;
	}
	t = clock() - t;
	cerr << "Large restart Runtime: " << t << " ( " << t/CLOCKS_PER_SEC << " ) seconds" << endl;
	

	std::vector<size_t> newdim({50, 50, 50, 50});
	t = clock();
	
	Slicer slicer2(newdim.size(), newdim.data());
//	slicer.updateDim(newdim.size(), newdim.data());
	ii = 0;
	for(slicer2.goBegin(); !slicer2.isEnd(); ++slicer2, ++ii) {
		sum += *slicer2;
		if(ii >= 50*50*50*50) {
			cerr << "Error should have finished!" << endl;
			return -1;
		}
	}
	t = clock() - t;
	cerr << "Large Area Runtime: " << t << " ( " << t/CLOCKS_PER_SEC << " ) seconds" << endl;
	

}
