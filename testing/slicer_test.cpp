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
	for(int64_t xx=0; xx<X; xx++) {
		for(int64_t yy=0; yy<Y; yy++) {
			for(int64_t zz=0; zz<Z; zz++) {
				for(int64_t ww=0; ww<W; ww++) {
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
	vector<int64_t> pos;
	int64_t p, xx, yy, zz, ww;

	std::vector<size_t> tdim({(size_t)X,(size_t)Y,(size_t)Z,(size_t)W});
	Slicer slicer(tdim.size(), tdim.data());

	cerr << "Classic Ordering" << endl;
	slicer.setOrder(order);
	for(slicer.goBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		int64_t sp = *slicer;
		pos = slicer.index();
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
	for(slicer.goBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		int64_t sp = *slicer;
		pos = slicer.index();
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
	for(slicer.goBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		int64_t sp = *slicer;
		pos = slicer.index();

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
	for(slicer.goBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		int64_t sp = *slicer;
		pos = slicer.index();

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
	for(slicer.goBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		int64_t sp = *slicer;
		pos = slicer.index();
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
	for(size_t ii=0 ; ii < ITERS; ii++) {
		for(slicer.goBegin(); !slicer.isEnd(); slicer++, ii++) 
			sum += *slicer;
	}
	t = clock() - t;
	cerr << "Large restart Runtime: " << t << " ( " << t/CLOCKS_PER_SEC << " ) seconds" << endl;
	

	std::vector<size_t> newdim({50, 50, 50, 50});
	t = clock();
	slicer.updateDim(newdim.size(), newdim.data());
	ii = 0;
	for(slicer.goBegin(); !slicer.isEnd(); slicer++, ii++) {
		sum += *slicer;
		if(ii >= 50*50*50*50) {
			cerr << "Error should have finished!" << endl;
			return -1;
		}
	}
	t = clock() - t;
	cerr << "Large Area Runtime: " << t << " ( " << t/CLOCKS_PER_SEC << " ) seconds" << endl;
	

}
