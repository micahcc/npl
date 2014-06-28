
#include "slicer.h"

#include <vector>
#include <list>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstdint>

using namespace std;

void linToIndex(size_t lpos, size_t& x, size_t& y, size_t& z, size_t& w,
		size_t sx, size_t sy, size_t sz, size_t sw)
{
	w = lpos%sw;
	lpos = lpos/sw;

	z = lpos%sz;
	lpos = lpos/sz;

	y = lpos%sy;
	lpos = lpos/sy;
	
	x = lpos%sx;
}

void indexToLin(size_t& lpos, size_t x, size_t y, size_t z, size_t w,
		size_t sx, size_t sy, size_t sz, size_t sw)
{
	(void)sx;

	lpos = 0;
	lpos = x*sy*sz*sw + y*sz*sw + z*sw + w;
}

int main()
{
	size_t X=2, Y=3, Z=4, W=5;
	std::vector<std::pair<size_t, size_t>> roi({{0,1},{0,1},{1,2},{1,3}});
	double array[X*Y*Z*W];
	size_t tx, ty, tz, tw;

	// canonical ordering
	size_t ii=0;
	for(size_t xx=0; xx<X; xx++) {
		for(size_t yy=0; yy<Y; yy++) {
			for(size_t zz=0; zz<Z; zz++) {
				for(size_t ww=0; ww<W; ww++) {
					size_t p;
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
	list<size_t> order({3,2,1,0});
	vector<size_t> pos;
	size_t p, xx, yy, zz, ww;

	std::vector<size_t> tdim({X,Y,Z,W});
	Slicer slicer(tdim);

	cerr << "Classic Ordering" << endl;
	slicer.setOrder(order);
	for(slicer.gotoBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		size_t sp = slicer.get(pos);
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
	order.push_back(order.front());
	order.pop_front();

	slicer.setOrder(order);
	for(slicer.gotoBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		size_t sp = slicer.get(pos);
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
	order.push_back(order.front());
	order.pop_front();

	slicer.setOrder(order);
	for(slicer.gotoBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		size_t sp = slicer.get(pos);
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
	order.push_back(order.front());
	order.pop_front();
	slicer.setOrder(order);
	for(slicer.gotoBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		size_t sp = slicer.get(pos);
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
	for(slicer.gotoBegin(); !slicer.isEnd(); slicer++, ii++) {
		
		size_t sp = slicer.get(pos);
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
	cerr << "PASS!" << endl;
}

