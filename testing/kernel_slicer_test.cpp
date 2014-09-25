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
 * @file kernel_slicer_test.cpp
 *
 *****************************************************************************/
#include <cassert>
#include <iostream>
#include "slicer.h"

using namespace std;

int64_t clamp(int64_t inf, int64_t sup, int64_t v)
{
  return std::max(inf, std::min(sup, v));
}

int main()
{
  size_t a,b,c;
  size_t sz[] = {10,20,30,10};
  std::vector<size_t> kern({1,1,1});
  npl::KSlicer it(4, sz);
  it.setRadius(kern);

  assert(it.isBegin());
  assert(!it.isEnd());

  std::vector<int64_t> index({1,1,1,0});
  it.goIndex(index);
  assert(!it.isBegin());
  assert(!it.isEnd());

  assert(!it.isEnd());

  it.goBegin();
  index.assign({3,4,5,6});
  it.goIndex(index);

  a = *++it;
  b = *it;
  ++it;
  c = *--it;
  cerr << a << " == " << b << " == " << c << endl;
  if(a != b || b != c) {
    return -1;
  }

  // compare with brute force method, note that zz will be the fastest
  // because it is the longest dimension
  it.goBegin();
  for(int64_t xx = 0; xx < sz[0]; ++xx){
    for(int64_t yy = 0; yy < sz[1]; ++yy){
      for(int64_t tt = 0; tt < sz[3]; ++tt){
        for(int64_t zz = 0; zz < sz[2]; ++zz, ++it){
          int64_t neighbor = 0;
          for(int64_t xxo = -1; xxo <= 1; ++xxo){
            for(int64_t yyo = -1; yyo <= 1; ++yyo){
              for(int64_t zzo = -1; zzo <= 1; ++zzo){
                for(int64_t tto = 0; tto <= 0; ++tto){
                  int64_t xeff = clamp(0, sz[0]-1, xx+xxo);
                  int64_t yeff = clamp(0, sz[1]-1, yy+yyo);
                  int64_t zeff = clamp(0, sz[2]-1, zz+zzo);
                  int64_t teff = clamp(0, sz[3]-1, tt+tto);
                  int64_t lin = teff+zeff*sz[3]+yeff*sz[3]*sz[2]+xeff*sz[3]*sz[2]*sz[1];

                  int64_t itlin = it.getK(neighbor);

                  it.indexK(neighbor, 1, index.data(), true);
                  if(xeff != index[0]) {
                    cerr << "Incorrect x map" << endl;
                    return -1;
                  }

                  it.indexK(neighbor, 2, index.data(), true);
                  if(yeff != index[1]) {
                    cerr << "Incorrect y map" << endl;
                    return -1;
                  }

                  it.indexK(neighbor, 3, index.data(), true);
                  if(zeff != index[2]) {
                    cerr << "Incorrect z map" << endl;
                    return -1;
                  }

                  it.indexK(neighbor, 4, index.data(), true);
                  if(teff != index[3]) {
                    cerr << "Incorrect t map" << endl;
                    return -1;
                  }

                  if(lin != itlin) {
                    cerr << "Incorrect linear mapping" << endl;
                    return -1;
                  }
				  ++neighbor;
                }
              }
            }
          }
        }
      }
    }
  }
  assert(it.isEnd());

  --it;
  // iterating backward
  for(int xx = sz[0]-1; xx >= 0; xx--){
    for(int yy = sz[1]-1; yy >= 0; yy--){
      for(int tt = sz[3]-1; tt >= 0; tt--){
        for(int zz = sz[2]-1; zz >= 0; zz--, --it){
          int64_t neighbor = 0;
          for(int64_t xxo = -1; xxo <= 1; ++xxo){
            for(int64_t yyo = -1; yyo <= 1; ++yyo){
              for(int64_t zzo = -1; zzo <= 1; ++zzo){
                for(int64_t tto = 0; tto <= 0; ++tto){
                  int64_t xeff = clamp(0, sz[0]-1, xx+xxo);
                  int64_t yeff = clamp(0, sz[1]-1, yy+yyo);
                  int64_t zeff = clamp(0, sz[2]-1, zz+zzo);
                  int64_t teff = clamp(0, sz[3]-1, tt+tto);
                  int64_t lin = teff+zeff*sz[3]+yeff*sz[3]*sz[2]+xeff*sz[3]*sz[2]*sz[1];

                  int64_t itlin = it.getK(neighbor);
                  it.indexK(neighbor, index.size(), index.data(), true);
				  ++neighbor;

                  if(xeff != index[0]) {
                    cerr << "Incorrect x map" << endl;
                    return -1;
                  }

                  if(yeff != index[1]) {
                    cerr << "Incorrect y map" << endl;
                    return -1;
                  }

                  if(zeff != index[2]) {
                    cerr << "Incorrect z map" << endl;
                    return -1;
                  }

                  if(teff != index[3]) {
                    cerr << "Incorrect t map" << endl;
                    return -1;
                  }

                  if(lin != itlin) {
                    cerr << "Incorrect linear mapping" << endl;
                    return -1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  assert(it.isBegin());

  // now make the iterator uneven
  std::vector<std::pair<int64_t,int64_t>> newk({{-2,1},{-1,0},{0,2},{0,2}});
  it.setWindow(newk);

  it.goBegin();
  for(int64_t xx = 0; xx < sz[0]; ++xx){
    for(int64_t yy = 0; yy < sz[1]; ++yy){
      for(int64_t tt = 0; tt < sz[3]; ++tt){
        for(int64_t zz = 0; zz < sz[2]; ++zz, ++it){
          int64_t neighbor = 0;
          for(int64_t xxo = -2; xxo <= 1; ++xxo){
            for(int64_t yyo = -1; yyo <= 0; ++yyo){
              for(int64_t zzo = 0; zzo <= 2; ++zzo){
                for(int64_t tto = 0; tto <= 2; ++tto){
                  int64_t xeff = clamp(0, sz[0]-1, xx+xxo);
                  int64_t yeff = clamp(0, sz[1]-1, yy+yyo);
                  int64_t zeff = clamp(0, sz[2]-1, zz+zzo);
                  int64_t teff = clamp(0, sz[3]-1, tt+tto);
                  int64_t lin = teff+zeff*sz[3]+yeff*sz[3]*sz[2]+xeff*sz[3]*sz[2]*sz[1];

				  int64_t itlin = it.getK(neighbor);
                  it.indexK(neighbor, index.size(), index.data(), true);
				  ++neighbor;

                  if(xeff != index[0]) {
                    cerr << "Incorrect x map" << endl;
                    return -1;
                  }

                  if(yeff != index[1]) {
                    cerr << "Incorrect y map" << endl;
                    return -1;
                  }

                  if(zeff != index[2]) {
                    cerr << "Incorrect z map" << endl;
                    return -1;
                  }

                  if(teff != index[3]) {
                    cerr << "Incorrect t map" << endl;
                    return -1;
                  }

                  if(lin != itlin) {
                    cerr << "Incorrect linear mapping" << endl;
                    return -1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  assert(it.isEnd());

  --it;
  // iterating backward
  for(int xx = sz[0]-1; xx >= 0; xx--){
    for(int yy = sz[1]-1; yy >= 0; yy--){
      for(int tt = sz[3]-1; tt >= 0; tt--){
        for(int zz = sz[2]-1; zz >= 0; zz--, --it){
          int64_t neighbor = 0;
          for(int64_t xxo = -2; xxo <= 1; ++xxo){
            for(int64_t yyo = -1; yyo <= 0; ++yyo){
              for(int64_t zzo = 0; zzo <= 2; ++zzo){
                for(int64_t tto = 0; tto <= 2; ++tto){
                  int64_t xeff = clamp(0, sz[0]-1, xx+xxo);
                  int64_t yeff = clamp(0, sz[1]-1, yy+yyo);
                  int64_t zeff = clamp(0, sz[2]-1, zz+zzo);
                  int64_t teff = clamp(0, sz[3]-1, tt+tto);
                  int64_t lin = teff+zeff*sz[3]+yeff*sz[3]*sz[2]+xeff*sz[3]*sz[2]*sz[1];
				
				  int64_t itlin = it.getK(neighbor);
                  it.indexK(neighbor, index.size(), index.data(), true);
				  ++neighbor;

                  if(xeff != index[0]) {
                    cerr << "Incorrect x map" << endl;
                    return -1;
                  }

                  if(yeff != index[1]) {
                    cerr << "Incorrect y map" << endl;
                    return -1;
                  }

                  if(zeff != index[2]) {
                    cerr << "Incorrect z map" << endl;
                    return -1;
                  }

                  if(teff != index[3]) {
                    cerr << "Incorrect t map" << endl;
                    return -1;
                  }

                  if(lin != itlin) {
                    cerr << "Incorrect linear mapping" << endl;
                    return -1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  assert(it.isBegin());

  // jumping to the current location
  for(int xx = sz[0]-1; xx >= 0; xx--){
    for(int yy = sz[1]-1; yy >= 0; yy--){
      for(int tt = sz[3]-1; tt >= 0; tt--){
        for(int zz = sz[2]-1; zz >= 0; zz--){
          int64_t neighbor = 0;
          index[0] = xx;
          index[1] = yy;
          index[2] = zz;
          index[3] = tt;
          it.goBegin(); // just to make sure jumping around doesn't screw with it
          it.goIndex(index);
          for(int64_t xxo = -2; xxo <= 1; ++xxo){
            for(int64_t yyo = -1; yyo <= 0; ++yyo){
              for(int64_t zzo = 0; zzo <= 2; ++zzo){
                for(int64_t tto = 0; tto <= 2; ++tto){
                  int64_t xeff = clamp(0, sz[0]-1, xx+xxo);
                  int64_t yeff = clamp(0, sz[1]-1, yy+yyo);
                  int64_t zeff = clamp(0, sz[2]-1, zz+zzo);
                  int64_t teff = clamp(0, sz[3]-1, tt+tto);
                  int64_t lin = teff+zeff*sz[3]+yeff*sz[3]*sz[2]+xeff*sz[3]*sz[2]*sz[1];

				  int64_t itlin = it.getK(neighbor);
                  it.indexK(neighbor, index.size(), index.data(), true);
				  ++neighbor;

                  if(xeff != index[0]) {
                    cerr << "Incorrect x map" << endl;
                    return -1;
                  }

                  if(yeff != index[1]) {
                    cerr << "Incorrect y map" << endl;
                    return -1;
                  }

                  if(zeff != index[2]) {
                    cerr << "Incorrect z map" << endl;
                    return -1;
                  }

                  if(teff != index[3]) {
                    cerr << "Incorrect t map" << endl;
                    return -1;
                  }

                  if(lin != itlin) {
                    cerr << "Incorrect linear mapping" << endl;
                    return -1;
                  }
                }
              }
            }
          }
          it.goEnd(); // just to make sure jumping around doesn't screw with it
        }
      }
    }
  }
}
