/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file pseudopolar_test.cpp Tests the ability of FFT and Zoom based pseudo
 * polar gridded fourier transform to match a brute-force linear interpolation
 * method, for highly variable (striped) fourier domain
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>
#include <random>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#define DEBUG 1

#include "mrimage.h"
#include "ndarray_utils.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "basic_functions.h"
#include "basic_plot.h"
#include "chirpz.h"

#include "fftw3.h"

clock_t brute_time = 0;
clock_t fft_time = 0;
clock_t zoom_time = 0;

using namespace npl;
using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::AngleAxisd;
using Eigen::EigenSolver;

/**
 * @brief Performs a rotation of the image first by rotating around z, then
 * around y, then around x.
 *
 * @param rx Rotation around x, radians
 * @param ry Rotation around y, radians
 * @param rz Rotation around z, radians
 * @param in Input image
 *
 * @return
 */
shared_ptr<MRImage> bruteForceRotate(Vector3d axis, double theta,
        shared_ptr<const MRImage> in)
{
    Matrix3d m;
    // negate because we are starting from the destination and mapping from
    // the source
    m = AngleAxisd(-theta, axis);
    LinInterp3DView<double> lin(in);
    auto out = dynamic_pointer_cast<MRImage>(in->copy());
    Vector3d ind;
    Vector3d cind;
    Vector3d center;
    for(size_t ii=0; ii<3 && ii<in->ndim(); ii++) {
        center[ii] = (in->dim(ii)-1)/2.;
    }

    for(Vector3DIter<double> it(out); !it.isEnd(); ++it) {
        it.index(3, ind.array().data());
        cind = m*(ind-center)+center;

        // set for each t
        for(size_t tt = 0; tt<in->tlen(); tt++)
            it.set(tt, lin(cind[0], cind[1], cind[2], tt));
    }

    return out;
}

double box(double x, double xmin, double xmax)
{
    return (x < xmax && x > xmin);
}

double boxGen(double x, double y, double z, double xsz, double ysz, double zsz)
{
    return (x/xsz > 0.25 && x/xsz < .35) && (y/ysz > 0.55 && y/ysz < .75) &&
        (z/zsz > 0.55 && z/zsz < .59);
}

double gaussGen(double x, double y, double z, double xsz, double ysz, double zsz)
{
    double v = exp(-pow(xsz/2-x,2)/9)*exp(-pow(ysz/2-y,2)/16)*exp(-pow(zsz/2-z,2)/64);
    if(v > 0.00001)
        return v;
    else
        return 0;
}

shared_ptr<MRImage> createTestImageRotated(size_t sz1, double rx, double ry,
        double rz)
{
    // create an image
    size_t sz[] = {sz1, sz1, sz1};
    auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, COMPLEX128);

    // create rotation matrix
    Matrix3d m;
    m = AngleAxisd(rx, Vector3d::UnitX())*AngleAxisd(ry, Vector3d::UnitY())*
        AngleAxisd(rz, Vector3d::UnitZ());
    m = m.inverse();

    double sum = 0;
    Vector3d ind;
    Vector3d cind;
    Vector3d center;
    for(size_t ii=0; ii<3 && ii<in->ndim(); ii++) {
        center[ii] = (in->dim(ii)-1)/2.;
    }

    for(Vector3DIter<double> it(in); !it.isEnd(); ++it) {
        it.index(3, ind.array().data());
        cind = m*(ind-center)+center;

        // set for each t
        for(size_t tt = 0; tt<in->tlen(); tt++) {
            double v = gaussGen(cind[0], cind[1], cind[2], sz1, sz1, sz1);
            it.set(tt, v);
            sum += v;
        }
    }

    for(OrderIter<double> sit(in); !sit.eof(); ++sit)
        sit.set(sit.get()/sum);

    return in;
}

shared_ptr<MRImage> createTestImage(size_t sz1)
{
    // create an image
    int64_t index[3];
    size_t sz[] = {sz1, sz1, sz1};
    auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, COMPLEX128);

    // fill with a shape that is somewhat unique when rotated.
    OrderIter<double> sit(in);
    double sum = 0;
    while(!sit.eof()) {
        sit.index(3, index);
        double v= gaussGen(index[0], index[1], index[2], sz1, sz1, sz1);
        sit.set(v);
        sum += v;
        ++sit;
    }

    for(sit.goBegin(); !sit.eof(); ++sit)
        sit.set(sit.get()/sum);

    return in;
}

Vector3d getAxis(shared_ptr<const MRImage> inimg1, shared_ptr<const MRImage> inimg2)
{

    auto img1 = dynamic_pointer_cast<MRImage>(inimg1->copy());
    auto img2 = dynamic_pointer_cast<MRImage>(inimg2->copy());
//
//    cerr << "Smoothing input." << endl;
//    for(size_t dd=0; dd<img1->ndim(); dd++)
//        gaussianSmooth1D(img1, dd, 5);
//    for(size_t dd=0; dd<img2->ndim(); dd++)
//        gaussianSmooth1D(img2, dd, 5);

    ostringstream oss;
    Vector3d axis;

    size_t ndim = 3;
    std::vector<int64_t> index(ndim);
    std::vector<int64_t> cindex(ndim);
    size_t pslope[ndim-1];
    double bestang0 = -1;
    double bestang1 = -1;
    double mineang0 = -1;
    double mineang1 = -1;
    double maxcor = 0;
    double minerr = INFINITY;

    for(size_t ii=0; ii<ndim; ii++) {
        size_t tmp = 0;
        for(size_t jj=0; jj<ndim; jj++) {
            if(jj != ii)
                pslope[tmp++] = jj;
        }

        cerr << "pseudo radius: " << ii <<
            ", pseudo slope 1: " << pslope[0] << ", pseudo slope 2: " <<
            pslope[1] << endl;

        auto s1_pp = dynamic_pointer_cast<MRImage>(pseudoPolar(img1, ii));
        auto s2_pp = dynamic_pointer_cast<MRImage>(pseudoPolar(img2, ii));

        writeComplex("s1_pp"+to_string(ii), s1_pp, true);
        writeComplex("s2_pp"+to_string(ii), s2_pp, true);

        vector<size_t> compdim(s1_pp->dim(), s1_pp->dim()+s1_pp->ndim());
        compdim[ii] = 1;
        auto corimg = dynamic_pointer_cast<MRImage>(
                s1_pp->copyCast(compdim.size(), compdim.data(), FLOAT64));
        auto errimg = dynamic_pointer_cast<MRImage>(
                s1_pp->copyCast(compdim.size(), compdim.data(), FLOAT64));

        ChunkIter<cdouble_t> it1(s1_pp);
        it1.setLineChunk(ii);
        ChunkIter<cdouble_t> it2(s2_pp);
        it2.setLineChunk(ii);

        OrderIter<double> cit(corimg);
        OrderIter<double> eit(errimg);
        for(it1.goBegin(), it2.goBegin(), cit.goBegin(), eit.goBegin();
                !it1.eof() && !it2.eof() && !cit.eof() && !eit.eof();
                it1.nextChunk(), it2.nextChunk(), ++cit, ++eit) {

            // double check the indices
            it1.index(index);
            cit.index(cindex);

            double ang0 = 2.*index[pslope[0]]/(s1_pp->dim(pslope[0]))-1;
            double ang1 = 2.*index[pslope[1]]/(s2_pp->dim(pslope[1]))-1;

            for(size_t ii=0; ii<index.size(); ii++){
                if(index[ii] != cindex[ii])
                    throw std::logic_error("Iteration order wrong!");
            }

            double corr = 0;
            double sum1 = 0, sum2 = 0;
            double ssq1 = 0, ssq2= 0;
            size_t count = 0;
            double err = 0;
            for(; !it1.eoc() && !it2.eoc(); ++it1, ++it2) {
                double m1 = abs(*it1);
                double m2 = abs(*it2);
                corr += m1*m2;
                sum1 += m1;
                sum2 += m2;
                ssq1 += m1*m1;
                ssq2 += m2*m2;
                err += pow(m1-m2,2);
                count++;
            }
            assert(it1.isChunkEnd());
            assert(it2.isChunkEnd());

            if(err < minerr) {
                minerr = err;
                cerr << "New Min Err: " << err << ", " << ang0 << ","
                    << ang1 << " or " << index[0] << "," << index[1] << ","
                    << index[2] << endl;
                mineang0 = ang0;
                mineang1 = ang1;
            }

            corr = sample_corr(count, sum1, sum2, ssq1, ssq2, corr);
            cit.set(corr);
            eit.set(err);

            if(fabs(corr) > maxcor) {
                maxcor = corr;
                cerr << "New Max Cor: " << corr << ", " << ang0 << ","
                    << ang1 << " or " << index[0] << "," << index[1]
                    << "," << index[2] << endl;
                axis[ii] = 1;
                bestang0 = ang0;
                bestang1 = ang1;
                axis[pslope[0]] = bestang0;
                axis[pslope[1]] = bestang1;
                axis.normalize();
            }

        }

        corimg->write("corimg_"+to_string(ii)+".nii.gz");
        errimg->write("errimg_"+to_string(ii)+".nii.gz");
        cerr << "pseudo radius: " << ii <<
            ", pseudo slope 1: " << pslope[0] << ", pseudo slope 2: "
            << pslope[1] << " best cor: " << maxcor << " at " << bestang0
            << ", " << bestang1 << ", best err: " << minerr << " at " <<
            mineang0 << ", " << mineang1 << endl;
        assert(it1.isEnd());
        assert(it2.isEnd());
    }


    return axis;
}

int testRotationAxis(double x, double y, double z, double theta)
{
    cerr << "Creating Test Image" << endl;
    size_t SIZE = 64;
    auto in = createTestImage(SIZE);
    cerr << "Done" << endl;

    cerr << "Rotating" << endl;

    Vector3d axis(x, y, z);
    axis.normalize();
    Matrix3d R = AngleAxisd(theta, axis).matrix();
    Vector3d euler = R.eulerAngles(0,1,2);
    cerr << "Axis:\n" << axis.transpose() << endl;
    cerr << "Matrix:\n" << R << endl;
    cerr << "Euler:\n" << euler.transpose() << endl;

    /// figure out which one would be the pseudopolar radius and slopes
    size_t rad = 0;
    size_t slopes[2];
    {
        double mrad = 0;
        for(size_t dd=0; dd<3; dd++) {
            if(axis[dd] > mrad) {
                mrad = axis[dd];
                rad = dd;
            }
        }
        size_t tmpd = 0;
        for(size_t dd=0; dd<3; dd++) {
            if(rad != dd)
                slopes[tmpd++] = dd;
        }
    }
    cerr << "Expected Pseudoradius: " << rad << endl;
    cerr << "Slope Dim: " << slopes[0] << " = " << axis[slopes[0]]/axis[rad] <<
        "(" << 64*(1+axis[slopes[0]]/axis[rad]) << ")" << endl;
    cerr << "Sl64 Dim: " << slopes[1] << " = " << axis[slopes[1]]/axis[rad] <<
        "(" << 64*(1+axis[slopes[1]]/axis[rad]) << ")" << endl;

    // rotate image
//    auto out = createTestImageRotated(SIZE, euler[0], euler[1], euler[2]);
    auto out = dynamic_pointer_cast<MRImage>(in->copy());
    rotateImageShearFFT(out, euler[0], euler[1], euler[2]);
    writeComplex("rotated", out);

    rotateImageShearFFT(in, euler[0], euler[1], euler[2]);
    rotateImageShearFFT(in, -euler[0], -euler[1], -euler[2]);
    writeComplex("input", in);

    cerr << "Done" << endl;

    Vector3d newax = getAxis(in, out);
    cerr << "Axis: " << newax.transpose() << endl;

    return 0;
}

int main(int argc, char** argv)
{
    double x = 1, y = 0.4, z = 0;
    double theta = 3.14159/20;

    if(argc == 4) {
        // axis
        x = atof(argv[1]);
        y = atof(argv[2]);
        z = atof(argv[3]);
    } else if(argc == 5) {
        // axis + rotation
        x = atof(argv[1]);
        y = atof(argv[2]);
        z = atof(argv[3]);
        theta = atof(argv[4]);
    }
    if(testRotationAxis(x, y, z, theta) != 0)
        return -1;

    return 0;
}

