/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * tracks.cpp Implementation of track class for loading tracks from trackvis,
 * or DFC formats.
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

namespace npl
{

class magic_error : public std::logic_error {
public:
	magic_error(const std::string& w) : std::logic_error(w) {};
	magic_error(const char* w) : std::logic_error(w) {} ;
};

/*********************************************************
 * DFT File Reader Functions
 ********************************************************/

/**
 * @brief Reads a BrainSuite DFT file. Throws INVALID_ARGUMENT if the magic is
 * wrong.
 *
 * @param filename trk file
 * @param ref Reference image
 *
 * @return vector of vector of points
 */
TrackSet readDFT(std::string tfile, std::string ref)
{
	DftHead head;
	uint8_t minversion[] = {1,0,0,3};
	int32_t npoints;
	float coord[3];
	double dcoord[3];
	TrackSet out;

	// read in the header
	ifstream infile(tfile.c_str(),  std::ifstream::in |  std::ifstream::binary);
	if(!infile.is_open())
		throw RUNTIME_ERROR("Error opening "+tfile+" for reading");

	infile.read(head.id_string, 8*sizeof(char));

	bool byteswap = false;
	if(strncmp(head.id_string, "DFT_BE", 6) == 0) {
		if(little_endian())
			byteswap = true;
		else if(big_endian())
			byteswap = false;
		else
			throw RUNTIME_ERROR("Broken Endianness");
	} else if(strncmp(head.id_string, "DFT_LE", 6) == 0) {
		if(little_endian())
			byteswap = false;
		else if(big_endian())
			byteswap = true;
		else
			throw RUNTIME_ERROR("Broken Endianness");
	} else {
		throw magic_error("Wrong Magic for DFT");
	}

	infile.read((char*)head.version, 4*sizeof(uint8_t));
	for(size_t ii=0; ii<4; ii++) {
		if(head.version[ii] < minversion[ii])
			throw RUNTIME_ERROR("DFT File Version too old!");
	}

	// read reference image, get spacing
	if(ref.empty())
		throw INVALID_ARGUMENT("No reference image provided to readDFT");
	auto refimg = readMRImage(ref);

	infile.read((char*)&head.header_size, sizeof(int32_t));
	infile.read((char*)&head.data_start, sizeof(int32_t));
	infile.read((char*)&head.metadata_offset, sizeof(int32_t));
	infile.read((char*)&head.subject_data_offset, sizeof(int32_t));
	infile.read((char*)&head.num_contours, sizeof(int32_t));
	infile.read((char*)&head.seedpoints, sizeof(int64_t));

	// Byte Swap
	if(byteswap) {
		swap(&head.header_size);
		swap(&head.data_start);
		swap(&head.metadata_offset);
		swap(&head.subject_data_offset);
		swap(&head.num_contours);
		swap(&head.seedpoints);
	}

	// jump to start of tracks
	infile.seekg(head.data_start, std::ios_base::beg);

	//for each track
	out.resize(head.num_contours);
	for(int ii = 0 ; ii < head.num_contours; ii++) {
		infile.read((char*)&npoints, sizeof(int32_t));

		out[ii].resize(npoints);
		//for each point in the track
		for(int jj = 0 ; jj < npoints; jj++) {

			// read coordinate, as index (remove spacing) and store the result
			// as a double (so that it can be converted to RAS)
			for(size_t kk = 0; kk<3; kk++) {
				infile.read((char*)&coord[kk], sizeof(float));
				if(byteswap) swap(&coord[kk]);
				dcoord[kk] = (double)coord[kk]/refimg->spacing(kk);
			}

			// convert index to physical point in image
			refimg->indexToPoint(3, dcoord, dcoord);

			for(size_t kk=0; kk<3; kk++)
				out[ii][jj][kk] = dcoord[kk];
		}
	}

	return out;
}

/************************************************
 * TrackVis Trk Reader
 ************************************************/

// Helper Classes and Functions
typedef struct
{
    int m;
    float point[0];
} TrkLine;

TrkLine* trkInc(TrkLine* cur, TrkHead* head)
{
	return (TrkLine*)(((float*)cur)+1 + cur->m*(3 + head->n_scalars) +
			head->n_properties);
}

float* trkPoint(TrkLine* cur, TrkHead* head, int index)
{
	assert(index < cur->m);
	return &cur->point[index*(3+head->n_scalars)];
}

/**
 * @brief Reads a trackvis trk file. Throws INVALID_ARGUMENT if the magic is
 * wrong.
 *
 * @param filename trk file
 * @param ref Reference image (in case the RAS is not valid)
 *
 * @return
 */
TrackSet readTrk(string tfile, string ref)
{
	ifstream infile(tfile, std::fstream::in | std::fstream::binary);
	if(!infile.is_open())
		throw RUNTIME_ERROR("Error opening "+tfile+" for reading");

	TrkHead head;
	bool byteswap = false;

	// Read Magic
	infile.read((char*)&head.id_string, sizeof(head.id_string));
	if(strncmp(head.id_string, "TRACK", 5) != 0)
		throw magic_error("Incorrect Magic for TrackVis");

	// Read Header
	infile.read((char*)&head.dim, sizeof(head.dim));
	infile.read((char*)&head.voxel_size, sizeof(head.voxel_size));
	infile.read((char*)&head.origin, sizeof(head.origin));
	infile.read((char*)&head.n_scalars, sizeof(head.n_scalars));
	infile.read((char*)&head.scalar_name, sizeof(head.scalar_name));
	infile.read((char*)&head.n_properties, sizeof(head.n_properties));
	infile.read((char*)&head.property_name, sizeof(head.property_name));
	infile.read((char*)&head.vox_to_ras, sizeof(head.vox_to_ras));
	infile.read((char*)&head.reserved, sizeof(head.reserved));
	infile.read((char*)&head.voxel_order, sizeof(head.voxel_order));
	infile.read((char*)&head.padA4, sizeof(head.padA4));
	infile.read((char*)&head.image_orientation_patient,
			sizeof(head.image_orientation_patient));
	infile.read((char*)&head.padB2, sizeof(head.padB2));
	infile.read((char*)&head.invert_x, sizeof(head.invert_x));
	infile.read((char*)&head.invert_y, sizeof(head.invert_y));
	infile.read((char*)&head.invert_z, sizeof(head.invert_z));
	infile.read((char*)&head.swap_xy, sizeof(head.swap_xy));
	infile.read((char*)&head.swap_yz, sizeof(head.swap_yz));
	infile.read((char*)&head.swap_zx, sizeof(head.swap_zx));
	infile.read((char*)&head.n_count, sizeof(head.n_count));
	infile.read((char*)&head.version, sizeof(head.version));
	infile.read((char*)&head.hdr_size, sizeof(head.hdr_size));

	// Check for byte swapping/Header Size
	if(head.hdr_size != 1000) {
		swap(&head.hdr_size);
		if(head.hdr_size != 1000)
			throw RUNTIME_ERROR("Error Invalid Header size in "+tfile);
		else
			byteswap = true;
	}

	if(byteswap) {
		// Read Header
		throw RUNTIME_ERROR("ERROR BYTE SWAPPING NOT YET IMPLEMENTED");
	}


#ifdef DEBUG
	cerr << head.id_string << endl;;
	cerr << "Dimensions" << endl;
	for(int i = 0 ; i < 3 ; i++)
		cerr << head.dim[i] << " ";
	cerr << endl;
	cerr << "Voxel Sizes" << endl;
	for(int i = 0 ; i < 3 ; i++)
		cerr << head.voxel_size[i] << " ";
	cerr << endl;
	cerr << "Number of Scalars: " << head.n_scalars << endl;
	for(int i = 0 ; i < head.n_scalars ; i++)
		cerr << "\t" << head.scalar_name[i] << endl;
	cerr << "Number of Properties: " << head.n_properties<< endl;
	for(int i = 0 ; i < head.n_properties; i++)
		cerr << "\t" << head.property_name[i] << endl;
#endif //DEBUG

	bool orient_valid = false;
	if(head.vox_to_ras[3][3] != 0)
		orient_valid = true;

#ifdef DEBUG
	if(!orient_valid)
		cerr << "Orient Invalid" << endl;
	cerr << "Origin" << endl;
	for(int i = 0 ; i < 3 ; i++) {
		cerr << head.origin[i] << " ";
	}
	cerr << endl;
	cerr << "Image Orientation (Patient): " << endl;
	for(int i = 0 ; i < 6 ; i++)
		cerr << head.image_orientation_patient[i] << "\t";
	cerr << endl;

	cerr << "Count: " << head.n_count << endl;
	cerr << "Version: " << head.version << endl;
	cerr << "Header Size: " << head.hdr_size << endl; //shouldbe 1000
#endif //DEBUG

	Eigen::Matrix4f reorient = Eigen::Matrix4f::Identity();
	if(orient_valid) {
		for(size_t ii=0; ii<4; ii++) {
			for(size_t jj=0; jj<4; jj++) {
				reorient(ii,jj) = head.vox_to_ras[ii][jj];
			}
		}
	} else if(!ref.empty()) {
		auto tmp = readMRImage(ref);
		if(tmp->ndim() < 3)
			throw RUNTIME_ERROR("Error input image "+ref+"does not "
					"have sufficient dimensions!");
		// set origin
		for(size_t ii=0; ii<3; ii++)
			reorient(ii, 3) = tmp->origin(ii);

		// set direction+ spacing
		for(size_t ii=0; ii<3; ii++) {
			for(size_t jj=0; jj<3; jj++) {
				reorient(ii,jj) = tmp->spacing(jj)*tmp->direction(ii,jj);
			}
		}
	}
#ifdef DEBUG
	cerr<<"Effective Index to RAS Matrix:\n"<<reorient<<endl;
#endif //DEBUG

	if(head.n_count == 0)
		return TrackSet();

	/* Actually Load Data */
	int trkpoints;
	TrackSet out(head.n_count);

	//for each track
	for(int ii = 0; ii < head.n_count; ii++) {
		// read tracklength
		infile.read((char*)&trkpoints, sizeof(trkpoints));

		//for each point in the track
		out[ii].resize(trkpoints);
		for(int jj = 0 ; jj < trkpoints; jj++) {
			float xyz;
			for(size_t kk=0; kk<3; kk++) {
				// convert space only to index
				infile.read((char*)&xyz, sizeof(float));
				out[ii][jj][kk] = (double)xyz/head.voxel_size[kk];
			}

			// Skip Scalars
			infile.seekg(head.n_scalars*sizeof(float), ios_base::cur);
		}

		// Skip Properties
		infile.seekg(head.n_properties*sizeof(float), ios_base::cur);
	}

	// Transform All Points
	Eigen::Vector3f y;
	for(size_t tt=0; tt<out.size(); tt++) {
		for(size_t pp=0; pp<out[tt].size(); pp++) {
			Eigen::Map<Eigen::Vector3f> x(out[tt][pp].data());
			y = reorient.topLeftCorner<3,3>()*x + reorient.topRightCorner<3,1>();
			for(size_t kk=0; kk<3; kk++)
				out[tt][pp][kk] =  y[kk];
		}
	}

	return out;
}

/**
 * @brief Reads tracks into a vector of tracks, where each track is a vector of
 * float[3].
 *
 * @param filename File storing tracks
 * @param ref Matching image used to construct the tracks (for orientation
 * information). This is mandatory for DFT files, which lack orientation
 * information of their own
 *
 * @return vector<vector<float[3]>> aka TrackSet
 */
TrackSet readTracks(std::string filename, std::string ref)
{
	try{
		return readDFT(filename, ref);
	} catch(magic_error r) { }

	try {
		return readTrk(filename, ref);
	} catch(magic_error r) { }

	throw INVALID_ARGUMENT("Error could not load "+filename+" unknown format");
	return TrackSet();
}

}
