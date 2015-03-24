/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * tracks.h Define track class
 *****************************************************************************/
#ifndef TRACKS_H
#define TRACKS_H

#include <iostream>
#include <vector>
#include <string>
#include <array>

namespace npl {

typedef std::vector<std::vector<std::array<float,3>>> TrackSet;

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
TrackSet readTracks(std::string filename, std::string ref = "");

/**
 * @brief Reads a BrainSuite DFT file. Throws INVALID_ARGUMENT if the magic is
 * wrong.
 *
 * @param tfile trk file
 * @param ref Reference image
 *
 * @return vector of vector of points
 */
TrackSet readDFT(std::string tfile, std::string ref = "");

/**
 * @brief Reads a trackvis trk file. Throws INVALID_ARGUMENT if the magic is
 * wrong.
 *
 * @param filename trk file
 * @param ref Reference image (in case the RAS is not valid)
 *
 * @return
 */
TrackSet readTrk(std::string filename, std::string ref = "");

}

#endif //TRACKS_H
