/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file macros.h Useful macros.
 *
 *****************************************************************************/

#ifndef MACROS_H
#define MACROS_H

#include <string>
#include <stdexcept>

#define INVALID_ARGUMENT(EXP) \
std::invalid_argument(__PRETTY_FUNCTION__+std::string(" -> ")+std::string(EXP))

#define RUNTIME_ERROR(EXP) \
std::runtime_error(__PRETTY_FUNCTION__+std::string(" -> ")+std::string(EXP))

#ifdef VERYDEBUG
#define DBG3(EXP) EXP
#else
#define DBG3(EXP)
#endif

#ifdef DEBUG
#define DBG2(EXP) EXP
#else
#define DBG2(EXP)
#endif

#ifndef NDEBUG
#define DBG1(EXP) EXP
#else
#define DBG1(EXP)
#endif

#endif
