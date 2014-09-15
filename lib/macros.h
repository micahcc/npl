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
 * @file macros.h Useful macros.
 *
 *****************************************************************************/

#ifndef MACROS_H
#define MACROS_H

#include <string>
#include <stdexcept>

#define INVALID_ARGUMENT(EXP) \
std::invalid_argument(__PRETTY_FUNCTION__+std::string(" -> ")+std::string(EXP))

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
