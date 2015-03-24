/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 ******************************************************************************/
#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.1.5-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.1.5-release"
	#else
		#define __version__ "3.1.5"
	#endif
#endif
