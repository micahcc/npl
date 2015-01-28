#ifndef __version__
	#if defined(DEBUG)
		#define __version__ "3.0.9-debug"
	#elif defined(NDEBUG)
		#define __version__ "3.0.9-release"
	#else
		#define __version__ "3.0.9"
	#endif
#endif
