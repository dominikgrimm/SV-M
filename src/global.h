/*
global.h

Author:		Dominik Gerhard Grimm
Year:		2011-2012
Group:		Machine Learning and Computational Biology Group (http://webdav.tuebingen.mpg.de/u/karsten/group/)
Institutes:	Max Planck Institute for Developmental Biology and Max Planck Institute for Intelligent Systems (72076 Tübingen, Germany)

This file is part of SV-M.

SV-M is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SV-M is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SV-M.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef GLOBAL
#define GLOBAL

#include <string>
#include "time.h"
#include "configs.h"

using namespace std;

#define LOG 	""
#define FERROR	"FATAL ERROR"
#define ERROR	"ERROR"
#define INFO	"INFORMATION"
#define WARNING "WARNING"
#define DEBUG	"DEBUG"


//Writing log files
#define logging(A,B,C) {\
	time_t rt; struct tm* ct;\
	time(&rt);\
	ct = localtime(&rt);\
	if(A > 0 && A <= 2/*config.logging*/){\
		cerr << wildcards[A] << "[" << ct->tm_mday << "." << ct->tm_mon+1 << "." << ct->tm_year + 1900\
		     << "," << ct->tm_hour << ":" << ct->tm_min << ":" << ct->tm_sec << "] " << B << " in "\
                     << __FILE__ << " at line " << __LINE__  << ": "\
                     << C << endl;\
	}\
} 

extern char* wildcards[];
extern const configurations& configs;


#endif //GLOBAL
