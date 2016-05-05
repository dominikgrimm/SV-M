/*
global.cc

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
#include <iostream>
#include <string>
#include "configs.h"

using namespace std;

//Important constant globals
const char* wildcards[] = {"","","\t>","\t>>","\t\t>>>","\t\t>>>>"};

//Configurations
configurations r_config;
const configurations& configs = r_config;

//global important functions
void parse_cmd_line(int& argc, char* argv[]) {
	return r_config.parse_cmd_line(argc,argv);
}
