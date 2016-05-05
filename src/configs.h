/*
configs.h

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
#ifndef CONFIG_FILE
#define CONFIG_FILE

#include <iostream>
#include <string>

using namespace std;

void parse_cmd_line(int& argc, char* argv[]);

typedef struct config_struct {
	string data_file;
	string output_dir;
	string output_file;
	string model_name;
	string norm_filename;
	unsigned nfold;
	unsigned experiments;
	unsigned eval_type;
	bool predict;
	void parse_cmd_line(int& argc,char* argv[]);
} configurations;

#endif
