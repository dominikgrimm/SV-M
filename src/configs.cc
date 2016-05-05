/*
configs.cc

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
#include <cstdlib>
#include "configs.h"
#include "StringHelper.h"
#include "svmclass.h"

using namespace std;

void configurations::parse_cmd_line(int& argc, char* argv[]) {
	//Init with defaults
	data_file = "";
	output_dir = "";
	output_file = "";
	model_name = "";
	norm_filename = "";
	nfold = 10;
	experiments = 1;
	eval_type = AUC;
	predict = false;
	
	//Parse command line arguments
	for(int i=1; i < argc; i++) {
		string cmd = argv[i];
		if(cmd == "-n") {
			if(argv[i+1] != NULL) {
				nfold = StringHelper::string_to<int>(argv[i+1]);
				i++;
			} else {
				cerr << "Argument missing for -n" << endl << endl;
				exit(-1);				 
			}
		}
		if(cmd == "-experiments") {
			if(argv[i+1] != NULL) {
				experiments = StringHelper::string_to<int>(argv[i+1]);
				i++;
			} else {
				cerr << "Argument missing for -experiments [default=1]" << endl << endl;
				exit(-1);				 
			}
		}
		if(cmd == "-train") {
			if(argv[i+2] != NULL) {
				data_file = argv[i+1];
				output_dir = argv[i+2];
				i+=2;
			} else {
				cerr << "Argument missing for -train <data_filename> <output_directory>" << endl << endl;
				exit(-1);
			}
		}
		if(cmd == "-predict") {
			if(argv[i+4] != NULL) {
				model_name = argv[i+1];
				norm_filename = argv[i+2];
				data_file = argv[i+3];
				output_file = argv[i+4];
				predict = true;
				i+=4;
			} else {
				cerr << "Argument missing for -predict <model_name> <normalization_parameters> <data_filename> <output filename>" << endl << endl;
				exit(-1);				 
			}

		}
	}
	//Check command line arguments
	if((argc <= 1) || (predict == false && (data_file == "" || output_dir == ""))) {
		cerr << endl;
		cerr << "Structural Variant Machine (SV-M)" << endl;
		cerr << "-----------------------------------" << endl << endl;
		cerr << "USAGE:\tsv-m\t" << endl << endl;;
		cerr << "training:\t-train <data_filename> <output_directory>" << endl;
		cerr << "\t\t-n [optional: k-fold (default=10)]" << endl;
		cerr << "\t\t-experiments [optional: number of experiments/repeats (default=1)]" << endl << endl;
		cerr << "prediction:\t-predict <model_name> <normalization_parameters> <data_filename> <output filename>" << endl << endl;
		cerr << "***********************************************************************************************************************************" << endl;
		cerr << "Version:\t0.1" << endl;
		cerr << "Date:\t\t01th of September 2012" << endl << endl;
		cerr << "Author:\t\tDominik Gerhard Grimm" << endl;
		cerr << "Mail:\t\tdominik.grimm@tuebingen.mpg.de" << endl;
		cerr << "Group:\t\tMachine Learning and Computational Biology Research Group (http://webdav.tuebingen.mpg.de/u/karsten/group/)" << endl;
		cerr << "Institute:\tMax Planck Institute for Developmental Biology and Max Planck Institute for Intelligent Systems (Tübingen Germany)" << endl << endl;
		cerr << "This tool make use of libSVM 3.0 (www.csie.ntu.edu.tw/~cjlin/libsvm/)" << endl;
		cerr << "***********************************************************************************************************************************" << endl << endl;
		exit(-1); 
	}
}
