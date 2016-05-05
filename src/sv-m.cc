/*
sv-m.cc

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
#include <cstdlib>
#include "svmclass.h"
#include "global.h"

using namespace std;

int main(int argc, char* argv[]) {
	void (*print_func)(const char*) = NULL;
	
	//No SVM processing output
	print_func = &print_null;
	svm_set_print_string_function(print_func);
	
    	//parse command line
    	parse_cmd_line(argc,argv);
    
    	cout << endl;
	cout << "Structural Variant Machine (SV-M)" << endl;
	cout << "-----------------------------------" << endl << endl;
	cout << "Data:\t" << configs.data_file << endl << endl;

	//Init SVM class object
	SVMClass svm(configs.eval_type);

	//If prediction 
	if(configs.predict) {
		svm.start_prediction();		
	} else { //else if new training is requiered
		svm.start_crossvalidation();
	}
	return 0;	
}
