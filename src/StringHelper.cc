/*
StringHelper.cc

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

//StringHelper class contains several methods to deal with string
#include "StringHelper.h"

using namespace std;

/*
*@Description: Split String by delimiter
*@param: String to split, Delimiter string
*@return: vector<string> with all substrings
*/
vector<string> StringHelper::split(string str,const string& delimiter) {
	vector<string> splitV;
	int l = 0;
	while(l!=-1) {
		l = str.find_first_of(delimiter,0);
		splitV.push_back(str.substr(0,l));
		str = str.substr(l+1,str.length());
	}
	return splitV;
}
