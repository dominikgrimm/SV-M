/*
StringHelper.h

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

#ifndef StringHelper_CLASS
#define StringHelper_CLASS

#include <string>
#include <vector>
#include <sstream>

using namespace std;

class StringHelper {
	public:
		StringHelper() {};
		~StringHelper() {};

		static vector<string> split(string ,const string&);
		/*
		*@Desciption: string to type <float>, <double>, <int>, <bool>, <long> ...
		*@param: variable from typ T to store the transformed string
		*        the string to transform
		*@return: true if everything was ok, otherwise false
		*/
		template <typename T>
		static bool string_to(T& t, const string& s) {
			istringstream iss(s);
 			return !(iss >> t).fail();
		}
		
		/*
		*@Desciption: string to type <float>, <double>, <int>, <bool>, <long> ...
		*@param:  the string to transform
		*@return: the transformed type
		*/
		template <typename T>
		static T string_to( const string& s) {
			istringstream iss(s);
			T t;
			iss >> t;
 			return t;
		}

		/*
		*@Desciption: type <float>, <double>, <int>, <bool>, <long> to string ...
		*@param: variable from typ T to transform 
		*        the transformed string
		*@return: true if everything was ok, otherwise false
		*/
		template <typename T>
		static bool to_string(string& s, const T& t) {
			stringstream iss;
			bool flag =(iss << t);
			s = iss.str();
 			return flag;
		}
	
		/*
		*@Desciption: type <float>, <double>, <int>, <bool>, <long> to string ...
		*@param: variable from typ T to transform
		*@return: string
		*/
		template <typename T>
		static string to_string(const T& t) {
			stringstream iss;
			iss << t;
			return iss.str();
		}

		/*
		*@Desciption: trim string
		*@param: string to trim
		*/
		static string& trim(string& s) {
			size_t pos;
			while((pos = s.find(' ')) != string::npos) {
				s.erase(pos,1);
			}
			return s;	
		}
		
		/*
		*@Desciption: trim string with a certain pattern
		*@param: string to trim, trim pattern
		*/
		static string& trim(string& s, const string& pattern) {
			size_t pos;
			while((pos = s.find(pattern)) != string::npos) {
				s.erase(pos,1);
			}
			return s;	
		}
		
		/*
		*@Desciption: count number of characters c in s
		*@param: number of characters
		*/
		static unsigned get_num_characters(string& s, const string& c) {
			unsigned num;
			for(unsigned i=0; i<s.length(); i++) {
				if(s[i] == c[0]) num++;
			}
			return num;	
		}
	
		/*
		*@Desciption: string to upper case 
		*@param: string s
		*/
		static string& to_upper(string& s) {
			for(unsigned int i=0; i<s.length();i++) {
				s[i] = toupper(s[i]);
			}
			return s;
		}
		
		/*
		*@Desciption: string to lower case 
		*@param: string s
		*/
		static string& to_lower(string& s) {
			for(unsigned int i=0; i<s.length();i++) {
				s[i] = tolower(s[i]);
			}
			return s;
		}
};
#endif //StringHelper_CLASS
