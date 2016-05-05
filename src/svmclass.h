/*
svmclass.h

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
#ifndef CLASS_SVMCLASS
#define CLASS_SVMCLASS

#define AUC				7
#define BREAK_EVEN_POINT		10

#define MAX_EVAL			(int)1000000000
#define MIN_EVAL			(int)0

#include "global.h"
#include "svm.h"
#include "StringHelper.h"

using namespace std;

typedef struct svm_data_struct {
	double label;
	vector<double> x;
	bool operator==(const svm_data_struct&) const;
} svm_data;

typedef struct svm_eval_struct {
	double auc;
	double tpr;
	double fpr;
	double break_even_point;
	double C;
	int eval_type;
	svm_eval_struct(const int&);
	svm_eval_struct();
	void init(const int&, const int&);
	void init(const int&);
	void set_eval_type(const int&);
	bool operator<(const svm_eval_struct&) const;
	bool operator>(const svm_eval_struct&) const;
} svm_eval;

void print_null(const char *s);
ptrdiff_t shuffle (ptrdiff_t i);

class AUC_Compare {
	private:
		const double *dec_val;
	public:
		AUC_Compare (const double *value);
	
	bool operator()(int i, int j) const;
};

class SVMClass {
	private:
		vector<svm_data> sdv_pos;		//Positive data
		vector<svm_data> sdv_neg;		//Negative data
		vector<svm_data> sdv_prediction;	//Prediction data set
		vector<double> max_data;		//Maximum value of each feature column
		vector<double> min_data;		//Minimum of each feature column
		unsigned short number_features; 	//Number of features
		svm_parameter param;
		svm_model* model;
		int eval_type;
		double* dec_values;
		double* yt;
		int* labels;
		vector<double> tpr;
		vector<double> fpr;
	
		double compute_auc(const double*, const double*, const int&);
		double compute_break_even_point();
		bool file_exists(const string&);

	public:
		SVMClass(const int& e_type);
		~SVMClass() {};
		
		bool load_data(const string&);
		double get_max_for_feature(const int&);
		double get_min_for_feature(const int&);
		unsigned short get_number_of_features();
		unsigned int size_positive_data();
		unsigned int size_negative_data();
		void normalize_data();
		void set_svm_parameters(const svm_parameter&);	
		svm_eval crossvalidation(const int&,const int&,const string&);
		struct svm_model *train_svm(const string&);	
		vector<double> get_weights();
		vector<double> get_weights(const svm_model&, const int&);
		void load_normalization_parameters(const string&);
		void save_normalization_parameters(const string&);
		void load_model(const string&);
		void start_crossvalidation();
		void start_prediction();
		long svm_predict(const string&, const string&);
};

#endif
