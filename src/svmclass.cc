/*
svmclass.cc

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
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <math.h>
#include <queue>

#include "svmclass.h"

using namespace std;

void print_null(const char *s) {};

ptrdiff_t shuffle (ptrdiff_t i) { 
	return rand()%i;
}

ptrdiff_t (*p_shuffle)(ptrdiff_t) = shuffle;

bool svm_data_struct::operator==(const svm_data_struct& s) const {
	if(label == s.label && x == s.x)
		return true;
	else
		return false;
}

svm_eval::svm_eval_struct(const int& type) {
	eval_type = type;
}

svm_eval::svm_eval_struct() {}

void svm_eval::init(const int& i, const int& type) {
	eval_type = type;
	auc = i;
	break_even_point = i;
	C = i;
	tpr = i;
	fpr = i;
}

void svm_eval::init(const int& i) {
	init(i,eval_type);
}

void svm_eval::set_eval_type(const int& type) {
	eval_type = type;
}

bool svm_eval::operator<(const svm_eval_struct& s) const {
	if(eval_type != s.eval_type)
		throw "Evaluation types of structs are not equal";
	if(eval_type == AUC) 
		return (auc < s.auc);
	if(eval_type == BREAK_EVEN_POINT) 
		return (break_even_point < s.break_even_point);

	return false;
}

bool svm_eval::operator>(const svm_eval_struct& s) const {
	if(eval_type != s.eval_type)
		throw "Evaluation types of structs are not equal";
	if(eval_type == AUC) 
		return (auc > s.auc);
	if(eval_type == BREAK_EVEN_POINT) 
		return (break_even_point > s.break_even_point);
	return false;
}

AUC_Compare::AUC_Compare(const double *value){
	dec_val = value;
}

bool AUC_Compare::operator()(int i, int j) const {
	return dec_val[i] > dec_val[j];
}

SVMClass::SVMClass(const int& e_type) {
	eval_type = e_type;
}

void SVMClass::start_crossvalidation() {
	//Read input training data from file
	cout << "Reading data from file...\t\t";
	load_data(configs.data_file);
	cout << "[OK]" << endl;
	
	//Normalize input data between 0 and 1
	cout << "Normalizing input data [0,1]...\t\t";
	normalize_data();
	cout << "[OK]" << endl << endl;

	cout << "#Positive training data:\t\t" << size_positive_data() << endl;
	cout << "#Negative training data:\t\t" << size_negative_data() << endl << endl;
	
	//Init svm parameters
	svm_parameter param;
	param.gamma =  1.0/get_number_of_features();
	param.svm_type = C_SVC;
	param.kernel_type = 0;
	param.coef0 = 0;
	param.C = 1;
	param.degree=0;
	param.shrinking = 1;
	param.nu = 0.5;
	param.cache_size = 100;
	param.eps = 1e-3;
	param.p = 0.1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	set_svm_parameters(param);
	//Start crossvalidation
	svm_eval max_eval(configs.eval_type);
	max_eval = crossvalidation(configs.nfold,configs.experiments,configs.output_dir);
	
	//Train linear model on the whole dataset with best performing C
	param.C = max_eval.C;
	//Train model with probability estimates
	param.probability = 1;
	set_svm_parameters(param);

	svm_model* model;
	model = train_svm(configs.output_dir);
	save_normalization_parameters(configs.output_dir + "/model_normalization.param");

	//Compute weights of the linear svm model
	vector<double> weights;
	weights = get_weights(*model,get_number_of_features());

	//Print Model Parameters and weights
	cout << endl;
	cout << "Offset b:\t\t\t" << -model->rho[0] << endl;
	cout << "C value:\t\t\t" << param.C << endl;
	cout << "Decision function:\t\ty = w'*x + b" << endl << endl;
	
	for(int i=0; i<get_number_of_features();i++) {
		cout << "Weight for feature " << i+1 << ":\t\t" << weights[i] << endl;
	}

	//Append results to result file
	ofstream result_out;
	string res_out_file = configs.output_dir + "/results.txt";
	result_out.open(res_out_file.c_str(),ofstream::app);
	result_out << endl;
	if(!result_out.is_open()) {
		logging(1,ERROR,"Can not open file " + res_out_file);
	}	
	result_out << "Offset b:\t\t\t" << -model->rho[0] << endl;
	result_out << "C value:\t\t\t" << param.C << endl;
	result_out << "Decision function:\t\ty = w'*x + b" << endl << endl;
	
	for(int i=0; i<get_number_of_features();i++) {
		result_out << "Weight for feature " << i+1 << ":\t\t" << weights[i] << endl;
	}

	result_out.close();
	cout << endl;
	cout << "Output files written to the directory: " << configs.output_dir << endl << endl;
}

void SVMClass::start_prediction() {
	//load model relevant files
	cout << "Loading model files...\t\t";
	load_normalization_parameters(configs.norm_filename);
	load_model(configs.model_name);
	cout << "[OK]" << endl;
	//Predict on data
	cout << "Predicting on data...\t\t";
	long predictions = svm_predict(configs.data_file,configs.output_file);
	cout << "[OK]" << endl << endl;
	cout << "Predicted " << predictions << " indel candidates." << endl;
	cout << "Output written to " << configs.output_file << endl << endl;
}

void SVMClass::set_svm_parameters(const svm_parameter& p) {
	param = p;
}

double SVMClass::compute_auc(const double* dec_v, const double* y, const int& n) {
	double auc  = 0;
	vector<int> indices(n);
	
	for(int i = 0; i < n; i++) {
		indices[i] = i;
	} 
	
	sort(indices.begin(), indices.end(), AUC_Compare(&dec_v[0]));
		
	double tp = 0;
	double fp = 0;
	queue<double> tp_q;
	queue<double> fp_q;
	tpr.clear();
	fpr.clear();
	
	for(int i = 0; i < n; i++) {
		if(y[indices[i]] == 1) {
			tp++;
			tp_q.push(tp);
			fp_q.push((fp_q.size()> 0)?fp_q.back():0);
		} else if(y[indices[i]] == -1) {
			auc += tp;
			fp++;
			tp_q.push((tp_q.size()> 0)?tp_q.back():0);
			fp_q.push(fp);
		}
	}	
	
	if(tp == 0 || fp == 0)
	{
		logging(1,WARNING,"Error: Too few negative or positive examples\n");
		auc = 0;
	} else {
		auc = auc / tp / fp;
		for(int i=0; i<n;i++) {
			tpr.push_back(tp_q.front()/tp);
			fpr.push_back(fp_q.front()/fp);
			tp_q.pop();
			fp_q.pop();
		}
	}
	return auc;
}

double SVMClass::compute_break_even_point() {
	vector<double> diffs;
	if (tpr.size() != fpr.size()) {
		logging(1,ERROR,"TPR size != FPR size\n");
		return -1.0;
	}
	//Compute diffs
	for(size_t i=0; i<tpr.size();i++) {
		diffs.push_back(fabs(tpr[i]-(1-fpr[i])));
	}
	//find index of minimum
	int index = 0;
	double tmp = MAX_EVAL;
	for(size_t i=0; i<diffs.size();i++) {
		if (diffs[i]<tmp) {
			index = i;
			tmp = diffs[i];
		} 
	}
	//Compute Break Even Point
	double fpr_tmp = 1-fpr[index];

	double tpr_tmp = tpr[index];
	
	return (tpr_tmp+fpr_tmp)/2.0;
}

bool SVMClass::load_data(const string& filename) {
	string line;
	ifstream ifs;
	//open inputstream
	ifs.open(filename.c_str(),ifstream::in);
	if(!ifs.is_open()) {
		logging(1,ERROR,"Could not open " + filename + " ... \n");
		return false;
	}
	while(!ifs.eof()) {
		getline(ifs,line);
		if(line != "") {
			vector<string> vs = StringHelper::split(line,"\t");
			max_data.resize(vs.size()-1,MIN_EVAL);
			min_data.resize(vs.size()-1,MAX_EVAL);
			number_features = vs.size()-1;
			if (number_features==0) {
				logging(1,ERROR,"Wrong input format...\n");
				return false;
			}
			svm_data sd;
			sd.label = StringHelper::string_to<double>(vs[0]);
			vector<double> x;
			for(unsigned int i=1; i<vs.size(); i++) {
				x.push_back(StringHelper::string_to<double>(vs[i])); 
				if(max_data[i-1] < StringHelper::string_to<double>(vs[i])) {
					max_data[i-1] = StringHelper::string_to<double>(vs[i]);
				}	
				if(min_data[i-1] > StringHelper::string_to<double>(vs[i])) {
					min_data[i-1] = StringHelper::string_to<double>(vs[i]);
				}
			}
			sd.x = x;
			if(sd.label == 1) {
				sdv_pos.push_back(sd);
			} else {
				sdv_neg.push_back(sd);
			}
		}
	}
	//close inputstream
	ifs.close();
	return true;	
}

unsigned int SVMClass::size_positive_data() {
	return sdv_pos.size();
}

unsigned int SVMClass::size_negative_data() {
	return sdv_neg.size();
}

double SVMClass::get_max_for_feature(const int& feature) {
	return max_data[feature];
}

double SVMClass::get_min_for_feature(const int& feature) {
	return min_data[feature];
}

unsigned short SVMClass::get_number_of_features() {
	return number_features;
}

void SVMClass::normalize_data() {
	for(unsigned i=0; i<sdv_pos.size(); i++) {
		for(unsigned j=0; j<sdv_pos[i].x.size();j++) {
			if(max_data[j]-min_data[j] != 0) 
				sdv_pos[i].x[j] = (sdv_pos[i].x[j]-min_data[j]) * (1.0/(max_data[j]-min_data[j]));
			else 
				sdv_pos[i].x[j] = 0;
		}
	}
	for(unsigned i=0; i<sdv_neg.size(); i++) {
		for(unsigned j=0; j<sdv_neg[i].x.size();j++) {
			if(max_data[j]-min_data[j] != 0) 
				sdv_neg[i].x[j] = (sdv_neg[i].x[j]-min_data[j]) * (1.0/(max_data[j]-min_data[j]));
			else
				sdv_neg[i].x[j] = 0;
		}	
	}
}

bool SVMClass::file_exists(const string& filename) {
	ifstream ifs(filename.c_str());
	return ifs;
}

svm_eval SVMClass::crossvalidation(const int& kfold, const int& experiments, const string& output_dir) {
	
	ofstream exp_out;
	string exp_out_file = output_dir + "/experiments.tab";
	if(file_exists(exp_out_file)) {
		logging(1,ERROR,"Output directory already exists! Please create a new one...\n");
		exit(-1);
	}
	exp_out.open(exp_out_file.c_str(),ofstream::out);
	if(!exp_out.is_open()) {
		logging(1,ERROR,"Can not open file " + exp_out_file );
	}	
	
	vector<svm_eval> eval_set;
	vector<svm_eval> eval_result;
	int ind = 0;

	srand(unsigned(time(NULL)));
	
	for(int ex=1; ex<=experiments;ex++) {
		cout << "Starting experiment: " << ex << endl; 
		vector<vector<svm_data> > pos_folds;
		vector<vector<svm_data> > neg_folds;
		
		//Shuffle data and split into k folds
		random_shuffle(sdv_pos.begin(), sdv_pos.end(), p_shuffle);
		random_shuffle(sdv_neg.begin(), sdv_neg.end(), p_shuffle);
		for(int i=0; i<kfold;i++) {
			vector<svm_data> d;
			pos_folds.push_back(d);
		}	
		for(int i=0; i<kfold;i++) {
			vector<svm_data> d;
			neg_folds.push_back(d);
		}	
		ind = 0;
		for(unsigned i=0; i<sdv_pos.size(); i++) {
			pos_folds[ind].push_back(sdv_pos[i]);
			ind = (ind<kfold-1)?ind+1:0;
		}
		ind = 0;
		for(unsigned i=0; i<sdv_neg.size(); i++) {
			neg_folds[ind].push_back(sdv_neg[i]);
			ind = (ind<kfold-1)?ind+1:0;
		}

		vector<svm_eval> eval_vector;
		int counter = 0;
		//For each fold do ...
		for(int i=0; i<kfold; i++) {
			vector<svm_data> pos_train_set;
			vector<svm_data> neg_train_set;	
			vector<svm_data> pos_test_set;
			vector<svm_data> neg_test_set;
			for(int j=0; j<kfold; j++) {
				if(i!=j) {
					for(unsigned k=0; k<pos_folds[j].size();k++) {
						svm_data s;
						s.label = pos_folds[j][k].label;
						s.x = pos_folds[j][k].x;
						pos_train_set.push_back(s);
					}
				}
			}	
			for(unsigned j=0; j<pos_folds[i].size();j++) {
				svm_data s;
				s.label = pos_folds[i][j].label;
				s.x = pos_folds[i][j].x;
				pos_test_set.push_back(s);
			}
			for(int j=0; j<kfold; j++) {
				if(i!=j) {
					for(unsigned k=0; k<neg_folds[j].size();k++) {
						svm_data s;
						s.label = neg_folds[j][k].label;
						s.x = neg_folds[j][k].x;
						neg_train_set.push_back(s);
					}
				}
			}
			for(unsigned j=0; j<neg_folds[i].size();j++) {
				svm_data s;
				s.label = neg_folds[i][j].label;
				s.x = neg_folds[i][j].x;
				neg_test_set.push_back(s);
			}

			//Copy training and testing data into svm data structure
			svm_problem prob_train;
			svm_problem prob_test;
			prob_train.l = pos_train_set.size()+neg_train_set.size();
			prob_train.y =  (double*)malloc((prob_train.l)*sizeof(double));
			prob_train.x = (struct svm_node**)malloc((prob_train.l)*sizeof(struct svm_node*));
			struct svm_node *node = (struct svm_node*)malloc((prob_train.l*pos_train_set[0].x.size()+prob_train.l)
						 *sizeof(struct svm_node));
			prob_test.l = neg_test_set.size()+pos_test_set.size();
			prob_test.y =  (double*)malloc((prob_test.l)*sizeof(double));
			prob_test.x = (struct svm_node**)malloc((prob_test.l)*sizeof(struct svm_node*));
			struct svm_node *node1 = (struct svm_node*)malloc((prob_test.l*pos_train_set[0].x.size()+prob_test.l)
						  *sizeof(struct svm_node));
			unsigned e=0;
			//add positive training examples
			for(unsigned l=0; l<pos_train_set.size(); l++) {
				prob_train.y[l]=pos_train_set[l].label;
				prob_train.x[l]=&node[e];
				for(unsigned k=0; k<pos_train_set[l].x.size(); k++) {
					node[e].index = (int)k+1;
					node[e].value = (double)pos_train_set[l].x[k];
					e++;
				}
				node[e++].index = -1;
			}
			//add negative training examples
			int index = pos_train_set.size();
			for(unsigned l=0; l<neg_train_set.size(); l++) {
				prob_train.y[index+l]=neg_train_set[l].label;
				prob_train.x[index+l]=&node[e];
				for(unsigned k=0; k<neg_train_set[l].x.size(); k++) {
					node[e].index = (int)k+1;
					node[e].value = (double)neg_train_set[l].x[k];
					e++;
				}
				node[e++].index = -1;
			}
			e = 0;
			//positive testing examples	
			for(unsigned l=0; l<pos_test_set.size(); l++) {
				prob_test.y[l]=pos_test_set[l].label;
				prob_test.x[l]=&node1[e];
				for(unsigned k=0; k<pos_test_set[l].x.size(); k++) {
					node1[e].index = (int)k+1;
					node1[e].value = (double)pos_test_set[l].x[k];
					e++;
				}
				node1[e++].index = -1;
			}
			index = pos_test_set.size();
			//add negative testing examples
			for(unsigned l=0; l<neg_test_set.size(); l++) {
				prob_test.y[index+l]=neg_test_set[l].label;
				prob_test.x[index+l]=&node1[e];
				for(unsigned k=0; k<neg_test_set[l].x.size(); k++) {
					node1[e].index = (int)k+1;
					node1[e].value = (double)neg_test_set[l].x[k];
					e++;
				}
				node1[e++].index = -1;
			}	
			
			//Inner Crossvalidation to determine C
			vector<vector<svm_data> > pos_folds_C;
			vector<vector<svm_data> > neg_folds_C;
			random_shuffle(pos_train_set.begin(), pos_train_set.end(), p_shuffle);
			random_shuffle(neg_train_set.begin(), neg_train_set.end(), p_shuffle);
			
			for(int j=0; j<kfold;j++) {
				vector<svm_data> d;
				pos_folds_C.push_back(d);
			}	
			for(int j=0; j<kfold;j++) {
				vector<svm_data> d;
				neg_folds_C.push_back(d);
			}	
			ind = 0;
			for(unsigned j=0; j<pos_train_set.size(); j++) {
				pos_folds_C[ind].push_back(pos_train_set[j]);
				ind = (ind<kfold-1)?ind+1:0;
			}
			ind = 0;
			for(unsigned j=0; j<neg_train_set.size(); j++) {
				neg_folds_C[ind].push_back(neg_train_set[j]);
				ind = (ind<kfold-1)?ind+1:0;
			}
			vector<svm_data> pos_train_set_C;
			vector<svm_data> neg_train_set_C;	
			vector<svm_data> pos_test_set_C;
			vector<svm_data> neg_test_set_C;
			//Prepare data...
			for(int j=0; j<kfold; j++) {
				if(i!=j) {
					for(unsigned k=0; k<pos_folds_C[j].size();k++) {
						svm_data s;
						s.label = pos_folds_C[j][k].label;
						s.x = pos_folds_C[j][k].x;
						pos_train_set_C.push_back(s);
					}
				}
			}	
			for(unsigned j=0; j<pos_folds_C[i].size();j++) {
				svm_data s;
				s.label = pos_folds_C[i][j].label;
				s.x = pos_folds_C[i][j].x;
				pos_test_set_C.push_back(s);
			}
			for(int j=0; j<kfold; j++) {
				if(i!=j) {
					for(unsigned k=0; k<neg_folds_C[j].size();k++) {
						svm_data s;
						s.label = neg_folds_C[j][k].label;
						s.x = neg_folds_C[j][k].x;
						neg_train_set_C.push_back(s);
					}
				}
			}
			for(unsigned j=0; j<neg_folds_C[i].size();j++) {
				svm_data s;
				s.label = neg_folds_C[i][j].label;
				s.x = neg_folds_C[i][j].x;
				neg_test_set_C.push_back(s);
			}
			//Copy training and testing examples to internal svm struct
			svm_problem prob_train_C;
			svm_problem prob_test_C;
			prob_train_C.l = pos_train_set_C.size()+neg_train_set_C.size();
			prob_train_C.y =  (double*)malloc((prob_train_C.l)*sizeof(double));
			prob_train_C.x = (struct svm_node**)malloc((prob_train_C.l)*sizeof(struct svm_node*));
			struct svm_node *node2 = (struct svm_node*)malloc((prob_train_C.l*pos_train_set[0].x.size()+prob_train_C.l)
						  *sizeof(struct svm_node));
			prob_test_C.l = pos_test_set_C.size()+neg_test_set_C.size();
			prob_test_C.y =  (double*)malloc((prob_test_C.l)*sizeof(double));
			prob_test_C.x = (struct svm_node**)malloc((prob_test_C.l)*sizeof(struct svm_node*));
			struct svm_node *node3 = (struct svm_node*)malloc((prob_test_C.l*pos_train_set[0].x.size()+prob_test_C.l)
						  *sizeof(struct svm_node));
			e=0;
			//add positive training examples to inner crossvalidation
			for(unsigned l=0; l<pos_train_set_C.size(); l++) {
				prob_train_C.y[l]=pos_train_set_C[l].label;
				prob_train_C.x[l]=&node2[e];
				for(unsigned k=0; k<pos_train_set_C[l].x.size(); k++) {
					node2[e].index = (int)k+1;
					node2[e].value = (double)pos_train_set_C[l].x[k];
					e++;
				}
				node2[e++].index = -1;
			}
			//add negative training examples to inner crossvalidation
			index = pos_train_set_C.size();
			for(unsigned l=0; l<neg_train_set_C.size(); l++) {
				prob_train_C.y[index+l]=neg_train_set_C[l].label;
				prob_train_C.x[index+l]=&node2[e];
				for(unsigned k=0; k<neg_train_set_C[l].x.size(); k++) {
					node2[e].index = (int)k+1;
					node2[e].value = (double)neg_train_set_C[l].x[k];
					e++;
				}
				node2[e++].index = -1;
			}
			e = 0;
			//positive testing examples for inner crossvalidation	
			for(unsigned l=0; l<pos_test_set_C.size(); l++) {
				prob_test_C.y[l]=pos_test_set_C[l].label;
				prob_test_C.x[l]=&node3[e];
				for(unsigned k=0; k<pos_test_set_C[l].x.size(); k++) {
					node3[e].index = (int)k+1;
					node3[e].value = (double)pos_test_set_C[l].x[k];
					e++;
				}
				node3[e++].index = -1;
			}
			index = pos_test_set_C.size();
			//negative testing examples for inner crossvaldiation
			for(unsigned l=0; l<neg_test_set_C.size(); l++) {
				prob_test_C.y[index+l]=neg_test_set_C[l].label;
				prob_test_C.x[index+l]=&node3[e];
				for(unsigned k=0; k<neg_test_set_C[l].x.size(); k++) {
					node3[e].index = (int)k+1;
					node3[e].value = (double)neg_test_set_C[l].x[k];
					e++;
				}
				node3[e++].index = -1;
			}
			
			map<int, svm_eval> eval_map;
			map<int, svm_eval>::iterator eval_it;
			double c_values[] = {0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000};			
			for(int C=0;C<11;C++){					
				dec_values = (double*)malloc((prob_test_C.l)*sizeof(double));
				yt = (double*)malloc((prob_test_C.l)*sizeof(double));
				labels = (int*)malloc((prob_test_C.l)*sizeof(int));
				param.C = c_values[C];
				svm_model* submodel = svm_train(&prob_train_C,&param);				
				unsigned correct_cl = 0;
				double tp = 0;
				double fp = 0;
				double tn = 0;
				double fn = 0;
				for(int l=0; l<prob_test_C.l;l++) {
					double d_class = svm_predict_values(submodel, prob_test_C.x[l],&dec_values[l]);
					if (prob_test_C.y[l] > 0) {
						yt[l] = 1;
					} else {
						yt[l] = -1;
					}
					if(d_class == prob_test_C.y[l])	{
						correct_cl++;
					}
					if(d_class == prob_test_C.y[l] && d_class == -1) {
						tn++;
					}
					if(d_class == prob_test_C.y[l] && d_class == 1) {
						tp++;
					}
					if(d_class != prob_test_C.y[l] && d_class == -1) {
						fn++;
					}
					if(d_class != prob_test_C.y[l] && d_class == 1) {
						fp++;
					}
				}
				svm_get_labels(submodel, labels);
				//For AUC
				if(labels[0] <= 0) {
					for(int j=0; j<prob_test.l;j++)
						dec_values[j] *= -1;
				}
				svm_eval ev(eval_type);
				ev.auc = 100.0*compute_auc(dec_values,yt,prob_test_C.l);
				ev.break_even_point = (ev.auc==0.0)?0.0:100.0*compute_break_even_point();
				ev.C = param.C;
				eval_map[param.C] = ev;
				
				svm_free_and_destroy_model(&submodel);
									
				free(labels);
				free(dec_values);
				free(yt);
			}
			//Train SVM for the C with the highest evaluation criterion and test performance
			svm_eval max_eval(eval_type);
			max_eval.init(MIN_EVAL);
			for(eval_it = eval_map.begin(); eval_it != eval_map.end(); eval_it++) {
				if(max_eval < eval_it->second) {
					max_eval = eval_it->second;
				}
			}
			param.C = max_eval.C;
			svm_model *model = svm_train(&prob_train,&param);
	
			double tp = 0;
			double fp = 0;
			double tn = 0;
			double fn = 0;
			double correct_cl = 0;
			
			dec_values = (double*)malloc((prob_test.l)*sizeof(double));
			yt = (double*)malloc((prob_test.l)*sizeof(double));
			labels = (int*)malloc((prob_test.l)*sizeof(int));
			
			for(int l=0; l<prob_test.l;l++) {
				double d_class = svm_predict_values(model, prob_test.x[l],&dec_values[l]);
				if (prob_test.y[l] > 0) {
					yt[l] = 1;
				} else {
					yt[l] = -1;
				}
				if(d_class == prob_test.y[l])	{
					correct_cl++;
				}
				if(d_class == prob_test.y[l] && d_class == -1) {
					tn++;
				}
				if(d_class == prob_test.y[l] && d_class == 1) {
					tp++;
				}
				if(d_class != prob_test.y[l] && d_class == -1) {
					fn++;
				}
				if(d_class != prob_test.y[l] && d_class == 1) {
					fp++;
				}
				counter++;
			}
			svm_get_labels(model, labels);
			//For AUC
			if(labels[0] <= 0) {
				for(int j=0; j<prob_test.l;j++)
					dec_values[j] *= -1;
			}
			svm_eval ev(eval_type);
			ev.auc = 100.0*compute_auc(dec_values,yt,prob_test.l);
			ev.break_even_point = (ev.auc==0.0)?0.0:100.0*compute_break_even_point();
			ev.C = max_eval.C;
			eval_vector.push_back(ev);	
			
			//Free memory
			svm_free_and_destroy_model(&model);
			free(labels);
			free(dec_values);
			free(yt);
			free(node2);
			free(prob_train_C.y);
			free(prob_train_C.x);
			free(node3);
			free(prob_test_C.y);
			free(prob_test_C.x);
			free(node);
			free(prob_train.y);
			free(prob_train.x);
			free(node1);
			free(prob_test.y);
			free(prob_test.x);
		}
		
		svm_eval eval_av(eval_type);
		eval_av.init(MIN_EVAL);
		svm_eval max_eval(eval_type);
		
		svm_eval tmp_res(eval_type);
		
		max_eval.init(MIN_EVAL);
		for(unsigned i=0; i<eval_vector.size(); i++) {
			if(max_eval < eval_vector[i]) 
				max_eval = eval_vector[i];
			eval_av.auc += eval_vector[i].auc;
			eval_av.break_even_point += eval_vector[i].break_even_point;
			eval_av.C += eval_vector[i].C;
		}
		tmp_res.C =  eval_av.C/(double)kfold;
		tmp_res.auc = eval_av.auc/(double)kfold;
		tmp_res.break_even_point = eval_av.break_even_point/(double)kfold;
		
		eval_set.push_back(max_eval);
		eval_result.push_back(tmp_res);
		
		exp_out << eval_av.C/(double)kfold << "\t" << tmp_res.auc << "\t" << tmp_res.break_even_point << endl;
	}
	
	svm_eval eval_av(eval_type);
	eval_av.init(MIN_EVAL);
	
	svm_eval max_eval(eval_type);
	max_eval.init(MIN_EVAL);
	for(unsigned i=0; i<eval_set.size(); i++) {
		if(max_eval < eval_set[i]) 
			max_eval = eval_set[i];
	}
	
	for(unsigned i=0; i<eval_result.size(); i++) {
		eval_av.auc += eval_result[i].auc;
		eval_av.break_even_point += eval_result[i].break_even_point;
		eval_av.C += eval_result[i].C;
	}
	
	max_eval.auc = eval_av.auc/(double)experiments;
	max_eval.break_even_point = eval_av.break_even_point/(double)experiments;
	
	cout << "Number of Experiments:\t" << experiments << endl;
	cout << "Max C-Value:\t\t" << max_eval.C << endl << endl;
	cout << "************************************************************" << endl;
	cout << "*     AVERAGE-VALUES over all k-folds and experiments      *" << endl;
	cout << "************************************************************" << endl;
	cout << "k-fold cross-validation:\t" << kfold << endl;
	cout << "Average C:\t\t\t" << eval_av.C/(double)experiments << endl;
	cout << "Average AUC:\t\t\t" << eval_av.auc/(double)experiments << endl;
	cout << "Average Break-Even-Point:\t" << eval_av.break_even_point/(double)experiments << endl;
	
	ofstream result_out;
	string res_out_file = output_dir + "/results.txt";
	result_out.open(res_out_file.c_str(),ofstream::out);
	if(!result_out.is_open()) {
		logging(1,ERROR,"Can not open file " + res_out_file);
	}	

	result_out << "Number of Experiments:\t" << experiments << endl;
	result_out << "Max C-Value:\t\t" << max_eval.C << endl << endl;
	result_out << "************************************************************" << endl;
	result_out << "*     AVERAGE-VALUES over all k-folds and experiments      *" << endl;
	result_out << "************************************************************" << endl;
	result_out << "k-fold cross-validation:\t" << kfold << endl;
	result_out << "Average C:\t\t\t" << eval_av.C/(double)experiments << endl;
	result_out << "Average AUC:\t\t\t" << eval_av.auc/(double)experiments << endl;
	result_out << "Average Break-Even-Point:\t" << eval_av.break_even_point/(double)experiments << endl;

	result_out.close();
	exp_out.close();
	return max_eval;
}

struct svm_model *SVMClass::train_svm(const string& output_dir) {
	vector<svm_data> svm_train_set;

	for(unsigned i=0; i<sdv_pos.size();i++) {
		svm_train_set.push_back(sdv_pos[i]);
	}
	for(unsigned i=0; i<sdv_neg.size();i++) {
		svm_train_set.push_back(sdv_neg[i]);
	}
	
	svm_problem prob_train;
	prob_train.l = svm_train_set.size();
	prob_train.y =  (double*)malloc((prob_train.l)*sizeof(double));
	prob_train.x = (struct svm_node**)malloc((prob_train.l)*sizeof(struct svm_node*));
	struct svm_node *node = (struct svm_node*)malloc((prob_train.l*svm_train_set[0].x.size()+prob_train.l)
				 *sizeof(struct svm_node));
	
	unsigned e=0;
	for(unsigned l=0; l<svm_train_set.size(); l++) {
		prob_train.y[l]=svm_train_set[l].label;
		prob_train.x[l]=&node[e];
		for(unsigned k=0; k<svm_train_set[l].x.size(); k++) {
			node[e].index = (int)k+1;
			node[e].value = (double)svm_train_set[l].x[k];
			e++;
		}
		node[e++].index = -1;
	}
	model = svm_train(&prob_train,&param);
	
	string output_name = output_dir + "/model.svm";
	svm_save_model(output_name.c_str(), model);
	free(node);
	free(prob_train.y);
	free(prob_train.x);

	return model;
}

vector<double> SVMClass::get_weights() {
	vector<double> weights(number_features,0);
	for(unsigned p=0; p<number_features;p++) {
		double w = 0;
		for(int t=0; t<model->l; t++) {
			w+= model->SV[t][p].value * model->sv_coef[0][t];
		}
		weights[p] = w;
	}
	return weights;
}

vector<double> SVMClass::get_weights(const svm_model& m, const int& num_features) {
	vector<double> weights(num_features,0);
	for(int p=0; p<num_features;p++) {
		double w = 0;
		for(int t=0; t<m.l; t++) {
			w+= m.SV[t][p].value * m.sv_coef[0][t];
		}
		weights[p] = w;
	}
	return weights;
}

void SVMClass::load_normalization_parameters(const string& filename) {
	ifstream ifs;
	ifs.open(filename.c_str(),iostream::in);
	if(!ifs.is_open()) {
		logging(1,ERROR, "Opening normalization parameter file " + filename + "\n\n");
		exit(-1);
	}
	string line;
	int counter = 0;
	while(!ifs.eof()) {
		getline(ifs, line);
		vector<string> sv = StringHelper::split(line,"\t");
		if(line!="") {
			if(counter==0) {
				for(unsigned i=0; i<sv.size(); i++) {
					min_data.push_back(StringHelper::string_to<double>(sv[i]));
				}
			} else if(counter == 1) {
				for(unsigned i=0; i<sv.size(); i++) {
					max_data.push_back(StringHelper::string_to<double>(sv[i]));
				}
			} else {
				logging(1,WARNING, "Wrong normalization parameter file format " + filename + " (Check format...)\n");
				exit(-1);
			}
			counter++;
		}	
	}
	ifs.close();
}

void SVMClass::save_normalization_parameters(const string& filename) {
	ofstream ofs;
	ofs.open(filename.c_str(),iostream::out);
	if(!ofs.is_open()) {
		logging(1,ERROR,"Could not open normalization output file to write " + filename + "\n");
		exit(-1);
	}
	if(min_data.size() >= 1 && max_data.size() >= 1) {
		ofs << min_data[0];
		for(unsigned i=1; i<min_data.size(); i++) {
			ofs << "\t" << min_data[i];
		}	
		ofs << endl;
		ofs << max_data[0];
		for(unsigned i=1; i<max_data.size(); i++) {
			ofs << "\t" << max_data[i];
		}
	} else {
		logging(1,WARNING,"No normalization parameters set\n");
	}
	ofs.close();
}

void SVMClass::load_model(const string& filename) {
	if((model=svm_load_model(filename.c_str())) == 0) {
		logging(1,ERROR, "Can't open model file " + filename + "\n");
		exit(-1);
	}
}

long SVMClass::svm_predict(const string& prediction_file, const string& output) {
	string line;
	ifstream ifs;
	//open inputstream
	ifs.open(prediction_file.c_str(),ifstream::in);
	if(!ifs.is_open()) {
		logging(1,ERROR,"Could not open " + prediction_file + " ... \n");
		exit(-1);
	}
	//Open outputstream
	ofstream ofs;
	ofs.open(output.c_str(),iostream::out);
	if(!ofs.is_open()) {
		logging(1,ERROR,"Could not open output file to write " + output + "\n");
		exit(-1);
	}

	//Use probablility estimates
	double* prob_estimates = (double *) malloc(2*sizeof(double));
	
	//Classifiy each indel
	long indel_counter=0;
	while(!ifs.eof()) {
		getline(ifs,line);
		if(line != "") {
			vector<string> vs = StringHelper::split(line,"\t");
		
			svm_problem prob;
			prob.l = 1;
			prob.y =  (double*)malloc((prob.l)*sizeof(double));
			prob.x = (struct svm_node**)malloc((prob.l)*sizeof(struct svm_node*));
			struct svm_node *node = (struct svm_node*)malloc((prob.l*vs.size()-3+prob.l)
					*sizeof(struct svm_node));

			unsigned e=0;
			prob.x[0]=&node[e];
			for(unsigned i=0; i<vs.size()-3; i++) {
				node[e].index = (int)i+1;
				if(max_data[i]-min_data[i] != 0) {
					node[e].value =  ((double)StringHelper::string_to<double>(vs[i+3])-min_data[i])
						 * (1.0/(max_data[i]-min_data[i]));
				} else {
					node[e].value = 0;	
				}
				e++;
			}
			node[e++].index = -1;
			
			//Predict
			double d_class = svm_predict_probability(model,prob.x[0],prob_estimates);
			ofs << d_class << "\t" << prob_estimates[0] << "\t";
			ofs << line << endl;
	
			//free memory
			free(prob.y);
			free(prob.x);
			free(node);
			indel_counter++;
		}
	}
	//close streams
	ifs.close();
	ofs.close();
	//free model
	svm_free_and_destroy_model(&model);

	return indel_counter;	
}
