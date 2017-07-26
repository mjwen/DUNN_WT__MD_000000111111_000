#ifndef DESCRIPTOR_H_
#define DESCRIPTOR_H_

#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include "helper.h"

#define MY_PI 3.1415926535897932


typedef double (*CutoffFunction)(double r, double rcut);
typedef double (*dCutoffFunction)(double r, double rcut);


class Descriptor
{
  public:
    Descriptor();
		~Descriptor();

		// initialization helper
		void set_cutfunc(char* name);
		void add_descriptor(char* name, double** values, int row, int col);

    int get_num_descriptors();

		// symmetry functions
    double sym_g2(double r, double rcut, double eta, double Rs);
    double sym_d_g2(double r, double rcut, double eta, double Rs);
    double sym_g3(double r, double rcut, double kappa);
    double sym_d_g3(double r, double rcut, double kappa);


//TODO delete; for debug purpose
    void echo_input() {
      std::cout<<"====================================="<<std::endl;
      for (size_t i=0; i<desc_name.size(); i++) {
        int rows = num_param_sets.at(i);
        int cols = num_params.at(i);
        std::cout<<"name: "<<desc_name.at(i)<<", rows: "<<rows<<", cols: "<<cols<<std::endl;
        for (int m=0; m<rows; m++) {
          for (int n=0; n<cols; n++) {
            std::cout<<desc_params.at(i)[m][n]<< " ";
          }
          std::cout<<std::endl;
        }
        std::cout<<std::endl;
      }
    }


	private:
		CutoffFunction cutoff;
		dCutoffFunction d_cutoff;
		std::vector<std::string> desc_name; // name of each descriptor
		std::vector<double**> desc_params;  // params of each descriptor
		std::vector<int> num_param_sets;    // number of parameter sets of each descriptor
		std::vector<int> num_params;        // size of parameters of each descriptor
};


// cutoffs
inline double cut_cos(double r, double rcut) {
	if (r < rcut)
		return 0.5 * (cos(MY_PI*r/rcut) + 1);
	else
		return 0.0;
}

inline double d_cut_cos(double r, double rcut) {
	if (r < rcut)
		return -0.5*MY_PI/rcut * sin(MY_PI*r/rcut);
	else
		return 0.0;
}

inline double cut_exp(double r, double rcut) {
	if (r < rcut)
		return 1;
	else
		return 0.0;
}

inline double d_cut_exp(double r, double rcut) {
	if (r < rcut)
		return 0.0;
	else
		return 0.0;
}



#endif // DESCRIPTOR_H_

