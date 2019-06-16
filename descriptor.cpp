//
// CDDL HEADER START
//
// The contents of this file are subject to the terms of the Common Development
// and Distribution License Version 1.0 (the "License").
//
// You can obtain a copy of the license at
// http://www.opensource.org/licenses/CDDL-1.0.  See the License for the
// specific language governing permissions and limitations under the License.
//
// When distributing Covered Code, include this CDDL HEADER in each file and
// include the License file in a prominent location with the name LICENSE.CDDL.
// If applicable, add the following below this CDDL HEADER, with the fields
// enclosed by brackets "[]" replaced with your own identifying information:
//
// Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
//
// CDDL HEADER END
//

//
// Copyright (c) 2019, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Mingjian Wen
//


#include "descriptor.h"

#define LOG_ERROR(msg) (std::cerr<<"ERROR (descriptor): "<< (msg) <<std::endl)

Descriptor::Descriptor() {
  has_three_body = false;
}

Descriptor::~Descriptor()
{
  for (size_t i = 0; i < params.size(); i++)
  {
    Deallocate2DArray(params.at(i));
  }
}

int Descriptor::read_parameter_file(FILE * const filePointer)
{
  int ier;
  int endOfFileFlag = 0;
  char nextLine[MAXLINE];
  char errorMsg[1024];
  char name[128];
  double cutoff;


  // descriptor
  int numDescTypes;
  int numDescs;
  int numParams;
  int numParamSets;
  double ** descParams = NULL;

  // cutoff
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%s %lf", name, &cutoff);
  if (ier != 2)
  {
    sprintf(errorMsg, "unable to read cutoff from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // register cutoff
  lowerCase(name);
  if (strcmp(name, "cos") != 0 && strcmp(name, "exp") != 0)
  {
    sprintf(errorMsg,
        "unsupported cutoff type. Expecting `cos', or `exp' "
        "given %s.\n",
        name);
    LOG_ERROR(errorMsg);
    return true;
  }
  set_cutoff(name, 1, &cutoff);


  // number of descriptor types
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%d", &numDescTypes);
  if (ier != 1)
  {
    sprintf(errorMsg, "unable to read number of descriptor types from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // descriptor
  for (int i = 0; i < numDescTypes; i++)
  {
    // descriptor name and parameter dimensions
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);

    // name of descriptor
    ier = sscanf(nextLine, "%s", name);
    if (ier != 1)
    {
      sprintf(errorMsg, "unable to read descriptor from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }
    lowerCase(name);  // change to lower case name
    if (strcmp(name, "g1") == 0)
    {
      add_descriptor(name, NULL, 1, 0);
    }
    else
    {
      // re-read name, and read number of param sets and number of params
      ier = sscanf(nextLine, "%s %d %d", name, &numParamSets, &numParams);
      if (ier != 3)
      {
        sprintf(errorMsg, "unable to read descriptor from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
      // change name to lower case
      lowerCase(name);

      // check size of params is correct w.r.t its name
      if (strcmp(name, "g2") == 0)
      {
        if (numParams != 2)
        {
          sprintf(errorMsg,
              "number of params for descriptor G2 is incorrect, "
              "expecting 2, but given %d.\n",
              numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g3") == 0)
      {
        if (numParams != 1)
        {
          sprintf(errorMsg,
              "number of params for descriptor G3 is incorrect, "
              "expecting 1, but given %d.\n",
              numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g4") == 0)
      {
        if (numParams != 3)
        {
          sprintf(errorMsg,
              "number of params for descriptor G4 is incorrect, "
              "expecting 3, but given %d.\n",
              numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g5") == 0)
      {
        if (numParams != 3)
        {
          sprintf(errorMsg,
              "number of params for descriptor G5 is incorrect, "
              "expecting 3, but given %d.\n",
              numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else
      {
        sprintf(errorMsg, "unsupported descriptor `%s' from line:\n", name);
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }

      // read descriptor params
      AllocateAndInitialize2DArray<double>(descParams, numParamSets, numParams);
      for (int j = 0; j < numParamSets; j++)
      {
        getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
        ier = getXdouble(nextLine, numParams, descParams[j]);
        if (ier)
        {
          sprintf(errorMsg,
              "unable to read descriptor parameters from line:\n");
          strcat(errorMsg, nextLine);
          LOG_ERROR(errorMsg);
          return true;
        }
      }

      // copy data to Descriptor
      add_descriptor(name, descParams, numParamSets, numParams);
      Deallocate2DArray(descParams);
    }
  }
  // number of descriptors
  numDescs = get_num_descriptors();

  // centering and normalizing params
  // flag, whether we use this feature
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%*s %s", name);
  if (ier != 1)
  {
    sprintf(errorMsg,
        "unable to read centering and normalization info from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  lowerCase(name);
  bool do_center_and_normalize;
  if (strcmp(name, "true") == 0) { do_center_and_normalize = true; }
  else
  {
    do_center_and_normalize = false;
  }

  int size = 0;
  double * means = NULL;
  double * stds = NULL;
  if (do_center_and_normalize)
  {
    // size of the data, this should be equal to numDescs
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
    ier = sscanf(nextLine, "%d", &size);
    if (ier != 1)
    {
      sprintf(errorMsg,
          "unable to read the size of centering and normalization "
          "data info from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }
    if (size != numDescs)
    {
      sprintf(errorMsg,
          "Size of centering and normalizing data inconsistent with "
          "the number of descriptors. Size = %d, num_descriptors=%d\n",
          size,
          numDescs);
      LOG_ERROR(errorMsg);
      return true;
    }

    // read means
    AllocateAndInitialize1DArray<double>(means, size);
    for (int i = 0; i < size; i++)
    {
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      ier = sscanf(nextLine, "%lf", &means[i]);
      if (ier != 1)
      {
        sprintf(errorMsg, "unable to read `means' from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }

    // read standard deviations
    AllocateAndInitialize1DArray<double>(stds, size);
    for (int i = 0; i < size; i++)
    {
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      ier = sscanf(nextLine, "%lf", &stds[i]);
      if (ier != 1)
      {
        sprintf(errorMsg, "unable to read `means' from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }
  }

  // store info into descriptor class
  set_center_and_normalize(
      do_center_and_normalize, size, means, stds);
  Deallocate1DArray(means);
  Deallocate1DArray(stds);

  // TODO delete
  //  echo_input();

  // everything is OK
  return false;
}


void Descriptor::set_cutoff(char * name, int ncutoff, double * value)
{
  if (strcmp(name, "cos") == 0)
  {
    cutoff_func = &cut_cos;
    d_cutoff_func = &d_cut_cos;
  }
  else if (strcmp(name, "exp") == 0)
  {
    cutoff_func = &cut_exp;
    d_cutoff_func = &d_cut_exp;
  }

  for (int i=0; i<ncutoff; i++) {
    cutoff.push_back(value[i]);
  }
}

void Descriptor::get_cutoff(int& ncutoff, double *& value)
{
  ncutoff = cutoff.size();
  value = cutoff.data();
}

void Descriptor::add_descriptor(char * name, double ** values, int row, int col)
{
  double ** params = 0;

  AllocateAndInitialize2DArray<double>(params, row, col);
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++) { params[i][j] = values[i][j]; }
  }

  std::string nm(name);

  int index = 0;
  for (size_t i = 0; i < num_param_sets.size(); i++)
  { index += num_param_sets[i]; }

  this->name.push_back(nm);
  this->params.push_back(params);
  num_param_sets.push_back(row);
  num_params.push_back(col);
  starting_index.push_back(index);

  // set t
  if (strcmp(name, "g4") == 0 || strcmp(name, "g5") == 0)
  { has_three_body = true; }
}

void Descriptor::set_center_and_normalize(bool do_center_and_normalize,
                                          int size,
                                          double * means,
                                          double * stds)
{
  center_and_normalize = do_center_and_normalize;
  for (int i = 0; i < size; i++)
  {
    features_mean.push_back(means[i]);
    features_std.push_back(stds[i]);
  }
}


int Descriptor::get_num_descriptors()
{
  int N = 0;

  for (size_t i = 0; i < num_param_sets.size(); i++)
  { N += num_param_sets.at(i); }
  return N;
}


//*****************************************************************************
// Symmetry functions: Jorg Behler, J. Chem. Phys. 134, 074106, 2011.
//*****************************************************************************

void Descriptor::sym_g1(double r, double rcut, double & phi)
{
  phi = cutoff_func(r, rcut);
}

void Descriptor::sym_d_g1(double r, double rcut, double & phi, double & dphi)
{
  phi = cutoff_func(r, rcut);
  dphi = d_cutoff_func(r, rcut);
}

void Descriptor::sym_g2(
    double eta, double Rs, double r, double rcut, double & phi)
{
  phi = exp(-eta * (r - Rs) * (r - Rs)) * cutoff_func(r, rcut);
}

void Descriptor::sym_d_g2(
    double eta, double Rs, double r, double rcut, double & phi, double & dphi)
{
  if (r > rcut)
  {
    phi = 0.;
    dphi = 0.;
  }
  else
  {
    double eterm = exp(-eta * (r - Rs) * (r - Rs));
    double determ = -2 * eta * (r - Rs) * eterm;
    double fc = cutoff_func(r, rcut);
    double dfc = d_cutoff_func(r, rcut);
    phi = eterm * fc;
    dphi = determ * fc + eterm * dfc;
  }
}

void Descriptor::sym_g3(double kappa, double r, double rcut, double & phi)
{
  phi = cos(kappa * r) * cutoff_func(r, rcut);
}

void Descriptor::sym_d_g3(
    double kappa, double r, double rcut, double & phi, double & dphi)
{
  double costerm = cos(kappa * r);
  double dcosterm = -kappa * sin(kappa * r);
  double fc = cutoff_func(r, rcut);
  double dfc = d_cutoff_func(r, rcut);

  phi = costerm * fc;
  dphi = dcosterm * fc + costerm * dfc;
}

void Descriptor::sym_g4(double zeta,
                        double lambda,
                        double eta,
                        const double * r,
                        const double * rcut,
                        double & phi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rcutjk = rcut[2];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik || rjk > rcutjk) { phi = 0.0; }
  else
  {
    // i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);

    double costerm;
    double base = 1 + lambda * cos_ijk;
    if (base <= 0)
    {  // prevent numerical instability (when lambda=-1 and cos_ijk=1)
      costerm = 0;
    }
    else
    {
      costerm = pow(base, zeta);
    }

    double eterm = exp(-eta * (rijsq + riksq + rjksq));

    phi = pow(2, 1 - zeta) * costerm * eterm * cutoff_func(rij, rcutij)
          * cutoff_func(rik, rcutik) * cutoff_func(rjk, rcutjk);
  }
}

void Descriptor::sym_d_g4(double zeta,
                          double lambda,
                          double eta,
                          const double * r,
                          const double * rcut,
                          double & phi,
                          double * const dphi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rcutjk = rcut[2];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik || rjk > rcutjk)
  {
    phi = 0.0;
    dphi[0] = 0.0;
    dphi[1] = 0.0;
    dphi[2] = 0.0;
  }
  else
  {
    // cosine term, i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
    double dcos_dij = (rijsq - riksq + rjksq) / (2 * rijsq * rik);
    double dcos_dik = (riksq - rijsq + rjksq) / (2 * rij * riksq);
    double dcos_djk = -rjk / (rij * rik);

    double costerm;
    double dcosterm_dcos;
    double base = 1 + lambda * cos_ijk;
    if (base <= 0)
    {  // prevent numerical instability (when lambda=-1 and cos_ijk=1)
      costerm = 0.0;
      dcosterm_dcos = 0.0;
    }
    else
    {
      double power = pow(base, zeta);
      double power_minus1 = power / base;
      costerm = power;
      dcosterm_dcos = zeta * power_minus1 * lambda;
    }
    double dcosterm_dij = dcosterm_dcos * dcos_dij;
    double dcosterm_dik = dcosterm_dcos * dcos_dik;
    double dcosterm_djk = dcosterm_dcos * dcos_djk;

    // exponential term
    double eterm = exp(-eta * (rijsq + riksq + rjksq));
    double determ_dij = -2 * eterm * eta * rij;
    double determ_dik = -2 * eterm * eta * rik;
    double determ_djk = -2 * eterm * eta * rjk;

    // power 2 term
    //double p2 = pow(2, 1 - zeta);
    int tmp = 1 << (int) zeta;  // compute 2^(zeta)
    double p2 = 2. / tmp;  // compute 2^(1-zeta)

    // cutoff_func
    double fcij = cutoff_func(rij, rcutij);
    double fcik = cutoff_func(rik, rcutik);
    double fcjk = cutoff_func(rjk, rcutjk);
    double fcprod = fcij * fcik * fcjk;
    double dfcprod_dij = d_cutoff_func(rij, rcutij) * fcik * fcjk;
    double dfcprod_dik = d_cutoff_func(rik, rcutik) * fcij * fcjk;
    double dfcprod_djk = d_cutoff_func(rjk, rcutjk) * fcij * fcik;

    // phi
    phi = p2 * costerm * eterm * fcprod;
    // dphi_dij
    dphi[0] = p2
              * (dcosterm_dij * eterm * fcprod + costerm * determ_dij * fcprod
                 + costerm * eterm * dfcprod_dij);
    // dphi_dik
    dphi[1] = p2
              * (dcosterm_dik * eterm * fcprod + costerm * determ_dik * fcprod
                 + costerm * eterm * dfcprod_dik);
    // dphi_djk
    dphi[2] = p2
              * (dcosterm_djk * eterm * fcprod + costerm * determ_djk * fcprod
                 + costerm * eterm * dfcprod_djk);
  }
}

void Descriptor::sym_g5(double zeta,
                        double lambda,
                        double eta,
                        const double * r,
                        const double * rcut,
                        double & phi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik) { phi = 0.0; }
  else
  {
    // i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);

    double costerm;
    double base = 1 + lambda * cos_ijk;
    if (base <= 0)
    {  // prevent numerical instability (when lambda=-1 and cos_ijk=1)
      costerm = 0;
    }
    else
    {
      costerm = pow(base, zeta);
    }

    double eterm = exp(-eta * (rijsq + riksq));

    phi = pow(2, 1 - zeta) * costerm * eterm * cutoff_func(rij, rcutij)
          * cutoff_func(rik, rcutik);
  }
}

void Descriptor::sym_d_g5(double zeta,
                          double lambda,
                          double eta,
                          const double * r,
                          const double * rcut,
                          double & phi,
                          double * const dphi)
{
  double rij = r[0];
  double rik = r[1];
  double rjk = r[2];
  double rcutij = rcut[0];
  double rcutik = rcut[1];
  double rijsq = rij * rij;
  double riksq = rik * rik;
  double rjksq = rjk * rjk;

  if (rij > rcutij || rik > rcutik)
  {
    phi = 0.0;
    dphi[0] = 0.0;
    dphi[1] = 0.0;
    dphi[2] = 0.0;
  }
  else
  {
    // cosine term, i is the apex atom
    double cos_ijk = (rijsq + riksq - rjksq) / (2 * rij * rik);
    double dcos_dij = (rijsq - riksq + rjksq) / (2 * rijsq * rik);
    double dcos_dik = (riksq - rijsq + rjksq) / (2 * rij * riksq);
    double dcos_djk = -rjk / (rij * rik);

    double costerm;
    double dcosterm_dcos;
    double base = 1 + lambda * cos_ijk;
    if (base <= 0)
    {  // prevent numerical instability (when lambda=-1 and cos_ijk=1)
      costerm = 0.0;
      dcosterm_dcos = 0.0;
    }
    else
    {
      costerm = pow(base, zeta);
      dcosterm_dcos = zeta * pow(base, zeta - 1) * lambda;
    }
    double dcosterm_dij = dcosterm_dcos * dcos_dij;
    double dcosterm_dik = dcosterm_dcos * dcos_dik;
    double dcosterm_djk = dcosterm_dcos * dcos_djk;

    // exponential term
    double eterm = exp(-eta * (rijsq + riksq));
    double determ_dij = -2 * eterm * eta * rij;
    double determ_dik = -2 * eterm * eta * rik;

    // power 2 term
    double p2 = pow(2, 1 - zeta);

    // cutoff_func
    double fcij = cutoff_func(rij, rcutij);
    double fcik = cutoff_func(rik, rcutik);
    double fcprod = fcij * fcik;
    double dfcprod_dij = d_cutoff_func(rij, rcutij) * fcik;
    double dfcprod_dik = d_cutoff_func(rik, rcutik) * fcij;

    // phi
    phi = p2 * costerm * eterm * fcprod;
    // dphi_dij
    dphi[0] = p2
              * (dcosterm_dij * eterm * fcprod + costerm * determ_dij * fcprod
                 + costerm * eterm * dfcprod_dij);
    // dphi_dik
    dphi[1] = p2
              * (dcosterm_dik * eterm * fcprod + costerm * determ_dik * fcprod
                 + costerm * eterm * dfcprod_dik);
    // dphi_djk
    dphi[2] = p2 * dcosterm_djk * eterm * fcprod;
  }
}
