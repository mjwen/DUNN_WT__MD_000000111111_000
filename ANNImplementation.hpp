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

#ifndef ANN_IMPLEMENTATION_HPP_
#define ANN_IMPLEMENTATION_HPP_

#include <vector>
#include "ANN.hpp"
#include "KIM_LogMacros.hpp"
#include "descriptor.h"
#include "helper.hpp"
#include "network.h"

#define DIM 3
#define ONE 1.0

#define MAX_PARAMETER_FILES 3

//==============================================================================
//
// Declaration of ANNImplementation class
//
//==============================================================================

//******************************************************************************
class ANNImplementation
{
 public:
  ANNImplementation(KIM::ModelDriverCreate * const modelDriverCreate,
                    KIM::LengthUnit const requestedLengthUnit,
                    KIM::EnergyUnit const requestedEnergyUnit,
                    KIM::ChargeUnit const requestedChargeUnit,
                    KIM::TemperatureUnit const requestedTemperatureUnit,
                    KIM::TimeUnit const requestedTimeUnit,
                    int * const ier);
  ~ANNImplementation();  // no explicit Destroy() needed here

  int Refresh(KIM::ModelRefresh * const modelRefresh);
  int Compute(KIM::ModelCompute const * const modelCompute,
              KIM::ModelComputeArguments const * const modelComputeArguments);
  int ComputeArgumentsCreate(KIM::ModelComputeArgumentsCreate * const
                                 modelComputeArgumentsCreate) const;
  int ComputeArgumentsDestroy(KIM::ModelComputeArgumentsDestroy * const
                                  modelComputeArgumentsDestroy) const;

 private:
  // Constant values that never change
  //   Set in constructor (via SetConstantValues)
  //
  //
  // ANNImplementation: constants

  // Constant values that are read from the input files and never change
  //   Set in constructor (via functions listed below)
  // none
  //
  // Private Model Parameters
  //   Memory allocated in AllocatePrivateParameterMemory() (from constructor)
  //   Memory deallocated in destructor
  //   Data set in ReadParameterFile routines
  // none
  //
  // KIM API: Model Parameters whose (pointer) values never change
  //   Memory allocated in AllocateParameterMemory() (from constructor)
  //   Memory deallocated in destructor
  //   Data set in ReadParameterFile routines OR by KIM Simulator
  // none

  // Mutable values that only change when Refresh() executes
  //   Set in Refresh (via SetRefreshMutableValues)
  //
  //
  // KIM API: Model Parameters (can be changed directly by KIM Simulator)
  // none
  //
  // ANNImplementation: values (changed only by Refresh())
  double influenceDistance_;
  int modelWillNotRequestNeighborsOfNoncontributingParticles_;

  // Mutable values that can change with each call to Refresh() and Compute()
  //   Memory may be reallocated on each call
  //
  //
  // ANNImplementation: values that change
  int cachedNumberOfParticles_;

  // descriptor;
  Descriptor * descriptor_;
  NeuralNetwork * network_;

  // Helper methods
  //
  //
  // Related to constructor
  void AllocatePrivateParameterMemory();
  void AllocateParameterMemory();

  static int
  OpenParameterFiles(KIM::ModelDriverCreate * const modelDriverCreate,
                     int const numberParameterFiles,
                     FILE * parameterFilePointers[MAX_PARAMETER_FILES]);
  int ProcessParameterFiles(
      KIM::ModelDriverCreate * const modelDriverCreate,
      int const numberParameterFiles,
      FILE * const parameterFilePointers[MAX_PARAMETER_FILES]);
  static void
  CloseParameterFiles(int const numberParameterFiles,
                      FILE * const parameterFilePointers[MAX_PARAMETER_FILES]);
  int ConvertUnits(KIM::ModelDriverCreate * const modelDriverCreate,
                   KIM::LengthUnit const requestedLengthUnit,
                   KIM::EnergyUnit const requestedEnergyUnit,
                   KIM::ChargeUnit const requestedChargeUnit,
                   KIM::TemperatureUnit const requestedTemperatureUnit,
                   KIM::TimeUnit const requestedTimeUnit);
  int RegisterKIMModelSettings(
      KIM::ModelDriverCreate * const modelDriverCreate) const;
  int RegisterKIMComputeArgumentsSettings(
      KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
      const;
  int RegisterKIMParameters(KIM::ModelDriverCreate * const modelDriverCreate);
  int RegisterKIMFunctions(
      KIM::ModelDriverCreate * const modelDriverCreate) const;

  //
  // Related to Refresh()
  template<class ModelObj>
  int SetRefreshMutableValues(ModelObj * const modelObj);

  //
  // Related to Compute()
  int SetComputeMutableValues(
      KIM::ModelComputeArguments const * const modelComputeArguments,
      bool & isComputeProcess_dEdr,
      bool & isComputeProcess_d2Edr2,
      bool & isComputeEnergy,
      bool & isComputeForces,
      bool & isComputeParticleEnergy,
      bool & isComputeVirial,
      bool & isComputeParticleVirial,
      int const *& particleSpeciesCodes,
      int const *& particleContributing,
      VectorOfSizeDIM const *& coordinates,
      double *& energy,
      VectorOfSizeDIM *& forces,
      double *& particleEnergy,
      VectorOfSizeSix *& virial,
      VectorOfSizeSix *& particleViral);
  int CheckParticleSpeciesCodes(KIM::ModelCompute const * const modelCompute,
                                int const * const particleSpeciesCodes) const;
  int GetComputeIndex(const bool & isComputeProcess_dEdr,
                      const bool & isComputeProcess_d2Edr2,
                      const bool & isComputeEnergy,
                      const bool & isComputeForces,
                      const bool & isComputeParticleEnergy,
                      const bool & isComputeVirial,
                      const bool & isComputeParticleVirial) const;

  // compute functions
  template<bool isComputeProcess_dEdr,
           bool isComputeProcess_d2Edr2,
           bool isComputeEnergy,
           bool isComputeForces,
           bool isComputeParticleEnergy,
           bool isComputeVirial,
           bool isComputeParticleVirial>
  int Compute(KIM::ModelCompute const * const modelCompute,
              KIM::ModelComputeArguments const * const modelComputeArguments,
              const int * const particleSpeciesCodes,
              const int * const particleContributing,
              const VectorOfSizeDIM * const coordinates,
              double * const energy,
              VectorOfSizeDIM * const forces,
              double * const particleEnergy,
              VectorOfSizeSix virial,
              VectorOfSizeSix * const particleVirial) const;

  // LJ functions
  void calc_phi(double const epsilon,
                double const sigma,
                double const cutoff,
                double const r,
                double * const phi) const;
  void calc_phi_dphi(double const epsilon,
                     double const sigma,
                     double const cutoff,
                     double const r,
                     double * const phi,
                     double * const dphi) const;
  void switch_fn(double const x_min,
                 double const x_max,
                 double const x,
                 double * const fn,
                 double * const fn_prime) const;
};

//==============================================================================
//
// Definition of ANNImplementation::Compute functions
//
// NOTE: Here we rely on the compiler optimizations to prune dead code
//       after the template expansions.  This provides high efficiency
//       and easy maintenance.
//
//==============================================================================

#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCompute
template<bool isComputeProcess_dEdr,
         bool isComputeProcess_d2Edr2,
         bool isComputeEnergy,
         bool isComputeForces,
         bool isComputeParticleEnergy,
         bool isComputeVirial,
         bool isComputeParticleVirial>
int ANNImplementation::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments,
    const int * const particleSpeciesCodes,
    const int * const particleContributing,
    const VectorOfSizeDIM * const coordinates,
    double * const energy,
    VectorOfSizeDIM * const forces,
    double * const particleEnergy,
    VectorOfSizeSix virial,
    VectorOfSizeSix * const particleVirial) const
{
  int ier = false;

  if ((isComputeEnergy == false) && (isComputeParticleEnergy == false)
      && (isComputeForces == false) && (isComputeProcess_dEdr == false)
      && (isComputeProcess_d2Edr2 == false) && (isComputeVirial == false)
      && (isComputeParticleVirial == false))
  { return ier; }


  if (isComputeProcess_dEdr == true)
  {
    LOG_ERROR("process_dEdr not supported by this driver");
    return true;
  }

  if (isComputeProcess_d2Edr2 == true)
  {
    LOG_ERROR("process_d2Edr2 not supported by this driver");
    return true;
  }

  bool need_dE = ((isComputeForces == true) || (isComputeVirial == true) ||
      (isComputeParticleVirial == true));


  // ANNImplementation: values that does not change
  int const Nparticles = cachedNumberOfParticles_;

  // initialize energy and forces
  if (isComputeEnergy == true) { *energy = 0.0; }

  if (isComputeParticleEnergy == true)
  {
    for (int i = 0; i < Nparticles; ++i) { particleEnergy[i] = 0.0; }
  }

  if (isComputeForces == true)
  {
    for (int i = 0; i < Nparticles; ++i)
    {
      for (int j = 0; j < DIM; ++j) { forces[i][j] = 0.0; }
    }
  }

  if (isComputeVirial == true)
  {
    for (int i = 0; i < 6; ++i) { virial[i] = 0.0; }
  }

  if (isComputeParticleVirial == true)
  {
    for (int i = 0; i < Nparticles; ++i)
    {
      for (int j = 0; j < 6; ++j) { particleVirial[i][j] = 0.0; }
    }
  }


  // calculate generalized coordinates
  //
  // Setup loop over contributing particles
  for (int i = 0; i < Nparticles; i++)
  {
    if (!particleContributing[i]) { continue; }

    // get neighbors of atom i
    int numnei = 0;
    int const * n1atom = 0;
    modelComputeArguments->GetNeighborList(0, i, &numnei, &n1atom);


    double * GC;
    double ** dGCdr;
    int const Ndescriptors = descriptor_->get_num_descriptors();
    AllocateAndInitialize1DArray<double>(GC, Ndescriptors);

    if (need_dE) {
    AllocateAndInitialize2DArray<double>(dGCdr, Ndescriptors, (numnei+1)*DIM);
  }



  descriptor_->generate_one_atom(i,
                                     coordinates,
                                         particleSpeciesCodes,
                                         n1atom,
                                         numnei,
                                         GC,
              reinterpret_cast <double *> (dGCdr),
                                         need_dE);








    // centering and normalization
    if (descriptor_->need_normalize())
    {
      for (int t = 0; t < Ndescriptors; t++)
      {
        double mean;
        double std;
        descriptor_->get_feature_mean_and_std(t, mean, std);
        GC[t] = (GC[t] - mean) / std;
      }

      // Done below when computing forces
      //      if (need_dE)
      //      {
      //        for (int s = 0; s < Npairs_two; s++)
      //        {
      //          for (int t = 0; t < Ndescriptors_two; t++)
      //          {
      //            int desc_idx = map_t_desc_two[t];
      //            dGCdr_two[s][t] /= descriptor_->feature_std_[desc_idx];
      //          }
      //        }

      //        for (int s = 0; s < Npairs_three; s++)
      //        {
      //          for (int t = 0; t < Ndescriptors_three; t++)
      //          {
      //            int desc_idx = map_t_desc_three[t];
      //            dGCdr_three[s][t][0] /= descriptor_->feature_std_[desc_idx];
      //            dGCdr_three[s][t][1] /= descriptor_->feature_std_[desc_idx];
      //            dGCdr_three[s][t][2] /= descriptor_->feature_std_[desc_idx];
      //          }
      //        }
      //      }
    }


    // NN feed forward
    int ensemble_index = 0;
    network_->forward(GC, 1, Ndescriptors, ensemble_index);

    // NN backprop to compute derivative of energy w.r.t generalized coords
    double * dEdGC = NULL;
    if (need_dE)
    {
      network_->backward();
      dEdGC = network_->get_grad_input();
    }

    double Ei = 0.;
    if (isComputeEnergy == true || isComputeParticleEnergy == true)
    { Ei = network_->get_sum_output(); }

    // Contribution to energy
    if (isComputeEnergy == true) { *energy += Ei; }

    // Contribution to particle energy
    if (isComputeParticleEnergy == true) { particleEnergy[i] += Ei; }

    // Contribution to forces
    if (isComputeForces == true) {
      // atom i itself
      for (int j = 0; j < Ndescriptors; j++) {
        for (int k = 0; k < numnei+1; k++) {
          int idx ;
          if (k==numnei) {
            idx = i;  // targeting atom itself
          }
          else {
            idx = n1atom[k]; // neighbors
          }
          for (int dim = 0; dim < DIM; dim++) {
            forces[idx][dim] += dEdGC[j] * dGCdr[j][k * DIM + dim];
          }
        }
      }
    }

    // Contribution to virial
    if (isComputeVirial == true) {
      //TODO
    }

    // Contribution to particleVirial
    if (isComputeParticleVirial == true) {
      //TODO
    }


    Deallocate1DArray(GC);
    Deallocate2DArray(dGCdr);

  }  // loop over i


  // everything is good
  ier = false;
  return ier;
}

#endif  // ANN_IMPLEMENTATION_HPP_
