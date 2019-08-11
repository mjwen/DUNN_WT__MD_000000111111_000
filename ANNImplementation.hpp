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

#include "ANN.hpp"
#include "KIM_LogMacros.hpp"
#include "descriptor.h"
#include "helper.hpp"
#include "network.h"
#include <vector>

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
  int ensemble_size_;
  int last_ensemble_size_;
  //
  // KIM API: Model Parameters whose (pointer) values never change
  //   Memory allocated in AllocateParameterMemory() (from constructor)
  //   Memory deallocated in destructor
  //   Data set in ReadParameterFile routines OR by KIM Simulator
  int active_member_id_;

  // Mutable values that only change when Refresh() executes
  //   Set in Refresh (via SetRefreshMutableValues)
  //
  //
  // KIM API: Model Parameters (can be changed directly by KIM Simulator)
  // none
  //
  // ANNImplementation: values (changed only by Refresh())
  int last_active_member_id_;
  double influenceDistance_;
  int modelWillNotRequestNeighborsOfNoncontributingParticles_;

  // Mutable values that can change with each call to Refresh() and Compute()
  //   Memory may be reallocated on each call
  //
  //
  // ANNImplementation: values that change
  int cachedNumberOfParticles_;

  // descriptor and network
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
      VectorOfSizeSix *& particleVirial);
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

  ier = true;

  if (isComputeProcess_dEdr == true)
  {
    LOG_ERROR("process_dEdr not supported by this driver");
    return ier;
  }

  if (isComputeProcess_d2Edr2 == true)
  {
    LOG_ERROR("process_d2Edr2 not supported by this driver");
    return ier;
  }

  bool need_dE = ((isComputeForces == true) || (isComputeVirial == true)
                  || (isComputeParticleVirial == true));

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

    double * GC = nullptr;
    double ** dGCdr = nullptr;
    int const Ndescriptors = descriptor_->get_num_descriptors();
    AllocateAndInitialize1DArray<double>(GC, Ndescriptors);
    if (need_dE)
    {
      AllocateAndInitialize2DArray<double>(
          dGCdr, Ndescriptors, (numnei + 1) * DIM);
    }

    descriptor_->generate_one_atom(i,
                                   coordinates,
                                   particleSpeciesCodes,
                                   n1atom,
                                   numnei,
                                   GC,
                                   dGCdr[0],
                                   need_dE);

    // centering and normalization
    if (descriptor_->need_normalize())
    {
      for (int j = 0; j < Ndescriptors; j++)
      {
        double mean;
        double std;
        descriptor_->get_feature_mean_and_std(j, mean, std);
        GC[j] = (GC[j] - mean) / std;

        if (need_dE)
        {
          for (int k = 0; k < (numnei + 1) * DIM; k++) { dGCdr[j][k] /= std; }
        }
      }
    }

    double E = 0;
    double * dEdGC = nullptr;

    // select a specific running mode
    if (ensemble_size_ == 0 || active_member_id_ == 0)
    {
      // fully-connected NN

      network_->set_fully_connected(true);
      int ensemble_index = 0;  // ignored by in fully-connected mode

      // NN forward
      network_->forward(GC, 1, Ndescriptors, ensemble_index);
      E = network_->get_sum_output();

      // NN backprop
      if (need_dE)
      {
        network_->backward();
        dEdGC = network_->get_grad_input();
      }
    }
    else if (active_member_id_ > 0 && active_member_id_ <= ensemble_size_)
    {
      // a specific member of the ensemble

      network_->set_fully_connected(false);
      int ensemble_index = active_member_id_ - 1;  // internally starts from 0

      // NN forward
      network_->forward(GC, 1, Ndescriptors, ensemble_index);
      E = network_->get_sum_output();

      // NN backprop
      if (need_dE)
      {
        network_->backward();
        dEdGC = network_->get_grad_input();
      }
    }
    else if (active_member_id_ == -1)
    {
      // average over ensemble

      network_->set_fully_connected(false);
      if (need_dE)
      { AllocateAndInitialize1DArray<double>(dEdGC, Ndescriptors); }

      for (int iev = 0; iev < ensemble_size_; iev++)
      {
        // NN forward
        network_->forward(GC, 1, Ndescriptors, iev);
        double eng = network_->get_sum_output();
        E += eng / ensemble_size_;

        // NN backprop
        if (need_dE)
        {
          network_->backward();
          double * deng = network_->get_grad_input();
          for (int j = 0; j < Ndescriptors; j++)
          { dEdGC[j] += deng[j] / ensemble_size_; }
        }
      }
    }
    else
    {
      char message[MAXLINE];
      sprintf(message,
              "`active_member_id=%d` out of range. Should be [-1, %d]",
              active_member_id_,
              ensemble_size_);
      LOG_ERROR(message);
      return ier;
    }

    // Contribution to energy
    if (isComputeEnergy == true) { *energy += E; }

    // Contribution to particle energy
    if (isComputeParticleEnergy == true) { particleEnergy[i] += E; }

    // Contribution to forces, particle virial, and virial
    if (need_dE == true)
    {
      for (int j = 0; j < Ndescriptors; j++)
      {
        for (int k = 0; k < numnei + 1; k++)
        {
          int idx;
          if (k == numnei)
          {
            idx = i;  // targeting atom itself
          }
          else
          {
            idx = n1atom[k];  // neighbors
          }
          VectorOfSizeDIM f;
          f[0] = -dEdGC[j] * dGCdr[j][k * DIM + 0];
          f[1] = -dEdGC[j] * dGCdr[j][k * DIM + 1];
          f[2] = -dEdGC[j] * dGCdr[j][k * DIM + 2];

          if (isComputeForces == true)
          {
            forces[idx][0] += f[0];
            forces[idx][1] += f[1];
            forces[idx][2] += f[2];
          }

          if (isComputeParticleVirial == true || isComputeVirial == true)
          {
            VectorOfSizeSix v;
            v[0] = -f[0] * coordinates[idx][0];
            v[1] = -f[1] * coordinates[idx][1];
            v[2] = -f[2] * coordinates[idx][2];
            v[3] = -f[1] * coordinates[idx][2];
            v[4] = -f[2] * coordinates[idx][0];
            v[5] = -f[0] * coordinates[idx][1];
            if (isComputeParticleVirial == true)
            {
              particleVirial[idx][0] += v[0];
              particleVirial[idx][1] += v[1];
              particleVirial[idx][2] += v[2];
              particleVirial[idx][3] += v[3];
              particleVirial[idx][4] += v[4];
              particleVirial[idx][5] += v[5];
            }
            if (isComputeVirial == true)
            {
              virial[0] += v[0];
              virial[1] += v[1];
              virial[2] += v[2];
              virial[3] += v[3];
              virial[4] += v[4];
              virial[5] += v[5];
            }
          }
        }
      }
    }

    // deallocate memory
    Deallocate1DArray(GC);
    if (need_dE) { Deallocate2DArray(dGCdr); }
    if (active_member_id_ == -1 and need_dE) { Deallocate1DArray(dEdGC); }

  }  // loop over i

  // everything is good
  ier = false;
  return ier;
}

#endif  // ANN_IMPLEMENTATION_HPP_
