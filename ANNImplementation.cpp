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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

#include "ANNImplementation.hpp"
#include "KIM_ModelDriverHeaders.hpp"

//==============================================================================
//
// Implementation of ANNImplementation public member functions
//
//==============================================================================

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
ANNImplementation::ANNImplementation(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit,
    int * const ier) :
    energyScale_(1.0),
    ensemble_size_(0),
    last_ensemble_size_(0),
    active_member_id_(-1),
    last_active_member_id_(-1),
    influenceDistance_(0.0),
    modelWillNotRequestNeighborsOfNoncontributingParticles_(1),
    cachedNumberOfParticles_(0)
{
  // create descriptor and network classes
  descriptor_ = new Descriptor();
  network_ = new NeuralNetwork();

  FILE * parameterFilePointers[MAX_PARAMETER_FILES];
  int numberParameterFiles;

  modelDriverCreate->GetNumberOfParameterFiles(&numberParameterFiles);
  *ier = OpenParameterFiles(
      modelDriverCreate, numberParameterFiles, parameterFilePointers);
  if (*ier) { return; }

  *ier = ProcessParameterFiles(
      modelDriverCreate, numberParameterFiles, parameterFilePointers);
  CloseParameterFiles(numberParameterFiles, parameterFilePointers);
  if (*ier) { return; }

  *ier = ConvertUnits(modelDriverCreate,
                      requestedLengthUnit,
                      requestedEnergyUnit,
                      requestedChargeUnit,
                      requestedTemperatureUnit,
                      requestedTimeUnit);
  if (*ier) { return; }

  *ier = SetRefreshMutableValues(modelDriverCreate);
  if (*ier) { return; }

  *ier = RegisterKIMModelSettings(modelDriverCreate);
  if (*ier) { return; }

  *ier = RegisterKIMParameters(modelDriverCreate);
  if (*ier) { return; }

  *ier = RegisterKIMFunctions(modelDriverCreate);
  if (*ier) { return; }

  // everything is good
  *ier = false;
  return;
}

//******************************************************************************
ANNImplementation::~ANNImplementation()
{  // note: it is ok to delete a null
   // pointer and we have ensured that
  // everything is initialized to null

  delete descriptor_;
  delete network_;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelRefresh
int ANNImplementation::Refresh(KIM::ModelRefresh * const modelRefresh)
{
  int ier;

  ier = SetRefreshMutableValues(modelRefresh);
  if (ier) { return ier; }

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int ANNImplementation::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments)
{
  int ier;

  // KIM API Model Input compute flags
  bool isComputeProcess_dEdr = false;
  bool isComputeProcess_d2Edr2 = false;
  //
  // KIM API Model Output compute flags
  bool isComputeEnergy = false;
  bool isComputeForces = false;
  bool isComputeParticleEnergy = false;
  bool isComputeVirial = false;
  bool isComputeParticleVirial = false;
  //
  // KIM API Model Input
  int const * particleSpeciesCodes = NULL;
  int const * particleContributing = NULL;
  VectorOfSizeDIM const * coordinates = NULL;
  //
  // KIM API Model Output
  double * energy = NULL;
  double * particleEnergy = NULL;
  VectorOfSizeDIM * forces = NULL;
  VectorOfSizeSix * virial = NULL;
  VectorOfSizeSix * particleVirial = NULL;

  ier = SetComputeMutableValues(modelComputeArguments,
                                isComputeProcess_dEdr,
                                isComputeProcess_d2Edr2,
                                isComputeEnergy,
                                isComputeForces,
                                isComputeParticleEnergy,
                                isComputeVirial,
                                isComputeParticleVirial,
                                particleSpeciesCodes,
                                particleContributing,
                                coordinates,
                                energy,
                                forces,
                                particleEnergy,
                                virial,
                                particleVirial);
  if (ier) { return ier; }

  // Skip this check for efficiency
  //
  // ier = CheckParticleSpecies(modelComputeArguments, particleSpeciesCodes);
  // if (ier) return ier;

#include "ANNImplementationComputeDispatch.cpp"

  return ier;
}

//******************************************************************************
int ANNImplementation::ComputeArgumentsCreate(
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate) const
{
  int ier;

  ier = RegisterKIMComputeArgumentsSettings(modelComputeArgumentsCreate);
  if (ier) { return ier; }

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int ANNImplementation::ComputeArgumentsDestroy(
    KIM::ModelComputeArgumentsDestroy * const modelComputeArgumentsDestroy)
    const
{
  int ier;

  (void) modelComputeArgumentsDestroy;  // avoid not used warning

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}

//==============================================================================
//
// Implementation of ANNImplementation private member functions
//
//==============================================================================

//******************************************************************************
void ANNImplementation::AllocatePrivateParameterMemory()
{
  // nothing to do for this case
}

//******************************************************************************
void ANNImplementation::AllocateParameterMemory()
{
  // nothing to do for this case
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
int ANNImplementation::OpenParameterFiles(
    KIM::ModelDriverCreate * const modelDriverCreate,
    int const numberParameterFiles,
    FILE * parameterFilePointers[MAX_PARAMETER_FILES])
{
  int ier;

  if (numberParameterFiles > MAX_PARAMETER_FILES)
  {
    ier = true;
    LOG_ERROR("ANN given too many parameter files");
    return ier;
  }

  for (int i = 0; i < numberParameterFiles; ++i)
  {
    std::string const * paramFileName;
    ier = modelDriverCreate->GetParameterFileName(i, &paramFileName);
    if (ier)
    {
      LOG_ERROR("Unable to get parameter file name");
      return ier;
    }

    parameterFilePointers[i] = fopen(paramFileName->c_str(), "r");
    if (parameterFilePointers[i] == 0)
    {
      char message[MAXLINE];
      sprintf(message, "ANN parameter file number %d cannot be opened", i);
      ier = true;
      LOG_ERROR(message);
      for (int j = i - 1; i <= 0; --i) { fclose(parameterFilePointers[j]); }
      return ier;
    }
  }

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
int ANNImplementation::ProcessParameterFiles(
    KIM::ModelDriverCreate * const modelDriverCreate,
    int const numberParameterFiles,
    FILE * const parameterFilePointers[MAX_PARAMETER_FILES])
{
  (void) numberParameterFiles;  // avoid not used warning

  int ier;
  char errorMsg[1024];

  //#######################
  // descriptor params
  //#######################
  ier = descriptor_->read_parameter_file(parameterFilePointers[0]);
  if (ier)
  {
    sprintf(errorMsg, "unable to read descriptor parameter file\n");
    LOG_ERROR(errorMsg);
    return true;
  }

  // set species
  int Nspecies = descriptor_->get_num_species();
  std::vector<std::string> species;
  descriptor_->get_species(species);
  for (int i = 0; i < Nspecies; i++)
  {
    KIM::SpeciesName const specName(species[i]);
    if (!specName.Known())
    {
      sprintf(errorMsg, "get unknown species\n");
      LOG_ERROR(errorMsg);
      return true;
    }
    ier = modelDriverCreate->SetSpeciesCode(specName, i);
    if (ier) { return ier; }
  }

  //#######################
  // model parameters
  //#######################

  int desc_size = descriptor_->get_num_descriptors();
  ier = network_->read_parameter_file(parameterFilePointers[1], desc_size);
  if (ier)
  {
    sprintf(errorMsg, "unable to read neural network parameter file\n");
    LOG_ERROR(errorMsg);
    return true;
  }

  //#######################
  // read dropout binary
  //#######################
  ier = network_->read_dropout_file(parameterFilePointers[2]);
  if (ier)
  {
    sprintf(errorMsg, "unable to read dropout file\n");
    LOG_ERROR(errorMsg);
    return true;
  }
  ensemble_size_ = last_ensemble_size_ = network_->get_ensemble_size();
  active_member_id_ = last_active_member_id_ = -1;  // default to average

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
void ANNImplementation::CloseParameterFiles(
    int const numberParameterFiles,
    FILE * const parameterFilePointers[MAX_PARAMETER_FILES])
{
  for (int i = 0; i < numberParameterFiles; ++i)
  { fclose(parameterFilePointers[i]); }
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
int ANNImplementation::ConvertUnits(
    KIM::ModelDriverCreate * const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit)
{
  int ier;

  // define default base units
  KIM::LengthUnit fromLength = KIM::LENGTH_UNIT::A;
  KIM::EnergyUnit fromEnergy = KIM::ENERGY_UNIT::eV;
  KIM::ChargeUnit fromCharge = KIM::CHARGE_UNIT::e;
  KIM::TemperatureUnit fromTemperature = KIM::TEMPERATURE_UNIT::K;
  KIM::TimeUnit fromTime = KIM::TIME_UNIT::ps;

  double convertLength = 1.0;

  ier = modelDriverCreate->ConvertUnit(fromLength,
                                       fromEnergy,
                                       fromCharge,
                                       fromTemperature,
                                       fromTime,
                                       requestedLengthUnit,
                                       requestedEnergyUnit,
                                       requestedChargeUnit,
                                       requestedTemperatureUnit,
                                       requestedTimeUnit,
                                       1.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       &convertLength);
  if (ier)
  {
    LOG_ERROR("Unable to convert length unit");
    return ier;
  }

  double convertEnergy = 1.0;
  ier = modelDriverCreate->ConvertUnit(fromLength,
                                       fromEnergy,
                                       fromCharge,
                                       fromTemperature,
                                       fromTime,
                                       requestedLengthUnit,
                                       requestedEnergyUnit,
                                       requestedChargeUnit,
                                       requestedTemperatureUnit,
                                       requestedTimeUnit,
                                       0.0,
                                       1.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       &convertEnergy);
  if (ier)
  {
    LOG_ERROR("Unable to convert energy unit");
    return ier;
  }

  // convert to active units
  if (convertEnergy != ONE or convertLength != ONE)
  {
    descriptor_->convert_units(convertEnergy, convertLength);
    energyScale_ = convertEnergy;
  }

  // register units
  ier = modelDriverCreate->SetUnits(requestedLengthUnit,
                                    requestedEnergyUnit,
                                    KIM::CHARGE_UNIT::unused,
                                    KIM::TEMPERATURE_UNIT::unused,
                                    KIM::TIME_UNIT::unused);
  if (ier)
  {
    LOG_ERROR("Unable to set units to requested values");
    return ier;
  }

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int ANNImplementation::RegisterKIMModelSettings(
    KIM::ModelDriverCreate * const modelDriverCreate) const
{
  // register numbering
  int error = modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);

  return error;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate
int ANNImplementation::RegisterKIMComputeArgumentsSettings(
    KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate) const
{
  // register arguments
  LOG_INFORMATION("Register argument supportStatus");

  int error = modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialForces,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
                  KIM::SUPPORT_STATUS::optional)
              || modelComputeArgumentsCreate->SetArgumentSupportStatus(
                  KIM::COMPUTE_ARGUMENT_NAME::partialParticleVirial,
                  KIM::SUPPORT_STATUS::optional);

  // register callbacks
  LOG_INFORMATION("Register callback supportStatus");
  error = error
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
              KIM::SUPPORT_STATUS::optional)
          || modelComputeArgumentsCreate->SetCallbackSupportStatus(
              KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
              KIM::SUPPORT_STATUS::optional);

  return error;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate
int ANNImplementation::RegisterKIMParameters(
    KIM::ModelDriverCreate * const modelDriverCreate)
{
  int ier = false;

  // publish parameters (order is important)
  ier = modelDriverCreate->SetParameterPointer(
            1,
            &ensemble_size_,
            "ensemble_size",
            "Size of the ensemble of models. `0` means this is a fully-"
            "connected neural network that does not support running in "
            "ensemble mode.")
        || modelDriverCreate->SetParameterPointer(
            1,
            &active_member_id_,
            "active_member_id",
            "Running mode of the ensemble of models, with available values of "
            "`-1, 0, 1, 2, ..., ensemble_size`. If `ensemble_size = 0`, "
            "this is ignored. Otherwise, `active_member_id = -1` means the "
            "output "
            "(energy, forces, etc.) will be obtained by averaging over all "
            "members of the ensemble (different dropout matrices); "
            "`active_member_id = 0` means the fully-connected network without "
            "dropout will be used; and `active_member_id = i` where i is an "
            "integer from 1 to `ensemble_size` means ensemble member i will be "
            "used to calculate the output.");

  if (ier)
  {
    LOG_ERROR("set_parameters");
    return ier;
  }

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int ANNImplementation::RegisterKIMFunctions(
    KIM::ModelDriverCreate * const modelDriverCreate) const
{
  int error;

  // register functions
  error = modelDriverCreate->SetRoutinePointer(
              KIM::MODEL_ROUTINE_NAME::Destroy,
              KIM::LANGUAGE_NAME::cpp,
              true,
              reinterpret_cast<KIM::Function *>(ANN::Destroy))
          || modelDriverCreate->SetRoutinePointer(
              KIM::MODEL_ROUTINE_NAME::Refresh,
              KIM::LANGUAGE_NAME::cpp,
              true,
              reinterpret_cast<KIM::Function *>(ANN::Refresh))
          || modelDriverCreate->SetRoutinePointer(
              KIM::MODEL_ROUTINE_NAME::Compute,
              KIM::LANGUAGE_NAME::cpp,
              true,
              reinterpret_cast<KIM::Function *>(ANN::Compute))
          || modelDriverCreate->SetRoutinePointer(
              KIM::MODEL_ROUTINE_NAME::ComputeArgumentsCreate,
              KIM::LANGUAGE_NAME::cpp,
              true,
              reinterpret_cast<KIM::Function *>(ANN::ComputeArgumentsCreate))
          || modelDriverCreate->SetRoutinePointer(
              KIM::MODEL_ROUTINE_NAME::ComputeArgumentsDestroy,
              KIM::LANGUAGE_NAME::cpp,
              true,
              reinterpret_cast<KIM::Function *>(ANN::ComputeArgumentsDestroy));

  return error;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelObj
template<class ModelObj>
int ANNImplementation::SetRefreshMutableValues(ModelObj * const modelObj)
{  // use (possibly) new values of parameters to
   // compute other quantities
  // NOTE: This function is templated because it's called with both a
  //       modelDriverCreate object during initialization and with a
  //       modelRefresh object when the Model's parameters have been altered
  int ier = true;

  // checks to make sure ensemble_size_ and active_member_id_ are correct
  if (ensemble_size_ != last_ensemble_size_)
  {
    LOG_ERROR("Value of `ensemble_size` changed.");
    return ier;
  }
  if (active_member_id_ < -1 || active_member_id_ > ensemble_size_)
  {
    char message[MAXLINE];
    sprintf(message,
            "`active_member_id=%d` out of range. Should be [-1, %d]",
            active_member_id_,
            ensemble_size_);
    LOG_ERROR(message);
    return ier;
  }
  if ((last_ensemble_size_ == 0)
      && (active_member_id_ != last_active_member_id_))
  { LOG_INFORMATION("`active_member_id`ignored since `ensemble_size=0`."); }
  last_active_member_id_ = active_member_id_;

  // update influence distance value in KIM API object
  int Nspecies = descriptor_->get_num_species();
  influenceDistance_ = 0.0;
  for (int i = 0; i < Nspecies; i++)
  {
    for (int j = 0; j < Nspecies; j++)
    {
      double cutoff = descriptor_->get_cutoff(i, j);
      if (influenceDistance_ < cutoff) { influenceDistance_ = cutoff; }
    }
  }

  modelObj->SetInfluenceDistancePointer(&influenceDistance_);
  modelObj->SetNeighborListPointers(
      1,
      &influenceDistance_,
      &modelWillNotRequestNeighborsOfNoncontributingParticles_);

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments
int ANNImplementation::SetComputeMutableValues(
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
    VectorOfSizeSix *& particleVirial)
{
  int ier = true;

  // get compute flags
  int compProcess_dEdr;
  int compProcess_d2Edr2;

  modelComputeArguments->IsCallbackPresent(
      KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm, &compProcess_dEdr);
  modelComputeArguments->IsCallbackPresent(
      KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term, &compProcess_d2Edr2);

  isComputeProcess_dEdr = compProcess_dEdr;
  isComputeProcess_d2Edr2 = compProcess_d2Edr2;

  int const * numberOfParticles;
  ier = modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles, &numberOfParticles)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
            &particleSpeciesCodes)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
            &particleContributing)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::coordinates,
            (double const **) &coordinates)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialEnergy, &energy)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialForces,
            (double const **) &forces)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy, &particleEnergy)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
            (double const **) &virial)
        || modelComputeArguments->GetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialParticleVirial,
            (double const **) &particleVirial);
  if (ier)
  {
    LOG_ERROR("GetArgumentPointer");
    return ier;
  }

  isComputeEnergy = (energy != NULL);
  isComputeForces = (forces != NULL);
  isComputeParticleEnergy = (particleEnergy != NULL);
  isComputeVirial = (virial != NULL);
  isComputeParticleVirial = (particleVirial != NULL);

  // update values
  cachedNumberOfParticles_ = *numberOfParticles;

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
// Assume that the particle species interge code starts from 0
#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCompute
int ANNImplementation::CheckParticleSpeciesCodes(
    KIM::ModelCompute const * const modelCompute,
    int const * const particleSpeciesCodes) const
{
  int ier;

  for (int i = 0; i < cachedNumberOfParticles_; ++i)
  {
    if ((particleSpeciesCodes[i] < 0)
        || (particleSpeciesCodes[i] >= descriptor_->get_num_species()))
    {
      ier = true;
      LOG_ERROR("unsupported particle species codes detected");
      return ier;
    }
  }

  // everything is good
  ier = false;
  return ier;
}

//******************************************************************************
int ANNImplementation::GetComputeIndex(
    const bool & isComputeProcess_dEdr,
    const bool & isComputeProcess_d2Edr2,
    const bool & isComputeEnergy,
    const bool & isComputeForces,
    const bool & isComputeParticleEnergy,
    const bool & isComputeVirial,
    const bool & isComputeParticleVirial) const
{
  // const int processdE = 2;
  const int processd2E = 2;
  const int energy = 2;
  const int force = 2;
  const int particleEnergy = 2;
  const int virial = 2;
  const int particleVirial = 2;

  int index = 0;

  // processdE
  index += (int(isComputeProcess_dEdr)) * processd2E * energy * force
           * particleEnergy * virial * particleVirial;

  // processd2E
  index += (int(isComputeProcess_d2Edr2)) * energy * force * particleEnergy
           * virial * particleVirial;

  // energy
  index += (int(isComputeEnergy)) * force * particleEnergy * virial
           * particleVirial;

  // force
  index += (int(isComputeForces)) * particleEnergy * virial * particleVirial;

  // particleEnergy
  index += (int(isComputeParticleEnergy)) * virial * particleVirial;

  // virial
  index += (int(isComputeVirial)) * particleVirial;

  // particleVirial
  index += (int(isComputeParticleVirial));

  return index;
}
