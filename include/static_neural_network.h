#ifndef static_neural_network_h
#define static_neural_newtork_h

#include <vector>
#include <array>
#include <string>

#include "data_handler.h"
#include "stm/vector.h"
#include "stm/matrix.h"

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
class StaticNeuralNetwork
{
public:
	StaticNeuralNetwork();
	~StaticNeuralNetwork();

	void SetUpData(unsigned int samples, float* inputData, float* outputData);
	void SetUpInputData	(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount));
	void SetUpOutputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount));

	void SetUpTrainingConfiguration(unsigned int sampleBatch, unsigned int sampleCount, unsigned int epochs, float learningRate, float momentum);
	void EnableMultibatch() { _multiBatch = true; }
	void DisableShuffling() { _shuffling = false; }
	void StartTraining();
	void StopTraining();
	
	using InData_t		= stm::vector<float, _INPUTS>;
	using OutData_t		= stm::vector<float, _OUTPUTS>;
	using InWeight_t	= stm::matrix<float, _NEURONS, _INPUTS>;
	using InBias_t		= stm::vector<float, _NEURONS>;
	using HidWeight_t	= stm::matrix<float, _NEURONS, _NEURONS>;
	using HidWeights_t	= std::array<HidWeight_t, _LAYERS>;
	using HidBias_t		= stm::vector<float, _NEURONS>;
	using HidBiases_t	= stm::matrix<float, _LAYERS, _NEURONS>;
	using OutWeight_t	= stm::matrix<float, _OUTPUTS, _NEURONS>;
	using OutBias_t		= stm::vector<float, _OUTPUTS>;

	std::pair<OutData_t, OutData_t> TestSample(unsigned int id) const;
	std::pair<InData_t, OutData_t> GetSampleData(unsigned int id);
	OutData_t ProcessSample(const InData_t& inputData) const;
	inline OutData_t GetCost(unsigned int id) const { const auto& sample = TestSample(id);  return Cost(sample.first, sample.second); }
	void PrintNetworkValues();

	inline constexpr unsigned int GetInputCount()	const { return _INPUTS; }
	inline constexpr unsigned int GetOutputCount()	const { return _OUTPUTS; }
	inline constexpr unsigned int GetLayerCount()	const { return _LAYERS; }
	inline constexpr unsigned int GetNeuronCount()	const { return _NEURONS; }

private:
	std::vector<unsigned int> ShuffleData();
	void InitializeNetwork();
	void BackPropagate(const InData_t& input, const OutData_t& output);
	void BackPropagateBatch(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output);
	void SaveAdjustments(const InWeight_t& inputWeightsAdjust,
						 const InBias_t& inputBiasesAdjust, 
						 const HidWeights_t& layersWeightsAdjust,
						 const HidBiases_t& layersBiasesAdjust,
						 const OutWeight_t& outputWeightsAdjust,
						 const OutBias_t& outputBiasesAdjust);
	void AdjustNetwork();

	inline static OutData_t quadratic(const OutData_t& calculated, const OutData_t& expected) { return stm::pow(calculated - expected, 2); }
	inline static OutData_t quadratic_prime(const OutData_t& calculated, const OutData_t& expected) { return calculated - expected; }

private:
	Data* _inputData;
	Data* _outputData;

	unsigned int _sampleBatch;
	unsigned int _trainingSampleCount;
	unsigned int _epochCount;
	float _learningRate;
	float _momentum;
	bool _gpuAccelerated;
	bool _parallelProcessing;
	bool _multiBatch;
	bool _shuffling;
	bool _resume;

	InWeight_t		_inputWeights;
	InBias_t		_inputBiases;
	HidWeights_t	_layersWeights;
	HidBiases_t		_layersBiases;
	OutWeight_t		_outputWeights;
	OutBias_t		_outputBiases;

	InWeight_t		_inputWeightsAdjust;
	InBias_t		_inputBiasesAdjust;
	HidWeights_t	_layersWeightsAdjust;
	HidBiases_t		_layersBiasesAdjust;
	OutWeight_t		_outputWeightsAdjust;
	OutBias_t		_outputBiasesAdjust;

	float(*Sigmoid)(float);
	float(*Sigmoid_Prime)(float);
	OutData_t(*Cost)(const OutData_t&, const OutData_t&);
	OutData_t(*Cost_Derivative)(const OutData_t&, const OutData_t&);
};

#include "static_neural_network.ipp"

#endif /* static_neural_network_h */
