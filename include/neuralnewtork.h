#ifndef neuralnetwork_h
#define neuralnetwork_h

#include <vector>

#include "stm/dynamic_vector.h"
#include "stm/dynamic_matrix.h"

#include "data_handler.h"

class NeuralNetwork
{
public:
	NeuralNetwork();
	~NeuralNetwork();

	void SetUpData(unsigned int samples, unsigned int inputs, unsigned int outputs, float* inputData, float* outputData);
	void SetUpInputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount));
	void SetUpOutputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount));

	void SetUpTrainingConfiguration(unsigned int hiddenLayers, unsigned int neurons, unsigned int sampleBatch, unsigned int sampleCount, unsigned int epochs, float learningRate, float momentum);
	void EnableMultibatch() { _multiBatch = true; }
	void DisableShuffling() { _shuffling = false; }
	void StartTraining();
	void StopTraining();

	using InData_t		= stm::dynamic_vector<float>;
	using OutData_t		= stm::dynamic_vector<float>;
	using InWeight_t	= stm::dynamic_matrix<float>;
	using InBias_t		= stm::dynamic_vector<float>;
	using HidWeight_t	= stm::dynamic_matrix<float>;
	using HidWeights_t	= std::vector<HidWeight_t>;
	using HidBias_t		= stm::dynamic_vector<float>;
	using HidBiases_t	= stm::dynamic_matrix<float>;
	using OutWeight_t	= stm::dynamic_matrix<float>;
	using OutBias_t		= stm::dynamic_vector<float>;
	

	void PrintNetworkValues();


	std::pair<OutData_t, OutData_t> TestSample(unsigned int id);
	OutData_t ProcessSample(const InData_t& inputData) const;

	inline unsigned int GetInputCount()			const { return _inputCount; }
	inline unsigned int GetOutputCount()		const { return _outputCount; }
	inline unsigned int GetHiddenLayerCount()	const { return _layerCount; }
	inline unsigned int GetNeuronCount()		const { return _neuronCount; }

private:
	std::vector<unsigned int> ShuffleData();
	void InitializeNetwork();
	void BackPropagate(const InData_t& input, const OutData_t& output);
	void BackPropagateBatch(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output);
	void AdjustNetwork();

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

	unsigned int _inputCount;
	unsigned int _outputCount;
	unsigned int _layerCount;
	unsigned int _neuronCount;

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

#endif /* neuralnetwork_h */
