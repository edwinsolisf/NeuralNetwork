#ifndef neuralnetwork_h
#define neuralnetwork_h

#include <vector>
#include <tuple>
#include "stm/dynamic_vector.h"
#include "stm/dynamic_matrix.h"

#include "data_handler.h"

class NeuralNetwork
{
public:
	NeuralNetwork();

	void SetUpInputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount));
	void SetUpOutputtData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount));

	void SetUpTrainingConfiguration(unsigned int hiddenLayers, unsigned int neurons, unsigned int sampleBatch, unsigned int sampleCount, unsigned int epochs, float learningRate, float momentum);

	void StartTraining();
	void StopTraining();

	std::pair<stm::dynamic_vector<float>, stm::dynamic_vector<float>> TestSample(unsigned int id);

	void PrintNetworkValues();

	~NeuralNetwork();

	stm::dynamic_vector<float> ProcessSample(const stm::dynamic_vector<float>& inputData) const;
private:
	void InitializeNetwork();
	stm::dynamic_matrix<float> BackPropagate(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output);
	void AdjustNetwork();

private:
	Data* _inputData;
	Data* _outputData;

	unsigned int _sampleBatch;
	unsigned int _trainingSampleCount;
	unsigned int _epochCount;
	float _learningRate;
	float _momentum;

	unsigned int _inputCount;
	unsigned int _outputCount;
	unsigned int _layerCount;
	unsigned int _neuronCount;

	stm::dynamic_matrix<float> _inputWeights;
	stm::dynamic_vector<float> _inputBiases;
	std::vector<stm::dynamic_matrix<float>> _layersWeights;
	stm::dynamic_matrix<float> _layersBiases;
	stm::dynamic_matrix<float> _outputWeights;
	stm::dynamic_vector<float> _outputBiases;

	stm::dynamic_matrix<float> _inputWeightsAdjust;
	stm::dynamic_vector<float> _inputBiasesAdjust;
	std::vector<stm::dynamic_matrix<float>> _layersWeightsAdjust;
	stm::dynamic_matrix<float> _layersBiasesAdjust;
	stm::dynamic_matrix<float> _outputWeightsAdjust;
	stm::dynamic_vector<float> _outputBiasesAdjust;

	bool _gpuAccelerated;
	bool _parallelProcessing;
	bool _multiBatch;

	float(*Sigmoid)(float);
	float(*Sigmoid_Prime)(float);
	stm::dynamic_vector<float>(*Cost_Derivative)(const stm::dynamic_vector<float>&, const stm::dynamic_vector<float>&);
};

#endif /* neuralnetwork_h */
