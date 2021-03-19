#include <iostream>
#include <future>

#include "stm/function.h"
#include "stm/distribution.h"
#include "stm/utilities.h"
#include "stm/avx_math.h"

#define TEMPLATE_DEFINE template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
#define TEMPLATE_ARGS _INPUTS, _OUTPUTS, _LAYERS, _NEURONS

TEMPLATE_DEFINE
using SNN = StaticNeuralNetwork<TEMPLATE_ARGS>;

TEMPLATE_DEFINE
SNN<TEMPLATE_ARGS>::StaticNeuralNetwork()
	:_inputData(nullptr), _outputData(nullptr),
	_sampleBatch(0), _trainingSampleCount(0), _epochCount(0), _learningRate(1.0f), _momentum(1.0f),
	_gpuAccelerated(false), _parallelProcessing(false), _multiBatch(false), _shuffling(true), _resume(false),
	_inputWeights(), _inputBiases(), _layersWeights(), _layersBiases(), _outputWeights(), _outputBiases(),
	_inputWeightsAdjust(), _inputBiasesAdjust(), _layersWeightsAdjust(), _layersBiasesAdjust(), _outputWeightsAdjust(),
	_outputBiasesAdjust(), Sigmoid(stm::logisticf), Sigmoid_Prime(stm::logistic_primef), Cost(quadratic), Cost_Derivative(quadratic_prime)
{
}

TEMPLATE_DEFINE
SNN<TEMPLATE_ARGS>::~StaticNeuralNetwork()
{
	delete _inputData;
	delete _outputData;
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::SetUpData(unsigned int samples, float* inputData, float* outputData)
{
	_inputData = new Data(_INPUTS, samples, DATA_TYPE::INPUT, inputData);
	_outputData = new Data(_OUTPUTS, samples, DATA_TYPE::OUTPUT, outputData);
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::SetUpInputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _inputData;
	_inputData = new Data(file, DATA_TYPE::INPUT, readfile);
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::SetUpOutputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _outputData;
	_outputData = new Data(file, DATA_TYPE::OUTPUT, readfile);
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::SetUpTrainingConfiguration(unsigned int sampleBatch, unsigned int sampleCount, unsigned int epochs, float learningRate, float momentum)
{
	_multiBatch = sampleBatch - 1;
	_sampleBatch = sampleBatch;
	_trainingSampleCount = sampleCount;
	_epochCount = epochs;
	_learningRate = learningRate;
	_momentum = momentum;
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::StartTraining()
{
	if (!_resume)
		InitializeNetwork();

	for (unsigned int i = 0; i < _epochCount; ++i)
	{
		for (unsigned int j = 0; j < _trainingSampleCount / _sampleBatch; ++j)
		{
			if (_multiBatch)
			{
				std::vector<unsigned int> batch = ShuffleData();
				BackPropagateBatch(_inputData->GetSampleBatch(batch), _outputData->GetSampleBatch(batch));
			}
			else
				BackPropagate(_inputData->GetSampleData(j), _outputData->GetSampleData(j));
			AdjustNetwork();
		}
	}
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::StopTraining()
{
}

TEMPLATE_DEFINE
typename SNN<TEMPLATE_ARGS>::OutData_t SNN<TEMPLATE_ARGS>::ProcessSample(const InData_t& inputData) const
{
	auto vec = stm::multiply(_inputWeights, inputData) + _inputBiases;
	vec.ApplyToVector(Sigmoid);

	for (unsigned int i = 0; i < _LAYERS; ++i)
	{
		vec = stm::multiply(_layersWeights[i], vec) + _layersBiases.GetRowVector(i);
		vec.ApplyToVector(Sigmoid);
	}

	auto out = stm::multiply(_outputWeights, vec) + _outputBiases;
	return out.ApplyToVector(Sigmoid);
}

TEMPLATE_DEFINE
inline std::pair<typename SNN<TEMPLATE_ARGS>::OutData_t, typename SNN<TEMPLATE_ARGS>::OutData_t> SNN<TEMPLATE_ARGS>::TestSample(unsigned int id) const
{
	return std::pair<OutData_t, OutData_t>(ProcessSample(InData_t(_inputData->GetSampleData(id))), OutData_t(_outputData->GetSampleData(id)));
}

TEMPLATE_DEFINE
inline std::pair<typename SNN<TEMPLATE_ARGS>::InData_t, typename SNN<TEMPLATE_ARGS>::OutData_t> SNN<TEMPLATE_ARGS>::GetSampleData(unsigned int id)
{
	return std::pair<InData_t, OutData_t>(InData_t(_inputData->GetSampleData(id)), OutData_t(_outputData->GetSampleData(id)));
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::PrintNetworkValues()
{
	std::cout << "\n\ninputweights\n";
	stm::Print(_inputWeights);

	std::cout << "\n\ninputbiases\n";
	stm::Print(_inputBiases);

	for (const auto& layer : _layersWeights)
	{
		std::cout << "\n\nlayerweights\n";
		stm::Print(layer);
	}

	std::cout << "\n\nlayerbiases\n";
	stm::Print(_layersBiases);

	std::cout << "\n\noutputweights\n";
	stm::Print(_outputWeights);

	std::cout << "\n\noutputbiases\n";
	stm::Print(_outputBiases);
}

TEMPLATE_DEFINE
std::vector<unsigned int> SNN<TEMPLATE_ARGS>::ShuffleData()
{
	std::vector<unsigned int> batch;
	batch.reserve(_sampleBatch);

	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	for (unsigned int i = 0; i < _sampleBatch; ++i)
		batch.push_back(rand() % _trainingSampleCount);
	return batch;
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::InitializeNetwork()
{
	for (unsigned int i = 0; i < _inputWeights.GetSize(); ++i)
		_inputWeights[0][i] = stm::normaldistr_randomf();
	//_inputWeights /= sqrtf(_INPUTS);

	//for (unsigned int i = 0; i < _inputBiases.GetSize(); ++i)
	//	_inputBiases[i] = stm::normaldistr_randomf();
	//_inputBiases /= sqrtf(_INPUTS);

	for (unsigned int j = 0; j < _LAYERS; ++j)
	{
		for (unsigned int i = 0; i < _layersWeights[j].GetSize(); ++i)
			_layersWeights[j][0][i] = stm::normaldistr_randomf();
		//_layersWeights[j] /= sqrtf(_NEURONS);
	}

	for (unsigned int i = 0; i < _layersBiases.GetSize(); ++i)
		_layersBiases[0][i] = stm::normaldistr_randomf();
	//_layersBiases /= sqrtf(_NEURONS);

	for (unsigned int i = 0; i < _outputWeights.GetSize(); ++i)
		_outputWeights[0][i] = stm::normaldistr_randomf();
	//_outputWeights /= sqrtf(_NEURONS);

	for (unsigned int i = 0; i < _outputBiases.GetSize(); ++i)
		_outputBiases[i] = stm::normaldistr_randomf();
	//_outputWeights /= sqrtf(_NEURONS);

	_resume = true;
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::BackPropagateBatch(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output)
{
	std::vector<std::future<void>> threads;
	for (unsigned int i = 0; i < _sampleBatch; ++i)
		threads.push_back(std::async(std::launch::async, &StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::BackPropagate, this, InData_t(input.GetColumnVector(i).GetData()), OutData_t(output.GetColumnVector(i).GetData())));
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::BackPropagate(const InData_t& input, const OutData_t& output)
{
	stm::matrix<float, _LAYERS + 1, _NEURONS> aValues;
	stm::matrix<float, _LAYERS + 1, _NEURONS> zValues;

	stm::vector<float, _NEURONS> vec = stm::avx::multiply256(_inputWeights, input) + _inputBiases;
	//stm::vector<float, _NEURONS> vec = stm::multiply(_inputWeights, input) + _inputBiases;

	for (unsigned int n = 0; n < _sampleBatch; ++n)
		aValues.SetRowVector(0, vec);
	vec.ApplyToVector(Sigmoid);
	for (unsigned int n = 0; n < _sampleBatch; ++n)
		zValues.SetRowVector(0, vec);


	for (unsigned int i = 0; i < _LAYERS; ++i)
	{
		vec = stm::multiply(_layersWeights[i], vec) + _layersBiases.GetRowVector(i);
		for (unsigned int n = 0; n < _sampleBatch; ++n)
			aValues.SetRowVector(i + 1, vec);
		vec.ApplyToVector(Sigmoid);
		for (unsigned int n = 0; n < _sampleBatch; ++n)
			zValues.SetRowVector(i + 1, vec);
	}

	OutData_t out = stm::multiply(_outputWeights, vec) + _outputBiases;
	OutData_t last_aValue = out;

	out.ApplyToVector(Sigmoid);
	last_aValue.ApplyToVector(Sigmoid_Prime);
	aValues.ApplyToMatrix(Sigmoid_Prime);


	OutBias_t outputBiasesAdjust = Cost_Derivative(out, output) * last_aValue;
	OutWeight_t outputWeightsAdjust = stm::multiply(stm::toColumnMatrix(outputBiasesAdjust), stm::toRowMatrix(zValues.GetRowVector(_LAYERS)));

	HidBiases_t layersBiasesAdjust;
	HidWeights_t layersWeightsAdjust;
	HidBias_t adjust = stm::multiply(_outputWeights.Transpose(), outputBiasesAdjust) * aValues.GetRowVector(_LAYERS) * _momentum;

	layersBiasesAdjust.SetRowVector(_LAYERS - 1, adjust);
	layersWeightsAdjust[_LAYERS - 1] = stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues.GetRowVector(_LAYERS - 1)));

	for (unsigned int i = 1; i < _LAYERS; ++i)
	{
		adjust = stm::multiply(_layersWeights[_LAYERS - i].Transpose(), adjust) * aValues.GetRowVector(_LAYERS - i) * _momentum;
		layersBiasesAdjust.SetRowVector(_LAYERS - 1 - i, adjust);
		layersWeightsAdjust[_LAYERS - 1 - i] = stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues.GetRowVector(_LAYERS - 1 - i)));
	}

	InBias_t inputBiasesAdjust = stm::multiply(_layersWeights[0].Transpose(), adjust) * aValues.GetRowVector(0) * _momentum;
	InWeight_t inputWeightsAdjust = stm::multiply(stm::toColumnMatrix(inputBiasesAdjust), stm::toRowMatrix(input));

	SaveAdjustments(inputWeightsAdjust, inputBiasesAdjust, layersWeightsAdjust, layersBiasesAdjust, outputWeightsAdjust, outputBiasesAdjust);
}

static std::mutex adjustLock;

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::SaveAdjustments(const InWeight_t& inputWeightsAdjust,
										 const InBias_t inputBiasesAdjust,
										 const HidWeights_t& layersWeightsAdjust,
										 const HidBiases_t& layersBiasesAdjust,
										 const OutWeight_t& outputWeightsAdjust,
										 const OutBias_t& outputBiasesAdjust)
{
	std::lock_guard<std::mutex> guard(adjustLock);
	_outputBiasesAdjust += outputBiasesAdjust;
	_outputWeightsAdjust += outputWeightsAdjust;
	_layersBiasesAdjust += layersBiasesAdjust;
	for (unsigned int i = 0; i < _LAYERS; ++i)
		_layersWeightsAdjust[i] += layersWeightsAdjust[i];
	_inputBiasesAdjust += inputBiasesAdjust;
	_inputWeightsAdjust += inputWeightsAdjust;
}

TEMPLATE_DEFINE
void SNN<TEMPLATE_ARGS>::AdjustNetwork()
{
	_inputBiases -= _inputBiasesAdjust * (_learningRate / _sampleBatch);
	_inputWeights -= _inputWeightsAdjust * (_learningRate / _sampleBatch);
	_layersBiases -= _layersBiasesAdjust * (_learningRate / _sampleBatch);
	for (unsigned int i = 0; i < _LAYERS; ++i)
		_layersWeights[i] -= _layersWeightsAdjust[i] * (_learningRate / _sampleBatch);
	_outputBiases -= _outputBiasesAdjust * (_learningRate / _sampleBatch);
	_outputWeights -= _outputWeightsAdjust * (_learningRate / _sampleBatch);


	_inputBiasesAdjust.SetAll(0.0f);
	_inputWeightsAdjust.SetAll(0.0f);
	_layersBiasesAdjust.SetAll(0.0f);
	for (unsigned int i = 0; i < _LAYERS; ++i)
		_layersWeightsAdjust[i].SetAll(0.0f);
	_outputBiasesAdjust.SetAll(0.0f);
	_outputWeightsAdjust.SetAll(0.0f);
}