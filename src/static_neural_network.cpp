#include "static_neural_network.h"

#include <iostream>
#include <future>

#include "stm/function.h"
#include "stm/distribution.h"
#include "stm/utilities.h"


template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::StaticNeuralNetwork()
	:_inputData(nullptr), _outputData(nullptr),
	_sampleBatch(0), _trainingSampleCount(0), _epochCount(0), _learningRate(0.0f), _momentum(0),
	_gpuAccelerated(false), _parallelProcessing(false), _multiBatch(false),
	_inputWeights(), _inputBiases(), _layersWeights(), _layersBiases(), _outputWeights(), _outputBiases(),
	_inputWeightsAdjust(), _inputBiasesAdjust(), _layersWeightsAdjust(), _layersBiasesAdjust(), _outputWeightsAdjust(),
	_outputBiasesAdjust(), Sigmoid(stm::logisticf), Sigmoid_Prime(stm::logistic_primef), Cost_Derivative(quadratic_prime)
{
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::~StaticNeuralNetwork()
{
	delete _inputData;
	delete _outputData;
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::SetUpData(unsigned int samples, float* inputData, float* outputData)
{
	_inputData = new Data(_INPUTS, samples, DATA_TYPE::INPUT, inputData);
	_outputData = new Data(_OUTPUTS, samples, DATA_TYPE::OUTPUT, outputData);
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::SetUpInputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _inputData;
	_inputData = new Data(file, DATA_TYPE::INPUT, readfile);
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::SetUpOutputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _outputData;
	_outputData = new Data(file, DATA_TYPE::OUTPUT, readfile);
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::SetUpTrainingConfiguration(unsigned int sampleBatch, unsigned int sampleCount, unsigned int epochs, float learningRate, float momentum)
{
	_sampleBatch = sampleBatch;
	_trainingSampleCount = sampleCount;
	_epochCount = epochs;
	_learningRate = learningRate;
	_momentum = momentum;
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::StartTraining()
{
	InitializeNetwork();
	for (unsigned int i = 0; i < _epochCount; ++i)
	{
		for (unsigned int j = 0; j < _trainingSampleCount / _sampleBatch; ++j)
		{
			std::vector<unsigned int> batch = ShuffleData();
			if (_multiBatch)
				BackPropagate(_inputData->GetSampleBatch(batch), _outputData->GetSampleBatch(batch));
			else
				BackPropagate(stm::toRowMatrix(_inputData->GetSample(j)), stm::toRowMatrix(_outputData->GetSample(j)));
			AdjustNetwork();
		}
	}
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::StopTraining()
{
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
stm::vector<float, _OUTPUTS> StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::ProcessSample(const stm::vector<float, _INPUTS>& inputData) const
{
	auto vec = stm::multiply(_inputWeights, inputData) + _inputBiases;
	vec.ApplyToVector(Sigmoid);

	for (unsigned int i = 0; i < _LAYERS; ++i)
	{
		vec = stm::multiply(_layersWeights, vec) + _layersBiases.GetRowVector(i);
		vec.ApplyToVector(Sigmoid);
	}

	auto out = stm::multiply(_outputWeights, vec) + _outputBiases;
	return out.ApplyToVector(Sigmoid);
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
std::pair<stm::vector<float, _OUTPUTS>, stm::vector<float, _OUTPUTS>> StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::TestSample(unsigned int id)
{
	return std::pair<stm::vector<float, _OUTPUTS>, stm::vector<float, _OUTPUTS>>(_outputData->GetSample(id), ProcessSample(_inputData->GetSample(id)));
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::PrintNetworkValues()
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

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
std::vector<unsigned int> StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::ShuffleData()
{
	std::vector<unsigned int> batch;
	batch.reserve(_sampleBatch);

	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	for (unsigned int i = 0; i < _sampleBatch; ++i)
		batch.push_back(rand() % _trainingSampleCount);
	return batch;
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::InitializeNetwork()
{
	for (unsigned int i = 0; i < _inputWeights.GetSize(); ++i)
		_inputWeights[0][i] = stm::normaldistr_randomf();
	_inputWeights /= sqrtf(_INPUTS);

	for (unsigned int i = 0; i < _inputBiases.GetSize(); ++i)
		_inputBiases[i] = stm::normaldistr_randomf();
	_inputBiases /= sqrtf(_INPUTS);

	for (unsigned int j = 0; j < _LAYERS; ++j)
	{
		for (unsigned int i = 0; i < _layersWeights[j].GetSize(); ++i)
			_layersWeights[j][0][i] = stm::normaldistr_randomf();
		_layersWeights[j] /= sqrtf(_NEURONS);
	}

	for (unsigned int i = 0; i < _layersBiases.GetSize(); ++i)
		_layersBiases[0][i] = stm::normaldistr_randomf();
	_layersBiases /= sqrtf(_NEURONS);

	for (unsigned int i = 0; i < _outputWeights.GetSize(); ++i)
		_outputWeights[0][i] = stm::normaldistr_randomf();
	_outputWeights /= sqrtf(_NEURONS);

	for (unsigned int i = 0; i < _outputBiases.GetSize(); ++i)
		_outputBiases[i] = stm::normaldistr_randomf();
	_outputWeights /= sqrtf(_NEURONS);

}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::BackPropagateBatch(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output)
{
	std::vector<std::future<void>> threads;
	for (unsigned int i = 0; i < _sampleBatch; ++i)
		threads.push_back(std::async(std::launch::async, &StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::BackPropagate, this, input.GetColumnVector(i), output.GetColumnVector(i)));
}

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::BackPropagate(const stm::vector<float, _INPUTS>& input, const stm::vector<float, _OUTPUTS>& output)
{
	stm::matrix<float, _LAYERS + 1, _NEURONS> aValues;
	stm::matrix<float, _LAYERS + 1, _NEURONS> zValues;

	//auto vec = avx::multiply256(_inputWeights, input) + _inputBiases;
	stm::vector<float, _NEURONS> vec = stm::multiply(_inputWeights, input) + _inputBiases;
	for (unsigned int n = 0; n < _sampleBatch; ++n)
		aValues.SetRowVector(0, vec);
	vec.ApplyToVector(Sigmoid);
	for (unsigned int n = 0; n < _sampleBatch; ++n)
		zValues.SetRowVector(0, vec);


	for (unsigned int i = 0; i < _LAYERS; ++i)
	{
		vec = stm::multiply(_layersWeights, vec) + _layersBiases.GetRowVector(i);
		for (unsigned int n = 0; n < _sampleBatch; ++n)
			aValues.SetRowVector(i + 1, vec);
		vec.ApplyToVector(Sigmoid);
		for (unsigned int n = 0; n < _sampleBatch; ++n)
			zValues.SetRowVector(i + 1, vec);
	}

	stm::vector<float, _OUTPUTS> out = stm::multiply(_outputWeights, vec) + _outputBiases;
	stm::vector<float, _OUTPUTS> last_aValue = out;

	out.ApplyToVector(Sigmoid);
	last_aValue.ApplyToVector(Sigmoid_Prime);
	aValues.ApplyToMatrix(Sigmoid_Prime);

	std::array<stm::matrix<float, _NEURONS, _NEURONS>, _LAYERS> layersWeightsAdjust;
	stm::matrix<float, _LAYERS, _NEURONS> layersBiasesAdjust;

	stm::vector<float, _OUTPUTS> outputBiasesAdjust = Cost_Derivative(out, output) * last_aValue;
	stm::matrix<float, _OUTPUTS, _NEURONS> outputWeightsAdjust = stm::multiply(stm::toColumnMatrix(outputBiasesAdjust), stm::toRowMatrix(zValues.GetRowVector(_LAYERS)));

	stm::vector<float, _NEURONS> adjust = stm::multiply(_outputWeights.Transpose(), adjust) * aValues.GetRowVector(_LAYERS);

	layersBiasesAdjust.SetRowVector(_LAYERS - 1, adjust);
	layersWeightsAdjust[_LAYERS - 1] = stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues.GetRowVector(_LAYERS - 1)));

	for (unsigned int i = 1; i < _LAYERS; ++i)
	{
		adjust = stm::multiply(_layersWeights[_LAYERS - i].Transpose(), adjust) * aValues.GetRowVector(_LAYERS - i);
		layersBiasesAdjust.SetRowVector(_LAYERS - 1 - i, adjust);
		layersWeightsAdjust[_LAYERS - 1 - i] = stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues.GetRowVector(_LAYERS- 1 - i)));
	}

	stm::vector<float, _INPUTS> inputBiasesAdjust = stm::multiply(_layersWeights[0].Transpose(), adjust) * aValues.GetRowVector(0);
	stm::matrix<float, _NEURONS, _INPUTS> inputWeightsAdjust = stm::multiply(stm::toColumnMatrix(inputBiasesAdjust), stm::toRowMatrix(input));

	SaveAdjustments(inputWeightsAdjust, inputBiasesAdjust, layersWeightsAdjust, layersBiasesAdjust, outputBiasesAdjust, outputWeightsAdjust);
}

static std::mutex adjustLock;
template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>
::SaveAdjustments(	const stm::matrix<float, _NEURONS, _INPUTS>& inputWeightsAdjust, 
					const stm::vector<float, _NEURONS>& inputBiasesAdjust, 
					const std::array<stm::matrix<float, _NEURONS, _NEURONS>, _LAYERS>& layersWeightsAdjust, 
					const stm::matrix<float, _LAYERS, _NEURONS>& layersBiasesAdjust, 
					const stm::matrix<float, _OUTPUTS, _NEURONS>& outputWeightsAdjust, 
					const stm::vector<float, _OUTPUTS>& outputBiasesAdjust)
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

template<unsigned int _INPUTS, unsigned int _OUTPUTS, unsigned int _LAYERS, unsigned int _NEURONS>
void StaticNeuralNetwork<_INPUTS, _OUTPUTS, _LAYERS, _NEURONS>::AdjustNetwork()
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

