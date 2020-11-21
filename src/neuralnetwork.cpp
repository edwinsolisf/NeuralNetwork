#include "neuralnewtork.h"

#include "stm/distribution.h"
#include "stm/function.h"
#include "stm/utilities.h"

stm::dynamic_vector<float> quadratic_prime(const stm::dynamic_vector<float>& calculated, const stm::dynamic_vector<float>& expected)
{
	return std::move(calculated - expected);
}

NeuralNetwork::NeuralNetwork()
	:_inputData(nullptr), _outputData(nullptr),
	_sampleBatch(0), _trainingSampleCount(0), _epochCount(0), _learningRate(0.0f), _momentum(0),
	_inputCount(0), _outputCount(0), _layerCount(0), _neuronCount(0),
	_inputWeights(1, 1), _inputBiases(1), _layersWeights(), _layersBiases(1, 1), _outputWeights(1, 1), _outputBiases(1),
	_inputWeightsAdjust(1, 1), _inputBiasesAdjust(1), _layersWeightsAdjust(), _layersBiasesAdjust(1, 1), _outputWeightsAdjust(1, 1), 
	_outputBiasesAdjust(1), _gpuAccelerated(false), _parallelProcessing(false), _multiBatch(false), Sigmoid(stm::logisticf), 
	Sigmoid_Prime(stm::logistic_primef), Cost_Derivative(quadratic_prime)
{
}

void NeuralNetwork::SetUpInputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _inputData;
	_inputData = new Data(file, DATA_TYPE::INPUT, readfile);
	_inputCount = _inputData->GetSampleSize();
}

void NeuralNetwork::SetUpOutputtData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _outputData;
	_outputData = new Data(file, DATA_TYPE::OUTPUT, readfile);
	_outputCount = _outputData->GetSampleSize();
}

void NeuralNetwork::SetUpTrainingConfiguration(unsigned int hiddedLayers, unsigned int neurons, unsigned int sampleBatch, unsigned int sampleCount, unsigned int epochs, float learningRate, float momentum)
{
	_layerCount = hiddedLayers;
	_neuronCount = neurons;
	_sampleBatch = sampleBatch;
	_trainingSampleCount = sampleCount;
	_epochCount = epochs;
	_learningRate = learningRate;
	_momentum = momentum;
}

void NeuralNetwork::InitializeNetwork()
{
	_inputWeights = std::move(stm::dynamic_matrix<float>(_inputCount, _neuronCount));
	for (unsigned int i = 0; i < _inputWeights.GetSize(); ++i)
		_inputWeights[0][i] = stm::normaldistr_randomf();
	_inputWeights /= sqrtf(_inputCount);

	_inputBiases = std::move(stm::dynamic_vector<float>(_neuronCount));
	for (unsigned int i = 0; i < _inputBiases.GetSize(); ++i)
		_inputBiases[i] = stm::normaldistr_randomf();
	_inputBiases /= sqrtf(_inputCount);

	_layersWeights.reserve(_layerCount);
	for (unsigned int j = 0; j < _layerCount; ++j)
	{
		_layersWeights.emplace_back(_neuronCount, _neuronCount);
		for (unsigned int i = 0; i < _layersWeights[j].GetSize(); ++i)
			_layersWeights[j][0][i] = stm::normaldistr_randomf();
		_layersWeights[j] /= sqrtf(_neuronCount);
	}

	_layersBiases = std::move(stm::dynamic_matrix<float>(_layerCount, _neuronCount));
	for (unsigned int i = 0; i < _layersBiases.GetSize(); ++i)
		_layersBiases[0][i] = stm::normaldistr_randomf();
	_layersBiases /= sqrtf(_neuronCount);

	_outputWeights = std::move(stm::dynamic_matrix<float>(_neuronCount, _outputCount));
	for (unsigned int i = 0; i < _outputWeights.GetSize(); ++i)
		_outputWeights[0][i] = stm::normaldistr_randomf();
	_outputWeights /= sqrtf(_neuronCount);

	_outputBiases = std::move(stm::dynamic_vector<float>(_outputCount));
	for (unsigned int i = 0; i < _outputBiases.GetSize(); ++i)
		_outputBiases[i] = stm::normaldistr_randomf();
	_outputWeights /= sqrtf(_neuronCount);

	_inputWeightsAdjust = std::move(stm::dynamic_matrix<float>(_inputCount, _neuronCount));
	_inputBiasesAdjust = std::move(stm::dynamic_vector<float>(_neuronCount));
	_layersWeightsAdjust.reserve(_layerCount);
	for (unsigned int i = 0; i < _layerCount; ++i)
		_layersWeightsAdjust.emplace_back(_neuronCount, _neuronCount);
	_layersBiasesAdjust = std::move(stm::dynamic_matrix<float>(_layerCount, _neuronCount));
	_outputWeightsAdjust = std::move(stm::dynamic_matrix<float>(_neuronCount, _outputCount));
	_outputBiasesAdjust = std::move(stm::dynamic_vector<float>(_outputCount));
}

std::pair<stm::dynamic_vector<float>, stm::dynamic_vector<float>> NeuralNetwork::TestSample(unsigned int id)
{
	return std::make_pair(_outputData->GetSample(id), ProcessSample(_inputData->GetSample(id)));
}

stm::dynamic_vector<float> NeuralNetwork::ProcessSample(const stm::dynamic_vector<float>& inputData) const
{
	//Input Layer
	stm::dynamic_vector<float> vec = stm::toRowVector(stm::multiply(stm::toRowMatrix(inputData), _inputWeights)) + _inputBiases;
	vec.ApplyToVector(Sigmoid);
	
	//Hidden Layers
	for (unsigned int i = 0; i < _layerCount; ++i)
	{
		vec = stm::multiply(_layersWeights[i], vec) + _layersBiases.GetRowVector(i);
		vec.ApplyToVector(Sigmoid);
	}

	//Output Layer
	stm::dynamic_vector<float> out = stm::toRowVector(stm::multiply(stm::toRowMatrix(vec), _outputWeights)) + _outputBiases;
	out.ApplyToVector(Sigmoid);
	return std::move(out);
}

stm::dynamic_matrix<float> NeuralNetwork::BackPropagate(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output)
{
	//auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::vector<stm::dynamic_matrix<float>> aValues(_sampleBatch, stm::dynamic_matrix<float>(_layerCount + 1, _neuronCount));
	std::vector<stm::dynamic_matrix<float>> zValues(_sampleBatch, stm::dynamic_matrix<float>(_layerCount + 1, _neuronCount));
	
	stm::dynamic_matrix<float> biases(_sampleBatch, _neuronCount);
	biases.SetAllRows(_inputBiases);
	stm::dynamic_matrix<float> values = stm::multiply(input, _inputWeights) + biases;

	for (unsigned int n = 0; n < _sampleBatch; ++n)
		aValues[n].SetRowVector(0, values.GetRowVector(n));
	values.ApplyToMatrix(Sigmoid);
	for (unsigned int n = 0; n < _sampleBatch; ++n)
		zValues[n].SetRowVector(0, values.GetRowVector(n));

	for (unsigned int i = 0; i < _layerCount; ++i)
	{
		biases.SetAllRows(_layersBiases.GetRowVector(i));
		values = stm::multiply(values, _layersWeights[i]) + biases;

		for (unsigned int n = 0; n < _sampleBatch; ++n)
			aValues[n].SetRowVector(i + 1, values.GetRowVector(n));
		values.ApplyToMatrix(Sigmoid);
		for (unsigned int n = 0; n < _sampleBatch; ++n)
			zValues[n].SetRowVector(i + 1, values.GetRowVector(n));
	}

	biases = stm::dynamic_matrix<float>(_sampleBatch, _outputCount);
	biases.SetAllRows(_outputBiases);
	stm::dynamic_matrix<float> out = stm::multiply(values, _outputWeights) + biases;
	stm::dynamic_matrix<float> last_aValue = out;
	out.ApplyToMatrix(Sigmoid);

	//auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	//std::cout << end - start << "ns\n";

	//start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	for (unsigned int n = 0; n < _sampleBatch; ++n)
	{
		stm::dynamic_vector<float> adjust = Cost_Derivative(out.GetRowVector(n), output.GetRowVector(n)) * last_aValue.GetRowVector(n).ApplyToVector(Sigmoid_Prime);

		_outputBiasesAdjust += adjust;

		_outputWeightsAdjust += stm::multiply(stm::toColumnMatrix(zValues[n].GetRowVector(_layerCount)), stm::toRowMatrix(adjust));

		adjust = stm::multiply(_outputWeights, adjust) * aValues[n].GetRowVector(_layerCount).ApplyToVector(Sigmoid_Prime);
		_layersBiasesAdjust.SetRowVector(_layerCount - 1, adjust + _layersBiasesAdjust.GetRowVector(_layerCount - 1 ));
		_layersWeightsAdjust[_layerCount - 1] += stm::multiply(stm::toColumnMatrix(zValues[n].GetRowVector(_layerCount - 1)), stm::toRowMatrix(adjust));

		for (unsigned int i = 1; i < _layerCount; ++i)
		{
			adjust = stm::multiply(_layersWeights[_layerCount - i], adjust) * aValues[n].GetRowVector(_layerCount - i).ApplyToVector(Sigmoid_Prime);
			_layersBiasesAdjust.SetRowVector(_layerCount - 1 - i, adjust + _layersBiasesAdjust.GetRowVector(_layerCount - 1 - i));
			_layersWeightsAdjust[_layerCount - 1 - i] += stm::multiply(stm::toColumnMatrix(zValues[n].GetRowVector(_layerCount - 1 - i)), stm::toRowMatrix(adjust));
		}

		adjust = stm::multiply(_layersWeights[0], adjust) * aValues[n].GetRowVector(0).ApplyToVector(Sigmoid_Prime);
		_inputBiasesAdjust += adjust;
		_inputWeightsAdjust += stm::multiply(stm::toColumnMatrix(input.GetRowVector(n)), stm::toRowMatrix(adjust));
	}
	//end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	//std::cout << end - start << "ns\n";
	return out;
}
void NeuralNetwork::AdjustNetwork()
{
	_inputBiases -= _inputBiasesAdjust * _learningRate / _sampleBatch;
	_inputWeights -= _inputWeightsAdjust * _learningRate / _sampleBatch;
	_layersBiases -= _layersBiasesAdjust * _learningRate / _sampleBatch;
	for (unsigned int i = 0; i < _layerCount; ++i)
		_layersWeights[i] -= _layersWeightsAdjust[i] * _learningRate / _sampleBatch;
	_outputBiases -= _outputBiasesAdjust * _learningRate / _sampleBatch;
	_outputWeights -= _outputWeightsAdjust * _learningRate / _sampleBatch;


	_inputBiasesAdjust.ApplyToVector([](float) { return 0.0f; });
	_inputWeightsAdjust.ApplyToMatrix([](float) { return 0.0f; });
	_layersBiasesAdjust.ApplyToMatrix([](float) { return 0.0f; });
	for (unsigned int i = 0; i < _layerCount; ++i)
		_layersWeightsAdjust[i].ApplyToMatrix([](float) { return 0.0f; });
	_outputBiasesAdjust.ApplyToVector([](float) { return 0.0f; });
	_outputWeightsAdjust.ApplyToMatrix([](float) { return 0.0f; });
}

void NeuralNetwork::StartTraining()
{
	InitializeNetwork();
	for (unsigned int i = 0; i < _epochCount; ++i)
	{
		for (unsigned int j = 0; j < _trainingSampleCount / _sampleBatch; ++j)
		{
			if (_multiBatch)
				BackPropagate(_inputData->GetSampleBatch(j, _sampleBatch), _outputData->GetSampleBatch(j, _sampleBatch));
			else
				BackPropagate(stm::toRowMatrix(_inputData->GetSample(j)), stm::toRowMatrix(_outputData->GetSample(j)));
			AdjustNetwork();
		}
	}
}

void NeuralNetwork::PrintNetworkValues()
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

NeuralNetwork::~NeuralNetwork()
{
	delete _inputData;
	delete _outputData;
}