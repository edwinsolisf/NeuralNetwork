#include "neuralnewtork.h"

#include <thread>
#include <future>

#include "stm/distribution.h"
#include "stm/function.h"
#include "stm/utilities.h"
#include "avxMath.h"

inline static stm::dynamic_vector<float> quadratic(const stm::dynamic_vector<float>& calculated, const stm::dynamic_vector<float>& expected)
{
	return stm::pow((calculated - expected), 2);
}

inline static stm::dynamic_vector<float> quadratic_prime(const stm::dynamic_vector<float>& calculated, const stm::dynamic_vector<float>& expected)
{
	return calculated - expected;
}

using NN = NeuralNetwork;

NN::NeuralNetwork()
	:_inputData(nullptr), _outputData(nullptr),
	_sampleBatch(0), _trainingSampleCount(0), _epochCount(0), _learningRate(0.0f), _momentum(0),
	_inputCount(0), _outputCount(0), _layerCount(0), _neuronCount(0),
	_inputWeights(1, 1), _inputBiases(1), _layersWeights(), _layersBiases(1, 1), _outputWeights(1, 1), _outputBiases(1),
	_inputWeightsAdjust(1, 1), _inputBiasesAdjust(1), _layersWeightsAdjust(), _layersBiasesAdjust(1, 1), _outputWeightsAdjust(1, 1), 
	_outputBiasesAdjust(1), _gpuAccelerated(false), _parallelProcessing(false), _multiBatch(false), _shuffling(true),
	Sigmoid(stm::logisticf), Sigmoid_Prime(stm::logistic_primef), Cost(quadratic), Cost_Derivative(quadratic_prime)
{
}

void NN::SetUpData(unsigned int samples, unsigned int inputs, unsigned int outputs, float* inputData, float* outputData)
{
	_inputCount = inputs;
	_outputCount = outputs;
	_inputData = new Data(inputs, samples, DATA_TYPE::INPUT, inputData);
	_outputData = new Data(outputs, samples, DATA_TYPE::OUTPUT, outputData);
}

void NN::SetUpInputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _inputData;
	_inputData = new Data(file, DATA_TYPE::INPUT, readfile);
	_inputCount = _inputData->GetSampleSize();
}

void NN::SetUpOutputData(const std::string& file, float* (*readfile)(const char* filePath, unsigned int& sampleSize, unsigned int& sampleCount))
{
	delete _outputData;
	_outputData = new Data(file, DATA_TYPE::OUTPUT, readfile);
	_outputCount = _outputData->GetSampleSize();
}

void NN::SetUpTrainingConfiguration(unsigned int hiddedLayers, unsigned int neurons, unsigned int sampleBatch, unsigned int sampleCount, unsigned int epochs, float learningRate, float momentum)
{
	_layerCount = hiddedLayers;
	_neuronCount = neurons;
	_sampleBatch = sampleBatch;
	_multiBatch = sampleBatch - 1;
	_trainingSampleCount = sampleCount;
	_epochCount = epochs;
	_learningRate = learningRate;
	_momentum = momentum;
}

//void NeuralNetwork::InitializeNetwork()
//{
//	_inputWeights = std::move(stm::dynamic_matrix<float>(_inputCount, _neuronCount));
//	for (unsigned int i = 0; i < _inputWeights.GetSize(); ++i)
//		_inputWeights[0][i] = stm::normaldistr_randomf();
//	_inputWeights /= sqrtf(_inputCount);
//
//	_inputBiases = std::move(stm::dynamic_vector<float>(_neuronCount));
//	for (unsigned int i = 0; i < _inputBiases.GetSize(); ++i)
//		_inputBiases[i] = stm::normaldistr_randomf();
//	_inputBiases /= sqrtf(_inputCount);
//
//	_layersWeights.reserve(_layerCount);
//	for (unsigned int j = 0; j < _layerCount; ++j)
//	{
//		_layersWeights.emplace_back(_neuronCount, _neuronCount);
//		for (unsigned int i = 0; i < _layersWeights[j].GetSize(); ++i)
//			_layersWeights[j][0][i] = stm::normaldistr_randomf();
//		_layersWeights[j] /= sqrtf(_neuronCount);
//	}
//
//	_layersBiases = std::move(stm::dynamic_matrix<float>(_layerCount, _neuronCount));
//	for (unsigned int i = 0; i < _layersBiases.GetSize(); ++i)
//		_layersBiases[0][i] = stm::normaldistr_randomf();
//	_layersBiases /= sqrtf(_neuronCount);
//
//	_outputWeights = std::move(stm::dynamic_matrix<float>(_neuronCount, _outputCount));
//	for (unsigned int i = 0; i < _outputWeights.GetSize(); ++i)
//		_outputWeights[0][i] = stm::normaldistr_randomf();
//	_outputWeights /= sqrtf(_neuronCount);
//
//	_outputBiases = std::move(stm::dynamic_vector<float>(_outputCount));
//	for (unsigned int i = 0; i < _outputBiases.GetSize(); ++i)
//		_outputBiases[i] = stm::normaldistr_randomf();
//	_outputWeights /= sqrtf(_neuronCount);
//
//	_inputWeightsAdjust = std::move(stm::dynamic_matrix<float>(_inputCount, _neuronCount));
//	_inputBiasesAdjust = std::move(stm::dynamic_vector<float>(_neuronCount));
//	_layersWeightsAdjust.reserve(_layerCount);
//	for (unsigned int i = 0; i < _layerCount; ++i)
//		_layersWeightsAdjust.emplace_back(_neuronCount, _neuronCount);
//	_layersBiasesAdjust = std::move(stm::dynamic_matrix<float>(_layerCount, _neuronCount));
//	_outputWeightsAdjust = std::move(stm::dynamic_matrix<float>(_neuronCount, _outputCount));
//	_outputBiasesAdjust = std::move(stm::dynamic_vector<float>(_outputCount));
//}
//
//std::pair<stm::dynamic_vector<float>, stm::dynamic_vector<float>> NeuralNetwork::TestSample(unsigned int id)
//{
//	return std::make_pair(_outputData->GetSample(id), stm::toRowVector(BackPropagate(_inputData->GetSampleBatch(id, 1), _outputData->GetSampleBatch(id, 1))));
//	//return std::make_pair(_outputData->GetSample(id), ProcessSample(_inputData->GetSample(id)));
//}
//
//stm::dynamic_vector<float> NeuralNetwork::ProcessSample(const stm::dynamic_vector<float>& inputData) const
//{
//	//Input Layer
//	stm::dynamic_vector<float> vec = stm::toRowVector(stm::multiply(stm::toRowMatrix(inputData), _inputWeights)) + _inputBiases;
//	vec.ApplyToVector(Sigmoid);
//	
//	//Hidden Layers
//	for (unsigned int i = 0; i < _layerCount; ++i)
//	{
//		vec = stm::multiply(_layersWeights[i], vec) + _layersBiases.GetRowVector(i);
//		vec.ApplyToVector(Sigmoid);
//	}
//
//	//Output Layer
//	stm::dynamic_vector<float> out = stm::toRowVector(stm::multiply(stm::toRowMatrix(vec), _outputWeights)) + _outputBiases;
//	out.ApplyToVector(Sigmoid);
//	return std::move(out);
//}
//
//stm::dynamic_matrix<float> NeuralNetwork::BackPropagate(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output)
//{
//	//auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//	std::vector<stm::dynamic_matrix<float>> aValues(_sampleBatch, stm::dynamic_matrix<float>(_layerCount + 1, _neuronCount));
//	std::vector<stm::dynamic_matrix<float>> zValues(_sampleBatch, stm::dynamic_matrix<float>(_layerCount + 1, _neuronCount));
//	
//	stm::dynamic_matrix<float> biases(_sampleBatch, _neuronCount);
//	biases.SetAllRows(_inputBiases);
//	stm::dynamic_matrix<float> values = stm::multiply(input, _inputWeights) + biases;
//
//	for (unsigned int n = 0; n < _sampleBatch; ++n)
//		aValues[n].SetRowVector(0, values.GetRowVector(n));
//	values.ApplyToMatrix(Sigmoid);
//	for (unsigned int n = 0; n < _sampleBatch; ++n)
//		zValues[n].SetRowVector(0, values.GetRowVector(n));
//
//	for (unsigned int i = 0; i < _layerCount; ++i)
//	{
//		biases.SetAllRows(_layersBiases.GetRowVector(i));
//		values = stm::multiply(values, _layersWeights[i]) + biases;
//
//		for (unsigned int n = 0; n < _sampleBatch; ++n)
//			aValues[n].SetRowVector(i + 1, values.GetRowVector(n));
//		values.ApplyToMatrix(Sigmoid);
//		for (unsigned int n = 0; n < _sampleBatch; ++n)
//			zValues[n].SetRowVector(i + 1, values.GetRowVector(n));
//	}
//
//	biases = stm::dynamic_matrix<float>(_sampleBatch, _outputCount);
//	biases.SetAllRows(_outputBiases);
//	stm::dynamic_matrix<float> out = stm::multiply(values, _outputWeights) + biases;
//	stm::dynamic_matrix<float> last_aValue = out;
//	out.ApplyToMatrix(Sigmoid);
//
//	//auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//	//std::cout << end - start << "ns\n";
//
//	//start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//	for (unsigned int n = 0; n < _sampleBatch; ++n)
//	{
//		stm::dynamic_vector<float> adjust = Cost_Derivative(out.GetRowVector(n), output.GetRowVector(n)) * last_aValue.GetRowVector(n).ApplyToVector(Sigmoid_Prime);
//
//		_outputBiasesAdjust += adjust;
//
//		_outputWeightsAdjust += stm::multiply(stm::toColumnMatrix(zValues[n].GetRowVector(_layerCount)), stm::toRowMatrix(adjust));
//
//		adjust = stm::multiply(_outputWeights, adjust) * aValues[n].GetRowVector(_layerCount).ApplyToVector(Sigmoid_Prime);
//		_layersBiasesAdjust.SetRowVector(_layerCount - 1, adjust + _layersBiasesAdjust.GetRowVector(_layerCount - 1 ));
//		_layersWeightsAdjust[_layerCount - 1] += stm::multiply(stm::toColumnMatrix(zValues[n].GetRowVector(_layerCount - 1)), stm::toRowMatrix(adjust));
//
//		for (unsigned int i = 1; i < _layerCount; ++i)
//		{
//			adjust = stm::multiply(_layersWeights[_layerCount - i], adjust) * aValues[n].GetRowVector(_layerCount - i).ApplyToVector(Sigmoid_Prime);
//			_layersBiasesAdjust.SetRowVector(_layerCount - 1 - i, adjust + _layersBiasesAdjust.GetRowVector(_layerCount - 1 - i));
//			_layersWeightsAdjust[_layerCount - 1 - i] += stm::multiply(stm::toColumnMatrix(zValues[n].GetRowVector(_layerCount - 1 - i)), stm::toRowMatrix(adjust));
//		}
//
//		adjust = stm::multiply(_layersWeights[0], adjust) * aValues[n].GetRowVector(0).ApplyToVector(Sigmoid_Prime);
//		_inputBiasesAdjust += adjust;
//		_inputWeightsAdjust += stm::multiply(stm::toColumnMatrix(input.GetRowVector(n)), stm::toRowMatrix(adjust));
//	}
//	//end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//	//std::cout << end - start << "ns\n";
//	return out;
//}

void NN::InitializeNetwork()
{
	_inputWeights = InWeight_t(_neuronCount, _inputCount);
	for (unsigned int i = 0; i < _inputWeights.GetSize(); ++i)
		_inputWeights[0][i] = stm::normaldistr_randomf();
	_inputWeights /= sqrtf(_inputCount);

	_inputBiases = InBias_t(_neuronCount);
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

	_layersBiases = HidBiases_t(_layerCount, _neuronCount);
	for (unsigned int i = 0; i < _layersBiases.GetSize(); ++i)
		_layersBiases[0][i] = stm::normaldistr_randomf();
	_layersBiases /= sqrtf(_neuronCount);

	_outputWeights = OutWeight_t(_outputCount, _neuronCount);
	for (unsigned int i = 0; i < _outputWeights.GetSize(); ++i)
		_outputWeights[0][i] = stm::normaldistr_randomf();
	_outputWeights /= sqrtf(_neuronCount);

	_outputBiases = OutBias_t(_outputCount);
	for (unsigned int i = 0; i < _outputBiases.GetSize(); ++i)
		_outputBiases[i] = stm::normaldistr_randomf();
	_outputWeights /= sqrtf(_neuronCount);

	_inputWeightsAdjust = InWeight_t(_neuronCount, _inputCount);
	_inputBiasesAdjust = InBias_t(_neuronCount);
	_layersWeightsAdjust.reserve(_layerCount);
	for (unsigned int i = 0; i < _layerCount; ++i)
		_layersWeightsAdjust.emplace_back(_neuronCount, _neuronCount);
	_layersBiasesAdjust = HidBiases_t(_layerCount, _neuronCount);
	_outputWeightsAdjust = OutWeight_t(_outputCount, _neuronCount);
	_outputBiasesAdjust = OutBias_t(_outputCount);
}

std::pair<NN::OutData_t, NN::OutData_t> NN::TestSample(unsigned int id)
{
	return std::make_pair(_outputData->GetSample(id), ProcessSample(_inputData->GetSample(id)));
}

NN::OutData_t NN::ProcessSample(const InData_t& inputData) const
{
	//Input Layer
	auto vec = stm::multiply(_inputWeights, inputData) + _inputBiases;
	vec.ApplyToVector(Sigmoid);

	//Hidden Layers
	for (unsigned int i = 0; i < _layerCount; ++i)
	{
		vec = stm::multiply(_layersWeights[i], vec) + _layersBiases.GetRowVector(i);
		vec.ApplyToVector(Sigmoid);
	}

	//Output Layer
	auto out = stm::multiply(_outputWeights, vec) + _outputBiases;
	out.ApplyToVector(Sigmoid);
	return std::move(out);
}

static std::mutex adjustLock;

/*void NeuralNetwork::BackPropagate(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output)
{
	std::vector<stm::dynamic_matrix<float>> aValues(_sampleBatch, stm::dynamic_matrix<float>(_layerCount + 1, _neuronCount));
	std::vector<stm::dynamic_matrix<float>> zValues(_sampleBatch, stm::dynamic_matrix<float>(_layerCount + 1, _neuronCount));

	stm::dynamic_matrix<float> biases(_neuronCount, _sampleBatch);
	biases.SetAllColumns(_inputBiases);
	stm::dynamic_matrix<float> values = stm::multiply(_inputWeights, input) + biases;

	for (unsigned int n = 0; n < _sampleBatch; ++n)
		aValues[n].SetRowVector(0, values.GetColumnVector(n));
	values.ApplyToMatrix(Sigmoid);
	for (unsigned int n = 0; n < _sampleBatch; ++n)
		zValues[n].SetRowVector(0, values.GetColumnVector(n));

	for (unsigned int i = 0; i < _layerCount; ++i)
	{
		biases.SetAllColumns(_layersBiases.GetRowVector(i));
		values = stm::multiply(_layersWeights[i], values) + biases;

		for (unsigned int n = 0; n < _sampleBatch; ++n)
			aValues[n].SetRowVector(i + 1, values.GetColumnVector(n));
		values.ApplyToMatrix(Sigmoid);
		for (unsigned int n = 0; n < _sampleBatch; ++n)
			zValues[n].SetRowVector(i + 1, values.GetColumnVector(n));
	}

	biases = stm::dynamic_matrix<float>(_outputCount, _sampleBatch);
	biases.SetAllColumns(_outputBiases);
	stm::dynamic_matrix<float> out = stm::multiply(_outputWeights, values) + biases;
	stm::dynamic_matrix<float> last_aValue = out;
	out.ApplyToMatrix(Sigmoid);
	
	auto func = [&](unsigned int n)
	{
		stm::dynamic_matrix<float> inputWeightsAdjust = stm::dynamic_matrix<float>(_neuronCount, _inputCount);
		stm::dynamic_vector<float> inputBiasesAdjust = stm::dynamic_vector<float>(_neuronCount);
		std::vector<stm::dynamic_matrix<float>> layersWeightsAdjust(_layerCount, stm::dynamic_matrix<float>(_neuronCount, _neuronCount));
		stm::dynamic_matrix<float> layersBiasesAdjust = stm::dynamic_matrix<float>(_layerCount, _neuronCount);
		stm::dynamic_matrix<float> outputWeightsAdjust = stm::dynamic_matrix<float>(_outputCount, _neuronCount);
		stm::dynamic_vector<float> outputBiasesAdjust = stm::dynamic_vector<float>(_outputCount);

		stm::dynamic_vector<float> adjust = Cost_Derivative(out.GetColumnVector(n), output.GetColumnVector(n)) * last_aValue.GetColumnVector(n).ApplyToVector(Sigmoid_Prime);

		outputBiasesAdjust += adjust;

		outputWeightsAdjust += stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues[n].GetRowVector(_layerCount)));

		adjust = stm::multiply(_outputWeights.Transpose(), adjust) * aValues[n].GetRowVector(_layerCount).ApplyToVector(Sigmoid_Prime);
		layersBiasesAdjust.SetRowVector(_layerCount - 1, adjust + layersBiasesAdjust.GetRowVector(_layerCount - 1));
		layersWeightsAdjust[_layerCount - 1] += stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues[n].GetRowVector(_layerCount - 1)));

		for (unsigned int i = 1; i < _layerCount; ++i)
		{
			adjust = stm::multiply(_layersWeights[_layerCount - i].Transpose(), adjust) * aValues[n].GetRowVector(_layerCount - i).ApplyToVector(Sigmoid_Prime);
			layersBiasesAdjust.SetRowVector(_layerCount - 1 - i, adjust + layersBiasesAdjust.GetRowVector(_layerCount - 1 - i));
			layersWeightsAdjust[_layerCount - 1 - i] += stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues[n].GetRowVector(_layerCount - 1 - i)));
		}

		adjust = stm::multiply(_layersWeights[0].Transpose(), adjust) * aValues[n].GetRowVector(0).ApplyToVector(Sigmoid_Prime);
		inputBiasesAdjust += adjust;
		inputWeightsAdjust += stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(input.GetColumnVector(n)));

		std::lock_guard<std::mutex> guard(adjustLock);
		_outputBiasesAdjust += outputBiasesAdjust;
		_outputWeightsAdjust += outputWeightsAdjust;
		_layersBiasesAdjust += layersBiasesAdjust;
		for (unsigned int i = 0; i < _layerCount; ++i)
			_layersWeightsAdjust[i] += layersWeightsAdjust[i];
		_inputBiasesAdjust += inputBiasesAdjust;
		_inputWeightsAdjust += inputWeightsAdjust;
	};

	std::vector<std::future<void>> threads;
	for(unsigned int i = 0; i < _sampleBatch; ++i)
		threads.push_back(std::async(std::launch::async, func, i));
}*/

void NN::BackPropagateBatch(const stm::dynamic_matrix<float>& input, const stm::dynamic_matrix<float>& output)
{
	std::vector<std::future<void>> threads;
	for (unsigned int i = 0; i < _sampleBatch; ++i)
		threads.push_back(std::async(std::launch::async, &NN::BackPropagate, this, input.GetColumnVector(i), output.GetColumnVector(i)));
}

void NN::BackPropagate(const InData_t& input, const OutData_t& output)
{
	stm::dynamic_matrix<float> aValues(_layerCount + 1, _neuronCount);
	stm::dynamic_matrix<float> zValues(_layerCount + 1, _neuronCount);

	stm::dynamic_vector<float> values = avx::multiply256(_inputWeights, input) + _inputBiases;

	for (unsigned int n = 0; n < _sampleBatch; ++n)
		aValues.SetRowVector(0, values);
	values.ApplyToVector(Sigmoid);
	for (unsigned int n = 0; n < _sampleBatch; ++n)
		zValues.SetRowVector(0, values);

	for (unsigned int i = 0; i < _layerCount; ++i)
	{
		values = stm::multiply(_layersWeights[i], values) + _layersBiases.GetRowVector(i);

		for (unsigned int n = 0; n < _sampleBatch; ++n)
			aValues.SetRowVector(i + 1, values);
		values.ApplyToVector(Sigmoid);
		for (unsigned int n = 0; n < _sampleBatch; ++n)
			zValues.SetRowVector(i + 1, values);
	}

	OutData_t out = stm::multiply(_outputWeights, values) + _outputBiases;
	OutData_t last_aValue = out;

	out.ApplyToVector(Sigmoid);
	last_aValue.ApplyToVector(Sigmoid_Prime);
	aValues.ApplyToMatrix(Sigmoid_Prime);

	OutBias_t outputBiasesAdjust = Cost_Derivative(out, output) * last_aValue;
	OutWeight_t outputWeightsAdjust = stm::multiply(stm::toColumnMatrix(outputBiasesAdjust), stm::toRowMatrix(zValues.GetRowVector(_layerCount)));

	HidWeights_t layersWeightsAdjust(_layerCount, HidWeight_t(_neuronCount, _neuronCount));
	HidBiases_t layersBiasesAdjust = HidBiases_t(_layerCount, _neuronCount);
	HidBias_t adjust = stm::multiply(_outputWeights.Transpose(), outputBiasesAdjust) * aValues.GetRowVector(_layerCount);

	layersBiasesAdjust.SetRowVector(_layerCount - 1, adjust);
	layersWeightsAdjust[_layerCount - 1] = stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues.GetRowVector(_layerCount - 1)));

	for (unsigned int i = 1; i < _layerCount; ++i)
	{
		adjust = stm::multiply(_layersWeights[_layerCount - i].Transpose(), adjust) * aValues.GetRowVector(_layerCount - i);
		layersBiasesAdjust.SetRowVector(_layerCount - 1 - i, adjust);
		layersWeightsAdjust[_layerCount - 1 - i] = stm::multiply(stm::toColumnMatrix(adjust), stm::toRowMatrix(zValues.GetRowVector(_layerCount - 1 - i)));
	}

	adjust = stm::multiply(_layersWeights[0].Transpose(), adjust) * aValues.GetRowVector(0);
	InBias_t inputBiasesAdjust = std::move(adjust);
	InWeight_t inputWeightsAdjust = stm::multiply(stm::toColumnMatrix(inputBiasesAdjust), stm::toRowMatrix(input));

	std::lock_guard<std::mutex> guard(adjustLock);
	_outputBiasesAdjust += outputBiasesAdjust;
	_outputWeightsAdjust += outputWeightsAdjust;
	_layersBiasesAdjust += layersBiasesAdjust;
	for (unsigned int i = 0; i < _layerCount; ++i)
		_layersWeightsAdjust[i] += layersWeightsAdjust[i];
	_inputBiasesAdjust += inputBiasesAdjust;
	_inputWeightsAdjust += inputWeightsAdjust;
}


void NN::AdjustNetwork()
{
	_inputBiases -= _inputBiasesAdjust * (_learningRate / _sampleBatch);
	_inputWeights -= _inputWeightsAdjust * (_learningRate / _sampleBatch);
	_layersBiases -= _layersBiasesAdjust * (_learningRate / _sampleBatch);
	for (unsigned int i = 0; i < _layerCount; ++i)
		_layersWeights[i] -= _layersWeightsAdjust[i] * (_learningRate / _sampleBatch);
	_outputBiases -= _outputBiasesAdjust * (_learningRate / _sampleBatch);
	_outputWeights -= _outputWeightsAdjust * (_learningRate / _sampleBatch);

	_inputBiasesAdjust.SetAll(0.0f);
	_inputWeightsAdjust.SetAll(0.0f);
	_layersBiasesAdjust.SetAll(0.0f);
	for (unsigned int i = 0; i < _layerCount; ++i)
		_layersWeightsAdjust[i].SetAll(0.0f);
	_outputBiasesAdjust.SetAll(0.0f);
	_outputWeightsAdjust.SetAll(0.0f);
}

void NN::StartTraining()
{
	InitializeNetwork();
	for (unsigned int i = 0; i < _epochCount; ++i)
	{
		for (unsigned int j = 0; j < _trainingSampleCount / _sampleBatch; ++j)
		{
			std::vector<unsigned int> batch = ShuffleData();
			if (_multiBatch)
				BackPropagateBatch(_inputData->GetSampleBatch(batch), _outputData->GetSampleBatch(batch));
			else
				BackPropagate(_inputData->GetSample(j), _outputData->GetSample(j));
			AdjustNetwork();
		}
	}
}

std::vector<unsigned int> NN::ShuffleData()
{
	std::vector<unsigned int> batch;
	batch.reserve(_sampleBatch);

	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	for (unsigned int i = 0; i < _sampleBatch; ++i)
		batch.push_back(rand() % _trainingSampleCount);
	return batch;
}

//void NeuralNetwork::ShuffleData()
//{
//	unsigned int count = _inputData->GetSampleCount(), inputSize = _inputData->GetSampleSize(), outputSize = _outputData->GetSampleSize();
//
//	std::default_random_engine rng;
//	std::vector<unsigned int> vec(count);
//	for (unsigned int i = 0; i < vec.size(); ++i)
//		vec[i] = i;
//	std::shuffle(vec.begin(), vec.end(), rng);
//
//	float* in = new float[count * inputSize];
//	float* out = new float[count * outputSize];
//	for (unsigned int i = 0; i < vec.size(); ++i)
//	{
//		memcpy(&in[i * inputSize], _inputData->GetSample(vec[i]).GetData(), sizeof(float) * inputSize);
//		memcpy(&out[i * outputSize], _outputData->GetSample(vec[i]).GetData(), sizeof(float) * outputSize);
//	}
//
//	_inputData->SetNewData(in);
//	_outputData->SetNewData(out);
//	//delete[] in;
//	//delete[] out;
//}

void NN::PrintNetworkValues()
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

NN::~NeuralNetwork()
{
	delete _inputData;
	delete _outputData;
}