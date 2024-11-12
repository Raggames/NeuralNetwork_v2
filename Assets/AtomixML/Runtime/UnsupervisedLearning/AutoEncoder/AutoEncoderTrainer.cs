using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.NeuralNetwork;
using Atom.MachineLearning.NeuralNetwork.V2;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using static Atom.MachineLearning.Unsupervised.AutoEncoder.AutoEncoderModel;

namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    [Serializable]
    public class AutoEncoderTrainer : IMLTrainer<AutoEncoderModel, NVector, NVector>, IEpochIteratable, ITrainIteratable, IBatchedTrainIteratable
    {
        [SerializeField] private AutoEncoderModel _autoEncoder;
        public AutoEncoderModel trainedModel { get => _autoEncoder; set => _autoEncoder = value; }

        [HyperParameter, SerializeField] private int _epochs = 1000;
        [HyperParameter, SerializeField] private int _batchSize = 10;
        [HyperParameter, SerializeField] private float _learningRate = .05f;
        [HyperParameter, SerializeField] private float _biasRate = 1f;
        [HyperParameter, SerializeField] private float _momentum = .01f;
        [HyperParameter, SerializeField] private float _weightDecay = .0001f;
        [HyperParameter, SerializeField] private AnimationCurve _learningRateCurve;

        [SerializeField] private bool _normalizeDataSet;

        [ShowInInspector, ReadOnly] private int _currentEpoch;
        [ShowInInspector, ReadOnly] private float _currentLearningRate;
        [ShowInInspector, ReadOnly] private float _currentLoss;

        public List<LayerInfo> LayerInfos;

        [Serializable]
        public class LayerInfo
        {
            public double AverageWeight;
            public double AverageBias;
        }

        private NVector[] _x_datas;
        private NVector[] _t_datas;
        private List<NVector> _x_datas_buffer;
        private StandardTrainingSupervisor _epochSupervisor;

        private double _errorSum = 0.0;
        private NVector _outputBuffer;

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            LayerInfos = new List<LayerInfo>();

            foreach (var layer in trainedModel.encoder.Layers)
                LayerInfos.Add(new LayerInfo());
            foreach (var layer in trainedModel.decoder.Layers)
                LayerInfos.Add(new LayerInfo());

            DatasetReader.Split_TrainTest_NVector(x_datas, .8f, out _x_datas, out _t_datas);

            _x_datas_buffer = new List<NVector>();
            _currentLearningRate = _learningRate;

            _outputBuffer = new NVector(trainedModel.tensorDimensions);

            _epochSupervisor = new StandardTrainingSupervisor();
            _epochSupervisor.SetEpochIteration(this);
            _epochSupervisor.SetTrainIteration(this);
            _epochSupervisor.SetTrainBatchIteration(this);
            _epochSupervisor.SetAutosave(_epochs / 100);

            await _epochSupervisor.RunBatchedAsync(_epochs, _x_datas.Length, _batchSize, true);

            // test train ? 
            // accuracy ?
            //ModelSerializer.SaveModel(trainedModel);

            return new TrainingResult();
        }

        public void OnBeforeEpoch(int epochIndex)
        {
            _currentEpoch = epochIndex;

            _errorSum = 0.0;
            _outputBuffer = new NVector(trainedModel.tensorDimensions);
        }

        public void OnTrainNextBatch(int[] batchIndexes)
        {
            foreach(var index in batchIndexes)
            {
                var input = _x_datas[index];

                _outputBuffer = trainedModel.Predict(input);

                // we try to reconstruct the input while autoencoding
                _errorSum += MLCostFunctions.MSE(input, _outputBuffer);

                var error = MLCostFunctions.MSE_Derivative(input, _outputBuffer);
                trainedModel.Backpropagate(error);

                if (index % 10 == 0)
                {
                    int ind = 0;

                    foreach (var layer in trainedModel.encoder.Layers)
                    {
                        LayerInfos[ind].AverageWeight = layer.GetAverageWeights();
                        LayerInfos[ind].AverageBias = layer.GetAverageBias();
                        ind++;
                    }
                    foreach (var layer in trainedModel.decoder.Layers)
                    {
                        LayerInfos[ind].AverageWeight = layer.GetAverageWeights();
                        LayerInfos[ind].AverageBias = layer.GetAverageBias();
                        ind++;
                    }
                }
            }

            trainedModel.AverageGradients(batchIndexes.Length);
            trainedModel.UpdateWeights(_currentLearningRate, _biasRate, _momentum, _weightDecay);

        }

        public void OnTrainNext(int index)
        {
            var _randomIndex = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
            var input = _x_datas_buffer[_randomIndex];
            _x_datas_buffer.RemoveAt(_randomIndex);

            _outputBuffer = trainedModel.Predict(input);

            // we try to reconstruct the input while autoencoding
            _errorSum += MLCostFunctions.MSE(input, _outputBuffer);

            var error = MLCostFunctions.MSE_Derivative(input, _outputBuffer);
            trainedModel.Backpropagate(error);
            trainedModel.UpdateWeights(_currentLearningRate, _biasRate, _momentum, _weightDecay);

            if (index % 10 == 0)
            {
                int ind = 0;

                foreach (var layer in trainedModel.encoder.Layers)
                {
                    LayerInfos[ind].AverageWeight = layer.GetAverageWeights();
                    LayerInfos[ind].AverageBias = layer.GetAverageBias();
                    ind++;
                }
                foreach (var layer in trainedModel.decoder.Layers)
                {
                    LayerInfos[ind].AverageWeight = layer.GetAverageWeights();
                    LayerInfos[ind].AverageBias = layer.GetAverageBias();
                    ind++;
                }
            }
        }

        public void OnAfterEpoch(int epochIndex)
        {
            _currentLoss = (float)_errorSum / _x_datas.Length;
            _currentLearningRate = _learningRateCurve.Evaluate(((float)_currentEpoch / (float)_epochs)) * _learningRate;
        }

        public Task<double> Score(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

        [Button]
        private void LoadLast()
        {
            trainedModel = ModelSerializer.LoadModel<AutoEncoderModel>("auto-encoder-mnist");
        }

        public void Cancel()
        {
            _epochSupervisor?.Cancel();
        }

    }
}
