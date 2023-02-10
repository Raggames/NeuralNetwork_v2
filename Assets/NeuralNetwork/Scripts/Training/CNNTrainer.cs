using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace NeuralNetwork
{
    public class CNNTrainer : NeuralNetworkTrainer
    {
        [Header("----CNN Trainer----")]
        public Vector2Int ImageDimensions;

        public int Padding = 1;
        public int Stride = 1;

        public CNN NeuralNetwork;

        private WaitForSeconds delay;
        private Coroutine ExecutionCoroutine;

        [Header("----RUNTIME----")]

        /// <summary>
        /// Feedback on computation 'speed'
        /// </summary>
        [ReadOnly] public double epochs_per_second = 0;
        /// <summary>
        /// Avoid unity from freezing on compute large epoch values
        /// </summary>
        [ReadOnly] public float max_frame_time = 2;

        public double[][,] x_datas;
        public double[][] t_datas;

        public double[,] run_inputs;
        public double[] run_outputs;
        public double[] run_test_outputs;

        private void Start()
        {
            Initialize();
        }

        private void OnGUI()
        {
            if (GUI.Button(new Rect(10, 10, 100, 30), "Train"))
            {
                //PrepareTraining();

                ExecutionCoroutine = StartCoroutine(Train());
            }

            if (GUI.Button(new Rect(10, 100, 100, 30), "Load"))
            {
                //PrepareExecution();
            }

            if (GUI.Button(new Rect(10, 150, 100, 30), "Test"))
            {
                //ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
            }

            if (GUI.Button(new Rect(10, 50, 100, 30), "Save"))
            {
                SaveBestTrainingWeightSet(NeuralNetwork);
            }
        }

        public void Initialize()
        {
            TrainingSetting.Init();

            NeuralNetwork = new CNN();
            // Convolute from 28x28 input to 27x27 feature map
            ConvolutionLayer convolutionLayer = new ConvolutionLayer(ImageDimensions.x, ImageDimensions.y, Padding, Stride)
                .AddFilter(KernelType.Identity);
            convolutionLayer.Initialize();

            NeuralNetwork.CNNLayers.Add(convolutionLayer);

            // Pool from 27x27 to 13x13
            var poolingLayer = new PoolingLayer(convolutionLayer.OutputWidth, convolutionLayer.OutputHeight, 1, 2, Padding, PoolingRule.Max);
            NeuralNetwork.CNNLayers.Add(poolingLayer);
/*
            ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(poolingLayer.OutputWidth, poolingLayer.OutputHeight, Padding, Stride)
                .AddFilter(KernelType.Identity);
            convolutionLayer2.Initialize();

            NeuralNetwork.CNNLayers.Add(convolutionLayer2);

            var poolingLayer2 = new PoolingLayer(convolutionLayer2.OutputWidth, convolutionLayer2.OutputHeight, 1, 2, Padding, PoolingRule.Max);
            NeuralNetwork.CNNLayers.Add(poolingLayer2);
*/
            // Pooling layer matrix out is 13x13 for 1 filter = 169 neurons
            NeuralNetwork.FlattenLayer = new FlattenLayer(poolingLayer.OutputWidth, poolingLayer.OutputHeight, 1);

            NeuralNetwork.DenseLayers.Add(new DenseLayer(LayerType.DenseHidden, ActivationFunctions.ReLU, NeuralNetwork.FlattenLayer.NodeCount, NeuralNetwork.FlattenLayer.NodeCount / 2));
            NeuralNetwork.DenseLayers.Add(new DenseLayer(LayerType.Output, ActivationFunctions.Softmax, NeuralNetwork.FlattenLayer.NodeCount, 10));

            NeuralNetwork.Initialize(this, null); 
            NeuralNetwork.InitializeWeights();
        }

        private IEnumerator Train()
        {
            CurrentEpoch = 0;

            double current_time = 0;

            // Get training datas from the setting
            TrainingSetting.GetMatrixTrainDatas(out x_datas, out t_datas);

            // Compute number of iterations 
            // Batchsize shouldn't be 0
            int iterations_per_epoch = x_datas.Length / BatchSize;

            int[] sequence_indexes = new int[x_datas.Length];
            for (int i = 0; i < sequence_indexes.Length; ++i)
                sequence_indexes[i] = i;

            for (int i = 0; i < Epochs; ++i)
            {
                int dataIndex = 0;

                // Shuffle datas each epoch
                Shuffle(sequence_indexes);

                // Going through all data batched, mini-batch or stochastic, depending on BatchSize value
                double mean_error_sum = 0;
                for (int d = 0; d < iterations_per_epoch; ++d)
                {
                    for (int j = 0; j < BatchSize; ++j)
                    {
                        run_inputs = x_datas[sequence_indexes[dataIndex]];
                        run_test_outputs = t_datas[sequence_indexes[dataIndex]];

                        run_outputs = NeuralNetwork.ComputeForward(run_inputs);

                        NeuralNetwork.ComputeGradients(run_test_outputs, run_outputs);

                        ComputeAccuracy(run_test_outputs, run_outputs);

                        mean_error_sum += GetLoss(run_outputs, run_test_outputs);
                        dataIndex++;

                        current_time += Time.deltaTime;
                        if (current_time > max_frame_time)
                        {
                            current_time = 0;
                            yield return null;
                        }
                    }

                    // Computing gradients average over batchsize
                    NeuralNetwork.MeanGradients(BatchSize);

                    // NeuralNetwork.MeanGradients
                    // Computing new weights and reseting gradients to 0 for next batch
                    NeuralNetwork.UpdateWeights(LearningRate, Momentum, WeightDecay, BiasRate);
                }

                double last_mean_error = Current_Mean_Error;
                // Computing the mean error
                Current_Mean_Error = mean_error_sum / x_datas.Length;

               /* if (Current_Mean_Error < last_mean_error)
                {
                    // Keeping a trace of the set
                    MemorizeBestSet(NeuralNetwork, Current_Mean_Error);
                }*/

                // If under target error, stop
                if (Current_Mean_Error < Target_Mean_Error)
                {
                    break;
                }

                //DecayLearningRate();

                CurrentEpoch++;
                epochs_per_second = CurrentEpoch / Time.realtimeSinceStartup;
            }
        }
    }
}
