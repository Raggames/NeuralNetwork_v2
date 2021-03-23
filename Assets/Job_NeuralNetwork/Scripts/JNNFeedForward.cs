using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using static Assets.Job_NeuralNetwork.Scripts.JNNMath;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class JNNFeedForward : MonoBehaviour
    {
        private JNNManager jnnManager;
        [Header("DONN Architecture")]
        /*public int InputLayerNeurons;
        public List<int> HiddenLayersNeurons;
        public int OutputLayerNeurons;*/

        [SerializeField]
        public JNNFeedForwardLayer InputLayer;
        public List<JNNFeedForwardLayer> HiddenLayers = new List<JNNFeedForwardLayer>();
        public JNNFeedForwardLayer OutputLayer;

        // Inputs

        public double[] networkInputs;

        // Input to Hidden / Hidden to Hidden
        public double[] hiddenLayerOutputs;

        public double[,] hiddenLayerWeights;
        public double[] hiddenLayerBias;

        public double[,] hiddenLayerPreviousWeightDelta;
        public double[] hiddenLayerPreviousBiasDelta;

        public double[] hiddenLayerGradients;

        // Hidden to Output

        public double[,] outputLayerWeights;
        public double[] outputLayerBias;

        public double[,] outputLayerPreviousWeightDelta;
        public double[] outputLayerPreviousBiasDelta;

        public double[] outputLayerGradients;


        private double[] networkOutputs;

        private double momentum;
        private double weightDecay;
        private double biasRate = 1f;


        [Header("DONN Rendering")]
        public int ScaleXY = 1;
        public int ScaleZ = 1;

        // JOB
        public JobHandle handle;


        // CREATING NETWORK ********************************************************************************************
        public void CreateNetwork(JNNManager manager)
        {
            jnnManager = manager;
            momentum = manager.Momentum;
            weightDecay = manager.WeightDecay;

            // Creating Arrays
            networkInputs = new double[InputLayer.NeuronsCount];

            hiddenLayerWeights = MakeMatrix(InputLayer.NeuronsCount, HiddenLayers[0].NeuronsCount);
            hiddenLayerBias = new double[HiddenLayers[0].NeuronsCount];

            hiddenLayerPreviousWeightDelta = MakeMatrix(InputLayer.NeuronsCount, HiddenLayers[0].NeuronsCount);
            hiddenLayerPreviousBiasDelta = new double[HiddenLayers[0].NeuronsCount];
            hiddenLayerGradients = new double[HiddenLayers[0].NeuronsCount];
            hiddenLayerOutputs = new double[HiddenLayers[0].NeuronsCount];


            outputLayerWeights = MakeMatrix(HiddenLayers[0].NeuronsCount, OutputLayer.NeuronsCount);
            outputLayerBias = new double[OutputLayer.NeuronsCount];

            outputLayerPreviousWeightDelta = MakeMatrix(HiddenLayers[0].NeuronsCount, OutputLayer.NeuronsCount);
            outputLayerPreviousBiasDelta = new double[OutputLayer.NeuronsCount];
            outputLayerGradients = new double[OutputLayer.NeuronsCount];
            networkOutputs = new double[HiddenLayers[0].NeuronsCount];

            // Initialize Random Weights

            for (int i = 0; i < hiddenLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hiddenLayerWeights.GetLength(1); ++j)
                {
                    hiddenLayerWeights[i, j] = GetRandomWeight();
                }
            }

            for(int i = 0; i < hiddenLayerBias.Length; ++i)
            {
                hiddenLayerBias[i] = GetRandomWeight();
            }

            for (int i = 0; i < outputLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < outputLayerWeights.GetLength(1); ++j)
                {
                    outputLayerWeights[i, j] = GetRandomWeight();
                }
            }

            for(int i = 0; i < outputLayerBias.Length; ++i)
            {
                outputLayerBias[i] = GetRandomWeight();
            }

            networkOutputs = new double[OutputLayer.NeuronsCount];
        }

        // EXECUTION ***************************************************************************************************
        #region Execution
        public void ComputeFeedForward(double[] inputs, out double[] results)
        {
            networkInputs = inputs;

            double[] hiddenSums = new double[HiddenLayers[0].NeuronsCount];
            double[] outputSums = new double[OutputLayer.NeuronsCount];

            for (int i = 0; i < hiddenLayerWeights.GetLength(1); ++i)
            {
                for (int j = 0; j < networkInputs.Length; ++j)
                {
                    hiddenSums[i] += inputs[j] * hiddenLayerWeights[j, i];
                }
                hiddenSums[i] += hiddenLayerBias[i];
                hiddenLayerOutputs[i] = JNNMath.ComputeActivation(HiddenLayers[0].ActivationFunction, false, hiddenSums[i]);
            }

            for (int i = 0; i < outputLayerWeights.GetLength(1); ++i)
            {
                for (int j = 0; j < hiddenLayerOutputs.Length; ++j)
                {
                    outputSums[i] += hiddenLayerOutputs[j] * outputLayerWeights[j, i];
                }
                outputSums[i] += outputLayerBias[i];

                if (OutputLayer.ActivationFunction != ActivationFunctions.Softmax)
                {
                    networkOutputs[i] = JNNMath.ComputeActivation(OutputLayer.ActivationFunction, false, outputSums[i]);  // Fonction de transformation ici;
                }
            }

            if (OutputLayer.ActivationFunction == ActivationFunctions.Softmax)
            {
                networkOutputs = Softmax(outputSums);
            }

            results = networkOutputs;
        }
        #endregion

        // WEIGHT SETTING **********************************************************************************************
        #region Weights

        private double[] dnaSave;
       
        public void LoadAndSetWeights()
        {
            NetworkData data = JNN_Load(saveName);

            dnaSave = data.dnaSave;
            int p = 0;
            for (int i = 0; i < hiddenLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hiddenLayerWeights.GetLength(1); ++j)
                {
                    hiddenLayerWeights[i, j] = dnaSave[p++];
                    
                }
            }
            for (int i = 0; i < hiddenLayerBias.Length; ++i)
            {
                hiddenLayerBias[i] = dnaSave[p++];
            }
            for (int i = 0; i < outputLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < outputLayerWeights.GetLength(1); ++j)
                {
                    outputLayerWeights[i, j] = dnaSave[p++];
                }
            }
            for (int i = 0; i < outputLayerBias.Length; ++i)
            {
                outputLayerBias[i] = dnaSave[p++];
            }

            jnnManager.LearningRate = (float)data.learningRate;
            jnnManager.Momentum = data.momentum;
            jnnManager.WeightDecay = data.weightDecay;
        }

        public void GetAndSaveWeights(double learningRate, double momentum, double weightDecay , double currentLoss, double accuracy)
        {
            int p = 0;
            int dnaLength = (hiddenLayerWeights.GetLength(0) * hiddenLayerWeights.GetLength(1)) + hiddenLayerBias.Length + (outputLayerWeights.GetLength(0) * outputLayerWeights.GetLength(1)) + outputLayerBias.Length;
            double[] weights = new double[dnaLength];

            for (int i = 0; i < hiddenLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hiddenLayerWeights.GetLength(1); ++j)
                {
                    weights[p++] = hiddenLayerWeights[i, j];
                }
            }
            for(int i = 0; i < hiddenLayerBias.Length; ++i)
            {
                weights[p++] = hiddenLayerBias[i];
            }
            for (int i = 0; i < outputLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < outputLayerWeights.GetLength(1); ++j)
                {
                    weights[p++] = outputLayerWeights[i, j];
                }
            }
            for(int i = 0; i < outputLayerBias.Length; ++i)
            {
                weights[p++] = outputLayerBias[i];
            }

            string version = "DNA_Architecture_" + InputLayer.NeuronsCount.ToString() + "_" + HiddenLayers[0].NeuronsCount.ToString() + "_" + OutputLayer.NeuronsCount.ToString() + "_cEpoch_" + jnnManager.currentEpoch.ToString();

            NetworkData data = new NetworkData
            {
                Version = version,
                dnaSave = weights,
                learningRate = learningRate,
                momentum = momentum,
                weightDecay = weightDecay,
                currentLoss = currentLoss,
                accuracy = accuracy,
            };
            JNN_Save(data);
        }

        #endregion

        // BACKPROPAGATION ********************************************************************************************
        #region BackPropagation
        public void BackPropagate(double[] costs, float learningRate)
        {
            for (int i = 0; i < outputLayerGradients.Length; ++i)
            {
                double derivative = JNNMath.ComputeActivation(OutputLayer.ActivationFunction, true, networkOutputs[i]);
                outputLayerGradients[i] = derivative * costs[i];
            }

            for (int i = 0; i < hiddenLayerGradients.Length; ++i)
            {
                double derivative = JNNMath.ComputeActivation(HiddenLayers[0].ActivationFunction, true, hiddenLayerOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < networkOutputs.Length; ++j)
                {
                    double x = outputLayerGradients[j] * outputLayerWeights[i, j];
                    sum += x;
                }
                hiddenLayerGradients[i] = derivative * sum;
            }

            for (int i = 0; i < hiddenLayerWeights.GetLength(0); ++i) 
            {
                for (int j = 0; j < hiddenLayerWeights.GetLength(1); ++j) 
                {
                    double delta = learningRate * hiddenLayerGradients[j] * networkInputs[i]; 
                    hiddenLayerWeights[i, j] += delta;
                    hiddenLayerWeights[i, j] += momentum * hiddenLayerPreviousWeightDelta[i, j];
                    hiddenLayerWeights[i, j] -= weightDecay * hiddenLayerWeights[i, j];
                    hiddenLayerWeights[i, j] = delta; 
                }
            }

            for (int i = 0; i < hiddenLayerBias.Length; ++i)
            {
                double delta = learningRate * hiddenLayerGradients[i] * biasRate;
                hiddenLayerBias[i] += delta;
                hiddenLayerBias[i] += momentum * hiddenLayerPreviousBiasDelta[i];
                hiddenLayerBias[i] -= weightDecay * hiddenLayerBias[i];                                                               
                hiddenLayerPreviousBiasDelta[i] = delta; 
            }

            for (int i = 0; i < outputLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < outputLayerWeights.GetLength(1); ++j)
                {
                    double delta = learningRate * outputLayerGradients[j] * hiddenLayerOutputs[i];
                    outputLayerWeights[i,j] += delta;
                    outputLayerWeights[i, j] += momentum * outputLayerPreviousWeightDelta[i,j];
                    outputLayerWeights[i, j] -= weightDecay * outputLayerWeights[i, j];                                               
                    outputLayerPreviousWeightDelta[i,j] = delta;
                }
            }

            for (int i = 0; i < outputLayerBias.Length; ++i)
            {
                double delta = learningRate * outputLayerGradients[i] * biasRate;
                outputLayerBias[i] += delta;
                outputLayerBias[i] += momentum * outputLayerPreviousBiasDelta[i]; 
                outputLayerBias[i] -= weightDecay * outputLayerBias[i];                                         
                outputLayerPreviousBiasDelta[i] = delta; 
            }
        } 
   
        #endregion
        

        // SERIALISATION **********************************************************************************************
        #region Serialisation

        public struct NetworkData
        {
            public string Version;

            public double learningRate;
            public double momentum;
            public double weightDecay;
            public double currentLoss;
            public double accuracy;

            public double[] dnaSave;
        }

        public string saveName;


        public string CreateSaveName(int version)
        {
            string newName = "";

            saveName = newName;
            return saveName;
        }

        private void JNN_Save(NetworkData data)
        {
            saveName = data.Version;
            // Serialize
            JNNSerializer.Save(data, data.Version);
        }

        private NetworkData JNN_Load(string fileName)
        {
            NetworkData loadedData = new NetworkData();
            loadedData = JNNSerializer.Load(loadedData, fileName);
            return loadedData;
        }

        #endregion

        // UTILS ******************************************************************************************************
        #region Utils
        public NativeArray<double> GetRandomValues()
        {
            NativeArray<double> inputs = new NativeArray<double>(InputLayer.NeuronsCount, Allocator.Persistent);
            for (int i = 0; i < inputs.Length; ++i)
            {
                inputs[i] = UnityEngine.Random.Range(0f, 10f);
            }
            return inputs;
        }

        public NativeArray<double> ToNativeArray(double[] arrayIn)
        {
            NativeArray<double> inputs = new NativeArray<double>(InputLayer.NeuronsCount, Allocator.Persistent);
            for (int i = 0; i < inputs.Length; ++i)
            {
                inputs[i] = arrayIn[i];
            }
            return inputs;
        }

        public double GetRandomWeight()
        {
            return UnityEngine.Random.Range(0.001f, 0.01f);
        }

        public static double[,] MakeMatrix(int x, int y)
        {
            double[,] matrix = new double[x, y];
            return matrix;
        }

        public static int SetID(int Layer, int index)
        {
            string ID = Layer.ToString() + index.ToString();
            int iD = int.Parse(ID);
            return iD;
        }
        #endregion
    }
}
