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

        /*   JNNNeuron[] inputLayer;
           List<JNNNeuron[]> hiddenLayers;
           JNNNeuron[] outputLayer;*/

        public double[,] inputLayerNeuronInput;
        public double[,] inputLayerWeights;
        public double[,] inputLayerPreviousDelta;
        public double[,] inputLayerGradients;
        public double[] inputLayerBias;


        public double[,] hiddenLayerNeuronInput;
        public double[,] hiddenLayerWeights;
        public double[,] hiddenLayerPreviousDelta;
        public double[,] hiddenLayerGradients;
        public double[] hiddenLayerBias;

        public double[,] outputLayerNeuronInput;
        public double[,] outputLayerWeights;
        public double[,] outputLayerPreviousDelta;
        public double[,] outputLayerGradients;
        public double[] outputLayerBias;


        public GameObject neuronPrefab;

        private double momentum;
        private double weightDecay;

        [Header("DONN Rendering")]
        public int ScaleXY = 1;
        public int ScaleZ = 1;

        // JOB
        public JobHandle handle;

        private double[] networkOutputs;

        // CREATING NETWORK ********************************************************************************************
        public void CreateNetwork(JNNManager manager)
        {
            jnnManager = manager;
            momentum = manager.Momentum;
            weightDecay = manager.WeightDecay;

            // Creating Arrays
            inputLayerNeuronInput = new double[InputLayer.NeuronsCount, 1];
            inputLayerWeights = new double[InputLayer.NeuronsCount, HiddenLayers[0].NeuronsCount];
            inputLayerPreviousDelta = new double[InputLayer.NeuronsCount, HiddenLayers[0].NeuronsCount];
            inputLayerGradients = new double[InputLayer.NeuronsCount, 1];
            inputLayerBias = new double[InputLayer.NeuronsCount];

            hiddenLayerNeuronInput = new double[HiddenLayers[0].NeuronsCount, InputLayer.NeuronsCount];
            hiddenLayerWeights = new double[HiddenLayers[0].NeuronsCount, OutputLayer.NeuronsCount];
            hiddenLayerPreviousDelta = new double[HiddenLayers[0].NeuronsCount, OutputLayer.NeuronsCount];
            hiddenLayerGradients = new double[HiddenLayers[0].NeuronsCount, 1];
            hiddenLayerBias = new double[HiddenLayers[0].NeuronsCount];

            outputLayerNeuronInput = new double[OutputLayer.NeuronsCount, HiddenLayers[0].NeuronsCount];
            outputLayerWeights = new double[OutputLayer.NeuronsCount, 1];
            outputLayerPreviousDelta = new double[OutputLayer.NeuronsCount, 1];
            outputLayerGradients = new double[OutputLayer.NeuronsCount, 1];
            outputLayerBias = new double[OutputLayer.NeuronsCount];

            // Initialize Random Weights

            for (int i = 0; i < inputLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < inputLayerWeights.GetLength(1); ++j)
                {
                    inputLayerWeights[i, j] = GetRandomWeight();
                }
                inputLayerBias[i] = GetRandomWeight();
            }

            for (int i = 0; i < hiddenLayerWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hiddenLayerWeights.GetLength(1); ++j)
                {
                    hiddenLayerWeights[i, j] = GetRandomWeight();
                }
                hiddenLayerBias[i] = GetRandomWeight();
            }

            for (int i = 0; i < outputLayerWeights.GetLength(0); ++i)
            {
                outputLayerBias[i] = GetRandomWeight();
                outputLayerWeights[i, 0] = GetRandomWeight();
            }

            networkOutputs = new double[OutputLayer.NeuronsCount];
        }

        // EXECUTION ***************************************************************************************************
        #region Execution
        public void ComputeFeedForward(double[] inputs, out double[] results)
        {
            for(int i = 0; i < inputLayerNeuronInput.GetLength(0); ++i)
            {
                inputLayerNeuronInput[i, 0] = inputs[i];
            }

            for (int i = 0; i < inputLayerWeights.GetLength(0); ++i)
            {
                // i = NeuronIndex in layer 
                for (int j = 0; j < inputLayerWeights.GetLength(1); ++j)
                {
                    double valueToHidden = inputLayerNeuronInput[i, 0] * inputLayerWeights[i, j];

                    hiddenLayerNeuronInput[j, i] = valueToHidden;
                }
            }

            for (int i = 0; i < hiddenLayerNeuronInput.GetLength(0); ++i)
            {
                double signal = 0f;
                for (int j = 0; j < hiddenLayerNeuronInput.GetLength(1); ++j)
                {
                    signal += hiddenLayerNeuronInput[i, j];
                }
                signal += hiddenLayerBias[i];
                signal /= hiddenLayerNeuronInput.GetLength(0);

                signal = JNNMath.ComputeActivation(HiddenLayers[0].ActivationFunction, false, signal);  // Fonction de transformation ici;

                // pour chaque neurone, signal calculé 

                for (int j = 0; j < hiddenLayerWeights.GetLength(1); ++j) // cette dimension est égale au neurones d'output
                {
                    double value = signal * hiddenLayerWeights[i, j];
                    outputLayerNeuronInput[j, i] = value;
                }
            }

            for (int i = 0; i < outputLayerNeuronInput.GetLength(0); ++i)
            {
                double signal = 0f;
                for (int j = 0; j < outputLayerNeuronInput.GetLength(1); ++j)
                {
                    signal += outputLayerNeuronInput[i, j];
                }
                signal += outputLayerBias[i];
                signal /= outputLayerNeuronInput.GetLength(0);

                signal = JNNMath.ComputeActivation(HiddenLayers[0].ActivationFunction, false, signal);  // Fonction de transformation ici;

                for (int j = 0; j < outputLayerWeights.GetLength(1); ++j) // cette dimension est égale au neurones d'output
                {
                    double value = signal * hiddenLayerWeights[i, j];
                    networkOutputs[i] = value;
                }
            }
            results = networkOutputs;
        }

        public void ComputeLayer(double[,] layer, double[] inputs, double[] bias, JNNFeedForwardLayer layerData, int nextLayerNeuronsCount, bool lastLayer = false)
        {

           


            #region oldstuff
            // On créer le job de traitement de la couche et on y passe sa data.
            /* switch (layerData.ActivationFunction)
             {
                 case ActivationFunctions.AxonsLinear:
                     AxonBasedLinearJob axonLinearJob = new AxonBasedLinearJob
                     {
                         Weights = flattenedWeights,
                         Bias = bias,
                         Inputs = inputs,
                         Outputs = outputs,
                     };
                     handle = axonLinearJob.Schedule(flattenedWeights.Length, layer.GetLength(1));

                     handle.Complete();
                     break;*/
            /* case ActivationFunctions.Linear:
                 LinearJob linearJob = new LinearJob
                 {
                     dataArray = neuronsData,
                     inputs = inputs,
                     outputs = outputs,
                 };

                 // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                 handle = linearJob.Schedule(neuronsData.Length, 1);

                 handle.Complete();

                 break;
             case ActivationFunctions.Sigmoid:
                 SigmoidJob sigmoidJob = new SigmoidJob
                 {
                     dataArray = neuronsData,
                     inputs = inputs,
                     outputs = outputs,
                 };

                 // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                 handle = sigmoidJob.Schedule(neuronsData.Length, 1);

                 handle.Complete();
                 break;

             case ActivationFunctions.Tanh:
                 TanhJob tanhJob = new TanhJob
                 {
                     dataArray = neuronsData,
                     inputs = inputs,
                     outputs = outputs,
                 };

                 // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                 handle = tanhJob.Schedule(neuronsData.Length, 1);

                 handle.Complete();
                 break;

             case ActivationFunctions.Sinusoid:
                 SinusoidJob sinusoidJob = new SinusoidJob
                 {
                     dataArray = neuronsData,
                     inputs = inputs,
                     outputs = outputs,
                 };

                 // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                 handle = sinusoidJob.Schedule(neuronsData.Length, 1);

                 handle.Complete();
                 break;

             case ActivationFunctions.ReLU:
                 ReLUJob reLUJob = new ReLUJob
                 {
                     dataArray = neuronsData,
                     inputs = inputs,
                     outputs = outputs,
                 };

                 // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                 handle = reLUJob.Schedule(neuronsData.Length, 1);

                 handle.Complete();
                 break;
             case ActivationFunctions.PReLU:
                 PReLUJob PreLUJob = new PReLUJob
                 {
                     dataArray = neuronsData,
                     inputs = inputs,
                     outputs = outputs,
                 };

                 // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                 handle = PreLUJob.Schedule(neuronsData.Length, 1);

                 handle.Complete();
                 break;
             case ActivationFunctions.Softmax:
                 LinearJob beforeSoftMax = new LinearJob
                 {
                     dataArray = neuronsData,
                     inputs = inputs,
                     outputs = outputs,
                 };

                 // On Schedule le Job. On peux utiliser => layerJob.Run(neuronsData.Length); pour traiter le Job en mainthread et debugger.
                 handle = beforeSoftMax.Schedule(neuronsData.Length, 1);

                 handle.Complete();

                 double[] results = JNNMath.Softmax(outputs.ToArray());
                 for(int i = 0; i < results.Length; ++i)
                 {
                     outputs[i] = results[i];
                 }

                 break;

          }

         // On sort les données de la couche
         outputData = outputs;

         for(int i = 0; i < layer.Length; ++i)
         {
             if(outputData[i] == 0)
             {
                 Debug.LogError(0);
             }
             layer[i].Output = outputData[i];
         }

         // Utilisation des arrays terminée, on rend la mémoire allouée.
         neuronsData.Dispose();
         inputs.Dispose();

         // Le run est fini, les données ont traversé toutes les couches. On sort les résultats et on Dispose() les outputs.
         if (lastLayer)
         {
             networkOutputs = new double[outputs.Length];

             for (int i = 0; i < outputs.Length; ++i)
             {
                 networkOutputs[i] = outputs[i];
             }

             outputs.Dispose();

            
         }
            */
            #endregion

        }


        // WEIGHT SETTING **********************************************************************************************
        #region Weights
        public void SetWeights()
        {

        }

        public void GetWeights()
        {

        }
        #endregion

        // BACKPROPAGATION ********************************************************************************************
        #region BackPropagation
        public void BackPropagate(double[] costs, float learningRate)
        {

            // Computing gradient descent
            for(int i = 0; i < outputLayerWeights.GetLength(0); ++i)
            {
                double gradient = costs[i] * outputLayerWeights[i, 0];
                double delta = gradient * learningRate;

                outputLayerWeights[i, 0] += delta;
                outputLayerWeights[i, 0] += outputLayerPreviousDelta[i, 0] * momentum;
                outputLayerWeights[i, 0] -= weightDecay * outputLayerWeights[i, 0];
                outputLayerPreviousDelta[i, 0] = delta;

                outputLayerBias[i] += delta;
                outputLayerBias[i] += outputLayerPreviousDelta[i, 0] * momentum;
                outputLayerBias[i] -= weightDecay * outputLayerWeights[i, 0];

                double derivative = JNNMath.ComputeActivation(OutputLayer.ActivationFunction, true, gradient);

                outputLayerGradients[i, 0] = derivative;
            }

            for(int i = 0; i < hiddenLayerWeights.GetLength(0); ++i)
            {
                double gradientSignal = 0f;
                for(int j = 0; j < hiddenLayerWeights.GetLength(1); ++j)
                {

                    double gradient = outputLayerGradients[j, 0] * hiddenLayerWeights[i, j];
                    double delta = gradient * learningRate;

                    hiddenLayerWeights[i, j] += delta;
                    hiddenLayerWeights[i, j] += hiddenLayerPreviousDelta[i, j] * momentum;
                    hiddenLayerWeights[i, j] -= weightDecay * hiddenLayerWeights[i, j];
                    hiddenLayerPreviousDelta[i, j] = delta;

                    gradientSignal += gradient; // or delta ?
                }
                gradientSignal /= hiddenLayerWeights.GetLength(1);

                double derivative = JNNMath.ComputeActivation(HiddenLayers[0].ActivationFunction, true, gradientSignal);

                hiddenLayerGradients[i, 0] = derivative; 
            }

            // Now calling weight and bias update on network
            UpdateWeights(learningRate);
        }
        
        public void UpdateWeights(float learningRate)
        {

           /* // Update Output Bias
            for (int i = 0; i < outputLayer.Length; ++i)
            {
                double delta = learningRate * outputLayer[i].grad;
                outputLayer[i].Bias += delta;
                outputLayer[i].Bias += outputLayer[i].PreviousDelta * momentum;
                outputLayer[i].Bias -= weightDecay * outputLayer[i].Bias;
                outputLayer[i].PreviousDelta = delta;
            }

            // Update Output Weight
            for (int i = 0; i < outputLayer.Length; ++i)
            {
                double delta = learningRate * outputLayer[i].grad * outputLayer[i].Weight;
                outputLayer[i].Weight += delta;
                outputLayer[i].Weight += outputLayer[i].PreviousDelta * momentum;
                outputLayer[i].Weight -= weightDecay * outputLayer[i].Weight;
                outputLayer[i].PreviousDelta = delta;
            }
            

            // Update Hidden Weights
            for (int i = 0; i < hiddenLayers.Count; ++i)
            {
                for (int k = 0; k < hiddenLayers[i].Length; ++k)
                {
                    double delta = learningRate * hiddenLayers[i][k].grad * hiddenLayers[i][k].Weight;
                    hiddenLayers[i][k].Weight += delta;
                    hiddenLayers[i][k].Weight += hiddenLayers[i][k].PreviousDelta * momentum;
                    hiddenLayers[i][k].Weight -= weightDecay * hiddenLayers[i][k].Weight;
                    hiddenLayers[i][k].PreviousDelta = delta;
                }
            }

            // Update Hidden Bias
            for (int i = 0; i < hiddenLayers.Count; ++i)
            {
                for (int k = 0; k < hiddenLayers[i].Length; ++k)
                {
                    double delta = learningRate * hiddenLayers[i][k].grad;
                    hiddenLayers[i][k].Bias += delta;
                    hiddenLayers[i][k].Bias += hiddenLayers[i][k].PreviousDelta * momentum;
                    hiddenLayers[i][k].Bias -= weightDecay * hiddenLayers[i][k].Bias;
                    hiddenLayers[i][k].PreviousDelta = delta;
                }
            }*/
        }
        #endregion
        #endregion

        // SERIALISATION **********************************************************************************************
        #region Serialisation

        public struct NetworkData
        {
            public int Version;

            public double[] weights;
            public double[] bias;
            public double[] momentum;

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
          
            // Serialize
            JNNSerializer.Save(data, CreateSaveName(data.Version));
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
            return UnityEngine.Random.Range(-0.1f, 0.1f);
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
